"""Pyro variational inference for the paper's separated rho/nu phases.

This module follows Algorithm 1 instead of optimizing mean and variance inside
one client call.  The server first collects local precision coordinates
``rho_{t,k}``, aggregates and broadcasts ``rho_{t+1}``, and only then starts a
second client call in which each client optimizes ``nu_{t,k}`` using the newly
broadcast covariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import SGD
from torch.func import functional_call
from torch.utils.data import DataLoader

from config import ModelConfig, TrainingConfig
from serialization import ParameterLayout


@dataclass
class PrecisionPhaseResult:
    precision: np.ndarray
    average_loss: float
    local_steps: int


@dataclass
class NaturalMeanPhaseResult:
    nu: np.ndarray
    implied_mean: np.ndarray
    average_loss: float
    local_steps: int


def _safe_precision(
    value: torch.Tensor,
    model_cfg: ModelConfig,
) -> torch.Tensor:
    return value.clamp(
        min=float(model_cfg.min_precision),
        max=float(model_cfg.max_precision),
    )


def _precision_from_log_parameter(
    log_precision: torch.Tensor,
    model_cfg: ModelConfig,
) -> torch.Tensor:
    """Positive precision parameterization used by Pyro's optimizer.

    Algorithm 1 writes SGD directly in rho.  In software, an unconstrained
    log-rho parameter is used so every SVI iterate remains a valid Gaussian.
    The returned quantity and all transmitted/aggregated values are rho.
    """
    lower = math.log(float(model_cfg.min_precision))
    upper = math.log(float(model_cfg.max_precision))
    return torch.exp(torch.clamp(log_precision, min=lower, max=upper))


class BayesianVITrainer:
    """Execute one local phase of Algorithm 1 with Pyro SVI."""

    def __init__(
        self,
        model: torch.nn.Module,
        layout: ParameterLayout,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.layout = layout
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device

    def train_precision_phase(
        self,
        *,
        global_mean: np.ndarray,
        global_precision: np.ndarray,
        loader: DataLoader,
        seed: int,
    ) -> PrecisionPhaseResult:
        """Optimize rho_{t,k} while keeping the mean fixed at mu_t.

        The guide is ``N(mu_t, diag(rho_{t,k})^{-1})`` and the model prior is
        ``N(mu_t, diag(rho_t)^{-1})``.  This implements Eqs. (23)-(25).
        """
        mean = torch.as_tensor(
            global_mean, dtype=torch.float32, device=self.device
        ).reshape(-1)
        prior_precision = _safe_precision(
            torch.as_tensor(
                global_precision, dtype=torch.float32, device=self.device
            ).reshape(-1),
            self.model_cfg,
        )
        self._validate_vector(mean, "global_mean")
        self._validate_vector(prior_precision, "global_precision")

        pyro.clear_param_store()
        pyro.set_rng_seed(int(seed))
        torch.manual_seed(int(seed))

        mean_named = self.layout.vector_to_named(mean)
        prior_precision_named = self.layout.vector_to_named(prior_precision)

        def model_fn(x: torch.Tensor, y: torch.Tensor | None = None) -> None:
            sampled: Dict[str, torch.Tensor] = {}
            with poutine.scale(scale=float(self.train_cfg.kl_weight)):
                for spec in self.layout.specs:
                    site = self.layout.site_name(spec.name)
                    prior_std = torch.rsqrt(prior_precision_named[spec.name])
                    sampled[spec.name] = pyro.sample(
                        site,
                        dist.Normal(
                            mean_named[spec.name],
                            prior_std,
                        ).to_event(len(spec.shape)),
                    )
            logits = functional_call(self.model, sampled, (x,))
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)

        def guide_fn(x: torch.Tensor, y: torch.Tensor | None = None) -> None:
            del x, y
            with poutine.scale(scale=float(self.train_cfg.kl_weight)):
                for spec in self.layout.specs:
                    site = self.layout.site_name(spec.name)
                    initial_log_rho = torch.log(prior_precision_named[spec.name])
                    log_rho = pyro.param(
                        f"{site}__log_precision",
                        initial_log_rho.clone(),
                    )
                    rho = _precision_from_log_parameter(log_rho, self.model_cfg)
                    pyro.sample(
                        site,
                        dist.Normal(
                            mean_named[spec.name],
                            torch.rsqrt(rho),
                        ).to_event(len(spec.shape)),
                    )

        average_loss, local_steps = self._run_svi(model_fn, guide_fn, loader)

        posterior_precision: Dict[str, torch.Tensor] = {}
        for spec in self.layout.specs:
            site = self.layout.site_name(spec.name)
            posterior_precision[spec.name] = _precision_from_log_parameter(
                pyro.param(f"{site}__log_precision"), self.model_cfg
            ).detach()
        precision_vector = self.layout.named_to_vector(posterior_precision)
        result = PrecisionPhaseResult(
            precision=precision_vector.cpu().numpy().astype(np.float32),
            average_loss=float(average_loss),
            local_steps=int(local_steps),
        )
        pyro.clear_param_store()
        return result

    def train_natural_mean_phase(
        self,
        *,
        global_mean: np.ndarray,
        next_global_precision: np.ndarray,
        local_precision: np.ndarray,
        loader: DataLoader,
        seed: int,
    ) -> NaturalMeanPhaseResult:
        """Optimize nu_{t,k} after the server broadcasts rho_{t+1}.

        Eqs. (33)-(35) are implemented element-wise for diagonal covariance:

        ``nu_init = (rho_local / rho_global_next) * mu_t``

        ``mu_local(nu) = (rho_global_next / rho_local) * nu``

        The phase-2 variational covariance is the newly aggregated global
        covariance ``diag(rho_{t+1})^{-1}``, exactly as stated around Eq. (34).
        """
        mean = torch.as_tensor(
            global_mean, dtype=torch.float32, device=self.device
        ).reshape(-1)
        next_rho = _safe_precision(
            torch.as_tensor(
                next_global_precision, dtype=torch.float32, device=self.device
            ).reshape(-1),
            self.model_cfg,
        )
        local_rho = _safe_precision(
            torch.as_tensor(
                local_precision, dtype=torch.float32, device=self.device
            ).reshape(-1),
            self.model_cfg,
        )
        self._validate_vector(mean, "global_mean")
        self._validate_vector(next_rho, "next_global_precision")
        self._validate_vector(local_rho, "local_precision")

        pyro.clear_param_store()
        pyro.set_rng_seed(int(seed))
        torch.manual_seed(int(seed))

        mean_named = self.layout.vector_to_named(mean)
        next_rho_named = self.layout.vector_to_named(next_rho)
        local_rho_named = self.layout.vector_to_named(local_rho)

        def model_fn(x: torch.Tensor, y: torch.Tensor | None = None) -> None:
            sampled: Dict[str, torch.Tensor] = {}
            with poutine.scale(scale=float(self.train_cfg.kl_weight)):
                for spec in self.layout.specs:
                    site = self.layout.site_name(spec.name)
                    global_std = torch.rsqrt(next_rho_named[spec.name])
                    sampled[spec.name] = pyro.sample(
                        site,
                        dist.Normal(
                            mean_named[spec.name],
                            global_std,
                        ).to_event(len(spec.shape)),
                    )
            logits = functional_call(self.model, sampled, (x,))
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)

        def guide_fn(x: torch.Tensor, y: torch.Tensor | None = None) -> None:
            del x, y
            with poutine.scale(scale=float(self.train_cfg.kl_weight)):
                for spec in self.layout.specs:
                    site = self.layout.site_name(spec.name)
                    init_nu = (
                        local_rho_named[spec.name]
                        / next_rho_named[spec.name]
                        * mean_named[spec.name]
                    )
                    nu = pyro.param(f"{site}__nu", init_nu.clone())
                    implied_mean = (
                        next_rho_named[spec.name]
                        / local_rho_named[spec.name]
                        * nu
                    )
                    global_std = torch.rsqrt(next_rho_named[spec.name])
                    pyro.sample(
                        site,
                        dist.Normal(implied_mean, global_std).to_event(
                            len(spec.shape)
                        ),
                    )

        average_loss, local_steps = self._run_svi(model_fn, guide_fn, loader)

        posterior_nu: Dict[str, torch.Tensor] = {}
        implied_mean_named: Dict[str, torch.Tensor] = {}
        for spec in self.layout.specs:
            site = self.layout.site_name(spec.name)
            nu = pyro.param(f"{site}__nu").detach()
            posterior_nu[spec.name] = nu
            implied_mean_named[spec.name] = (
                next_rho_named[spec.name]
                / local_rho_named[spec.name]
                * nu
            ).detach()

        nu_vector = self.layout.named_to_vector(posterior_nu)
        implied_mean_vector = self.layout.named_to_vector(implied_mean_named)
        result = NaturalMeanPhaseResult(
            nu=nu_vector.cpu().numpy().astype(np.float32),
            implied_mean=implied_mean_vector.cpu().numpy().astype(np.float32),
            average_loss=float(average_loss),
            local_steps=int(local_steps),
        )
        pyro.clear_param_store()
        return result

    def _run_svi(
        self,
        model_fn,
        guide_fn,
        loader: DataLoader,
    ) -> Tuple[float, int]:
        optimizer = SGD(
            {"lr": float(self.train_cfg.learning_rate)},
            clip_args={"clip_norm": float(self.train_cfg.gradient_clip_norm)},
        )
        svi = SVI(
            model_fn,
            guide_fn,
            optimizer,
            loss=TraceMeanField_ELBO(
                num_particles=int(self.train_cfg.mc_train_samples),
                vectorize_particles=False,
            ),
        )

        total_loss = 0.0
        total_examples = 0
        local_steps = 0
        non_blocking = bool(self.device.type == "cuda" and loader.pin_memory)
        for _ in range(int(self.train_cfg.local_epochs)):
            for features, targets in loader:
                features = features.to(self.device, non_blocking=non_blocking)
                targets = targets.to(self.device, non_blocking=non_blocking)
                batch_loss = float(svi.step(features, targets))
                total_loss += batch_loss
                total_examples += int(targets.numel())
                local_steps += 1
        return total_loss / max(1, total_examples), local_steps

    def _validate_vector(self, value: torch.Tensor, name: str) -> None:
        if value.numel() != self.layout.total_numel:
            raise ValueError(
                f"{name} contains {value.numel()} values; "
                f"expected {self.layout.total_numel}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
