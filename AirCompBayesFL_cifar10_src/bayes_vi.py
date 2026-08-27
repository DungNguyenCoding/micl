"""Pyro variational inference for the paper's separated rho/nu phases.

Version 1.3.1 keeps the two server-controlled phases introduced in v1.3.0,
but corrects two mathematical details:

1. ``rho`` is optimized directly, as written in Eq. (25), instead of updating
   an unconstrained ``log(rho)`` coordinate.
2. In phase 2, the variational covariance is the newly aggregated
   ``Sigma_{t+1}``, while the KL prior remains the round-start global posterior
   ``q_{theta_t}``, as required by Eqs. (13), (15), and (35).

Pyro still supplies the probabilistic model, guide, Monte-Carlo ELBO, and
reparameterized Gaussian samples. PyTorch SGD is used on the explicit ``rho``
and ``nu`` tensors so the optimized coordinates match Algorithm 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import TraceMeanField_ELBO
from torch.func import functional_call
from torch.utils.data import DataLoader

from config import ModelConfig, TrainingConfig
from serialization import ParameterLayout


@dataclass
class PrecisionPhaseResult:
    precision: np.ndarray
    average_loss: float
    local_steps: int
    precision_delta_l2: float
    precision_delta_max_abs: float
    precision_changed_fraction: float
    applied_gradient_l2_mean: float
    applied_gradient_max_abs: float


@dataclass
class NaturalMeanPhaseResult:
    nu: np.ndarray
    implied_mean: np.ndarray
    average_loss: float
    local_steps: int


def _safe_precision(value: torch.Tensor, model_cfg: ModelConfig) -> torch.Tensor:
    """Return a finite positive precision in the configured interval."""
    return value.clamp(
        min=float(model_cfg.min_precision),
        max=float(model_cfg.max_precision),
    )


class BayesianVITrainer:
    """Execute one local phase of Algorithm 1 with Pyro's differentiable ELBO."""

    def __init__(
        self,
        model: torch.nn.Module,
        layout: ParameterLayout,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        device: torch.device,
        learning_rate: float | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        # ``functional_call`` supplies sampled weights. The module's own stored
        # parameters are not optimization variables in Bayesian local training.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.layout = layout
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device
        self.learning_rate = (
            float(train_cfg.learning_rate)
            if learning_rate is None
            else float(learning_rate)
        )

    def train_precision_phase(
        self,
        *,
        global_mean: np.ndarray,
        global_precision: np.ndarray,
        loader: DataLoader,
        seed: int,
    ) -> PrecisionPhaseResult:
        """Optimize ``rho_{t,k}`` with ``mu_t`` fixed; Eqs. (23)-(25).

        The prior is ``N(mu_t, diag(rho_t)^-1)`` and the guide is
        ``N(mu_t, diag(rho_{t,k})^-1)``. The trainable tensor is ``rho`` itself,
        not ``log(rho)``, so one optimizer step is the software equivalent of
        ``rho <- rho - eta * grad_rho L_k`` followed by a positivity projection.
        """
        # Keep the variational distribution and the rho master coordinate in
        # float64.  At rho=400, one float32 ULP is about 3e-5, while the direct
        # Eq. (25) update can be much smaller.  A float32 rho therefore appears
        # exactly frozen even when the ELBO gradient is non-zero.  Samples are
        # cast to the model dtype immediately before the CNN forward pass.
        mean = torch.as_tensor(
            global_mean, dtype=torch.float64, device=self.device
        ).reshape(-1)
        prior_precision = _safe_precision(
            torch.as_tensor(
                global_precision, dtype=torch.float64, device=self.device
            ).reshape(-1),
            self.model_cfg,
        )
        self._validate_vector(mean, "global_mean")
        self._validate_vector(prior_precision, "global_precision")

        self._seed(seed)
        mean_named = self.layout.vector_to_named(mean)
        prior_precision_named = self.layout.vector_to_named(prior_precision)

        initial_rho = prior_precision.detach().clone()
        rho = torch.nn.Parameter(initial_rho.clone())

        def model_fn(
            x: torch.Tensor,
            y: torch.Tensor | None = None,
            likelihood_scale: float = 1.0,
        ) -> None:
            sampled: Dict[str, torch.Tensor] = {}
            with poutine.scale(scale=float(self.train_cfg.kl_weight)):
                for spec in self.layout.specs:
                    site = self.layout.site_name(spec.name)
                    sampled_weight = pyro.sample(
                        site,
                        dist.Normal(
                            mean_named[spec.name],
                            torch.rsqrt(prior_precision_named[spec.name]),
                        ).to_event(len(spec.shape)),
                    )
                    sampled[spec.name] = sampled_weight.to(dtype=torch.float32)
            logits = functional_call(self.model, sampled, (x,))
            with poutine.scale(scale=float(likelihood_scale)):
                with pyro.plate("data", x.shape[0]):
                    pyro.sample("obs", dist.Categorical(logits=logits), obs=y)

        def guide_fn(
            x: torch.Tensor,
            y: torch.Tensor | None = None,
            likelihood_scale: float = 1.0,
        ) -> None:
            del x, y, likelihood_scale
            rho_named = self.layout.vector_to_named(
                _safe_precision(rho, self.model_cfg)
            )
            with poutine.scale(scale=float(self.train_cfg.kl_weight)):
                for spec in self.layout.specs:
                    site = self.layout.site_name(spec.name)
                    pyro.sample(
                        site,
                        dist.Normal(
                            mean_named[spec.name],
                            torch.rsqrt(rho_named[spec.name]),
                        ).to_event(len(spec.shape)),
                    )

        def project_precision() -> None:
            with torch.no_grad():
                rho.clamp_(
                    min=float(self.model_cfg.min_precision),
                    max=float(self.model_cfg.max_precision),
                )

        (
            average_loss,
            local_steps,
            applied_gradient_l2_mean,
            applied_gradient_max_abs,
        ) = self._run_coordinate_sgd(
            model_fn=model_fn,
            guide_fn=guide_fn,
            loader=loader,
            parameters=[rho],
            after_step=project_precision,
        )
        precision = _safe_precision(rho.detach(), self.model_cfg)
        delta = precision - initial_rho
        result = PrecisionPhaseResult(
            # Keep rho in float64 across client state, server aggregation, and
            # AirComp.  Casting here was the second source of the frozen-rho bug.
            precision=precision.cpu().numpy().astype(np.float64),
            average_loss=float(average_loss),
            local_steps=int(local_steps),
            precision_delta_l2=float(torch.linalg.vector_norm(delta).cpu()),
            precision_delta_max_abs=float(torch.max(torch.abs(delta)).cpu()),
            precision_changed_fraction=float(
                torch.count_nonzero(delta).cpu() / max(1, delta.numel())
            ),
            applied_gradient_l2_mean=float(applied_gradient_l2_mean),
            applied_gradient_max_abs=float(applied_gradient_max_abs),
        )
        pyro.clear_param_store()
        return result

    def train_natural_mean_phase(
        self,
        *,
        global_mean: np.ndarray,
        prior_global_precision: np.ndarray,
        next_global_precision: np.ndarray,
        local_precision: np.ndarray,
        loader: DataLoader,
        seed: int,
    ) -> NaturalMeanPhaseResult:
        """Optimize ``nu_{t,k}`` after the server broadcasts ``rho_{t+1}``.

        Coordinate transforms for diagonal covariance are:

        ``nu_init = (rho_local / rho_next) * mu_t``

        ``mu_local(nu) = (rho_next / rho_local) * nu``

        The guide covariance is ``diag(rho_next)^-1``. The KL prior is the
        round-start global posterior ``N(mu_t, diag(rho_t)^-1)``; therefore the
        function receives both ``prior_global_precision`` and
        ``next_global_precision``.
        """
        # Distribution parameters use float64 so the phase-1 precision update
        # is not rounded away before phase 2.  The actual CNN forward still uses
        # float32 sampled weights.
        mean = torch.as_tensor(
            global_mean, dtype=torch.float64, device=self.device
        ).reshape(-1)
        prior_rho = _safe_precision(
            torch.as_tensor(
                prior_global_precision, dtype=torch.float64, device=self.device
            ).reshape(-1),
            self.model_cfg,
        )
        next_rho = _safe_precision(
            torch.as_tensor(
                next_global_precision, dtype=torch.float64, device=self.device
            ).reshape(-1),
            self.model_cfg,
        )
        local_rho = _safe_precision(
            torch.as_tensor(
                local_precision, dtype=torch.float64, device=self.device
            ).reshape(-1),
            self.model_cfg,
        )
        self._validate_vector(mean, "global_mean")
        self._validate_vector(prior_rho, "prior_global_precision")
        self._validate_vector(next_rho, "next_global_precision")
        self._validate_vector(local_rho, "local_precision")

        self._seed(seed)
        mean_named = self.layout.vector_to_named(mean)
        prior_rho_named = self.layout.vector_to_named(prior_rho)
        next_rho_named = self.layout.vector_to_named(next_rho)
        local_rho_named = self.layout.vector_to_named(local_rho)

        nu_init = (local_rho / next_rho * mean).to(dtype=torch.float32)
        nu = torch.nn.Parameter(nu_init.detach().clone())

        def model_fn(
            x: torch.Tensor,
            y: torch.Tensor | None = None,
            likelihood_scale: float = 1.0,
        ) -> None:
            sampled: Dict[str, torch.Tensor] = {}
            # Eq. (15) continues to regularize against q_{theta_t}, whose
            # covariance is the round-start global covariance.
            with poutine.scale(scale=float(self.train_cfg.kl_weight)):
                for spec in self.layout.specs:
                    site = self.layout.site_name(spec.name)
                    sampled_weight = pyro.sample(
                        site,
                        dist.Normal(
                            mean_named[spec.name],
                            torch.rsqrt(prior_rho_named[spec.name]),
                        ).to_event(len(spec.shape)),
                    )
                    sampled[spec.name] = sampled_weight.to(dtype=torch.float32)
            logits = functional_call(self.model, sampled, (x,))
            with poutine.scale(scale=float(likelihood_scale)):
                with pyro.plate("data", x.shape[0]):
                    pyro.sample("obs", dist.Categorical(logits=logits), obs=y)

        def guide_fn(
            x: torch.Tensor,
            y: torch.Tensor | None = None,
            likelihood_scale: float = 1.0,
        ) -> None:
            del x, y, likelihood_scale
            nu_named = self.layout.vector_to_named(nu)
            with poutine.scale(scale=float(self.train_cfg.kl_weight)):
                for spec in self.layout.specs:
                    site = self.layout.site_name(spec.name)
                    implied_mean = (
                        next_rho_named[spec.name]
                        / local_rho_named[spec.name]
                        * nu_named[spec.name]
                    )
                    pyro.sample(
                        site,
                        dist.Normal(
                            implied_mean,
                            torch.rsqrt(next_rho_named[spec.name]),
                        ).to_event(len(spec.shape)),
                    )

        (
            average_loss,
            local_steps,
            _nu_gradient_l2_mean,
            _nu_gradient_max_abs,
        ) = self._run_coordinate_sgd(
            model_fn=model_fn,
            guide_fn=guide_fn,
            loader=loader,
            parameters=[nu],
            after_step=None,
        )

        with torch.no_grad():
            implied_mean = next_rho / local_rho * nu
        result = NaturalMeanPhaseResult(
            nu=nu.detach().cpu().numpy().astype(np.float32),
            implied_mean=implied_mean.detach().cpu().numpy().astype(np.float32),
            average_loss=float(average_loss),
            local_steps=int(local_steps),
        )
        pyro.clear_param_store()
        return result

    def _run_coordinate_sgd(
        self,
        *,
        model_fn,
        guide_fn,
        loader: DataLoader,
        parameters: Sequence[torch.nn.Parameter],
        after_step,
    ) -> Tuple[float, int, float, float]:
        """Minimize Pyro's differentiable negative ELBO in paper coordinates.

        The final two return values summarize the *applied* (post-clipping)
        gradients.  They make a numerically frozen rho immediately visible in
        client_metrics.csv instead of requiring inference from accuracy curves.
        """
        optimizer = torch.optim.SGD(
            parameters,
            lr=float(self.learning_rate),
            momentum=float(self.train_cfg.momentum),
            weight_decay=float(self.train_cfg.weight_decay),
        )
        elbo = TraceMeanField_ELBO(
            num_particles=int(self.train_cfg.mc_train_samples),
            vectorize_particles=False,
        )

        dataset_size = max(1, int(len(loader.dataset)))
        total_normalized_loss = 0.0
        local_steps = 0
        applied_gradient_l2_sum = 0.0
        applied_gradient_max_abs = 0.0
        non_blocking = bool(self.device.type == "cuda" and loader.pin_memory)

        for _ in range(int(self.train_cfg.local_epochs)):
            for features, targets in loader:
                features = features.to(self.device, non_blocking=non_blocking)
                targets = targets.to(self.device, non_blocking=non_blocking)
                batch_size = max(1, int(targets.numel()))
                # Eq. (14) uses the likelihood of the complete local dataset.
                # This turns a mini-batch likelihood into an unbiased full-data
                # estimator while leaving the KL term applied once.
                likelihood_scale = float(dataset_size) / float(batch_size)

                optimizer.zero_grad(set_to_none=True)
                loss = elbo.differentiable_loss(
                    model_fn,
                    guide_fn,
                    features,
                    targets,
                    likelihood_scale,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite variational loss encountered: {loss.detach().item()}"
                    )
                loss.backward()

                clip_norm = float(self.train_cfg.gradient_clip_norm)
                if clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(parameters, max_norm=clip_norm)

                gradient_sq = 0.0
                gradient_max = 0.0
                for parameter in parameters:
                    if parameter.grad is None:
                        continue
                    grad = parameter.grad.detach()
                    gradient_sq += float(torch.sum(grad.double() ** 2).cpu())
                    gradient_max = max(
                        gradient_max,
                        float(torch.max(torch.abs(grad)).cpu()),
                    )
                applied_gradient_l2_sum += gradient_sq ** 0.5
                applied_gradient_max_abs = max(
                    applied_gradient_max_abs, gradient_max
                )

                optimizer.step()
                if after_step is not None:
                    after_step()

                total_normalized_loss += float(loss.detach().cpu()) / dataset_size
                local_steps += 1

        return (
            total_normalized_loss / max(1, local_steps),
            local_steps,
            applied_gradient_l2_sum / max(1, local_steps),
            applied_gradient_max_abs,
        )

    def _seed(self, seed: int) -> None:
        pyro.clear_param_store()
        pyro.set_rng_seed(int(seed))
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

    def _validate_vector(self, value: torch.Tensor, name: str) -> None:
        if value.numel() != self.layout.total_numel:
            raise ValueError(
                f"{name} contains {value.numel()} values; "
                f"expected {self.layout.total_numel}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
