"""Pyro mean-field variational inference for local Bayesian clients."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro import poutine
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import SGD
from torch.func import functional_call
from torch.utils.data import DataLoader

from config import ModelConfig, TrainingConfig
from serialization import ParameterLayout


@dataclass
class PosteriorResult:
    mean: np.ndarray
    precision: np.ndarray
    average_loss: float
    phase1_loss: float
    phase2_loss: float


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, min=1.0e-8, max=50.0)
    return torch.log(torch.expm1(value))


class BayesianVITrainer:
    """Train a diagonal Gaussian posterior against a global Gaussian prior.

    The Pyro model and guide use matching Normal sample sites. Scaling the
    latent sample sites in both model and guide makes the optimized objective
    equal to task_loss + kl_weight * KL(q_local || q_global), matching Eq. (15)
    in the paper.
    """

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

    def fit(
        self,
        global_mean: np.ndarray,
        global_precision: np.ndarray,
        loader: DataLoader,
        seed: int,
    ) -> PosteriorResult:
        mean = torch.as_tensor(global_mean, dtype=torch.float32, device=self.device)
        precision = torch.as_tensor(
            global_precision, dtype=torch.float32, device=self.device
        ).clamp(
            min=self.model_cfg.min_precision,
            max=self.model_cfg.max_precision,
        )
        prior_std = torch.rsqrt(precision).clamp(
            min=self.model_cfg.min_posterior_std,
            max=self.model_cfg.max_posterior_std,
        )

        if self.train_cfg.bayesian_local_mode == "joint":
            local_mean, local_std, loss = self._run_phase(
                prior_mean=mean,
                prior_std=prior_std,
                initial_mean=mean,
                initial_std=prior_std,
                loader=loader,
                seed=seed,
                train_mean=True,
                train_std=True,
            )
            phase1_loss = 0.0
            phase2_loss = loss
        else:
            # Paper-inspired two-phase local optimization: first optimize
            # uncertainty with the global mean fixed, then optimize the mean
            # while holding the locally learned uncertainty fixed.
            _, local_std, phase1_loss = self._run_phase(
                prior_mean=mean,
                prior_std=prior_std,
                initial_mean=mean,
                initial_std=prior_std,
                loader=loader,
                seed=seed,
                train_mean=False,
                train_std=True,
            )
            local_mean, local_std, phase2_loss = self._run_phase(
                prior_mean=mean,
                prior_std=prior_std,
                initial_mean=mean,
                initial_std=local_std,
                loader=loader,
                seed=seed + 1_000_003,
                train_mean=True,
                train_std=False,
            )
            loss = phase1_loss + phase2_loss

        local_std = local_std.clamp(
            min=self.model_cfg.min_posterior_std,
            max=self.model_cfg.max_posterior_std,
        )
        local_precision = torch.reciprocal(local_std.square()).clamp(
            min=self.model_cfg.min_precision,
            max=self.model_cfg.max_precision,
        )
        return PosteriorResult(
            mean=local_mean.detach().cpu().numpy().astype(np.float32),
            precision=local_precision.detach().cpu().numpy().astype(np.float32),
            average_loss=float(loss),
            phase1_loss=float(phase1_loss),
            phase2_loss=float(phase2_loss),
        )

    def _run_phase(
        self,
        *,
        prior_mean: torch.Tensor,
        prior_std: torch.Tensor,
        initial_mean: torch.Tensor,
        initial_std: torch.Tensor,
        loader: DataLoader,
        seed: int,
        train_mean: bool,
        train_std: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        pyro.clear_param_store()
        pyro.set_rng_seed(int(seed))
        torch.manual_seed(int(seed))

        prior_mean_named = self.layout.vector_to_named(prior_mean)
        prior_std_named = self.layout.vector_to_named(prior_std)
        initial_mean_named = self.layout.vector_to_named(initial_mean)
        initial_std_named = self.layout.vector_to_named(initial_std)

        def model_fn(x: torch.Tensor, y: torch.Tensor | None = None) -> None:
            sampled: Dict[str, torch.Tensor] = {}
            with poutine.scale(scale=float(self.train_cfg.kl_weight)):
                for spec in self.layout.specs:
                    site = self.layout.site_name(spec.name)
                    sampled[spec.name] = pyro.sample(
                        site,
                        dist.Normal(
                            prior_mean_named[spec.name],
                            prior_std_named[spec.name],
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
                    if train_mean:
                        loc = pyro.param(
                            f"{site}__loc", initial_mean_named[spec.name].clone()
                        )
                    else:
                        loc = initial_mean_named[spec.name]

                    if train_std:
                        raw = pyro.param(
                            f"{site}__raw_scale",
                            _inverse_softplus(initial_std_named[spec.name]).clone(),
                        )
                        scale = F.softplus(raw) + self.model_cfg.min_posterior_std
                        scale = torch.clamp(
                            scale,
                            min=self.model_cfg.min_posterior_std,
                            max=self.model_cfg.max_posterior_std,
                        )
                    else:
                        scale = initial_std_named[spec.name]

                    pyro.sample(
                        site,
                        dist.Normal(loc, scale).to_event(len(spec.shape)),
                    )

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
        non_blocking = bool(self.device.type == "cuda" and loader.pin_memory)
        for _ in range(self.train_cfg.local_epochs):
            for features, targets in loader:
                features = features.to(self.device, non_blocking=non_blocking)
                targets = targets.to(self.device, non_blocking=non_blocking)
                batch_loss = float(svi.step(features, targets))
                total_loss += batch_loss
                total_examples += int(targets.numel())

        posterior_mean: Dict[str, torch.Tensor] = {}
        posterior_std: Dict[str, torch.Tensor] = {}
        for spec in self.layout.specs:
            site = self.layout.site_name(spec.name)
            if train_mean:
                posterior_mean[spec.name] = pyro.param(f"{site}__loc").detach()
            else:
                posterior_mean[spec.name] = initial_mean_named[spec.name].detach()
            if train_std:
                raw = pyro.param(f"{site}__raw_scale")
                posterior_std[spec.name] = (
                    F.softplus(raw) + self.model_cfg.min_posterior_std
                ).detach()
            else:
                posterior_std[spec.name] = initial_std_named[spec.name].detach()

        mean_vector = self.layout.named_to_vector(posterior_mean)
        std_vector = self.layout.named_to_vector(posterior_std)
        average_loss = total_loss / max(1, total_examples)
        return mean_vector, std_vector, float(average_loss)
