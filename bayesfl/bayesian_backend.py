"""Factory for local Bayesian inference backends.

The default remains the validated Pyro implementation.
"""

from __future__ import annotations

import torch

from config import (
    ModelConfig,
    TrainingConfig,
)
from serialization import ParameterLayout


def normalize_bayesian_backend(
    value: str,
) -> str:
    backend = (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
    )

    if backend == "bayesiantorch":
        backend = "bayesian_torch"

    if backend not in {
        "pyro",
        "bayesian_torch",
    }:
        raise ValueError(
            "Unsupported Bayesian backend "
            f"{value!r}; expected 'pyro' "
            "or 'bayesian_torch'"
        )

    return backend


def create_bayesian_trainer(
    *,
    model: torch.nn.Module,
    layout: ParameterLayout,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    learning_rate: float | None = None,
):
    backend = normalize_bayesian_backend(
        getattr(
            train_cfg,
            "bayesian_backend",
            "pyro",
        )
    )

    if backend == "pyro":
        from bayes_vi import (
            BayesianVITrainer,
        )

        cls = BayesianVITrainer

    else:
        from bayesian_torch_vi import (
            BayesianTorchVITrainer,
        )

        cls = BayesianTorchVITrainer

    return cls(
        model=model,
        layout=layout,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        device=device,
        learning_rate=learning_rate,
    )
