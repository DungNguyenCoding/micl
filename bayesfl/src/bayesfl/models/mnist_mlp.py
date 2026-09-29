"""MNIST MLP architectures used by the BayesFL experiments."""

from __future__ import annotations

import torch
from torch import nn

from .bayesian_layers import bayesian_forward, make_bayesian_linear


def _linear_factory(
    *,
    bayesian: bool,
    posterior_mu_init: float,
    posterior_rho_init: float,
):
    if bayesian:
        return lambda a, b: make_bayesian_linear(
            a,
            b,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
    return lambda a, b: nn.Linear(a, b)


class MNISTMLP(nn.Module):
    """Legacy 784-500-300-10 MLP kept for archived experiments."""

    def __init__(
        self,
        *,
        bayesian: bool = False,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.bayesian = bayesian
        make = _linear_factory(
            bayesian=bayesian,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
        self.fc1 = make(28 * 28, 500)
        self.fc2 = make(500, 300)
        self.fc3 = make(300, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = torch.relu(bayesian_forward(self.fc1, x))
        x = torch.relu(bayesian_forward(self.fc2, x))
        return bayesian_forward(self.fc3, x)


class MNISTMLP256x5(nn.Module):
    """Six-linear-layer MNIST MLP used by the updated experiment.

    Architecture:
        784 -> 256 -> 256 -> 256 -> 256 -> 256 -> 10

    The five hidden layers use ReLU.  Including biases, the deterministic
    network has exactly 466,698 trainable parameters.
    """

    def __init__(
        self,
        *,
        bayesian: bool = False,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.bayesian = bayesian
        make = _linear_factory(
            bayesian=bayesian,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
        self.fc1 = make(28 * 28, 256)
        self.fc2 = make(256, 256)
        self.fc3 = make(256, 256)
        self.fc4 = make(256, 256)
        self.fc5 = make(256, 256)
        self.fc6 = make(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = torch.relu(bayesian_forward(self.fc1, x))
        x = torch.relu(bayesian_forward(self.fc2, x))
        x = torch.relu(bayesian_forward(self.fc3, x))
        x = torch.relu(bayesian_forward(self.fc4, x))
        x = torch.relu(bayesian_forward(self.fc5, x))
        return bayesian_forward(self.fc6, x)
