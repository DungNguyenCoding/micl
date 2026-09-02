"""MNIST 784-500-300-10 MLP."""

from __future__ import annotations

import torch
from torch import nn

from .bayesian_layers import bayesian_forward, make_bayesian_linear


class MNISTMLP(nn.Module):
    def __init__(
        self,
        *,
        bayesian: bool = False,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.bayesian = bayesian
        if bayesian:
            make = lambda a, b: make_bayesian_linear(
                a,
                b,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init,
            )
        else:
            make = lambda a, b: nn.Linear(a, b)
        self.fc1 = make(28 * 28, 500)
        self.fc2 = make(500, 300)
        self.fc3 = make(300, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = torch.relu(bayesian_forward(self.fc1, x))
        x = torch.relu(bayesian_forward(self.fc2, x))
        return bayesian_forward(self.fc3, x)
