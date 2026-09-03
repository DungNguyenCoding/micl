"""CIFAR-10 BasicCNN used by the Online Laplace Approximation BFL paper code."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .bayesian_layers import bayesian_forward, make_bayesian_conv2d, make_bayesian_linear


def _xavier_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class CifarPaperBasicCNN(nn.Module):
    """Architecture matching the authors' released ``BasicCNN``.

    Deterministic path:
        Conv(3,32,5) -> ReLU -> MaxPool(2)
        Conv(32,64,5) -> ReLU -> MaxPool(2)
        Linear(1600,512) -> ReLU -> Linear(512,10)

    For BBB, deterministic layers are replaced by Bayesian-Torch
    reparameterization layers. With ``match_deterministic_init=True`` the
    posterior means are copied from the exact Xavier deterministic model.
    """

    def __init__(
        self,
        *,
        bayesian: bool = False,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.bayesian = bool(bayesian)

        if self.bayesian:
            self.conv1 = make_bayesian_conv2d(
                3,
                32,
                5,
                bias=True,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init,
            )
            self.conv2 = make_bayesian_conv2d(
                32,
                64,
                5,
                bias=True,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init,
            )
            self.fc1 = make_bayesian_linear(
                64 * 5 * 5,
                512,
                bias=True,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init,
            )
            self.fc2 = make_bayesian_linear(
                512,
                10,
                bias=True,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init,
            )
        else:
            self.conv1 = nn.Conv2d(3, 32, 5, bias=True)
            self.conv2 = nn.Conv2d(32, 64, 5, bias=True)
            self.fc1 = nn.Linear(64 * 5 * 5, 512, bias=True)
            self.fc2 = nn.Linear(512, 10, bias=True)
            self.apply(_xavier_init)

        self.pool = nn.MaxPool2d(2, 2)

    def _layer(self, layer, x):
        return bayesian_forward(layer, x) if self.bayesian else layer(x)

    def forward(self, x):
        x = self.pool(F.relu(self._layer(self.conv1, x)))
        x = self.pool(F.relu(self._layer(self.conv2, x)))
        x = x.view(x.size(0), 64 * 5 * 5)
        x = F.relu(self._layer(self.fc1, x))
        return self._layer(self.fc2, x)
