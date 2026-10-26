"""Compact four-trainable-layer MNIST CNN for controlled architecture studies.

No batch-normalization, dropout, running state, stochastic sampling, or
custom parameter transformations. The deterministic path is used unchanged
by both FedAvg and FOLA; Bayesian layers are optional for method=bbb.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .bayesian_layers import bayesian_forward, make_bayesian_conv2d, make_bayesian_linear


class MNISTCompactCNN(nn.Module):
    """Conv(1,16,5) -> pool -> Conv(16,32,5) -> pool -> FC(512,128) -> FC(128,10).

    Convolutions use stride=1/padding=0, 2x2 max pooling uses stride=2.
    For 28x28 MNIST inputs: 28 -> 24 -> 12 -> 8 -> 4 spatially.
    Exactly 80,202 trainable scalar parameters in deterministic mode.
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
            def conv(a, b):
                return make_bayesian_conv2d(
                    a, b, kernel_size=5, stride=1, padding=0, bias=True,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init,
                )
            def linear(a, b):
                return make_bayesian_linear(
                    a, b, bias=True,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init,
                )
        else:
            def conv(a, b):
                return nn.Conv2d(a, b, kernel_size=5, stride=1, padding=0, bias=True)
            def linear(a, b):
                return nn.Linear(a, b, bias=True)

        self.conv1 = conv(1, 16)
        self.conv2 = conv(16, 32)
        self.fc1 = linear(32 * 4 * 4, 128)
        self.fc2 = linear(128, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"MNISTCompactCNN expects NCHW image tensors [N,1,28,28], got {tuple(x.shape)}"
            )
        x = self.pool(F.relu(bayesian_forward(self.conv1, x)))
        x = self.pool(F.relu(bayesian_forward(self.conv2, x)))
        x = torch.flatten(x, 1)
        x = F.relu(bayesian_forward(self.fc1, x))
        return bayesian_forward(self.fc2, x)
