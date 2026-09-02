"""CIFAR ResNet-56 using GroupNorm with eight groups."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from .bayesian_layers import (
    bayesian_forward,
    make_bayesian_conv2d,
    make_bayesian_linear,
)


class _BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        conv_factory: Callable[..., nn.Module],
        groups: int,
    ) -> None:
        super().__init__()
        self.conv1 = conv_factory(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = conv_factory(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(groups, out_channels)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = conv_factory(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else bayesian_forward(self.shortcut, x)
        out = bayesian_forward(self.conv1, x)
        out = F.relu(self.gn1(out), inplace=True)
        out = bayesian_forward(self.conv2, out)
        out = self.gn2(out)
        return F.relu(out + identity, inplace=True)


class CifarResNet56GN8(nn.Module):
    """6n+2 CIFAR ResNet with n=9 and projection shortcuts."""

    def __init__(
        self,
        *,
        bayesian: bool = False,
        groups: int = 8,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
    ) -> None:
        super().__init__()
        if groups != 8:
            raise ValueError("This research baseline fixes GroupNorm to 8 groups.")
        self.bayesian = bayesian
        if bayesian:
            def conv_factory(in_c, out_c, **kwargs):
                return make_bayesian_conv2d(
                    in_c,
                    out_c,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init,
                    **kwargs,
                )
        else:
            conv_factory = lambda in_c, out_c, **kwargs: nn.Conv2d(in_c, out_c, **kwargs)

        self.conv1 = conv_factory(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, 16)
        self.stage1 = self._make_stage(16, 16, blocks=9, stride=1, conv_factory=conv_factory, groups=groups)
        self.stage2 = self._make_stage(16, 32, blocks=9, stride=2, conv_factory=conv_factory, groups=groups)
        self.stage3 = self._make_stage(32, 64, blocks=9, stride=2, conv_factory=conv_factory, groups=groups)
        if bayesian:
            self.fc = make_bayesian_linear(
                64,
                10,
                bias=True,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init,
            )
        else:
            self.fc = nn.Linear(64, 10)

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        *,
        blocks: int,
        stride: int,
        conv_factory: Callable[..., nn.Module],
        groups: int,
    ) -> nn.Sequential:
        layers = [_BasicBlock(in_channels, out_channels, stride, conv_factory, groups)]
        layers.extend(
            _BasicBlock(out_channels, out_channels, 1, conv_factory, groups)
            for _ in range(blocks - 1)
        )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = bayesian_forward(self.conv1, x)
        x = F.relu(self.gn1(x), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = torch.flatten(x, 1)
        return bayesian_forward(self.fc, x)
