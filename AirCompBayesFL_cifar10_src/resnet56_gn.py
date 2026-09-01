"""CIFAR ResNet-56 with GroupNorm for federated/Bayesian experiments."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

CIFAR10_RESNET56_GN_PARAMETER_COUNT = 855_770


def _group_count(channels: int) -> int:
    if channels == 16:
        return 4
    if channels in {32, 64}:
        return 8
    raise ValueError(f"Unsupported ResNet-56 channel count: {channels}")


class CIFARResNetBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False,
        )
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False,
        )
        self.gn2 = nn.GroupNorm(groups, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.GroupNorm(groups, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.relu(x + identity)


class CIFAR10ResNet56GN(nn.Module):
    """CIFAR ResNet-56 (6n+2, n=9) with GroupNorm instead of BatchNorm.

    Stem: 3x3 Conv 3->16 + GN
    Stage 1: 9 basic blocks, 16 channels
    Stage 2: 9 basic blocks, 32 channels; first block stride 2
    Stage 3: 9 basic blocks, 64 channels; first block stride 2
    Head: global average pooling + Linear(64, 10)

    Projection shortcuts use 1x1 Conv + GroupNorm at stage transitions.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(4, 16)
        self.stage1 = self._make_stage(16, 16, blocks=9, first_stride=1)
        self.stage2 = self._make_stage(16, 32, blocks=9, first_stride=2)
        self.stage3 = self._make_stage(32, 64, blocks=9, first_stride=2)
        self.fc = nn.Linear(64, num_classes)

        actual = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if num_classes == 10 and actual != CIFAR10_RESNET56_GN_PARAMETER_COUNT:
            raise RuntimeError(
                f"ResNet-56-GN parameter count changed: {actual:,} != "
                f"{CIFAR10_RESNET56_GN_PARAMETER_COUNT:,}"
            )

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        *,
        blocks: int,
        first_stride: int,
    ) -> nn.Sequential:
        layers = [
            CIFARResNetBasicBlock(in_channels, out_channels, first_stride)
        ]
        for _ in range(1, blocks):
            layers.append(CIFARResNetBasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.fc(x)
