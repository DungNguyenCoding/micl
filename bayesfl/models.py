"""Dataset-selectable neural networks for the unified baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


MNIST_PAPER_CNN_PARAMETER_COUNT = 62_346
CIFAR10_RESNET56_GN_PARAMETER_COUNT = 855_770
CIFAR10_RESNET56_GN_BAYESIAN_PARAMETER_COUNT = 851_514
CIFAR10_RESNET56_GN_GROUPNORM_PARAMETER_COUNT = 4_256


class PaperCNN(nn.Module):
    """Original MNIST CNN from the v1.6.1 reproduction path."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = torch.flatten(x, 1)
        return self.fc(x)


class CIFARResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(_gn_groups(out_channels), out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(_gn_groups(out_channels), out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + identity)


def _gn_groups(channels: int) -> int:
    if channels == 16:
        return 4
    return 8


class CIFAR10ResNet56GN(nn.Module):
    """CIFAR ResNet-56 (6n+2, n=9) with GroupNorm instead of BatchNorm.

    The projection shortcuts use Conv1x1 + GroupNorm.  The resulting model has
    exactly 855,770 trainable parameters.  Native Bayesian-Torch conversion
    Bayesianizes Conv2d and Linear modules but leaves GroupNorm deterministic,
    so the Bayesian dimension is 851,514.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, 16)
        self.stage1 = self._make_stage(16, 16, blocks=9, first_stride=1)
        self.stage2 = self._make_stage(16, 32, blocks=9, first_stride=2)
        self.stage3 = self._make_stage(32, 64, blocks=9, first_stride=2)
        self.fc = nn.Linear(64, num_classes)

        actual = count_parameters(self)
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
        layers = [CIFARResNetBasicBlock(in_channels, out_channels, first_stride)]
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


def build_model(dataset: str, name: str | None = None, num_classes: int = 10) -> nn.Module:
    dataset = str(dataset).strip().lower()
    if dataset == "mnist":
        expected = "paper_cnn"
        if name is not None and str(name).strip().lower() != expected:
            raise ValueError("MNIST requires model paper_cnn")
        model = PaperCNN(num_classes=num_classes)
        if num_classes == 10 and count_parameters(model) != MNIST_PAPER_CNN_PARAMETER_COUNT:
            raise RuntimeError("PaperCNN parameter count changed")
        return model

    if dataset == "cifar10":
        expected = "resnet56_gn"
        if name is not None and str(name).strip().lower() != expected:
            raise ValueError("CIFAR-10 requires model resnet56_gn")
        return CIFAR10ResNet56GN(num_classes=num_classes)

    raise ValueError("dataset must be mnist or cifar10")


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
