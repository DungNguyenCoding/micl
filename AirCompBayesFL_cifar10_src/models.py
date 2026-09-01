"""Neural network architecture used in the paper simulations."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PaperCNN(nn.Module):
    """MNIST CNN with exactly 62,346 trainable parameters.

    The parameter count implied by the paper is obtained by two unpadded 5x5
    convolution layers, each followed by 2x2 max pooling, and a final
    1024-to-10 linear classifier:

    * Conv(1, 32, 5): 832 parameters
    * Conv(32, 64, 5): 51,264 parameters
    * Linear(1024, 10): 10,250 parameters
    * Total: 62,346 parameters
    """

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


def build_model(name: str = "paper_cnn", num_classes: int = 10) -> nn.Module:
    if name != "paper_cnn":
        raise ValueError(f"Unsupported model: {name}")
    model = PaperCNN(num_classes=num_classes)
    expected = 62_346 if num_classes == 10 else None
    if expected is not None:
        actual = count_parameters(model)
        if actual != expected:
            raise RuntimeError(f"PaperCNN parameter count mismatch: {actual} != {expected}")
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


# ============================================================
# AIRCOMP_RAY_SAFE_CIFAR_MODEL
#
# Ray actors import models.py independently from the driver.
# Install the same CIFAR model override based on the
# process-visible AIRCOMP_DATASET environment variable.
# ============================================================

import os as _aircomp_os

if (
    _aircomp_os.environ
    .get("AIRCOMP_DATASET", "mnist")
    .strip()
    .lower()
    == "cifar10"
):
    from cifar10_support import (
        CIFAR10PaperCNN as _AirCompCIFAR10PaperCNN,
        CIFAR10_PARAMETER_COUNT as _AIRCOMP_CIFAR_D,
        CIFAR10_PARAMETER_COUNTS as _AIRCOMP_CIFAR_D_BY_MODEL,
        _cifar_build_model as _aircomp_cifar_build_model,
    )

    _AIRCOMP_CIFAR_D_VALUES = set(_AIRCOMP_CIFAR_D_BY_MODEL.values())

    # Replace model factory.
    build_model = _aircomp_cifar_build_model

    # Replace common model class symbol when present.
    if "PaperCNN" in globals():
        PaperCNN = _AirCompCIFAR10PaperCNN

    # Replace known expected-dimension constants when present.
    for _name in (
        "EXPECTED_PARAMETER_COUNT",
        "PAPER_PARAMETER_COUNT",
        "EXPECTED_D",
        "MODEL_DIMENSION",
    ):
        if _name in globals():
            globals()[_name] = _AIRCOMP_CIFAR_D

    def _aircomp_assert_cifar_parameter_count(model):
        count = int(
            sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )
        )

        if count not in _AIRCOMP_CIFAR_D_VALUES:
            raise AssertionError(
                f"Unsupported CIFAR-10 model dimension {count:,}; "
                f"expected one of {sorted(_AIRCOMP_CIFAR_D_VALUES)}"
            )

        return count

    for _name in (
        "assert_expected_parameter_count",
        "assert_paper_parameter_count",
        "assert_parameter_count",
    ):
        if _name in globals():
            globals()[_name] = (
                _aircomp_assert_cifar_parameter_count
            )

