from __future__ import annotations

"""Opt-in CIFAR-10 extension for AirCompBayesFL.

This module deliberately does not change the MNIST/paper reproduction path.
``main_cifar10.py`` installs these overrides before importing the existing
``main.py``.  The rest of the FL, Bayesian VI, sparse posterior, AirComp,
wireless, and metrics code is reused unchanged.

CIFAR-10 model (minimal native-RGB analogue of the paper CNN):
    Conv2d(3, 32, kernel_size=5)
    max-pool + ReLU
    Conv2d(32, 64, kernel_size=5)
    max-pool + ReLU
    flatten 64*5*5 = 1600
    Linear(1600, 10)

Trainable parameters: 69,706.
At F=1024 this is ceil(69706/1024)=69 OFDM groups per phase.
"""

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets as tv_datasets
from torchvision import transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_PARAMETER_COUNT = 78_042

from resnet56_gn import (
    CIFAR10ResNet56GN,
    CIFAR10_RESNET56_GN_PARAMETER_COUNT,
)

CIFAR10_PARAMETER_COUNTS = {
    "paper_cnn": CIFAR10_PARAMETER_COUNT,
    "cifar_residual_cnn": CIFAR10_PARAMETER_COUNT,
    "resnet56_gn": CIFAR10_RESNET56_GN_PARAMETER_COUNT,
}


def cifar10_transform() -> transforms.Compose:
    # No augmentation by design: this extension changes the dataset/model only,
    # while leaving the learning/wireless setup as close as possible to MNIST.
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


class CIFAR10AsMNIST(Dataset):
    """CIFAR-10 dataset exposing the constructor/attributes used for MNIST.

    The existing dataset.py calls ``datasets.MNIST(...)`` in several places.
    The CIFAR launcher replaces that symbol with this adapter.  The transform
    supplied by the MNIST path is intentionally ignored because it is
    one-channel MNIST normalization and is invalid for RGB CIFAR-10.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Any | None = None,
        target_transform: Any | None = None,
        download: bool = False,
        **_: Any,
    ) -> None:
        self._dataset = tv_datasets.CIFAR10(
            root=root,
            train=bool(train),
            transform=cifar10_transform(),
            target_transform=target_transform,
            download=bool(download),
        )
        # Preserve attributes commonly used by the existing partition code.
        self.targets = self._dataset.targets
        self.data = self._dataset.data
        self.classes = self._dataset.classes
        self.class_to_idx = self._dataset.class_to_idx
        self.train = bool(train)
        self.root = root
        self.transform = self._dataset.transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        return self._dataset[index]


class CIFAR10ResidualBlock(nn.Module):
    """Small residual block using GroupNorm.

    GroupNorm is preferred over BatchNorm here because federated
    clients use small batches and highly non-IID one-class data.
    It also avoids client-specific BatchNorm running statistics.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        if out_channels == 16:
            groups = 4
        else:
            groups = 8

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.gn1 = nn.GroupNorm(
            groups,
            out_channels,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.gn2 = nn.GroupNorm(
            groups,
            out_channels,
        )

        if (
            stride != 1
            or in_channels != out_channels
        ):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(
                    groups,
                    out_channels,
                ),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.gn2(x)

        x = x + identity
        x = F.relu(x)

        return x


class CIFAR10ResidualCNN(nn.Module):
    """Compact residual CNN designed for CIFAR-10.

    Architecture:

        RGB 3x32x32
          -> Conv(3,16,3x3) + GroupNorm
          -> Residual 16->16
          -> Residual 16->32, stride 2
          -> Residual 32->64, stride 2
          -> AdaptiveAvgPool(1x1)
          -> Linear(64,10)

    The model intentionally remains small enough for Bayesian
    VI and AirComp while being substantially more appropriate
    for CIFAR-10 than the original MNIST-style two-conv CNN.
    """

    def __init__(
        self,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        # Keep the name "conv1" because existing diagnostics
        # inspect model.conv1.in_channels.
        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.gn1 = nn.GroupNorm(
            4,
            16,
        )

        self.block1 = CIFAR10ResidualBlock(
            16,
            16,
            stride=1,
        )

        self.block2 = CIFAR10ResidualBlock(
            16,
            32,
            stride=2,
        )

        self.block3 = CIFAR10ResidualBlock(
            32,
            64,
            stride=2,
        )

        self.fc = nn.Linear(
            64,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = F.adaptive_avg_pool2d(
            x,
            output_size=1,
        )

        x = torch.flatten(
            x,
            1,
        )

        return self.fc(x)


# Compatibility alias.
#
# models.py and the Ray-safe override already import this name.
# Keeping it means the MNIST core source does not need to change.
CIFAR10PaperCNN = CIFAR10ResidualCNN


def cifar10_parameter_count(model_name: str = "paper_cnn") -> int:
    model = _cifar_build_model(model_name, 10)
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _cifar_build_model(
    *args: Any,
    **kwargs: Any,
) -> nn.Module:
    name = str(kwargs.get("name", args[0] if len(args) >= 1 else "paper_cnn"))
    name = name.strip().lower()
    num_classes = int(kwargs.get("num_classes", args[1] if len(args) >= 2 else 10))

    if num_classes != 10:
        raise ValueError("CIFAR-10 extension requires num_classes=10")

    if name in {"paper_cnn", "cifar_residual_cnn"}:
        model = CIFAR10ResidualCNN(num_classes=num_classes)
        expected = CIFAR10_PARAMETER_COUNT
    elif name == "resnet56_gn":
        model = CIFAR10ResNet56GN(num_classes=num_classes)
        expected = CIFAR10_RESNET56_GN_PARAMETER_COUNT
    else:
        raise ValueError(
            f"Unsupported CIFAR-10 model: {name!r}; expected "
            "paper_cnn, cifar_residual_cnn, or resnet56_gn"
        )

    count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if count != expected:
        raise RuntimeError(
            f"CIFAR-10 model parameter count changed: {count:,} != {expected:,}"
        )
    return model


def install_cifar10_overrides() -> None:
    """Install CIFAR-10 dataset/model overrides for this Python process only."""
    import dataset as dataset_module
    import models as models_module

    # Dataset: every existing MNIST construction now creates CIFAR-10.
    dataset_module.datasets.MNIST = CIFAR10AsMNIST

    # Model: replace the factory and the common class symbol used by the core.
    models_module.build_model = _cifar_build_model
    if hasattr(models_module, "PaperCNN"):
        models_module.PaperCNN = CIFAR10PaperCNN

    # Update common expected-dimension constants if they exist in this version.
    for name in (
        "EXPECTED_PARAMETER_COUNT",
        "PAPER_PARAMETER_COUNT",
        "EXPECTED_D",
        "MODEL_DIMENSION",
    ):
        if hasattr(models_module, name):
            setattr(models_module, name, CIFAR10_PARAMETER_COUNT)

    # Some versions expose an explicit assertion helper.  Replace only those
    # well-known names inside the opt-in CIFAR process.
    def _assert_cifar_count(model: nn.Module) -> int:
        count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        supported = set(CIFAR10_PARAMETER_COUNTS.values())
        if count not in supported:
            raise AssertionError(
                f"Unsupported CIFAR-10 model dimension {count:,}; "
                f"expected one of {sorted(supported)}"
            )
        return count

    for name in (
        "assert_expected_parameter_count",
        "assert_paper_parameter_count",
        "assert_parameter_count",
    ):
        if hasattr(models_module, name):
            setattr(models_module, name, _assert_cifar_count)
