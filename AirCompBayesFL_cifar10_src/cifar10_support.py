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
CIFAR10_PARAMETER_COUNT = 69_706


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


class CIFAR10PaperCNN(nn.Module):
    """Native-RGB CIFAR-10 analogue of the paper's small CNN."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc = nn.Linear(64 * 5 * 5, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, 1)
        return self.fc(x)


def cifar10_parameter_count() -> int:
    model = CIFAR10PaperCNN()
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _cifar_build_model(*args: Any, **kwargs: Any) -> nn.Module:
    del args, kwargs
    model = CIFAR10PaperCNN()
    count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if count != CIFAR10_PARAMETER_COUNT:
        raise RuntimeError(
            f"CIFAR-10 model parameter count changed: {count:,} != "
            f"{CIFAR10_PARAMETER_COUNT:,}"
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
        if count != CIFAR10_PARAMETER_COUNT:
            raise AssertionError(
                f"Expected CIFAR-10 model dimension {CIFAR10_PARAMETER_COUNT:,}, got {count:,}"
            )
        return count

    for name in (
        "assert_expected_parameter_count",
        "assert_paper_parameter_count",
        "assert_parameter_count",
    ):
        if hasattr(models_module, name):
            setattr(models_module, name, _assert_cifar_count)
