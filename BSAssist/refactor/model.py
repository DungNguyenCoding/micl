"""CIFAR-10 CNN model, training, testing, and parameter conversion helpers."""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common.typing import NDArrays
from torch.utils.data import DataLoader

from config import SimConfig


class Cifar10CNN(nn.Module):
    """CNN with exactly D = 307,498 communicated trainable parameters.

    The architecture uses six 3x3 convolutional layers and one linear classifier.
    BatchNorm layers are stateless/non-affine so they do not add communicated
    parameters, which keeps D aligned with the CIFAR-10 target screenshot.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.4),
        )
        self.classifier = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def configure_torch_threads(num_threads: int) -> None:
    """Avoid CPU oversubscription when many Ray actors train clients."""
    n = max(1, int(num_threads))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def count_trainable_params(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))


def get_parameters(net: nn.Module) -> NDArrays:
    return [val.detach().cpu().numpy().copy() for _, val in net.state_dict().items()]


def set_parameters(net: nn.Module, parameters: NDArrays, device: torch.device) -> None:
    keys = list(net.state_dict().keys())
    if len(keys) != len(parameters):
        raise ValueError(f"Expected {len(keys)} tensors, received {len(parameters)} tensors")
    state_dict = OrderedDict()
    for key, value in zip(keys, parameters):
        state_dict[key] = torch.as_tensor(value, device=device)
    net.load_state_dict(state_dict, strict=True)


def flatten_parameters(parameters: NDArrays) -> np.ndarray:
    return np.concatenate([p.reshape(-1) for p in parameters]).astype(np.float32, copy=False)


def unflatten_parameters(flat: np.ndarray, shapes: Sequence[Tuple[int, ...]]) -> NDArrays:
    arrays: NDArrays = []
    cursor = 0
    for shape in shapes:
        size = int(np.prod(shape))
        arrays.append(flat[cursor : cursor + size].reshape(shape).astype(np.float32, copy=False))
        cursor += size
    if cursor != len(flat):
        raise ValueError(f"Flat vector has {len(flat)} values, but shapes consume {cursor}")
    return arrays


def clone_parameters(parameters: NDArrays) -> NDArrays:
    return [p.copy() for p in parameters]


def make_optimizer(net: nn.Module, cfg: SimConfig) -> optim.Optimizer:
    optimizer_name = cfg.optimizer.lower()
    if optimizer_name == "sgd":
        return optim.SGD(
            net.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    if optimizer_name == "adam":
        return optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError("Unsupported optimizer. Use 'sgd' or 'adam'.")


def train(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    cfg: SimConfig,
    deterministic_seed: int | None = None,
) -> None:
    """Local update function fLU(M, theta) from the paper."""
    if deterministic_seed is not None:
        set_seed(deterministic_seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(net, cfg)
    net.train()
    for _ in range(cfg.local_epochs):
        for images, labels in trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def test(net: nn.Module, testloader: DataLoader, device: torch.device) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    net.eval()
    for images, labels in testloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = net(images)
        total_loss += float(criterion(logits, labels).item())
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    return total_loss / max(total, 1), correct / max(total, 1)
