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
