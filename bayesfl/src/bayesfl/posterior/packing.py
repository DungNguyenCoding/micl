"""Stable parameter packing for Flower NumPy transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ParameterLayout:
    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]

    @classmethod
    def from_model(cls, model: nn.Module) -> "ParameterLayout":
        pairs = list(model.named_parameters())
        return cls(
            names=tuple(name for name, _ in pairs),
            shapes=tuple(tuple(param.shape) for _, param in pairs),
        )

    @property
    def size(self) -> int:
        return len(self.names)

    def validate(self, arrays: Sequence[np.ndarray]) -> None:
        if len(arrays) != self.size:
            raise ValueError(f"Expected {self.size} arrays, got {len(arrays)}")
        for name, expected, array in zip(self.names, self.shapes, arrays):
            if tuple(array.shape) != expected:
                raise ValueError(f"Shape mismatch for {name}: expected {expected}, got {array.shape}")


def model_to_ndarrays(model: nn.Module) -> list[np.ndarray]:
    return [param.detach().cpu().numpy().copy() for _, param in model.named_parameters()]


def ndarrays_to_model(model: nn.Module, arrays: Sequence[np.ndarray]) -> None:
    layout = ParameterLayout.from_model(model)
    layout.validate(arrays)
    with torch.no_grad():
        for (_, param), array in zip(model.named_parameters(), arrays):
            tensor = torch.as_tensor(array, device=param.device, dtype=param.dtype)
            param.copy_(tensor)


def pack_fola(mean_arrays: Sequence[np.ndarray], precision_arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    if len(mean_arrays) != len(precision_arrays):
        raise ValueError("FOLA mean and precision lists must have the same length")
    return [np.asarray(a).copy() for a in mean_arrays] + [np.asarray(a).copy() for a in precision_arrays]


def unpack_fola(arrays: Sequence[np.ndarray], layout: ParameterLayout) -> tuple[list[np.ndarray], list[np.ndarray]]:
    expected = 2 * layout.size
    if len(arrays) != expected:
        raise ValueError(f"Expected {expected} FOLA arrays, got {len(arrays)}")
    means = [np.asarray(a) for a in arrays[: layout.size]]
    precisions = [np.asarray(a) for a in arrays[layout.size :]]
    layout.validate(means)
    layout.validate(precisions)
    return means, precisions


def initial_fola_state(model: nn.Module, initial_precision: float) -> list[np.ndarray]:
    means = model_to_ndarrays(model)
    precisions = [np.full_like(a, float(initial_precision), dtype=np.float32) for a in means]
    return pack_fola(means, precisions)
