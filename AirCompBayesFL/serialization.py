"""Stable vector serialization for model and posterior parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    numel: int


class ParameterLayout:
    """Map named PyTorch parameters to and from one flat vector."""

    def __init__(self, model: nn.Module) -> None:
        self.specs: List[TensorSpec] = [
            TensorSpec(name, tuple(parameter.shape), parameter.numel())
            for name, parameter in model.named_parameters()
        ]
        self.total_numel = sum(spec.numel for spec in self.specs)

    @staticmethod
    def site_name(parameter_name: str) -> str:
        return "w__" + parameter_name.replace(".", "__")

    def flatten_model(self, model: nn.Module) -> torch.Tensor:
        named = dict(model.named_parameters())
        return torch.cat([named[spec.name].detach().reshape(-1) for spec in self.specs])

    def load_model_vector(self, model: nn.Module, vector: torch.Tensor | np.ndarray) -> None:
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector)
        if vector.numel() != self.total_numel:
            raise ValueError(
                f"Vector contains {vector.numel()} values, expected {self.total_numel}"
            )
        offset = 0
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                spec = next(item for item in self.specs if item.name == name)
                chunk = vector[offset : offset + spec.numel].reshape(spec.shape)
                parameter.copy_(chunk.to(device=parameter.device, dtype=parameter.dtype))
                offset += spec.numel

    def vector_to_named(
        self,
        vector: torch.Tensor,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Dict[str, torch.Tensor]:
        if vector.numel() != self.total_numel:
            raise ValueError(
                f"Vector contains {vector.numel()} values, expected {self.total_numel}"
            )
        result: Dict[str, torch.Tensor] = {}
        offset = 0
        for spec in self.specs:
            chunk = vector[offset : offset + spec.numel].reshape(spec.shape)
            if device is not None or dtype is not None:
                chunk = chunk.to(device=device or chunk.device, dtype=dtype or chunk.dtype)
            result[spec.name] = chunk
            offset += spec.numel
        return result

    def named_to_vector(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = [spec.name for spec in self.specs if spec.name not in values]
        if missing:
            raise KeyError(f"Missing tensors: {missing}")
        return torch.cat([values[spec.name].reshape(-1) for spec in self.specs])

    def split_vector(self, vector: torch.Tensor) -> List[torch.Tensor]:
        result: List[torch.Tensor] = []
        offset = 0
        for spec in self.specs:
            result.append(vector[offset : offset + spec.numel].reshape(spec.shape))
            offset += spec.numel
        return result


def initial_model_vector(model: nn.Module, seed: int) -> np.ndarray:
    """Deterministically initialize and serialize a model."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for module in model.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
    layout = ParameterLayout(model)
    return layout.flatten_model(model).cpu().numpy().astype(np.float32)
