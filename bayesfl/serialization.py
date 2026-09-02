"""Stable flat-vector serialization for deterministic model parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    numel: int


class ParameterLayout:
    def __init__(self, model: nn.Module) -> None:
        self.specs: List[TensorSpec] = [
            TensorSpec(name, tuple(parameter.shape), int(parameter.numel()))
            for name, parameter in model.named_parameters()
        ]
        self.total_numel = int(sum(spec.numel for spec in self.specs))

    def flatten_model(self, model: nn.Module) -> torch.Tensor:
        named = dict(model.named_parameters())
        return torch.cat([named[spec.name].detach().reshape(-1) for spec in self.specs])

    def load_model_vector(self, model: nn.Module, vector: torch.Tensor | np.ndarray) -> None:
        value = torch.as_tensor(vector).reshape(-1)
        if int(value.numel()) != self.total_numel:
            raise ValueError(
                f"Vector contains {value.numel()} values, expected {self.total_numel}"
            )
        named = dict(model.named_parameters())
        offset = 0
        with torch.no_grad():
            for spec in self.specs:
                parameter = named[spec.name]
                chunk = value[offset : offset + spec.numel].reshape(spec.shape)
                parameter.copy_(chunk.to(device=parameter.device, dtype=parameter.dtype))
                offset += spec.numel

    def vector_to_named(
        self,
        vector: torch.Tensor | np.ndarray,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Dict[str, torch.Tensor]:
        value = torch.as_tensor(vector).reshape(-1)
        if int(value.numel()) != self.total_numel:
            raise ValueError(
                f"Vector contains {value.numel()} values, expected {self.total_numel}"
            )
        result: Dict[str, torch.Tensor] = {}
        offset = 0
        for spec in self.specs:
            chunk = value[offset : offset + spec.numel].reshape(spec.shape)
            if device is not None or dtype is not None:
                chunk = chunk.to(device=device or chunk.device, dtype=dtype or chunk.dtype)
            result[spec.name] = chunk
            offset += spec.numel
        return result

    def named_to_vector(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([values[spec.name].reshape(-1) for spec in self.specs])
