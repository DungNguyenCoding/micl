"""Native Bayesian-Torch state adapter for one-phase federated BayesAvg.

Conv2d and Linear parameters are Bayesian. GroupNorm affine parameters remain
ordinary deterministic parameters, matching native dnn_to_bnn behavior.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

from config import ModelConfig, VariationalConfig
from models import (
    CIFAR10_RESNET56_GN_BAYESIAN_PARAMETER_COUNT,
    CIFAR10_RESNET56_GN_GROUPNORM_PARAMETER_COUNT,
    build_model,
)
from serialization import ParameterLayout, TensorSpec


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, min=torch.finfo(value.dtype).eps)
    return torch.where(value > 20.0, value, torch.log(torch.expm1(value)))


@dataclass(frozen=True)
class Binding:
    spec: TensorSpec
    module_path: str
    kind: str  # bayesian | deterministic
    mu_name: str | None = None
    rho_name: str | None = None
    deterministic_name: str | None = None


class BayesianTorchStateAdapter:
    def __init__(
        self,
        deterministic_model: nn.Module,
        layout: ParameterLayout,
        variational_cfg: VariationalConfig,
    ) -> None:
        self.source_model = deterministic_model
        self.layout = layout
        self.variational_cfg = variational_cfg
        self.model = copy.deepcopy(deterministic_model)

        dnn_to_bnn(
            self.model,
            {
                "prior_mu": float(variational_cfg.prior_mu),
                "prior_sigma": float(variational_cfg.prior_sigma),
                "posterior_mu_init": float(variational_cfg.posterior_mu_init),
                "posterior_rho_init": float(variational_cfg.posterior_rho_init),
                "type": "Reparameterization",
                "moped_enable": False,
                "moped_delta": 0.5,
            },
        )

        self.bindings = self._build_bindings()
        self.bayesian_dimension = int(
            sum(b.spec.numel for b in self.bindings if b.kind == "bayesian")
        )
        self.deterministic_dimension = int(
            sum(b.spec.numel for b in self.bindings if b.kind == "deterministic")
        )
        self.state_dimension = int(
            2 * self.bayesian_dimension + self.deterministic_dimension
        )

    def _module(self, path: str) -> nn.Module:
        return self.model if path == "" else self.model.get_submodule(path)

    def _source_module(self, path: str) -> nn.Module:
        return self.source_model if path == "" else self.source_model.get_submodule(path)

    def _build_bindings(self) -> List[Binding]:
        result: List[Binding] = []
        for spec in self.layout.specs:
            module_path, sep, parameter_name = spec.name.rpartition(".")
            if not sep:
                module_path = ""
                parameter_name = spec.name
            source = self._source_module(module_path)
            converted = self._module(module_path)

            if isinstance(source, nn.Conv2d):
                if parameter_name == "weight":
                    binding = Binding(
                        spec,
                        module_path,
                        "bayesian",
                        mu_name="mu_kernel",
                        rho_name="rho_kernel",
                    )
                elif parameter_name == "bias":
                    binding = Binding(
                        spec,
                        module_path,
                        "bayesian",
                        mu_name="mu_bias",
                        rho_name="rho_bias",
                    )
                else:
                    raise RuntimeError(f"Unsupported Conv2d parameter {spec.name}")
            elif isinstance(source, nn.Linear):
                if parameter_name == "weight":
                    binding = Binding(
                        spec,
                        module_path,
                        "bayesian",
                        mu_name="mu_weight",
                        rho_name="rho_weight",
                    )
                elif parameter_name == "bias":
                    binding = Binding(
                        spec,
                        module_path,
                        "bayesian",
                        mu_name="mu_bias",
                        rho_name="rho_bias",
                    )
                else:
                    raise RuntimeError(f"Unsupported Linear parameter {spec.name}")
            elif isinstance(source, nn.GroupNorm):
                binding = Binding(
                    spec,
                    module_path,
                    "deterministic",
                    deterministic_name=parameter_name,
                )
            else:
                raise RuntimeError(
                    f"Unsupported parameterized module {type(source).__name__}: {spec.name}"
                )

            if binding.kind == "bayesian":
                mu = getattr(converted, str(binding.mu_name))
                rho = getattr(converted, str(binding.rho_name))
                if tuple(mu.shape) != spec.shape or tuple(rho.shape) != spec.shape:
                    raise RuntimeError(f"Bayesian shape mismatch for {spec.name}")
            else:
                parameter = getattr(converted, str(binding.deterministic_name))
                if tuple(parameter.shape) != spec.shape:
                    raise RuntimeError(f"Deterministic shape mismatch for {spec.name}")
            result.append(binding)
        return result

    def to(self, device: torch.device) -> "BayesianTorchStateAdapter":
        self.model.to(device)
        return self

    def state_vector(self) -> np.ndarray:
        mus: List[torch.Tensor] = []
        rhos: List[torch.Tensor] = []
        deterministic: List[torch.Tensor] = []
        for binding in self.bindings:
            module = self._module(binding.module_path)
            if binding.kind == "bayesian":
                mus.append(getattr(module, str(binding.mu_name)).detach().reshape(-1))
                rhos.append(getattr(module, str(binding.rho_name)).detach().reshape(-1))
            else:
                deterministic.append(
                    getattr(module, str(binding.deterministic_name)).detach().reshape(-1)
                )
        parts = [torch.cat(mus), torch.cat(rhos)]
        if deterministic:
            parts.append(torch.cat(deterministic))
        return torch.cat(parts).cpu().numpy().astype(np.float32)

    def split_state(self, state: np.ndarray | torch.Tensor):
        value = torch.as_tensor(state).reshape(-1)
        if int(value.numel()) != self.state_dimension:
            raise ValueError(
                f"Bayesian state has {value.numel()} values, expected {self.state_dimension}"
            )
        d = self.bayesian_dimension
        return (
            value[:d],
            value[d : 2 * d],
            value[2 * d :],
        )

    def load_state(self, state: np.ndarray | torch.Tensor) -> None:
        mu_vector, rho_vector, deterministic_vector = self.split_state(state)
        mu_offset = 0
        rho_offset = 0
        det_offset = 0
        with torch.no_grad():
            for binding in self.bindings:
                module = self._module(binding.module_path)
                n = binding.spec.numel
                if binding.kind == "bayesian":
                    mu = getattr(module, str(binding.mu_name))
                    rho = getattr(module, str(binding.rho_name))
                    mu_chunk = mu_vector[mu_offset : mu_offset + n].reshape(binding.spec.shape)
                    rho_chunk = rho_vector[rho_offset : rho_offset + n].reshape(binding.spec.shape)
                    mu.copy_(mu_chunk.to(device=mu.device, dtype=mu.dtype))
                    rho.copy_(rho_chunk.to(device=rho.device, dtype=rho.dtype))
                    mu_offset += n
                    rho_offset += n
                else:
                    parameter = getattr(module, str(binding.deterministic_name))
                    chunk = deterministic_vector[
                        det_offset : det_offset + n
                    ].reshape(binding.spec.shape)
                    parameter.copy_(chunk.to(device=parameter.device, dtype=parameter.dtype))
                    det_offset += n

    def mean_model_vector(self) -> np.ndarray:
        chunks: List[torch.Tensor] = []
        for binding in self.bindings:
            module = self._module(binding.module_path)
            if binding.kind == "bayesian":
                value = getattr(module, str(binding.mu_name))
            else:
                value = getattr(module, str(binding.deterministic_name))
            chunks.append(value.detach().reshape(-1))
        return torch.cat(chunks).cpu().numpy().astype(np.float32)

    def rho_vector(self) -> torch.Tensor:
        values = []
        for binding in self.bindings:
            if binding.kind != "bayesian":
                continue
            module = self._module(binding.module_path)
            values.append(getattr(module, str(binding.rho_name)).reshape(-1))
        return torch.cat(values)

    def mu_vector(self) -> torch.Tensor:
        values = []
        for binding in self.bindings:
            if binding.kind != "bayesian":
                continue
            module = self._module(binding.module_path)
            values.append(getattr(module, str(binding.mu_name)).reshape(-1))
        return torch.cat(values)

    def sigma_vector(self) -> torch.Tensor:
        return F.softplus(self.rho_vector())

    def trainable_parameters(self) -> List[nn.Parameter]:
        values: List[nn.Parameter] = []
        seen = set()
        for binding in self.bindings:
            module = self._module(binding.module_path)
            names = (
                [binding.mu_name, binding.rho_name]
                if binding.kind == "bayesian"
                else [binding.deterministic_name]
            )
            for name in names:
                parameter = getattr(module, str(name))
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    values.append(parameter)
        return values

    def kl_sum(self) -> torch.Tensor:
        prior_mu = float(self.variational_cfg.prior_mu)
        prior_sigma = float(self.variational_cfg.prior_sigma)
        result = None
        for binding in self.bindings:
            if binding.kind != "bayesian":
                continue
            module = self._module(binding.module_path)
            mu = getattr(module, str(binding.mu_name))
            sigma = F.softplus(getattr(module, str(binding.rho_name)))
            sigma = torch.clamp(sigma, min=1.0e-12)
            value = (
                torch.log(torch.as_tensor(prior_sigma, device=mu.device, dtype=mu.dtype) / sigma)
                + (sigma.square() + (mu - prior_mu).square()) / (2.0 * prior_sigma**2)
                - 0.5
            ).sum()
            result = value if result is None else result + value
        if result is None:
            raise RuntimeError("No Bayesian parameters were found")
        return result

    def apply_variance_floor(
        self,
        global_rho_vector: torch.Tensor,
        ratio: float,
    ) -> float:
        """Apply sigma_local >= ratio * sigma_global and return clipped fraction."""
        global_rho = torch.as_tensor(global_rho_vector).reshape(-1)
        if int(global_rho.numel()) != self.bayesian_dimension:
            raise ValueError("global_rho_vector has wrong dimension")
        offset = 0
        clipped = 0
        total = 0
        with torch.no_grad():
            for binding in self.bindings:
                if binding.kind != "bayesian":
                    continue
                module = self._module(binding.module_path)
                rho = getattr(module, str(binding.rho_name))
                n = binding.spec.numel
                global_chunk = global_rho[offset : offset + n].reshape(binding.spec.shape)
                global_chunk = global_chunk.to(device=rho.device, dtype=rho.dtype)
                floor_sigma = float(ratio) * F.softplus(global_chunk)
                local_sigma = F.softplus(rho)
                mask = local_sigma < floor_sigma
                clipped += int(torch.count_nonzero(mask).detach().cpu())
                total += int(mask.numel())
                if bool(torch.any(mask)):
                    rho.copy_(torch.where(mask, _inverse_softplus(floor_sigma), rho))
                offset += n
        return float(clipped / max(1, total))

    def sigma_stats(self) -> tuple[float, float, float]:
        sigma = self.sigma_vector().detach()
        return (
            float(sigma.mean().cpu()),
            float(sigma.min().cpu()),
            float(sigma.max().cpu()),
        )


def build_initial_states(
    *,
    dataset: str,
    model_cfg: ModelConfig,
    variational_cfg: VariationalConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return (Bayes state, matched FedAvg mean vector, bayesian_d, det_d)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        base = build_model(dataset, model_cfg.name, model_cfg.num_classes)
        layout = ParameterLayout(base)
        adapter = BayesianTorchStateAdapter(base, layout, variational_cfg)
        bayes_state = adapter.state_vector()
        mean_state = adapter.mean_model_vector()

    if str(dataset).lower() == "cifar10":
        if adapter.bayesian_dimension != CIFAR10_RESNET56_GN_BAYESIAN_PARAMETER_COUNT:
            raise RuntimeError(
                f"Expected Bayesian d={CIFAR10_RESNET56_GN_BAYESIAN_PARAMETER_COUNT}, "
                f"got {adapter.bayesian_dimension}"
            )
        if adapter.deterministic_dimension != CIFAR10_RESNET56_GN_GROUPNORM_PARAMETER_COUNT:
            raise RuntimeError(
                f"Expected deterministic GN d={CIFAR10_RESNET56_GN_GROUPNORM_PARAMETER_COUNT}, "
                f"got {adapter.deterministic_dimension}"
            )
    return (
        bayes_state,
        mean_state,
        adapter.bayesian_dimension,
        adapter.deterministic_dimension,
    )
