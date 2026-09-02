"""Bayesian-Torch adapter for the AirComp Gaussian posterior state.

This module is intentionally independent of the FL client/server code.

The AirComp protocol communicates one Gaussian mean vector and one
Gaussian precision vector in the deterministic model's ParameterLayout.

Bayesian-Torch internally stores:
    posterior mean: mu_*
    posterior scale: softplus(rho_*)

This adapter maps between those two representations while preserving
the exact deterministic parameter ordering.

Important:
    Never construct ParameterLayout from the Bayesian-Torch model.
    The Bayesian model contains both mu and rho parameters and therefore
    does not have the communication dimension of the deterministic model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

from serialization import ParameterLayout, TensorSpec


_MIN_SIGMA = 1.0e-8


def _inverse_softplus(
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Stable inverse of softplus for strictly positive sigma."""

    sigma = torch.clamp(
        sigma,
        min=_MIN_SIGMA,
    )

    # log(expm1(x)) is accurate for the small scales used here.
    return torch.log(torch.expm1(sigma))


def _sigma_from_rho(
    rho: torch.Tensor,
) -> torch.Tensor:
    return F.softplus(rho)


def _gaussian_kl_sum(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
) -> torch.Tensor:
    """Sum KL[N(mu_q,sigma_q^2) || N(mu_p,sigma_p^2)].

    Bayesian-Torch's BaseVariationalLayer_.kl_div() uses the same
    elementwise expression but returns its mean.  The AirComp/Pyro
    implementation uses a full parameter-vector KL, so for a fair
    backend comparison we use the sum here.
    """

    sigma_q = torch.clamp(
        sigma_q,
        min=_MIN_SIGMA,
    )
    sigma_p = torch.clamp(
        sigma_p,
        min=_MIN_SIGMA,
    )

    elementwise = (
        torch.log(sigma_p)
        - torch.log(sigma_q)
        + (
            sigma_q.square()
            + (mu_q - mu_p).square()
        )
        / (
            2.0
            * sigma_p.square()
        )
        - 0.5
    )

    return elementwise.sum()


class VariationalGroupNorm(nn.Module):
    """Bayesian affine GroupNorm compatible with Bayesian-Torch.

    Bayesian-Torch 0.5.0 does not convert nn.GroupNorm.  This wrapper
    preserves the GroupNorm operation while assigning independent
    Gaussian posteriors to its affine weight and bias coordinates.

    Parameter naming intentionally mirrors Bayesian-Torch Linear layers:
        mu_weight
        rho_weight
        mu_bias
        rho_bias
        prior_weight_mu
        prior_weight_sigma
        prior_bias_mu
        prior_bias_sigma
    """

    def __init__(
        self,
        source: nn.GroupNorm,
    ) -> None:
        super().__init__()

        self.num_groups = int(
            source.num_groups
        )
        self.num_channels = int(
            source.num_channels
        )
        self.eps = float(
            source.eps
        )
        self.affine = bool(
            source.affine
        )

        if not self.affine:
            self.register_parameter(
                "mu_weight",
                None,
            )
            self.register_parameter(
                "rho_weight",
                None,
            )
            self.register_parameter(
                "mu_bias",
                None,
            )
            self.register_parameter(
                "rho_bias",
                None,
            )

            self.register_buffer(
                "prior_weight_mu",
                None,
                persistent=False,
            )
            self.register_buffer(
                "prior_weight_sigma",
                None,
                persistent=False,
            )
            self.register_buffer(
                "prior_bias_mu",
                None,
                persistent=False,
            )
            self.register_buffer(
                "prior_bias_sigma",
                None,
                persistent=False,
            )

            return

        weight = source.weight.detach().clone()
        bias = source.bias.detach().clone()

        initial_sigma = torch.full_like(
            weight,
            0.01,
        )

        initial_rho = _inverse_softplus(
            initial_sigma
        )

        self.mu_weight = nn.Parameter(
            weight.clone()
        )
        self.rho_weight = nn.Parameter(
            initial_rho.clone()
        )

        self.mu_bias = nn.Parameter(
            bias.clone()
        )
        self.rho_bias = nn.Parameter(
            initial_rho.clone()
        )

        self.register_buffer(
            "prior_weight_mu",
            weight.clone(),
            persistent=False,
        )
        self.register_buffer(
            "prior_weight_sigma",
            initial_sigma.clone(),
            persistent=False,
        )

        self.register_buffer(
            "prior_bias_mu",
            bias.clone(),
            persistent=False,
        )
        self.register_buffer(
            "prior_bias_sigma",
            initial_sigma.clone(),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if not self.affine:
            return F.group_norm(
                x,
                self.num_groups,
                None,
                None,
                self.eps,
            )

        sigma_weight = _sigma_from_rho(
            self.rho_weight
        )
        sigma_bias = _sigma_from_rho(
            self.rho_bias
        )

        weight = (
            self.mu_weight
            + sigma_weight
            * torch.randn_like(
                self.mu_weight
            )
        )

        bias = (
            self.mu_bias
            + sigma_bias
            * torch.randn_like(
                self.mu_bias
            )
        )

        return F.group_norm(
            x,
            self.num_groups,
            weight,
            bias,
            self.eps,
        )

    def kl_loss(
        self,
    ) -> torch.Tensor:
        """Bayesian-Torch-style mean KL for compatibility."""

        if not self.affine:
            return torch.tensor(
                0.0
            )

        kl_weight = (
            _gaussian_kl_sum(
                self.mu_weight,
                _sigma_from_rho(
                    self.rho_weight
                ),
                self.prior_weight_mu,
                self.prior_weight_sigma,
            )
            / self.mu_weight.numel()
        )

        kl_bias = (
            _gaussian_kl_sum(
                self.mu_bias,
                _sigma_from_rho(
                    self.rho_bias
                ),
                self.prior_bias_mu,
                self.prior_bias_sigma,
            )
            / self.mu_bias.numel()
        )

        return kl_weight + kl_bias


def _replace_group_norms(
    module: nn.Module,
) -> None:
    """Replace GroupNorm recursively, preserving module paths."""

    for name, child in list(
        module.named_children()
    ):
        if isinstance(
            child,
            nn.GroupNorm,
        ):
            setattr(
                module,
                name,
                VariationalGroupNorm(
                    child
                ),
            )
        else:
            _replace_group_norms(
                child
            )


@dataclass(frozen=True)
class CoordinateBinding:
    """Map one deterministic TensorSpec to Bayesian-Torch tensors."""

    spec: TensorSpec
    module_path: str
    parameter_name: str

    mu_name: str
    rho_name: str

    prior_mu_name: str
    prior_sigma_name: str

    module_kind: str


class BayesianTorchParameterAdapter:
    """Bridge deterministic AirComp vectors and Bayesian-Torch tensors."""

    def __init__(
        self,
        model: nn.Module,
        layout: ParameterLayout,
    ) -> None:

        self.layout = layout

        # Source model is used only for deterministic parameter topology.
        self._source_model = model

        # Bayesian model is a private local-training copy.
        self.model = copy.deepcopy(
            model
        )

        # Bayesianize GroupNorm first.  The standard Bayesian-Torch
        # converter will leave these custom modules alone.
        _replace_group_norms(
            self.model
        )

        # Placeholder values only.
        #
        # All coordinate-wise priors and posteriors are overwritten
        # explicitly by this adapter before local training.
        prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -4.6,
            "type": "Reparameterization",
            "moped_enable": False,
            "moped_delta": 0.5,
        }

        dnn_to_bnn(
            self.model,
            prior_parameters,
        )

        self.bindings: List[
            CoordinateBinding
        ] = self._build_bindings()

        mapped = sum(
            binding.spec.numel
            for binding
            in self.bindings
        )

        if mapped != self.layout.total_numel:
            raise RuntimeError(
                "Bayesian-Torch coordinate mapping "
                f"covers {mapped:,} parameters, "
                f"expected {self.layout.total_numel:,}"
            )

    def _build_bindings(
        self,
    ) -> List[CoordinateBinding]:

        bindings: List[
            CoordinateBinding
        ] = []

        for spec in self.layout.specs:

            module_path, sep, parameter_name = (
                spec.name.rpartition(".")
            )

            if not sep:
                module_path = ""
                parameter_name = (
                    spec.name
                )

            source_module = (
                self._source_model
                if module_path == ""
                else self._source_model.get_submodule(
                    module_path
                )
            )

            bayesian_module = (
                self.model
                if module_path == ""
                else self.model.get_submodule(
                    module_path
                )
            )

            if isinstance(
                source_module,
                nn.Conv2d,
            ):
                if parameter_name == "weight":
                    binding = CoordinateBinding(
                        spec=spec,
                        module_path=module_path,
                        parameter_name=parameter_name,
                        mu_name="mu_kernel",
                        rho_name="rho_kernel",
                        prior_mu_name="prior_weight_mu",
                        prior_sigma_name="prior_weight_sigma",
                        module_kind="conv2d",
                    )
                elif parameter_name == "bias":
                    binding = CoordinateBinding(
                        spec=spec,
                        module_path=module_path,
                        parameter_name=parameter_name,
                        mu_name="mu_bias",
                        rho_name="rho_bias",
                        prior_mu_name="prior_bias_mu",
                        prior_sigma_name="prior_bias_sigma",
                        module_kind="conv2d",
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported Conv2d parameter: {spec.name}"
                    )

            elif isinstance(
                source_module,
                nn.Linear,
            ):
                if parameter_name == "weight":
                    binding = CoordinateBinding(
                        spec=spec,
                        module_path=module_path,
                        parameter_name=parameter_name,
                        mu_name="mu_weight",
                        rho_name="rho_weight",
                        prior_mu_name="prior_weight_mu",
                        prior_sigma_name="prior_weight_sigma",
                        module_kind="linear",
                    )
                elif parameter_name == "bias":
                    binding = CoordinateBinding(
                        spec=spec,
                        module_path=module_path,
                        parameter_name=parameter_name,
                        mu_name="mu_bias",
                        rho_name="rho_bias",
                        prior_mu_name="prior_bias_mu",
                        prior_sigma_name="prior_bias_sigma",
                        module_kind="linear",
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported Linear parameter: {spec.name}"
                    )

            elif isinstance(
                source_module,
                nn.GroupNorm,
            ):
                if parameter_name == "weight":
                    binding = CoordinateBinding(
                        spec=spec,
                        module_path=module_path,
                        parameter_name=parameter_name,
                        mu_name="mu_weight",
                        rho_name="rho_weight",
                        prior_mu_name="prior_weight_mu",
                        prior_sigma_name="prior_weight_sigma",
                        module_kind="groupnorm",
                    )
                elif parameter_name == "bias":
                    binding = CoordinateBinding(
                        spec=spec,
                        module_path=module_path,
                        parameter_name=parameter_name,
                        mu_name="mu_bias",
                        rho_name="rho_bias",
                        prior_mu_name="prior_bias_mu",
                        prior_sigma_name="prior_bias_sigma",
                        module_kind="groupnorm",
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported GroupNorm parameter: {spec.name}"
                    )

            else:
                raise RuntimeError(
                    "Bayesian-Torch adapter encountered "
                    "an unsupported parameterized module:\n"
                    f"  parameter={spec.name}\n"
                    f"  module={type(source_module).__name__}"
                )

            # Validate the converted module immediately.
            for attribute in (
                binding.mu_name,
                binding.rho_name,
                binding.prior_mu_name,
                binding.prior_sigma_name,
            ):
                if not hasattr(
                    bayesian_module,
                    attribute,
                ):
                    raise RuntimeError(
                        f"{spec.name}: converted module "
                        f"{type(bayesian_module).__name__} "
                        f"is missing attribute {attribute!r}"
                    )

            mu = getattr(
                bayesian_module,
                binding.mu_name,
            )

            if mu is None:
                raise RuntimeError(
                    f"{spec.name}: Bayesian posterior mean is None"
                )

            if tuple(mu.shape) != tuple(
                spec.shape
            ):
                raise RuntimeError(
                    f"{spec.name}: Bayesian tensor shape "
                    f"{tuple(mu.shape)} != expected {spec.shape}"
                )

            bindings.append(
                binding
            )

        return bindings

    def _module_for(
        self,
        binding: CoordinateBinding,
    ) -> nn.Module:
        if binding.module_path == "":
            return self.model

        return self.model.get_submodule(
            binding.module_path
        )

    def _chunks(
        self,
        vector: np.ndarray | torch.Tensor,
    ) -> Iterator[
        tuple[
            CoordinateBinding,
            torch.Tensor,
        ]
    ]:

        value = torch.as_tensor(
            vector
        ).reshape(-1)

        if (
            value.numel()
            != self.layout.total_numel
        ):
            raise ValueError(
                f"Vector contains {value.numel()} values, "
                f"expected {self.layout.total_numel}"
            )

        offset = 0

        for binding in self.bindings:

            count = binding.spec.numel

            chunk = value[
                offset:
                offset + count
            ].reshape(
                binding.spec.shape
            )

            yield (
                binding,
                chunk,
            )

            offset += count

    def set_prior(
        self,
        mean: np.ndarray | torch.Tensor,
        precision: np.ndarray | torch.Tensor,
    ) -> None:
        """Set coordinate-wise Gaussian prior N(mean, precision^-1)."""

        precision_vector = torch.as_tensor(
            precision,
            dtype=torch.float64,
        ).reshape(-1)

        if (
            precision_vector.numel()
            != self.layout.total_numel
        ):
            raise ValueError(
                "Prior precision has wrong dimension"
            )

        if torch.any(
            ~torch.isfinite(
                precision_vector
            )
        ):
            raise ValueError(
                "Prior precision contains non-finite values"
            )

        if torch.any(
            precision_vector <= 0
        ):
            raise ValueError(
                "Prior precision must be strictly positive"
            )

        sigma_vector = (
            torch.rsqrt(
                precision_vector
            )
        )

        mean_chunks = list(
            self._chunks(mean)
        )

        sigma_chunks = list(
            self._chunks(
                sigma_vector
            )
        )

        with torch.no_grad():

            for (
                binding,
                mean_chunk,
            ), (
                sigma_binding,
                sigma_chunk,
            ) in zip(
                mean_chunks,
                sigma_chunks,
            ):

                if (
                    binding
                    != sigma_binding
                ):
                    raise RuntimeError(
                        "Internal coordinate mapping mismatch"
                    )

                module = self._module_for(
                    binding
                )

                prior_mu = getattr(
                    module,
                    binding.prior_mu_name,
                )

                prior_sigma = getattr(
                    module,
                    binding.prior_sigma_name,
                )

                prior_mu.copy_(
                    mean_chunk.to(
                        device=prior_mu.device,
                        dtype=prior_mu.dtype,
                    )
                )

                prior_sigma.copy_(
                    sigma_chunk.to(
                        device=prior_sigma.device,
                        dtype=prior_sigma.dtype,
                    )
                )

    def set_posterior_mean(
        self,
        mean: np.ndarray | torch.Tensor,
    ) -> None:

        with torch.no_grad():

            for binding, chunk in self._chunks(
                mean
            ):

                module = self._module_for(
                    binding
                )

                parameter = getattr(
                    module,
                    binding.mu_name,
                )

                parameter.copy_(
                    chunk.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )

    def set_posterior_precision(
        self,
        precision: np.ndarray | torch.Tensor,
    ) -> None:

        precision_vector = torch.as_tensor(
            precision,
            dtype=torch.float64,
        ).reshape(-1)

        if (
            precision_vector.numel()
            != self.layout.total_numel
        ):
            raise ValueError(
                "Posterior precision has wrong dimension"
            )

        if torch.any(
            ~torch.isfinite(
                precision_vector
            )
        ):
            raise ValueError(
                "Posterior precision contains non-finite values"
            )

        if torch.any(
            precision_vector <= 0
        ):
            raise ValueError(
                "Posterior precision must be strictly positive"
            )

        sigma_vector = torch.rsqrt(
            precision_vector
        )

        rho_vector = _inverse_softplus(
            sigma_vector
        )

        with torch.no_grad():

            for binding, chunk in self._chunks(
                rho_vector
            ):

                module = self._module_for(
                    binding
                )

                parameter = getattr(
                    module,
                    binding.rho_name,
                )

                parameter.copy_(
                    chunk.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )

    def posterior_mean_vector(
        self,
    ) -> torch.Tensor:

        values = []

        for binding in self.bindings:

            module = self._module_for(
                binding
            )

            mu = getattr(
                module,
                binding.mu_name,
            )

            values.append(
                mu.reshape(-1)
            )

        return torch.cat(
            values
        )

    def posterior_precision_vector(
        self,
    ) -> torch.Tensor:

        values = []

        for binding in self.bindings:

            module = self._module_for(
                binding
            )

            rho = getattr(
                module,
                binding.rho_name,
            )

            sigma = _sigma_from_rho(
                rho
            )

            precision = torch.reciprocal(
                sigma.square()
            )

            # Preserve server's precision representation as float64.
            values.append(
                precision.reshape(-1)
                .to(torch.float64)
            )

        return torch.cat(
            values
        )

    def mean_parameters(
        self,
    ) -> List[nn.Parameter]:

        values = []

        for binding in self.bindings:

            module = self._module_for(
                binding
            )

            parameter = getattr(
                module,
                binding.mu_name,
            )

            values.append(
                parameter
            )

        # A single tensor may theoretically occur multiple times in
        # unusual parameter-sharing models.  Remove duplicates.
        return _unique_parameters(
            values
        )

    def scale_parameters(
        self,
    ) -> List[nn.Parameter]:

        values = []

        for binding in self.bindings:

            module = self._module_for(
                binding
            )

            parameter = getattr(
                module,
                binding.rho_name,
            )

            values.append(
                parameter
            )

        return _unique_parameters(
            values
        )

    def set_trainable(
        self,
        *,
        mean: bool,
        scale: bool,
    ) -> None:

        for parameter in self.mean_parameters():
            parameter.requires_grad_(
                bool(mean)
            )

        for parameter in self.scale_parameters():
            parameter.requires_grad_(
                bool(scale)
            )

    def total_kl_sum(
        self,
    ) -> torch.Tensor:
        """Full coordinate-sum Gaussian KL."""

        result = None

        for binding in self.bindings:

            module = self._module_for(
                binding
            )

            mu = getattr(
                module,
                binding.mu_name,
            )

            rho = getattr(
                module,
                binding.rho_name,
            )

            prior_mu = getattr(
                module,
                binding.prior_mu_name,
            )

            prior_sigma = getattr(
                module,
                binding.prior_sigma_name,
            )

            value = _gaussian_kl_sum(
                mu,
                _sigma_from_rho(
                    rho
                ),
                prior_mu,
                prior_sigma,
            )

            result = (
                value
                if result is None
                else result + value
            )

        if result is None:
            raise RuntimeError(
                "Bayesian model has no mapped parameters"
            )

        return result

    def coordinate_count(
        self,
        module_kind: str | None = None,
    ) -> int:

        return sum(
            binding.spec.numel
            for binding in self.bindings
            if (
                module_kind is None
                or binding.module_kind
                == module_kind
            )
        )


def _unique_parameters(
    parameters: List[nn.Parameter],
) -> List[nn.Parameter]:

    result: List[
        nn.Parameter
    ] = []

    seen = set()

    for parameter in parameters:

        key = id(parameter)

        if key in seen:
            continue

        seen.add(key)
        result.append(
            parameter
        )

    return result
