"""Thin constructors around bayesian-torch reparameterization layers."""

from __future__ import annotations

from typing import Any


def _layers():
    try:
        from bayesian_torch.layers import Conv2dReparameterization, LinearReparameterization
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "bayesian-torch is required for method=bbb. Install requirements.txt first."
        ) from exc
    return Conv2dReparameterization, LinearReparameterization


def make_bayesian_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int = 0,
    bias: bool = False,
    posterior_mu_init: float = 0.0,
    posterior_rho_init: float = -3.0,
) -> Any:
    conv_cls, _ = _layers()
    return conv_cls(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        prior_mean=0.0,
        prior_variance=1.0,
        posterior_mu_init=posterior_mu_init,
        posterior_rho_init=posterior_rho_init,
        bias=bias,
    )


def make_bayesian_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    posterior_mu_init: float = 0.0,
    posterior_rho_init: float = -3.0,
) -> Any:
    _, linear_cls = _layers()
    return linear_cls(
        in_features=in_features,
        out_features=out_features,
        prior_mean=0.0,
        prior_variance=1.0,
        posterior_mu_init=posterior_mu_init,
        posterior_rho_init=posterior_rho_init,
        bias=bias,
    )


def bayesian_forward(layer: Any, x):
    """Call a bayesian-torch layer without its built-in Gaussian KL."""
    if hasattr(layer, "rho_kernel") or hasattr(layer, "rho_weight"):
        return layer(x, return_kl=False)
    return layer(x)
