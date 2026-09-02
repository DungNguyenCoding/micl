"""Scale-mixture prior and BBB complexity cost.

The sampled weights are reconstructed from bayesian-torch's last epsilon buffers,
so the likelihood and complexity terms use the same Monte Carlo draw.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


_LOG_2PI = math.log(2.0 * math.pi)


def normal_log_prob(x: torch.Tensor, mean: torch.Tensor | float, sigma: torch.Tensor | float) -> torch.Tensor:
    sigma_t = torch.as_tensor(sigma, device=x.device, dtype=x.dtype).clamp_min(1e-12)
    mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
    return -0.5 * _LOG_2PI - torch.log(sigma_t) - 0.5 * ((x - mean_t) / sigma_t).pow(2)


def scale_mixture_log_prob(
    x: torch.Tensor,
    *,
    pi: float,
    sigma1: float,
    sigma2: float,
) -> torch.Tensor:
    log_p1 = math.log(pi) + normal_log_prob(x, 0.0, sigma1)
    log_p2 = math.log1p(-pi) + normal_log_prob(x, 0.0, sigma2)
    return torch.logaddexp(log_p1, log_p2)


def _sampled_terms(module: nn.Module):
    if hasattr(module, "mu_kernel") and hasattr(module, "rho_kernel"):
        sigma = F.softplus(module.rho_kernel).clamp_min(1e-12)
        weight = module.mu_kernel + sigma * module.eps_kernel
        yield weight, module.mu_kernel, sigma
        if getattr(module, "mu_bias", None) is not None:
            sigma_b = F.softplus(module.rho_bias).clamp_min(1e-12)
            bias = module.mu_bias + sigma_b * module.eps_bias
            yield bias, module.mu_bias, sigma_b
    elif hasattr(module, "mu_weight") and hasattr(module, "rho_weight"):
        sigma = F.softplus(module.rho_weight).clamp_min(1e-12)
        weight = module.mu_weight + sigma * module.eps_weight
        yield weight, module.mu_weight, sigma
        if getattr(module, "mu_bias", None) is not None:
            sigma_b = F.softplus(module.rho_bias).clamp_min(1e-12)
            bias = module.mu_bias + sigma_b * module.eps_bias
            yield bias, module.mu_bias, sigma_b


def bbb_complexity_cost(
    model: nn.Module,
    *,
    pi: float,
    sigma1: float,
    sigma2: float,
) -> torch.Tensor:
    """Monte Carlo estimate sum_j [log q(w_j) - log p(w_j)]."""
    total = None
    for module in model.modules():
        for sampled, mu, sigma in _sampled_terms(module):
            log_q = normal_log_prob(sampled, mu, sigma)
            log_p = scale_mixture_log_prob(sampled, pi=pi, sigma1=sigma1, sigma2=sigma2)
            value = (log_q - log_p).sum()
            total = value if total is None else total + value
    if total is None:
        param = next(model.parameters())
        return torch.zeros((), device=param.device, dtype=param.dtype)
    return total
