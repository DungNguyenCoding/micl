"""Model factory and parameter-count helpers."""

from __future__ import annotations

import torch
from torch import nn

from bayesfl.config import ExperimentConfig
from .cifar_resnet import CifarResNet56GN8
from .mnist_mlp import MNISTMLP


def build_model(cfg: ExperimentConfig) -> nn.Module:
    bayesian = cfg.method == "bbb"
    kwargs = dict(
        bayesian=bayesian,
        posterior_mu_init=cfg.bbb.posterior_mu_init,
        posterior_rho_init=cfg.bbb.posterior_rho_init,
    )
    if cfg.data.dataset == "mnist":
        return MNISTMLP(**kwargs)
    if cfg.data.dataset == "cifar10":
        return CifarResNet56GN8(groups=cfg.model.group_norm_groups, **kwargs)
    raise ValueError(f"Unsupported dataset: {cfg.data.dataset}")


def count_bayesian_random_variables(model: nn.Module) -> int:
    """Count one random variable per Bayesian weight/bias element."""
    total = 0
    for name, param in model.named_parameters():
        if any(token in name for token in ("mu_kernel", "mu_weight", "mu_bias")):
            total += param.numel()
    return int(total)


def initialize_model(cfg: ExperimentConfig) -> nn.Module:
    """Build a deterministic initialization reproducibly."""
    torch.manual_seed(cfg.runtime.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.runtime.seed)
    return build_model(cfg)
