"""Model factory and parameter-count helpers."""

from __future__ import annotations

import torch
from torch import nn

from bayesfl.config import ExperimentConfig
from .cifar_resnet import CifarResNet56GN8
from .mnist_mlp import MNISTMLP
from .paper_cnn import CifarPaperBasicCNN


def _build_model(cfg: ExperimentConfig, *, bayesian: bool) -> nn.Module:
    kwargs = dict(
        bayesian=bayesian,
        posterior_mu_init=cfg.bbb.posterior_mu_init,
        posterior_rho_init=cfg.bbb.posterior_rho_init,
    )
    if cfg.data.dataset == "mnist":
        return MNISTMLP(**kwargs)
    if cfg.data.dataset == "cifar10":
        if cfg.model.name == "paper_basiccnn":
            return CifarPaperBasicCNN(**kwargs)
        if cfg.model.name == "resnet56_gn8":
            return CifarResNet56GN8(groups=cfg.model.group_norm_groups, **kwargs)
    raise ValueError(f"Unsupported dataset/model: {cfg.data.dataset}/{cfg.model.name}")


def build_model(cfg: ExperimentConfig) -> nn.Module:
    return _build_model(cfg, bayesian=(cfg.method == "bbb"))


def _deterministic_parameter_name(bayesian_name: str) -> str:
    if bayesian_name.endswith(".mu_kernel"):
        return bayesian_name[: -len(".mu_kernel")] + ".weight"
    if bayesian_name.endswith(".mu_weight"):
        return bayesian_name[: -len(".mu_weight")] + ".weight"
    if bayesian_name.endswith(".mu_bias"):
        return bayesian_name[: -len(".mu_bias")] + ".bias"
    return bayesian_name


def _copy_deterministic_initialization(
    deterministic_model: nn.Module,
    bayesian_model: nn.Module,
) -> None:
    deterministic_params = dict(deterministic_model.named_parameters())
    copied = 0
    with torch.no_grad():
        for bayes_name, bayes_param in bayesian_model.named_parameters():
            if "rho_" in bayes_name:
                continue
            deterministic_name = _deterministic_parameter_name(bayes_name)
            if deterministic_name not in deterministic_params:
                raise RuntimeError(
                    f"No deterministic parameter for {bayes_name} -> {deterministic_name}"
                )
            source = deterministic_params[deterministic_name]
            if source.shape != bayes_param.shape:
                raise RuntimeError(
                    f"Shape mismatch for {bayes_name}: "
                    f"{tuple(bayes_param.shape)} vs {tuple(source.shape)}"
                )
            bayes_param.copy_(source.to(device=bayes_param.device, dtype=bayes_param.dtype))
            copied += 1
    if copied == 0:
        raise RuntimeError("No deterministic parameters were copied into BBB model")


def count_bayesian_random_variables(model: nn.Module) -> int:
    """Count one random variable per Bayesian weight/bias element."""
    total = 0
    for name, param in model.named_parameters():
        if any(token in name for token in ("mu_kernel", "mu_weight", "mu_bias")):
            total += param.numel()
    return int(total)


def _seed_model_initialization(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_model(cfg: ExperimentConfig) -> nn.Module:
    """Build the reproducible initial server model.

    When BBB matched initialization is enabled, the deterministic network is
    initialized first and copied exactly into the Bayesian posterior means.
    This keeps the common paper architecture/initialization identical across
    FedAvg, FOLA, and the BBB extension.
    """
    seed = int(cfg.runtime.seed)
    _seed_model_initialization(seed)
    if cfg.method != "bbb" or not cfg.bbb.match_deterministic_init:
        return build_model(cfg)

    deterministic_model = _build_model(cfg, bayesian=False)
    _seed_model_initialization(seed)
    bayesian_model = _build_model(cfg, bayesian=True)
    _copy_deterministic_initialization(deterministic_model, bayesian_model)
    return bayesian_model
