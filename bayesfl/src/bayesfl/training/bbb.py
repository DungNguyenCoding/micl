"""Local Bayes-by-Backprop training with a Gaussian scale-mixture prior."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from bayesfl.config import ExperimentConfig, round_learning_rate
from bayesfl.models.factory import count_bayesian_random_variables
from bayesfl.posterior.gaussian import inverse_softplus_np, softplus_np
from bayesfl.posterior.scale_mixture import bbb_complexity_cost


def _kl_beta(batch_index: int, num_batches: int, scheme: str) -> float:
    if num_batches <= 0:
        return 1.0
    if scheme == "equal_minibatch":
        return 1.0 / float(num_batches)
    if scheme == "blundell_geometric":
        # Eq. (9) special weighting discussed by Blundell et al.
        weights = np.power(2.0, np.arange(num_batches - 1, -1, -1, dtype=np.float64))
        weights /= weights.sum()
        return float(weights[batch_index])
    raise ValueError(f"Unknown BBB KL scheme: {scheme}")


def _warmup_multiplier(cfg: ExperimentConfig, server_round: int) -> float:
    if not cfg.bbb.kl_weight_schedule:
        return 1.0
    warm = max(1, cfg.bbb.kl_warmup_rounds)
    return min(1.0, server_round / float(warm))


def _make_optimizer(model: nn.Module, cfg: ExperimentConfig, lr: float):
    rho_params = []
    base_params = []
    for name, param in model.named_parameters():
        if "rho_" in name:
            rho_params.append(param)
        else:
            base_params.append(param)
    groups = [{"params": base_params, "lr": lr}]
    if rho_params:
        groups.append({"params": rho_params, "lr": lr * cfg.bbb.rho_lr_multiplier})
    return torch.optim.SGD(
        groups,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )


def apply_bbb_variance_floor(
    model: nn.Module,
    global_parameters: dict[str, np.ndarray],
    ratio: float,
) -> float:
    """Enforce local sigma >= ratio * global sigma on every rho tensor."""
    total = 0
    changed = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "rho_" not in name:
                continue
            global_rho = np.asarray(global_parameters[name])
            local_rho = param.detach().cpu().numpy()
            sigma_local = softplus_np(local_rho)
            sigma_global = softplus_np(global_rho)
            floor = ratio * sigma_global
            mask = sigma_local < floor
            sigma_new = np.maximum(sigma_local, floor)
            rho_new = inverse_softplus_np(sigma_new)
            param.copy_(torch.as_tensor(rho_new, device=param.device, dtype=param.dtype))
            total += mask.size
            changed += int(np.count_nonzero(mask))
    return changed / float(total) if total else 0.0


def train_bbb(
    model: nn.Module,
    loader: DataLoader,
    cfg: ExperimentConfig,
    *,
    server_round: int,
    device: torch.device,
    client_size: int,
    average_client_size: float,
    global_parameter_arrays: Sequence[np.ndarray],
) -> Dict[str, float]:
    model.train()
    names = [name for name, _ in model.named_parameters()]
    global_map = {name: np.asarray(array).copy() for name, array in zip(names, global_parameter_arrays)}
    d = count_bayesian_random_variables(model)
    if d <= 0:
        raise RuntimeError("BBB model contains no Bayesian random variables")

    lr = round_learning_rate(cfg.training, server_round)
    optimizer = _make_optimizer(model, cfg, lr)
    base_kl_weight = cfg.resolved_kl_weight(d)
    size_scale = client_size / max(1e-12, average_client_size) if cfg.bbb.lambda_scale_by_size else 1.0
    kl_weight = base_kl_weight * size_scale * _warmup_multiplier(cfg, server_round)

    objective_sum = 0.0
    task_sum = 0.0
    complexity_per_weight_sum = 0.0
    seen = 0
    correct = 0
    steps = 0
    rho_grad_l2_sum = 0.0
    rho_grad_nonzero_steps = 0
    num_batches = max(1, len(loader))

    for _ in range(cfg.training.local_epochs):
        for batch_index, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            beta = _kl_beta(batch_index, num_batches, cfg.bbb.kl_scheme)
            mc_loss = torch.zeros((), device=device)
            mc_task = torch.zeros((), device=device)
            mc_complexity = torch.zeros((), device=device)
            last_logits = None
            for _mc in range(cfg.bbb.mc_train):
                logits = model(x)
                task_loss = F.cross_entropy(logits, y, reduction="mean")
                complexity = bbb_complexity_cost(
                    model,
                    prior_type=cfg.bbb.prior_type,
                    prior_mean=cfg.bbb.prior_mean,
                    prior_sigma=cfg.bbb.prior_sigma,
                    pi=cfg.bbb.prior_pi,
                    sigma1=cfg.bbb.prior_sigma1,
                    sigma2=cfg.bbb.prior_sigma2,
                )
                loss = task_loss + (kl_weight * beta * complexity)
                mc_loss = mc_loss + loss
                mc_task = mc_task + task_loss.detach()
                mc_complexity = mc_complexity + complexity.detach()
                last_logits = logits
            mc_loss = mc_loss / float(cfg.bbb.mc_train)
            mc_loss.backward()

            rho_grad_sq = 0.0
            for name, param in model.named_parameters():
                if "rho_" in name and param.grad is not None:
                    rho_grad_sq += float(param.grad.detach().pow(2).sum())
            rho_grad_l2 = rho_grad_sq ** 0.5
            rho_grad_l2_sum += rho_grad_l2
            if rho_grad_l2 > 0.0:
                rho_grad_nonzero_steps += 1

            if cfg.training.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            optimizer.step()

            batch = int(y.numel())
            objective_sum += float(mc_loss.detach()) * batch
            task_sum += float(mc_task / cfg.bbb.mc_train) * batch
            complexity_per_weight_sum += float(mc_complexity / cfg.bbb.mc_train) / float(d)
            if last_logits is not None:
                correct += int((last_logits.argmax(dim=1) == y).sum().detach())
            seen += batch
            steps += 1

    floor_fraction = apply_bbb_variance_floor(model, global_map, cfg.bbb.variance_floor_ratio)

    rho_update_sq = 0.0
    for name, param in model.named_parameters():
        if "rho_" not in name:
            continue
        before = np.asarray(global_map[name], dtype=np.float64)
        after = param.detach().cpu().numpy().astype(np.float64, copy=False)
        delta = after - before
        rho_update_sq += float(np.sum(delta * delta))

    return {
        "train_loss": objective_sum / max(1, seen),
        "task_loss": task_sum / max(1, seen),
        "train_accuracy": correct / max(1, seen),
        "complexity_per_weight": complexity_per_weight_sum / max(1, steps),
        "resolved_kl_weight": float(base_kl_weight),
        "effective_kl_weight": float(kl_weight),
        "variance_floor_fraction": float(floor_fraction),
        "bayesian_dimension": float(d),
        "lr": lr,
        "local_steps": float(steps),
        "rho_grad_l2_mean": rho_grad_l2_sum / max(1, steps),
        "rho_grad_nonzero_fraction": rho_grad_nonzero_steps / max(1, steps),
        "rho_update_l2": float(rho_update_sq ** 0.5),
    }
