"""Federated Online Laplace Approximation local training."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from bayesfl.config import ExperimentConfig, round_learning_rate
from bayesfl.posterior.gaussian import apply_precision_variance_floor, fola_local_precision


def train_fola(
    model: nn.Module,
    loader: DataLoader,
    cfg: ExperimentConfig,
    *,
    server_round: int,
    device: torch.device,
    global_mean_arrays: Sequence[np.ndarray],
    global_precision_arrays: Sequence[np.ndarray],
    client_size: int,
    average_client_size: float,
) -> tuple[list[np.ndarray], Dict[str, float]]:
    """Train the local MAP mean and estimate diagonal online precision.

    The Fisher accumulator follows the paper's Algorithm 1 operationally: the
    task-gradient square is accumulated at every local optimization step. The
    gradient is the minibatch task-loss gradient, a practical diagonal-Fisher
    approximation for large networks.
    """
    model.train()
    params = [p for _, p in model.named_parameters()]
    if len(params) != len(global_mean_arrays) or len(params) != len(global_precision_arrays):
        raise ValueError("FOLA state does not match model parameters")

    global_means = [torch.as_tensor(a, device=device, dtype=p.dtype) for p, a in zip(params, global_mean_arrays)]
    global_precs = [torch.as_tensor(a, device=device, dtype=p.dtype).clamp_min(cfg.fola.precision_min) for p, a in zip(params, global_precision_arrays)]
    fisher_accum = [torch.zeros_like(p) for p in params]

    lr = round_learning_rate(cfg.training, server_round)
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )
    size_scale = client_size / max(average_client_size, 1e-12) if cfg.fola.lambda_scale_by_size else 1.0
    prior_lambda = cfg.fola.prior_lambda * size_scale

    objective_sum = 0.0
    task_sum = 0.0
    prior_sum = 0.0
    correct = 0
    seen = 0
    steps = 0

    for _ in range(cfg.training.local_epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            task_loss = F.cross_entropy(logits, y, reduction="mean")

            # Curvature is estimated from task gradients only, not from the prior.
            task_grads = torch.autograd.grad(
                task_loss,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            for acc, grad in zip(fisher_accum, task_grads):
                if grad is not None:
                    acc.add_(grad.detach().pow(2))

            prior_loss = torch.zeros((), device=device)
            for p, mu_g, prec_g in zip(params, global_means, global_precs):
                prior_loss = prior_loss + 0.5 * torch.sum(prec_g * (p - mu_g).pow(2))
            objective = task_loss + prior_lambda * prior_loss
            objective.backward()
            optimizer.step()

            batch = int(y.numel())
            objective_sum += float(objective.detach()) * batch
            task_sum += float(task_loss.detach()) * batch
            prior_sum += float(prior_loss.detach()) * batch
            correct += int((logits.argmax(dim=1) == y).sum().detach())
            seen += batch
            steps += 1

    precision_arrays: list[np.ndarray] = []
    floor_total = 0.0
    for fisher_t, global_p in zip(fisher_accum, global_precision_arrays):
        fisher = (fisher_t / max(1, steps)).detach().cpu().numpy()
        local_p = fola_local_precision(
            fisher,
            np.asarray(global_p),
            server_round,
            precision_min=cfg.fola.precision_min,
            precision_max=cfg.fola.precision_max,
        )
        local_p, floor_fraction = apply_precision_variance_floor(
            local_p,
            np.asarray(global_p),
            cfg.fola.variance_floor_ratio,
            precision_min=cfg.fola.precision_min,
            precision_max=cfg.fola.precision_max,
        )
        precision_arrays.append(local_p.astype(np.float32, copy=False))
        floor_total += floor_fraction * local_p.size

    total_elements = sum(p.size for p in precision_arrays)
    return precision_arrays, {
        "train_loss": objective_sum / max(1, seen),
        "task_loss": task_sum / max(1, seen),
        "prior_loss": prior_sum / max(1, seen),
        "train_accuracy": correct / max(1, seen),
        "effective_prior_lambda": float(prior_lambda),
        "variance_floor_fraction": floor_total / max(1, total_elements),
        "lr": lr,
        "local_steps": float(steps),
    }
