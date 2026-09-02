"""One-phase local Bayesian-Torch optimization for BayesAvg."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bayesian_torch_backend import BayesianTorchStateAdapter
from config import ModelConfig, TrainingConfig, VariationalConfig
from serialization import ParameterLayout


@dataclass
class BayesianTrainingResult:
    state_vector: np.ndarray
    mean_model_vector: np.ndarray
    average_loss: float
    average_ce: float
    average_kl_sum: float
    local_steps: int
    kl_weight_base: float
    kl_weight_client: float
    kl_warmup_factor: float
    mu_update_l2: float
    rho_update_l2: float
    deterministic_update_l2: float
    sigma_mean: float
    sigma_min: float
    sigma_max: float
    variance_floor_clipped_fraction: float


def resolved_base_kl_weight(variational_cfg: VariationalConfig, bayesian_dimension: int) -> float:
    if variational_cfg.kl_weight is None:
        return 1.0 / float(bayesian_dimension)
    return float(variational_cfg.kl_weight)


def resolved_kl_weight(
    variational_cfg: VariationalConfig,
    bayesian_dimension: int,
    *,
    server_round: int,
    client_size: int,
    average_client_size: float,
) -> tuple[float, float, float]:
    base = resolved_base_kl_weight(variational_cfg, bayesian_dimension)
    if bool(variational_cfg.kl_weight_schedule):
        warmup = max(1, int(variational_cfg.kl_warmup_rounds))
        factor = min(1.0, max(0.0, float(server_round) / float(warmup)))
    else:
        factor = 1.0
    size_scale = (
        float(client_size) / max(float(average_client_size), 1.0e-12)
        if bool(variational_cfg.lambda_scale_by_size)
        else 1.0
    )
    return base, factor, base * factor * size_scale


def train_bayesavg(
    *,
    deterministic_model: torch.nn.Module,
    layout: ParameterLayout,
    global_state: np.ndarray,
    loader: DataLoader,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    variational_cfg: VariationalConfig,
    device: torch.device,
    seed: int,
    server_round: int,
    learning_rate: float,
    client_size: int,
    average_client_size: float,
) -> BayesianTrainingResult:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    adapter = BayesianTorchStateAdapter(deterministic_model, layout, variational_cfg).to(device)
    adapter.load_state(global_state)
    adapter.model.train()

    initial_mu, initial_rho, initial_det = adapter.split_state(global_state)
    initial_mu = initial_mu.detach().cpu().to(torch.float64)
    initial_rho = initial_rho.detach().cpu().to(torch.float64)
    initial_det = initial_det.detach().cpu().to(torch.float64)
    global_rho = adapter.rho_vector().detach().clone()

    base_kl, warmup_factor, client_kl = resolved_kl_weight(
        variational_cfg,
        adapter.bayesian_dimension,
        server_round=int(server_round),
        client_size=int(client_size),
        average_client_size=float(average_client_size),
    )

    parameters = adapter.trainable_parameters()
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(learning_rate),
        momentum=float(train_cfg.momentum),
        weight_decay=float(train_cfg.weight_decay),
    )

    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_examples = 0
    local_steps = 0
    floor_fractions = []
    mc_train = int(variational_cfg.mc_train)

    for _ in range(int(train_cfg.local_epochs)):
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            ce = torch.zeros((), device=device, dtype=torch.float32)
            for _sample in range(mc_train):
                logits = adapter.model(features)
                if isinstance(logits, tuple):
                    logits = logits[0]
                ce = ce + F.cross_entropy(logits, targets, reduction="mean")
            ce = ce / float(mc_train)
            kl_sum = adapter.kl_sum()
            loss = ce + float(client_kl) * kl_sum
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite Bayesian loss: {loss}")
            loss.backward()

            if float(train_cfg.gradient_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters, max_norm=float(train_cfg.gradient_clip_norm)
                )
            optimizer.step()
            floor_fractions.append(
                adapter.apply_variance_floor(
                    global_rho,
                    float(variational_cfg.variance_floor_ratio),
                )
            )

            n = int(targets.numel())
            total_loss += float(loss.detach().cpu()) * n
            total_ce += float(ce.detach().cpu()) * n
            total_kl += float(kl_sum.detach().cpu()) * n
            total_examples += n
            local_steps += 1

    final_state = adapter.state_vector()
    final_mu, final_rho, final_det = adapter.split_state(final_state)
    mu_delta = final_mu.detach().cpu().to(torch.float64) - initial_mu
    rho_delta = final_rho.detach().cpu().to(torch.float64) - initial_rho
    det_delta = final_det.detach().cpu().to(torch.float64) - initial_det
    sigma_mean, sigma_min, sigma_max = adapter.sigma_stats()

    return BayesianTrainingResult(
        state_vector=final_state,
        mean_model_vector=adapter.mean_model_vector(),
        average_loss=float(total_loss / max(1, total_examples)),
        average_ce=float(total_ce / max(1, total_examples)),
        average_kl_sum=float(total_kl / max(1, total_examples)),
        local_steps=int(local_steps),
        kl_weight_base=float(base_kl),
        kl_weight_client=float(client_kl),
        kl_warmup_factor=float(warmup_factor),
        mu_update_l2=float(torch.linalg.vector_norm(mu_delta).cpu()),
        rho_update_l2=float(torch.linalg.vector_norm(rho_delta).cpu()),
        deterministic_update_l2=(
            float(torch.linalg.vector_norm(det_delta).cpu()) if det_delta.numel() else 0.0
        ),
        sigma_mean=float(sigma_mean),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        variance_floor_clipped_fraction=float(
            np.mean(floor_fractions) if floor_fractions else 0.0
        ),
    )
