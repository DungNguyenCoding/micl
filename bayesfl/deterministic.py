"""Deterministic local SGD used by the FedAvg baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import TrainingConfig
from serialization import ParameterLayout


@dataclass
class DeterministicResult:
    model_vector: np.ndarray
    average_loss: float
    local_steps: int
    update_l2: float
    update_max_abs: float


def train_fedavg(
    *,
    model: torch.nn.Module,
    layout: ParameterLayout,
    global_vector: np.ndarray,
    loader: DataLoader,
    train_cfg: TrainingConfig,
    device: torch.device,
    seed: int,
    learning_rate: float,
) -> DeterministicResult:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = model.to(device)
    model.train()
    global_np = np.asarray(global_vector, dtype=np.float32).reshape(-1)
    layout.load_model_vector(model, global_np)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(learning_rate),
        momentum=float(train_cfg.momentum),
        weight_decay=float(train_cfg.weight_decay),
    )
    total_loss = 0.0
    total_examples = 0
    steps = 0

    for _ in range(int(train_cfg.local_epochs)):
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            if float(train_cfg.gradient_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_cfg.gradient_clip_norm)
                )
            optimizer.step()
            n = int(targets.numel())
            total_loss += float(loss.detach().cpu()) * n
            total_examples += n
            steps += 1

    local = layout.flatten_model(model).detach().cpu().numpy().astype(np.float32)
    delta = local.astype(np.float64) - global_np.astype(np.float64)
    return DeterministicResult(
        model_vector=local,
        average_loss=float(total_loss / max(1, total_examples)),
        local_steps=int(steps),
        update_l2=float(np.linalg.norm(delta)),
        update_max_abs=float(np.max(np.abs(delta))) if delta.size else 0.0,
    )
