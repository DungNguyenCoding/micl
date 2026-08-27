"""Deterministic local training for FedAvg, FedProx, and SCAFFOLD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    new_client_control: Optional[np.ndarray] = None
    control_delta: Optional[np.ndarray] = None


def train_deterministic(
    *,
    model: torch.nn.Module,
    layout: ParameterLayout,
    global_vector: np.ndarray,
    loader: DataLoader,
    train_cfg: TrainingConfig,
    device: torch.device,
    method: str,
    seed: int,
    learning_rate: Optional[float] = None,
    global_control: Optional[np.ndarray] = None,
    client_control: Optional[np.ndarray] = None,
) -> DeterministicResult:
    if method not in {"fedavg", "fedprox", "scaffold"}:
        raise ValueError(f"Unsupported deterministic method: {method}")

    torch.manual_seed(int(seed))
    model = model.to(device)
    model.train()
    global_tensor = torch.as_tensor(global_vector, dtype=torch.float32, device=device)
    layout.load_model_vector(model, global_tensor)

    global_reference = global_tensor.detach().clone()
    if method == "scaffold":
        if global_control is None or client_control is None:
            raise ValueError("SCAFFOLD requires global and client control variates")
        global_control_t = torch.as_tensor(
            global_control, dtype=torch.float32, device=device
        )
        client_control_t = torch.as_tensor(
            client_control, dtype=torch.float32, device=device
        )
        global_control_parts = layout.split_vector(global_control_t)
        client_control_parts = layout.split_vector(client_control_t)
    else:
        global_control_parts = []
        client_control_parts = []

    effective_lr = (
        float(train_cfg.learning_rate)
        if learning_rate is None
        else float(learning_rate)
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=effective_lr,
        momentum=float(train_cfg.momentum),
        weight_decay=float(train_cfg.weight_decay),
    )
    total_loss = 0.0
    total_examples = 0
    local_steps = 0

    for _ in range(train_cfg.local_epochs):
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            task_loss = F.cross_entropy(logits, targets)
            loss = task_loss

            if method == "fedprox":
                current = layout.flatten_model(model)
                proximal = 0.5 * train_cfg.fedprox_mu * torch.sum(
                    (current - global_reference).square()
                )
                loss = loss + proximal

            loss.backward()

            if method == "scaffold":
                for parameter, global_c, local_c in zip(
                    model.parameters(), global_control_parts, client_control_parts
                ):
                    if parameter.grad is None:
                        continue
                    parameter.grad.add_(
                        global_c.to(parameter.grad.device)
                        - local_c.to(parameter.grad.device)
                    )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=train_cfg.gradient_clip_norm
            )
            optimizer.step()

            batch_size = int(targets.numel())
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            local_steps += 1

    local_vector_t = layout.flatten_model(model).detach()
    average_loss = total_loss / max(1, total_examples)

    if method != "scaffold":
        return DeterministicResult(
            model_vector=local_vector_t.cpu().numpy().astype(np.float32),
            average_loss=float(average_loss),
            local_steps=local_steps,
        )

    assert global_control is not None and client_control is not None
    denominator = max(1, local_steps) * effective_lr
    new_client_control_t = (
        torch.as_tensor(client_control, dtype=torch.float32, device=device)
        - torch.as_tensor(global_control, dtype=torch.float32, device=device)
        + (global_reference - local_vector_t) / denominator
    )
    old_client_control_t = torch.as_tensor(
        client_control, dtype=torch.float32, device=device
    )
    control_delta_t = new_client_control_t - old_client_control_t
    return DeterministicResult(
        model_vector=local_vector_t.cpu().numpy().astype(np.float32),
        average_loss=float(average_loss),
        local_steps=local_steps,
        new_client_control=new_client_control_t.cpu().numpy().astype(np.float32),
        control_delta=control_delta_t.cpu().numpy().astype(np.float32),
    )
