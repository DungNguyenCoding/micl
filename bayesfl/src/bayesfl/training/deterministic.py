"""Standard local SGD used by FedAvg."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from bayesfl.config import ExperimentConfig, round_learning_rate


def train_fedavg(
    model: nn.Module,
    loader: DataLoader,
    cfg: ExperimentConfig,
    *,
    server_round: int,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    lr = round_learning_rate(cfg.training, server_round)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )
    loss_sum = 0.0
    correct = 0
    seen = 0
    steps = 0
    for _ in range(cfg.training.local_epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            batch = int(y.numel())
            loss_sum += float(loss.detach()) * batch
            correct += int((logits.argmax(dim=1) == y).sum().detach())
            seen += batch
            steps += 1
    return {
        "train_loss": loss_sum / max(1, seen),
        "train_accuracy": correct / max(1, seen),
        "lr": lr,
        "local_steps": float(steps),
    }
