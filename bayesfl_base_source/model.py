"""Torch models, local training, evaluation, and parameter utilities."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import RunConfig


class MLP(nn.Module):
    """Simple fully-connected classifier used by FedAvg, OLA, and VI."""

    def __init__(self, input_shape: Sequence[int], num_classes: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        input_dim = int(np.prod(tuple(input_shape)))
        dims = [input_dim] + [int(x) for x in hidden_dims] + [int(num_classes)]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class SmallCNN(nn.Module):
    """Small CNN for deterministic FedAvg/OLA runs on MNIST or CIFAR-10."""

    def __init__(self, input_shape: Sequence[int], num_classes: int) -> None:
        super().__init__()
        channels, height, width = map(int, input_shape)
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        pooled_h = height // 4
        pooled_w = width // 4
        self.fc1 = nn.Linear(64 * pooled_h * pooled_w, 256)
        self.fc2 = nn.Linear(256, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_threads(num_threads: int) -> None:
    """Avoid CPU oversubscription in Ray/Flower workers."""
    if num_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
        torch.set_num_threads(int(num_threads))


def resolve_device(device_arg: str) -> torch.device:
    """Resolve ``auto`` to CUDA if available, otherwise CPU."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return torch.device(device_arg)


def build_model(cfg: RunConfig, input_shape: Sequence[int], num_classes: int) -> nn.Module:
    """Build the configured deterministic model."""
    if cfg.model == "mlp":
        return MLP(input_shape=input_shape, num_classes=num_classes, hidden_dims=cfg.normalized_hidden())
    if cfg.model == "cnn":
        if cfg.method == "vi":
            raise ValueError("VI scaffold currently supports --model mlp only")
        return SmallCNN(input_shape=input_shape, num_classes=num_classes)
    raise ValueError(f"Unsupported model type {cfg.model!r}")


def trainable_parameters(net: nn.Module) -> list[nn.Parameter]:
    return [p for p in net.parameters() if p.requires_grad]


def parameter_shapes(net: nn.Module) -> list[tuple[int, ...]]:
    return [tuple(p.shape) for p in trainable_parameters(net)]


def parameter_metadata(net: nn.Module) -> list[dict[str, object]]:
    """Return stable flat slices for trainable named parameters."""
    rows: list[dict[str, object]] = []
    cursor = 0
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        n = int(param.numel())
        rows.append({"name": name, "shape": tuple(param.shape), "start": cursor, "end": cursor + n, "num_params": n})
        cursor += n
    return rows


def num_parameters(net: nn.Module) -> int:
    return int(sum(p.numel() for p in trainable_parameters(net)))


def flatten_parameters_tensor(net: nn.Module) -> torch.Tensor:
    """Differentiably flatten trainable parameters into one vector."""
    return torch.cat([p.reshape(-1) for p in trainable_parameters(net)])


def flatten_parameters(net: nn.Module) -> np.ndarray:
    """Flatten trainable parameters into a NumPy vector."""
    with torch.no_grad():
        return flatten_parameters_tensor(net).detach().cpu().numpy().astype(np.float32, copy=True)


def set_flat_parameters(net: nn.Module, flat: np.ndarray | torch.Tensor, device: torch.device | None = None) -> None:
    """Load a flat parameter vector into a model."""
    params = trainable_parameters(net)
    target_device = device if device is not None else next(net.parameters()).device
    flat_t = torch.as_tensor(flat, dtype=torch.float32, device=target_device).flatten()
    expected = sum(p.numel() for p in params)
    if flat_t.numel() != expected:
        raise ValueError(f"Flat parameter vector has {flat_t.numel()} values, expected {expected}")
    cursor = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(flat_t[cursor : cursor + n].view_as(p))
            cursor += n


def build_optimizer(cfg: RunConfig, net: nn.Module) -> torch.optim.Optimizer:
    params = trainable_parameters(net)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


def _stat(arr: np.ndarray | None, key: str) -> dict[str, float]:
    if arr is None:
        return {f"{key}_mean": float("nan"), f"{key}_std": float("nan"), f"{key}_p50": float("nan"), f"{key}_p90": float("nan"), f"{key}_max": float("nan")}
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return _stat(None, key)
    return {
        f"{key}_mean": float(x.mean()),
        f"{key}_std": float(x.std()),
        f"{key}_p50": float(np.percentile(x, 50)),
        f"{key}_p90": float(np.percentile(x, 90)),
        f"{key}_max": float(x.max()),
    }


def train_deterministic(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
    prior_mu: np.ndarray | None = None,
    prior_precision: np.ndarray | None = None,
    prior_lambda: float = 0.0,
    collect_fisher: bool = False,
) -> tuple[float, np.ndarray | None, dict[str, float]]:
    """Train a deterministic model locally and return a rich training summary.

    Returns:
        ``(avg_total_loss, fisher_diag_or_none, stats)``. ``avg_total_loss`` is
        the optimized local objective, while ``stats['task_loss']`` is the pure
        data likelihood/cross-entropy component. For OLA/FOLA, ``prior_loss`` is
        the prior-iteration quadratic penalty before multiplying by
        ``ola_prior_lambda``.
    """
    net.to(device)
    net.train()
    optimizer = build_optimizer(cfg, net)
    params = trainable_parameters(net)
    fisher_accum: torch.Tensor | None = None
    fisher_steps = 0
    task_loss_sum = 0.0
    prior_loss_sum = 0.0
    total_loss_sum = 0.0
    correct = 0
    example_sum = 0
    batch_count = 0

    prior_mu_t: torch.Tensor | None = None
    prior_precision_t: torch.Tensor | None = None
    if prior_mu is not None and prior_precision is not None and prior_lambda > 0:
        prior_mu_t = torch.as_tensor(prior_mu, dtype=torch.float32, device=device).flatten()
        prior_precision_t = torch.as_tensor(prior_precision, dtype=torch.float32, device=device).flatten()

    for _epoch in range(int(cfg.local_epochs)):
        for x, y in trainloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = net(x)
            task_loss = F.cross_entropy(logits, y)

            if collect_fisher:
                task_grads = torch.autograd.grad(task_loss, params, retain_graph=True, allow_unused=False)
                grad_flat = torch.cat([g.detach().reshape(-1) for g in task_grads])
                if cfg.fisher_clip > 0:
                    grad_flat = torch.clamp(grad_flat, min=-cfg.fisher_clip, max=cfg.fisher_clip)
                fisher_accum = grad_flat.pow(2) if fisher_accum is None else fisher_accum + grad_flat.pow(2)
                fisher_steps += 1

            prior_loss = torch.tensor(0.0, device=device)
            if prior_mu_t is not None and prior_precision_t is not None:
                flat = flatten_parameters_tensor(net)
                prior_loss = 0.5 * torch.sum(prior_precision_t * (flat - prior_mu_t).pow(2)) / max(flat.numel(), 1)

            total_loss = task_loss + float(prior_lambda) * prior_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            batch_n = int(y.size(0))
            batch_count += 1
            example_sum += batch_n
            task_loss_sum += float(task_loss.detach().cpu()) * batch_n
            prior_loss_sum += float(prior_loss.detach().cpu()) * batch_n
            total_loss_sum += float(total_loss.detach().cpu()) * batch_n
            correct += int((logits.argmax(dim=1) == y).sum().item())

    avg_task_loss = task_loss_sum / max(example_sum, 1)
    avg_prior_loss = prior_loss_sum / max(example_sum, 1)
    avg_total_loss = total_loss_sum / max(example_sum, 1)
    fisher_np: np.ndarray | None = None
    if collect_fisher:
        if fisher_accum is None or fisher_steps == 0:
            fisher_np = np.zeros(num_parameters(net), dtype=np.float32)
        else:
            fisher_np = (fisher_accum / float(fisher_steps)).detach().cpu().numpy().astype(np.float32)

    stats: dict[str, float] = {
        "train_loss": float(avg_total_loss),
        "task_loss": float(avg_task_loss),
        "prior_loss": float(avg_prior_loss),
        "regularization_loss": float(float(prior_lambda) * avg_prior_loss),
        "accuracy_local_train_estimate": float(correct / max(example_sum, 1)),
        "loss_local_train_estimate": float(avg_task_loss),
        "num_batches": float(batch_count),
        "num_examples_seen": float(example_sum),
    }
    if fisher_np is not None:
        stats.update(_stat(fisher_np, "ola_fisher"))
    return avg_total_loss, fisher_np, stats


@torch.no_grad()
def evaluate(net: nn.Module, testloader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Evaluate cross-entropy loss and accuracy."""
    net.to(device)
    net.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in testloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = net(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += float(loss.detach().cpu())
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return loss_sum / max(total, 1), correct / max(total, 1)
