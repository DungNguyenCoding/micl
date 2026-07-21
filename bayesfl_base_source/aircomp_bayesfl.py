"""Distribution-level AirComp simulator for Bayesian FL.

This module implements a standalone simulator for the paper
"Distribution-Level AirComp for Wireless Federated Learning under Data Scarcity
and Heterogeneity" (arXiv:2506.06090).  It is intentionally separate from the
Flower simulator used by ``main.py`` because AirComp assumes simultaneous analog
transmission and waveform superposition, which is not naturally represented by
Flower's request/response client API.

Implemented algorithms
----------------------
* AirComp FedAvg baseline.
* AirComp FedProx baseline.
* AirComp SCAFFOLD baseline, mainly for the default comparison.
* Proposed distribution-level AirComp Bayesian FL using a mean-field Gaussian
  posterior and two AirComp phases:

  Phase 1: update posterior precision rho = diag(Sigma^{-1}).
  Phase 2: update posterior mean mu.

The simulator is designed to reproduce the *structure* of Section VI in the
paper: scarce non-IID MNIST, 40 devices, one/few labels per device, Poisson local
sample counts, OFDM/AirComp channel distortion, accuracy-vs-channel-use curves,
power-budget curves, and reliability/ECE diagrams.

The exact numerical curves can vary from the paper due to random realizations,
software versions, and several implementation choices not fully specified in the
paper.  For paper-grade reproduction, run multiple realizations and compare the
averaged curves.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from torchvision import datasets, transforms
except Exception as exc:  # pragma: no cover
    datasets = None
    transforms = None
    _TORCHVISION_IMPORT_ERROR = exc
else:
    _TORCHVISION_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    if str(v).lower() in {"1", "true", "yes", "y", "on"}:
        return True
    if str(v).lower() in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return torch.device(device)


def dbm_to_watt(dbm: float) -> float:
    return float(10.0 ** ((float(dbm) - 30.0) / 10.0))


def db_to_linear(db: float) -> float:
    return float(10.0 ** (float(db) / 10.0))


def safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Paper CNN: exactly 62,346 trainable parameters for MNIST.
# Conv5-valid -> pool -> Conv5-valid -> pool -> FC(1024 -> 10)
# ---------------------------------------------------------------------------


class PaperCNN(nn.Module):
    """CNN architecture used by the AirComp Bayesian FL paper.

    Parameter count on MNIST:
    conv1: 32 * 1 * 5 * 5 + 32 = 832
    conv2: 64 * 32 * 5 * 5 + 64 = 51,264
    fc:    10 * 1024 + 10 = 10,250
    total = 62,346
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 4 * 4, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


@dataclass(frozen=True)
class FlatSpec:
    names: Tuple[str, ...]
    shapes: Tuple[Tuple[int, ...], ...]
    starts: Tuple[int, ...]
    ends: Tuple[int, ...]

    @property
    def dim(self) -> int:
        return int(self.ends[-1])


def paper_cnn_spec() -> FlatSpec:
    model = PaperCNN()
    names: list[str] = []
    shapes: list[tuple[int, ...]] = []
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for name, param in model.named_parameters():
        names.append(name)
        shape = tuple(param.shape)
        shapes.append(shape)
        starts.append(cursor)
        cursor += int(param.numel())
        ends.append(cursor)
    return FlatSpec(tuple(names), tuple(shapes), tuple(starts), tuple(ends))


SPEC = paper_cnn_spec()


def flatten_model(net: nn.Module) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([p.detach().reshape(-1).cpu() for p in net.parameters()]).numpy().astype(np.float32)


def set_flat_model(net: nn.Module, flat: np.ndarray | torch.Tensor, device: torch.device) -> None:
    flat_t = torch.as_tensor(flat, dtype=torch.float32, device=device).flatten()
    cursor = 0
    with torch.no_grad():
        for p in net.parameters():
            n = int(p.numel())
            p.copy_(flat_t[cursor : cursor + n].view_as(p))
            cursor += n


def init_flat_model(seed: int, device: torch.device) -> np.ndarray:
    torch.manual_seed(int(seed))
    net = PaperCNN().to(device)
    return flatten_model(net)


def split_flat_torch(flat: torch.Tensor) -> Dict[str, torch.Tensor]:
    flat = flat.flatten()
    return {name: flat[s:e].view(shape) for name, shape, s, e in zip(SPEC.names, SPEC.shapes, SPEC.starts, SPEC.ends)}


def paper_cnn_forward_flat(x: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
    p = split_flat_torch(flat)
    x = F.conv2d(x, p["conv1.weight"], p["conv1.bias"])
    x = F.max_pool2d(F.relu(x), 2)
    x = F.conv2d(x, p["conv2.weight"], p["conv2.bias"])
    x = F.max_pool2d(F.relu(x), 2)
    x = torch.flatten(x, start_dim=1)
    x = F.linear(x, p["fc.weight"], p["fc.bias"])
    return x


# ---------------------------------------------------------------------------
# Data scarcity / heterogeneity partitioning
# ---------------------------------------------------------------------------


def load_mnist(data_dir: str) -> tuple[Dataset, Dataset, np.ndarray]:
    if datasets is None or transforms is None:
        raise RuntimeError(f"torchvision import failed: {_TORCHVISION_IMPORT_ERROR}")
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root=data_dir, train=True, transform=tfm, download=True)
    test = datasets.MNIST(root=data_dir, train=False, transform=tfm, download=True)
    targets = np.asarray(train.targets, dtype=np.int64)
    return train, test, targets


def make_scarce_label_skew_partitions(
    targets: np.ndarray,
    num_devices: int,
    mean_examples: float,
    local_classes: int,
    rng: np.random.Generator,
) -> tuple[list[list[int]], list[dict[str, object]]]:
    """Create scarce non-IID partitions matching Section VI.

    Each device receives samples from ``local_classes`` labels. The number of
    local samples is Poisson with mean ``mean_examples``. We sample without
    replacement inside each label pool when possible; if a label pool is
    exhausted, we fall back to sampling with replacement for robustness.
    """
    labels = np.arange(10, dtype=np.int64)
    pools: dict[int, list[int]] = {int(c): rng.permutation(np.where(targets == c)[0]).tolist() for c in labels}
    cursors = {int(c): 0 for c in labels}
    partitions: list[list[int]] = []
    rows: list[dict[str, object]] = []

    for did in range(int(num_devices)):
        cls = rng.choice(labels, size=int(local_classes), replace=False)
        total_n = max(1, int(rng.poisson(float(mean_examples))))
        per_class = [total_n // int(local_classes)] * int(local_classes)
        for i in range(total_n % int(local_classes)):
            per_class[i] += 1

        idxs: list[int] = []
        label_counts = {int(c): 0 for c in labels}
        for c, n_c in zip(cls, per_class):
            c = int(c)
            start = cursors[c]
            end = start + int(n_c)
            if end <= len(pools[c]):
                chosen = pools[c][start:end]
                cursors[c] = end
            else:
                # In rare cases for large mean/local realizations.
                chosen = rng.choice(np.where(targets == c)[0], size=int(n_c), replace=True).tolist()
            idxs.extend(int(i) for i in chosen)
            label_counts[c] += int(n_c)

        rng.shuffle(idxs)
        partitions.append(idxs)
        probs = np.asarray([label_counts[c] for c in labels], dtype=np.float64)
        probs = probs / max(probs.sum(), 1.0)
        entropy = -float(np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
        rows.append(
            {
                "physical_client_id": did,
                "num_examples": len(idxs),
                "local_classes": int(local_classes),
                "assigned_labels": ";".join(map(str, sorted(map(int, cls)))),
                "label_entropy": entropy,
                **{f"label_{c}_count": int(label_counts[c]) for c in labels},
            }
        )
    return partitions, rows


def make_client_loaders(
    train: Dataset,
    partitions: Sequence[Sequence[int]],
    batch_size: int,
    seed: int,
    num_workers: int,
) -> list[DataLoader]:
    loaders: list[DataLoader] = []
    for cid, idxs in enumerate(partitions):
        gen = torch.Generator()
        gen.manual_seed(int(seed) + int(cid))
        subset = Subset(train, list(map(int, idxs)))
        loaders.append(DataLoader(subset, batch_size=int(batch_size), shuffle=True, generator=gen, num_workers=int(num_workers)))
    return loaders


def sample_device_distances(num_devices: int, coverage_radius_m: float, rng: np.random.Generator) -> np.ndarray:
    # Uniform in disk. Clamp minimum distance to avoid unrealistically huge gain.
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=int(num_devices))) * float(coverage_radius_m)
    return np.maximum(radius, 1.0).astype(np.float64)


# ---------------------------------------------------------------------------
# Evaluation and calibration
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_deterministic_flat(flat: np.ndarray, test_loader: DataLoader, device: torch.device) -> dict[str, float]:
    net = PaperCNN().to(device)
    set_flat_model(net, flat, device)
    net.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    probs_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        logits = net(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        loss_sum += float(loss.detach().cpu())
        probs_all.append(probs.detach().cpu())
        labels_all.append(y.detach().cpu())
    probs_cat = torch.cat(probs_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    ece, mce, bins = calibration_from_probs(probs_cat, labels_cat, bins=10)
    return {
        "accuracy": float(correct / max(total, 1)),
        "loss": float(loss_sum / max(total, 1)),
        "ece": float(ece),
        "mce": float(mce),
        "num_examples": int(total),
        "calibration_bins": bins,
    }


@torch.no_grad()
def evaluate_bayesian_posterior(
    mu: np.ndarray,
    rho: np.ndarray,
    test_loader: DataLoader,
    device: torch.device,
    mc_samples: int,
    sample_scale: float,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    mu_t = torch.as_tensor(mu, dtype=torch.float32, device=device)
    sigma = 1.0 / np.sqrt(np.maximum(np.asarray(rho, dtype=np.float64), 1e-12))
    sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    total = 0
    correct = 0
    loss_sum = 0.0
    probs_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        sample_probs: list[torch.Tensor] = []
        for _ in range(max(1, int(mc_samples))):
            if int(mc_samples) == 1 or float(sample_scale) == 0.0:
                flat = mu_t
            else:
                eps_np = rng.normal(size=mu.shape).astype(np.float32)
                eps = torch.as_tensor(eps_np, dtype=torch.float32, device=device)
                flat = mu_t + float(sample_scale) * sigma_t * eps
            logits = paper_cnn_forward_flat(x, flat)
            sample_probs.append(F.softmax(logits, dim=1))
        probs = torch.stack(sample_probs, dim=0).mean(dim=0)
        loss = F.nll_loss(torch.log(torch.clamp(probs, min=1e-12)), y, reduction="sum")
        pred = probs.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        loss_sum += float(loss.detach().cpu())
        probs_all.append(probs.detach().cpu())
        labels_all.append(y.detach().cpu())
    probs_cat = torch.cat(probs_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    ece, mce, bins = calibration_from_probs(probs_cat, labels_cat, bins=10)
    return {
        "accuracy": float(correct / max(total, 1)),
        "loss": float(loss_sum / max(total, 1)),
        "ece": float(ece),
        "mce": float(mce),
        "num_examples": int(total),
        "calibration_bins": bins,
    }


def calibration_from_probs(probs: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> tuple[float, float, list[dict[str, float]]]:
    conf, pred = torch.max(probs, dim=1)
    corr = (pred == labels).to(torch.float32)
    conf_np = conf.numpy()
    corr_np = corr.numpy()
    ids = np.minimum((conf_np * int(bins)).astype(np.int64), int(bins) - 1)
    rows: list[dict[str, float]] = []
    ece = 0.0
    mce = 0.0
    total = max(int(labels.numel()), 1)
    for bid in range(int(bins)):
        mask = ids == bid
        count = int(mask.sum())
        if count > 0:
            acc = float(corr_np[mask].mean())
            avg_conf = float(conf_np[mask].mean())
            gap = abs(acc - avg_conf)
            contrib = gap * count / total
        else:
            acc = avg_conf = gap = contrib = 0.0
        ece += contrib
        mce = max(mce, gap)
        rows.append(
            {
                "bin_id": int(bid),
                "bin_left": float(bid / bins),
                "bin_right": float((bid + 1) / bins),
                "bin_count": int(count),
                "bin_accuracy": float(acc),
                "bin_confidence": float(avg_conf),
                "bin_gap": float(gap),
                "ece_contribution": float(contrib),
            }
        )
    return float(ece), float(mce), rows


# ---------------------------------------------------------------------------
# AirComp channel and power control
# ---------------------------------------------------------------------------


@dataclass
class AirCompStats:
    update_l2: float
    distortion_l2: float
    distortion_ratio: float
    avg_update_power: float
    mean_channel_gain: float
    clipped_fraction: float
    noise_l2: float


def solve_power_control(abs_delta: np.ndarray, u: np.ndarray, power_watt: float, exact: bool = True) -> tuple[np.ndarray, bool]:
    """Solve Eq. (42)/(43) for one client and one OFDM group."""
    abs_delta = np.asarray(abs_delta, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if not np.any(abs_delta):
        return abs_delta, False
    current_power = float(np.sum(u * abs_delta * abs_delta))
    if current_power <= float(power_watt) or float(power_watt) <= 0:
        return abs_delta, False
    if not exact:
        scale = math.sqrt(max(float(power_watt), 1e-30) / max(current_power, 1e-30))
        return abs_delta * scale, True

    # Find lambda such that sum u*(abs_delta/(1+lambda*u))^2 = P.
    lo = 0.0
    hi = 1.0
    for _ in range(60):
        v = abs_delta / (1.0 + hi * u)
        val = float(np.sum(u * v * v))
        if val <= float(power_watt):
            break
        hi *= 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        v = abs_delta / (1.0 + mid * u)
        val = float(np.sum(u * v * v))
        if val > float(power_watt):
            lo = mid
        else:
            hi = mid
    v = abs_delta / (1.0 + hi * u)
    return v, True


def aircomp_aggregate(
    updates: np.ndarray,
    weights: np.ndarray,
    distances_m: np.ndarray,
    num_subchannels: int,
    power_watt: float,
    noise_watt: float,
    gamma_linear: float,
    pathloss_alpha: float,
    rng: np.random.Generator,
    exact_power_control: bool = True,
) -> tuple[np.ndarray, AirCompStats]:
    """Aggregate client updates via analog AirComp.

    Args:
        updates: shape [K, d], local update vectors.
        weights: shape [K], aggregation weights pi_k, should sum to 1.
        distances_m: shape [K], device-BS distances.

    Returns:
        estimated weighted update and AirComp distortion metrics.
    """
    updates = np.asarray(updates, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = weights / max(float(weights.sum()), 1e-12)
    K, d = updates.shape
    Fch = int(num_subchannels)
    N = int(math.ceil(d / Fch))
    padded_d = N * Fch
    padded = np.zeros((K, padded_d), dtype=np.float64)
    padded[:, :d] = updates
    weighted_true_padded = np.sum(weights[:, None] * padded, axis=0)
    out = np.zeros(padded_d, dtype=np.float64)

    bar_delta = float(np.sum(weights * np.sum(updates * updates, axis=1) / max(d, 1)))
    if bar_delta <= 1e-30:
        return np.zeros(d, dtype=np.float32), AirCompStats(0.0, 0.0, 0.0, 0.0, float("nan"), 0.0, 0.0)

    pathloss_var = np.maximum(np.asarray(distances_m, dtype=np.float64), 1.0) ** (-float(pathloss_alpha))
    clipped = 0
    total_groups = K * N
    channel_gain_sum = 0.0
    noise_l2 = 0.0

    for n in range(N):
        start = n * Fch
        end = start + Fch
        group = padded[:, start:end]
        # |h|^2 for CN(0, r^-alpha) is exponential with mean r^-alpha.
        h_abs_sq = rng.exponential(scale=pathloss_var[:, None], size=(K, Fch))
        h_abs_sq = np.maximum(h_abs_sq, 1e-30)
        channel_gain_sum += float(np.mean(h_abs_sq))
        g = 1.0 / h_abs_sq
        u = (weights[:, None] ** 2) * float(gamma_linear) * g / max(bar_delta, 1e-30)
        signed_limited = np.zeros_like(group)
        for k in range(K):
            v, was_clipped = solve_power_control(np.abs(group[k]), u[k], power_watt, exact=exact_power_control)
            clipped += int(was_clipped)
            signed_limited[k] = np.sign(group[k]) * v
        noiseless = np.sum(weights[:, None] * signed_limited, axis=0)
        # Eq. (31)/(36): scaled complex channel noise. We simulate the real part.
        noise_std = math.sqrt(max(bar_delta, 1e-30) / max(float(gamma_linear), 1e-30) * max(float(noise_watt), 0.0) / 2.0)
        noise = rng.normal(loc=0.0, scale=noise_std, size=Fch)
        noise_l2 += float(np.sum(noise * noise))
        out[start:end] = noiseless + noise

    est = out[:d]
    true = weighted_true_padded[:d]
    distortion = est - true
    true_l2 = float(np.linalg.norm(true))
    distortion_l2 = float(np.linalg.norm(distortion))
    stats = AirCompStats(
        update_l2=true_l2,
        distortion_l2=distortion_l2,
        distortion_ratio=float(distortion_l2 / max(true_l2, 1e-12)),
        avg_update_power=float(bar_delta),
        mean_channel_gain=float(channel_gain_sum / max(N, 1)),
        clipped_fraction=float(clipped / max(total_groups, 1)),
        noise_l2=float(math.sqrt(noise_l2)),
    )
    return est.astype(np.float32), stats


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------


def train_fedavg_or_fedprox_local(
    global_flat: np.ndarray,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    local_epochs: int,
    fedprox_mu: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    net = PaperCNN().to(device)
    set_flat_model(net, global_flat, device)
    global_t = torch.as_tensor(global_flat, dtype=torch.float32, device=device)
    opt = torch.optim.SGD(net.parameters(), lr=float(lr))
    total_loss = 0.0
    total_examples = 0
    correct = 0
    for _ in range(int(local_epochs)):
        net.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = net(x)
            task_loss = F.cross_entropy(logits, y)
            prox = torch.tensor(0.0, device=device)
            if float(fedprox_mu) > 0:
                flat_now = torch.cat([p.reshape(-1) for p in net.parameters()])
                prox = 0.5 * float(fedprox_mu) * torch.sum((flat_now - global_t).pow(2))
            loss = task_loss + prox
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n = int(y.numel())
            total_examples += n
            total_loss += float(loss.detach().cpu()) * n
            correct += int((logits.argmax(dim=1) == y).sum().item())
    flat_new = flatten_model(net)
    stats = {
        "train_loss": float(total_loss / max(total_examples, 1)),
        "train_accuracy": float(correct / max(total_examples, 1)),
        "num_examples": int(total_examples),
    }
    return flat_new, stats


def train_scaffold_local(
    global_flat: np.ndarray,
    c_global: np.ndarray,
    c_client: np.ndarray,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    local_epochs: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    net = PaperCNN().to(device)
    set_flat_model(net, global_flat, device)
    opt = torch.optim.SGD(net.parameters(), lr=float(lr))
    correction = torch.as_tensor(c_global - c_client, dtype=torch.float32, device=device)
    total_steps = 0
    total_loss = 0.0
    total_examples = 0
    correct = 0
    for _ in range(int(local_epochs)):
        net.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = net(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Apply SCAFFOLD correction to gradients.
            cursor = 0
            with torch.no_grad():
                for p in net.parameters():
                    n = int(p.numel())
                    corr = correction[cursor : cursor + n].view_as(p)
                    p.grad.add_(corr)
                    cursor += n
            opt.step()
            total_steps += 1
            n_ex = int(y.numel())
            total_examples += n_ex
            total_loss += float(loss.detach().cpu()) * n_ex
            correct += int((logits.argmax(dim=1) == y).sum().item())
    local_flat = flatten_model(net)
    # c_i^+ = c_i - c + (x - y)/(K*eta) where K here is local steps.
    denom = max(total_steps * float(lr), 1e-12)
    c_new = c_client - c_global + (global_flat - local_flat) / denom
    c_delta = c_new - c_client
    return local_flat, c_delta.astype(np.float32), {
        "train_loss": float(total_loss / max(total_examples, 1)),
        "train_accuracy": float(correct / max(total_examples, 1)),
        "num_examples": int(total_examples),
        "local_steps": int(total_steps),
    }


def gaussian_kl_diag(mu1: torch.Tensor, rho1: torch.Tensor, mu2: torch.Tensor, rho2: torch.Tensor) -> torch.Tensor:
    """KL(N(mu1, diag(1/rho1)) || N(mu2, diag(1/rho2)))."""
    rho1 = torch.clamp(rho1, min=1e-12)
    rho2 = torch.clamp(rho2, min=1e-12)
    var1 = 1.0 / rho1
    # log |Sigma2|/|Sigma1| = sum(log var2 - log var1) = sum(log rho1 - log rho2)
    term = torch.log(rho1) - torch.log(rho2) - 1.0 + rho2 * var1 + rho2 * (mu2 - mu1).pow(2)
    return 0.5 * torch.sum(term)


def vi_task_loss_mc(x: torch.Tensor, y: torch.Tensor, mu: torch.Tensor, rho: torch.Tensor, mc_samples: int) -> torch.Tensor:
    rho = torch.clamp(rho, min=1e-8)
    losses: list[torch.Tensor] = []
    sigma = torch.rsqrt(rho)
    for _ in range(max(1, int(mc_samples))):
        eps = torch.randn_like(mu)
        flat = mu + sigma * eps
        logits = paper_cnn_forward_flat(x, flat)
        losses.append(F.cross_entropy(logits, y))
    return torch.stack(losses).mean()


def softplus_inverse_np(x: np.ndarray, min_value: float = 1e-8) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=np.float64) - float(min_value), 1e-12)
    return (x + np.log(-np.expm1(-x))).astype(np.float32)


def train_proposed_phase_rho(
    global_mu: np.ndarray,
    global_rho: np.ndarray,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    local_epochs: int,
    vi_lambda: float,
    mc_samples: int,
    rho_floor: float,
) -> tuple[np.ndarray, dict[str, float]]:
    mu = torch.as_tensor(global_mu, dtype=torch.float32, device=device)
    rho_prior = torch.as_tensor(np.maximum(global_rho, rho_floor), dtype=torch.float32, device=device)
    raw = torch.nn.Parameter(torch.as_tensor(softplus_inverse_np(global_rho, rho_floor), dtype=torch.float32, device=device))
    opt = torch.optim.SGD([raw], lr=float(lr))
    total_loss = 0.0
    total_task = 0.0
    total_kl = 0.0
    steps = 0
    examples = 0
    for _ in range(int(local_epochs)):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            rho = F.softplus(raw) + float(rho_floor)
            task = vi_task_loss_mc(x, y, mu, rho, int(mc_samples))
            kl = gaussian_kl_diag(mu, rho, mu, rho_prior)
            loss = task + float(vi_lambda) * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n = int(y.numel())
            examples += n
            total_loss += float(loss.detach().cpu())
            total_task += float(task.detach().cpu())
            total_kl += float(kl.detach().cpu())
            steps += 1
    rho_new = (F.softplus(raw) + float(rho_floor)).detach().cpu().numpy().astype(np.float32)
    stats = {
        "phase1_loss": float(total_loss / max(steps, 1)),
        "phase1_task_loss": float(total_task / max(steps, 1)),
        "phase1_kl": float(total_kl / max(steps, 1)),
        "phase1_examples": int(examples),
        "rho_mean": float(np.mean(rho_new)),
        "rho_p50": float(np.percentile(rho_new, 50)),
        "rho_p90": float(np.percentile(rho_new, 90)),
    }
    return rho_new, stats


def train_proposed_phase_mu(
    global_mu: np.ndarray,
    global_rho: np.ndarray,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    local_epochs: int,
    vi_lambda: float,
    mc_samples: int,
    rho_floor: float,
) -> tuple[np.ndarray, dict[str, float]]:
    prior_mu = torch.as_tensor(global_mu, dtype=torch.float32, device=device)
    rho = torch.as_tensor(np.maximum(global_rho, rho_floor), dtype=torch.float32, device=device)
    mu = torch.nn.Parameter(torch.as_tensor(global_mu, dtype=torch.float32, device=device).clone())
    opt = torch.optim.SGD([mu], lr=float(lr))
    total_loss = 0.0
    total_task = 0.0
    total_kl = 0.0
    steps = 0
    examples = 0
    for _ in range(int(local_epochs)):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            task = vi_task_loss_mc(x, y, mu, rho, int(mc_samples))
            kl = gaussian_kl_diag(mu, rho, prior_mu, rho)
            loss = task + float(vi_lambda) * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n = int(y.numel())
            examples += n
            total_loss += float(loss.detach().cpu())
            total_task += float(task.detach().cpu())
            total_kl += float(kl.detach().cpu())
            steps += 1
    mu_new = mu.detach().cpu().numpy().astype(np.float32)
    stats = {
        "phase2_loss": float(total_loss / max(steps, 1)),
        "phase2_task_loss": float(total_task / max(steps, 1)),
        "phase2_kl": float(total_kl / max(steps, 1)),
        "phase2_examples": int(examples),
        "mu_l2": float(np.linalg.norm(mu_new)),
    }
    return mu_new, stats


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


@dataclass
class AirCompConfig:
    output_dir: str = "outputs/aircomp_bayesfl_mnist_paper"
    data_dir: str = "./data"
    experiment: str = "default"
    methods: str = "fedavg,fedprox,scaffold,proposed"
    seed: int = 42
    realizations: int = 1
    max_channel_uses: int = 3_000_000
    eval_every: int = 1
    num_devices: int = 40
    coverage_radius_m: float = 200.0
    mean_client_examples: float = 10.0
    local_classes: int = 1
    power_dbm: float = 23.0
    noise_dbm: float = -74.0
    num_subchannels: int = 1024
    pathloss_alpha: float = 4.0
    gamma_db: float = 10.0
    lr: float = 0.1
    batch_size: int = 10
    local_epochs: int = 3
    vi_lambda: float = 1.0 / 50000.0
    mc_samples: int = 5
    rho_init: float = 100.0
    rho_floor: float = 1e-6
    posterior_sample_scale: float = 1.0
    fedprox_mu: float = 0.001
    exact_power_control: bool = True
    num_workers: int = 0
    device: str = "auto"
    save_model: bool = False


def parse_args() -> AirCompConfig:
    p = argparse.ArgumentParser(description="Distribution-level AirComp Bayesian FL simulator")
    p.add_argument("--output_dir", default=AirCompConfig.output_dir)
    p.add_argument("--data_dir", default=AirCompConfig.data_dir)
    p.add_argument("--experiment", choices=["default", "label_skew", "dataset_size", "power", "full", "custom"], default="default")
    p.add_argument("--methods", default=AirCompConfig.methods, help="Comma-separated: fedavg,fedprox,scaffold,proposed")
    p.add_argument("--seed", type=int, default=AirCompConfig.seed)
    p.add_argument("--realizations", type=int, default=AirCompConfig.realizations)
    p.add_argument("--max_channel_uses", type=int, default=AirCompConfig.max_channel_uses)
    p.add_argument("--eval_every", type=int, default=AirCompConfig.eval_every)
    p.add_argument("--num_devices", type=int, default=AirCompConfig.num_devices)
    p.add_argument("--coverage_radius_m", type=float, default=AirCompConfig.coverage_radius_m)
    p.add_argument("--mean_client_examples", type=float, default=AirCompConfig.mean_client_examples)
    p.add_argument("--local_classes", type=int, default=AirCompConfig.local_classes)
    p.add_argument("--power_dbm", type=float, default=AirCompConfig.power_dbm)
    p.add_argument("--noise_dbm", type=float, default=AirCompConfig.noise_dbm)
    p.add_argument("--num_subchannels", type=int, default=AirCompConfig.num_subchannels)
    p.add_argument("--pathloss_alpha", type=float, default=AirCompConfig.pathloss_alpha)
    p.add_argument("--gamma_db", type=float, default=AirCompConfig.gamma_db)
    p.add_argument("--lr", type=float, default=AirCompConfig.lr)
    p.add_argument("--batch_size", type=int, default=AirCompConfig.batch_size)
    p.add_argument("--local_epochs", type=int, default=AirCompConfig.local_epochs)
    p.add_argument("--vi_lambda", type=float, default=AirCompConfig.vi_lambda)
    p.add_argument("--mc_samples", type=int, default=AirCompConfig.mc_samples)
    p.add_argument("--rho_init", type=float, default=AirCompConfig.rho_init)
    p.add_argument("--rho_floor", type=float, default=AirCompConfig.rho_floor)
    p.add_argument("--posterior_sample_scale", type=float, default=AirCompConfig.posterior_sample_scale)
    p.add_argument("--fedprox_mu", type=float, default=AirCompConfig.fedprox_mu)
    p.add_argument("--exact_power_control", type=str2bool, default=AirCompConfig.exact_power_control)
    p.add_argument("--num_workers", type=int, default=AirCompConfig.num_workers)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=AirCompConfig.device)
    p.add_argument("--save_model", type=str2bool, default=AirCompConfig.save_model)
    ns = p.parse_args()
    cfg = AirCompConfig(**vars(ns))
    if cfg.local_classes < 1 or cfg.local_classes > 10:
        raise ValueError("--local_classes must be in [1, 10]")
    if cfg.realizations < 1:
        raise ValueError("--realizations must be positive")
    if cfg.max_channel_uses <= 0:
        raise ValueError("--max_channel_uses must be positive")
    return cfg


def scenario_grid(cfg: AirCompConfig) -> list[dict[str, object]]:
    if cfg.experiment == "custom":
        return [
            {
                "scenario": "custom",
                "condition_name": f"classes{cfg.local_classes}_mean{cfg.mean_client_examples:g}_P{cfg.power_dbm:g}",
                "local_classes": cfg.local_classes,
                "mean_client_examples": cfg.mean_client_examples,
                "power_dbm": cfg.power_dbm,
                "methods": cfg.methods.split(","),
            }
        ]
    default = {
        "scenario": "default",
        "condition_name": "default",
        "local_classes": 1,
        "mean_client_examples": 10.0,
        "power_dbm": 23.0,
        "methods": cfg.methods.split(","),
    }
    if cfg.experiment == "default":
        return [default]
    if cfg.experiment == "label_skew":
        return [
            {**default, "scenario": "label_skew", "condition_name": f"{c}_class_local", "local_classes": c, "methods": [m for m in cfg.methods.split(",") if m != "scaffold"]}
            for c in [1, 2, 10]
        ]
    if cfg.experiment == "dataset_size":
        return [
            {**default, "scenario": "dataset_size", "condition_name": f"mean{m:g}", "mean_client_examples": float(m), "methods": [x for x in cfg.methods.split(",") if x != "scaffold"]}
            for m in [10, 20, 50]
        ]
    if cfg.experiment == "power":
        return [
            {**default, "scenario": "power", "condition_name": f"P{p:g}dBm", "power_dbm": float(p), "methods": [x for x in cfg.methods.split(",") if x != "scaffold"]}
            for p in [3, 23, 33]
        ]
    if cfg.experiment == "full":
        old = cfg.experiment
        out: list[dict[str, object]] = []
        for exp in ["default", "label_skew", "dataset_size", "power"]:
            cfg.experiment = exp
            out.extend(scenario_grid(cfg))
        cfg.experiment = old
        return out
    raise ValueError(cfg.experiment)


def symbols_per_round(method: str, d: int, Fch: int) -> int:
    N = int(math.ceil(d / int(Fch)))
    if method in {"fedavg", "fedprox"}:
        return N
    if method in {"scaffold", "proposed"}:
        return 2 * N
    raise ValueError(method)


def run_method(
    cfg: AirCompConfig,
    scenario: Mapping[str, object],
    realization: int,
    method: str,
    loaders: Sequence[DataLoader],
    test_loader: DataLoader,
    init_flat: np.ndarray,
    distances_m: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    method = method.strip().lower()
    d = int(init_flat.size)
    Fch = int(cfg.num_subchannels)
    per_round_channel_uses = symbols_per_round(method, d, Fch) * Fch
    max_rounds = max(1, int(math.ceil(float(cfg.max_channel_uses) / float(per_round_channel_uses))))
    power_watt = dbm_to_watt(float(scenario["power_dbm"]))
    noise_watt = dbm_to_watt(float(cfg.noise_dbm))
    gamma_linear = db_to_linear(float(cfg.gamma_db))
    weights = np.asarray([len(loader.dataset) for loader in loaders], dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1.0)

    metrics: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    state_for_save: dict[str, np.ndarray] = {}

    if method in {"fedavg", "fedprox", "scaffold"}:
        global_flat = init_flat.astype(np.float32, copy=True)
        c_global = np.zeros_like(global_flat, dtype=np.float32)
        c_clients = [np.zeros_like(global_flat, dtype=np.float32) for _ in loaders]
        for round_idx in range(1, max_rounds + 1):
            t0 = time.time()
            local_flats: list[np.ndarray] = []
            local_c_deltas: list[np.ndarray] = []
            train_losses: list[float] = []
            for cid, loader in enumerate(loaders):
                if method == "scaffold":
                    local_flat, c_delta, st = train_scaffold_local(global_flat, c_global, c_clients[cid], loader, device, cfg.lr, cfg.local_epochs)
                    c_clients[cid] = c_clients[cid] + c_delta
                    local_c_deltas.append(c_delta)
                else:
                    mu = cfg.fedprox_mu if method == "fedprox" else 0.0
                    local_flat, st = train_fedavg_or_fedprox_local(global_flat, loader, device, cfg.lr, cfg.local_epochs, fedprox_mu=mu)
                local_flats.append(local_flat)
                train_losses.append(float(st.get("train_loss", float("nan"))))
            deltas = np.stack([lf - global_flat for lf in local_flats], axis=0)
            agg_delta, air_stats = aircomp_aggregate(
                deltas,
                weights,
                distances_m,
                Fch,
                power_watt,
                noise_watt,
                gamma_linear,
                cfg.pathloss_alpha,
                rng,
                exact_power_control=cfg.exact_power_control,
            )
            global_flat = (global_flat + agg_delta).astype(np.float32)
            c_distortion = float("nan")
            if method == "scaffold":
                c_delta_stack = np.stack(local_c_deltas, axis=0)
                agg_c_delta, c_stats = aircomp_aggregate(
                    c_delta_stack,
                    weights,
                    distances_m,
                    Fch,
                    power_watt,
                    noise_watt,
                    gamma_linear,
                    cfg.pathloss_alpha,
                    rng,
                    exact_power_control=cfg.exact_power_control,
                )
                c_global = (c_global + agg_c_delta).astype(np.float32)
                c_distortion = c_stats.distortion_ratio

            channel_uses = int(round_idx * per_round_channel_uses)
            if round_idx == 1 or round_idx == max_rounds or round_idx % int(cfg.eval_every) == 0:
                ev = evaluate_deterministic_flat(global_flat, test_loader, device)
                for b in ev.pop("calibration_bins"):
                    calibration_rows.append({**base_row(cfg, scenario, realization, method, round_idx, channel_uses), **b, "eval_scope": "global_test"})
                metrics.append(
                    {
                        **base_row(cfg, scenario, realization, method, round_idx, channel_uses),
                        "accuracy": ev["accuracy"],
                        "loss": ev["loss"],
                        "ece": ev["ece"],
                        "mce": ev["mce"],
                        "train_loss_mean": float(np.nanmean(train_losses)),
                        "aircomp_update_l2": air_stats.update_l2,
                        "aircomp_distortion_l2": air_stats.distortion_l2,
                        "aircomp_distortion_ratio": air_stats.distortion_ratio,
                        "aircomp_avg_update_power": air_stats.avg_update_power,
                        "aircomp_clipped_fraction": air_stats.clipped_fraction,
                        "aircomp_noise_l2": air_stats.noise_l2,
                        "scaffold_control_distortion_ratio": c_distortion,
                        "round_time_sec": float(time.time() - t0),
                    }
                )
        state_for_save = {"global_flat": global_flat}
        if method == "scaffold":
            state_for_save["c_global"] = c_global

    elif method == "proposed":
        global_mu = init_flat.astype(np.float32, copy=True)
        global_rho = np.full_like(global_mu, fill_value=float(cfg.rho_init), dtype=np.float32)
        for round_idx in range(1, max_rounds + 1):
            t0 = time.time()
            rho_updates: list[np.ndarray] = []
            phase1_losses: list[float] = []
            for loader in loaders:
                local_rho, st = train_proposed_phase_rho(
                    global_mu,
                    global_rho,
                    loader,
                    device,
                    cfg.lr,
                    cfg.local_epochs,
                    cfg.vi_lambda,
                    cfg.mc_samples,
                    cfg.rho_floor,
                )
                rho_updates.append(local_rho - global_rho)
                phase1_losses.append(float(st.get("phase1_loss", float("nan"))))
            agg_drho, rho_air = aircomp_aggregate(
                np.stack(rho_updates, axis=0),
                weights,
                distances_m,
                Fch,
                power_watt,
                noise_watt,
                gamma_linear,
                cfg.pathloss_alpha,
                rng,
                exact_power_control=cfg.exact_power_control,
            )
            global_rho = np.maximum(global_rho + agg_drho, float(cfg.rho_floor)).astype(np.float32)

            mu_updates: list[np.ndarray] = []
            phase2_losses: list[float] = []
            for loader in loaders:
                local_mu, st = train_proposed_phase_mu(
                    global_mu,
                    global_rho,
                    loader,
                    device,
                    cfg.lr,
                    cfg.local_epochs,
                    cfg.vi_lambda,
                    cfg.mc_samples,
                    cfg.rho_floor,
                )
                mu_updates.append(local_mu - global_mu)
                phase2_losses.append(float(st.get("phase2_loss", float("nan"))))
            agg_dmu, mu_air = aircomp_aggregate(
                np.stack(mu_updates, axis=0),
                weights,
                distances_m,
                Fch,
                power_watt,
                noise_watt,
                gamma_linear,
                cfg.pathloss_alpha,
                rng,
                exact_power_control=cfg.exact_power_control,
            )
            global_mu = (global_mu + agg_dmu).astype(np.float32)

            channel_uses = int(round_idx * per_round_channel_uses)
            if round_idx == 1 or round_idx == max_rounds or round_idx % int(cfg.eval_every) == 0:
                ev_mean = evaluate_deterministic_flat(global_mu, test_loader, device)
                ev_mc = evaluate_bayesian_posterior(
                    global_mu,
                    global_rho,
                    test_loader,
                    device,
                    mc_samples=max(1, int(cfg.mc_samples)),
                    sample_scale=float(cfg.posterior_sample_scale),
                    seed=int(cfg.seed) + 999 * int(realization) + int(round_idx),
                )
                for b in ev_mean.pop("calibration_bins"):
                    calibration_rows.append({**base_row(cfg, scenario, realization, method, round_idx, channel_uses), **b, "eval_scope": "global_test"})
                for b in ev_mc.pop("calibration_bins"):
                    calibration_rows.append({**base_row(cfg, scenario, realization, method, round_idx, channel_uses), **b, "eval_scope": "global_test_mc"})
                sigma = 1.0 / np.sqrt(np.maximum(global_rho.astype(np.float64), 1e-12))
                snr = np.abs(global_mu.astype(np.float64)) / (sigma + 1e-12)
                metrics.append(
                    {
                        **base_row(cfg, scenario, realization, method, round_idx, channel_uses),
                        "accuracy": ev_mean["accuracy"],
                        "loss": ev_mean["loss"],
                        "ece": ev_mean["ece"],
                        "mce": ev_mean["mce"],
                        "mc_accuracy": ev_mc["accuracy"],
                        "mc_loss": ev_mc["loss"],
                        "mc_ece": ev_mc["ece"],
                        "phase1_loss_mean": float(np.nanmean(phase1_losses)),
                        "phase2_loss_mean": float(np.nanmean(phase2_losses)),
                        "rho_mean": float(np.mean(global_rho)),
                        "rho_p50": float(np.percentile(global_rho, 50)),
                        "rho_p90": float(np.percentile(global_rho, 90)),
                        "sigma_mean": float(np.mean(sigma)),
                        "sigma_p50": float(np.percentile(sigma, 50)),
                        "snr_raw_p50": float(np.percentile(snr, 50)),
                        "snr_raw_p90": float(np.percentile(snr, 90)),
                        "aircomp_rho_distortion_ratio": rho_air.distortion_ratio,
                        "aircomp_mu_distortion_ratio": mu_air.distortion_ratio,
                        "aircomp_rho_clipped_fraction": rho_air.clipped_fraction,
                        "aircomp_mu_clipped_fraction": mu_air.clipped_fraction,
                        "round_time_sec": float(time.time() - t0),
                    }
                )
        state_for_save = {"global_mu": global_mu, "global_rho": global_rho}
    else:
        raise ValueError(f"Unknown method {method!r}")

    return metrics, calibration_rows, state_for_save


def base_row(cfg: AirCompConfig, scenario: Mapping[str, object], realization: int, method: str, round_idx: int, channel_uses: int) -> dict[str, object]:
    return {
        "run_id": "aircomp_bayesfl",
        "scenario": str(scenario["scenario"]),
        "condition_name": str(scenario["condition_name"]),
        "realization": int(realization),
        "method": str(method),
        "round": int(round_idx),
        "channel_uses": int(channel_uses),
        "num_devices": int(cfg.num_devices),
        "local_classes": int(scenario["local_classes"]),
        "mean_client_examples": float(scenario["mean_client_examples"]),
        "power_dbm": float(scenario["power_dbm"]),
    }


def run_experiment(cfg: AirCompConfig) -> dict[str, Path]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    train_ds, test_ds, targets = load_mnist(cfg.data_dir)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=int(cfg.num_workers))

    all_metrics: list[dict[str, object]] = []
    all_calibration: list[dict[str, object]] = []
    all_clients: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []

    config_rows = [{"key": k, "value": v} for k, v in asdict(cfg).items()]
    write_csv(out_dir / "config.csv", config_rows, ["key", "value"])

    for scen_idx, scenario in enumerate(scenario_grid(cfg)):
        scenario_methods = [m.strip().lower() for m in scenario["methods"] if str(m).strip()]
        for realization in range(int(cfg.realizations)):
            scenario_seed = int(cfg.seed) + 10000 * scen_idx + 1000 * realization
            rng = np.random.default_rng(scenario_seed)
            partitions, client_rows = make_scarce_label_skew_partitions(
                targets=targets,
                num_devices=int(cfg.num_devices),
                mean_examples=float(scenario["mean_client_examples"]),
                local_classes=int(scenario["local_classes"]),
                rng=rng,
            )
            for row in client_rows:
                all_clients.append({"scenario": scenario["scenario"], "condition_name": scenario["condition_name"], "realization": realization, **row})
            loaders = make_client_loaders(train_ds, partitions, int(cfg.batch_size), scenario_seed, int(cfg.num_workers))
            distances_m = sample_device_distances(int(cfg.num_devices), float(cfg.coverage_radius_m), rng)
            init_flat = init_flat_model(seed=scenario_seed, device=device)
            for method in scenario_methods:
                print(f"[aircomp] scenario={scenario['scenario']} condition={scenario['condition_name']} realization={realization} method={method}", flush=True)
                method_rng = np.random.default_rng(scenario_seed + abs(hash(method)) % 100000)
                metrics, calib_rows, state = run_method(cfg, scenario, realization, method, loaders, test_loader, init_flat, distances_m, device, method_rng)
                all_metrics.extend(metrics)
                all_calibration.extend(calib_rows)
                if metrics:
                    final = metrics[-1]
                    best = max(metrics, key=lambda r: safe_float(r.get("accuracy")))
                    best_ece = min(metrics, key=lambda r: safe_float(r.get("ece")))
                    all_summary.append(
                        {
                            "scenario": scenario["scenario"],
                            "condition_name": scenario["condition_name"],
                            "realization": realization,
                            "method": method,
                            "final_round": final.get("round"),
                            "final_channel_uses": final.get("channel_uses"),
                            "final_accuracy": final.get("accuracy"),
                            "best_accuracy": best.get("accuracy"),
                            "best_accuracy_round": best.get("round"),
                            "final_loss": final.get("loss"),
                            "final_ece": final.get("ece"),
                            "best_ece": best_ece.get("ece"),
                            "best_ece_round": best_ece.get("round"),
                        }
                    )
                if bool(cfg.save_model):
                    model_dir = out_dir / "models" / str(scenario["scenario"]) / str(scenario["condition_name"]) / f"realization_{realization:02d}" / method
                    model_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(model_dir / "final_state.npz", **state)

    paths = {
        "metrics": out_dir / "metrics.csv",
        "calibration": out_dir / "calibration_bins.csv",
        "client_data": out_dir / "client_data_summary.csv",
        "summary": out_dir / "run_summary.csv",
    }
    write_csv(paths["metrics"], all_metrics)
    write_csv(paths["calibration"], all_calibration)
    write_csv(paths["client_data"], all_clients)
    write_csv(paths["summary"], all_summary)
    print("[aircomp] wrote:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return paths


def main() -> None:
    cfg = parse_args()
    run_experiment(cfg)


if __name__ == "__main__":
    main()
