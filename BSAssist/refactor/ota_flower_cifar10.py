"""Flower simulation for BS-dataset-assisted broadband OTA-FL on CIFAR-10.

This is a refactor of the provided notebook into a runnable Python module using
Flower.  It implements Algorithm 3 from the paper: BS initial update +
superimposed over-the-air update report + optimized power allocation.  The
TCI benchmark is intentionally omitted.

Example quick smoke test:
    python ota_flower_cifar10.py --rounds 1 --num-devices 5 --m0-values 20 \
        --mean-client-size 20 --client-cpus 1

Example closer to the attached CIFAR-10 Fig. 1 setting:
    python ota_flower_cifar10.py --rounds 1000 --num-devices 300 \
        --coverage-m 550 --m0-values 1600 160 20 --mean-client-size 160 \
        --plot --output-dir outputs/cifar10_fig1
"""

from __future__ import annotations

import argparse
import csv
import copy
import math
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import flwr as fl
from flwr.common import FitIns, Metrics, NDArrays, Parameters, Scalar
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SimConfig:
    """Simulation parameters corresponding to the screenshot/paper notation."""

    num_devices: int = 300  # K
    coverage_m: float = 550.0  # r_cvge in the attached CIFAR-10 screenshot
    max_symbol_power_dbm: float = 20.0  # P [dBm]
    noise_power_dbm: float = -50.0  # sigma_z^2 [dBm]
    power_scaling_db: float = -10.0  # gamma [dB]
    num_subchannels: int = 1024  # F
    path_loss_exp: float = 4.0  # alpha
    mean_client_size: int = 160  # Poisson mean for |M_k| on CIFAR-10
    min_client_size: int = 10
    classes_per_client: int = 3
    batch_size: int = 10
    local_epochs: int = 3
    lr: float = 0.05
    optimizer: str = "sgd"  # "sgd" keeps the local-update logic close to Algorithm 1
    momentum: float = 0.0
    weight_decay: float = 0.0
    num_workers: int = 0
    seed: int = 42
    track_distortion: bool = True
    channel_eps: float = 1e-30
    rho_eps: float = 1e-30
    power_tol: float = 1e-5
    power_max_iters: int = 60
    device: str = "auto"

    @property
    def max_symbol_power_mw(self) -> float:
        return 10.0 ** (self.max_symbol_power_dbm / 10.0)

    @property
    def noise_power_mw(self) -> float:
        return 10.0 ** (self.noise_power_dbm / 10.0)

    @property
    def power_scaling_linear(self) -> float:
        return 10.0 ** (self.power_scaling_db / 10.0)


# -----------------------------------------------------------------------------
# Model: 7 trainable-layer CIFAR-10 CNN with D = 307,498 trainable parameters.
# -----------------------------------------------------------------------------


class Cifar10CNN(nn.Module):
    """7 trainable-layer CNN used for the CIFAR-10 OTA-FL simulation.

    Architecture:
      - two 3x3x32 conv layers + BN/ReLU + pool/dropout
      - two 3x3x64 conv layers + BN/ReLU + pool/dropout
      - two 3x3x128 conv layers + BN/ReLU + pool/dropout
      - one linear classifier with 10 outputs

    BatchNorm is configured with affine=False and track_running_stats=False so
    the communicated trainable state has exactly 307,498 parameters:
    6 convolutional layers + 1 fully connected layer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.4),
        )
        self.classifier = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# -----------------------------------------------------------------------------
# Utilities: reproducibility, model parameter conversion, training, evaluation
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def get_model_parameters(model: nn.Module) -> NDArrays:
    """Return model state tensors as NumPy arrays in state_dict order."""
    return [val.detach().cpu().numpy().copy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: NDArrays, device: torch.device) -> None:
    """Load model state tensors from NumPy arrays in state_dict order."""
    keys = list(model.state_dict().keys())
    if len(keys) != len(parameters):
        raise ValueError(f"Expected {len(keys)} tensors, received {len(parameters)} tensors")
    state_dict = OrderedDict()
    for key, value in zip(keys, parameters):
        state_dict[key] = torch.tensor(value, device=device)
    model.load_state_dict(state_dict, strict=True)


def flatten_parameters(parameters: NDArrays) -> np.ndarray:
    return np.concatenate([p.reshape(-1) for p in parameters]).astype(np.float32, copy=False)


def unflatten_parameters(flat: np.ndarray, shapes: Sequence[Tuple[int, ...]]) -> NDArrays:
    arrays: NDArrays = []
    cursor = 0
    for shape in shapes:
        size = int(np.prod(shape))
        arrays.append(flat[cursor : cursor + size].reshape(shape).astype(np.float32, copy=False))
        cursor += size
    if cursor != len(flat):
        raise ValueError(f"Flat vector has {len(flat)} values but shapes consume {cursor}")
    return arrays


def clone_parameters(parameters: NDArrays) -> NDArrays:
    return [p.copy() for p in parameters]


def make_optimizer(model: nn.Module, cfg: SimConfig) -> optim.Optimizer:
    opt = cfg.optimizer.lower()
    if opt == "sgd":
        return optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    if opt == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError("Unsupported optimizer. Use --optimizer sgd or --optimizer adam")


def train_local_model(
    model: nn.Module,
    trainloader: DataLoader,
    cfg: SimConfig,
    device: torch.device,
) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, cfg)

    for _ in range(cfg.local_epochs):
        for images, labels in trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_model(model: nn.Module, testloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in testloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(criterion(logits, labels).item())
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
    return total_loss / max(total, 1), correct / max(total, 1)


# -----------------------------------------------------------------------------
# CIFAR-10 data partitioning
# -----------------------------------------------------------------------------


def load_cifar10_datasets(data_dir: str) -> Tuple[Dataset, Dataset]:
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    return trainset, testset


def partition_cifar10_non_iid(
    trainset: Dataset,
    cfg: SimConfig,
    max_m0: int,
) -> Tuple[Subset, List[Subset]]:
    """Create BS and device datasets following the screenshot logic.

    BS data is sampled i.i.d. from the CIFAR-10 training set.  Each device has an
    imbalanced non-i.i.d. local set: a Poisson-random number of samples from
    exactly three of the ten classes.  No BS sample is reused by edge devices.
    """
    rng = np.random.default_rng(cfg.seed)
    all_indices = np.arange(len(trainset))
    rng.shuffle(all_indices)

    if max_m0 >= len(trainset):
        raise ValueError("max_m0 must be smaller than the CIFAR-10 training size")

    bs_indices = all_indices[:max_m0]
    remaining_indices = all_indices[max_m0:]
    targets = np.asarray(getattr(trainset, "targets"))

    class_bins: Dict[int, List[int]] = {label: [] for label in range(10)}
    for idx in remaining_indices:
        class_bins[int(targets[idx])].append(int(idx))
    for label in range(10):
        rng.shuffle(class_bins[label])

    # Pools restricted to remaining_indices, used only if a class bin runs out.
    class_replacement_pools: Dict[int, np.ndarray] = {
        label: np.asarray([idx for idx in remaining_indices if int(targets[idx]) == label], dtype=np.int64)
        for label in range(10)
    }

    client_datasets: List[Subset] = []
    for _ in range(cfg.num_devices):
        num_samples = max(cfg.min_client_size, int(rng.poisson(cfg.mean_client_size)))
        selected_classes = rng.choice(10, size=cfg.classes_per_client, replace=False)

        per_class = [num_samples // cfg.classes_per_client] * cfg.classes_per_client
        for i in range(num_samples % cfg.classes_per_client):
            per_class[i] += 1

        client_indices: List[int] = []
        for label, quota in zip(selected_classes, per_class):
            label = int(label)
            available = class_bins[label]
            take = min(quota, len(available))
            if take > 0:
                client_indices.extend(available[:take])
                del available[:take]

            deficit = quota - take
            if deficit > 0:
                pool = class_replacement_pools[label]
                if len(pool) == 0:
                    continue
                # Duplicates can occur only in deficit cases; BS overlap is still avoided.
                client_indices.extend(rng.choice(pool, size=deficit, replace=True).tolist())

        rng.shuffle(client_indices)
        client_datasets.append(Subset(trainset, client_indices))

    return Subset(trainset, bs_indices.tolist()), client_datasets


def label_distribution(subset: Subset, num_classes: int = 10) -> np.ndarray:
    targets = np.asarray(getattr(subset.dataset, "targets"))
    labels = targets[np.asarray(subset.indices, dtype=np.int64)]
    return np.bincount(labels, minlength=num_classes)


# -----------------------------------------------------------------------------
# OTA PHY simulation: optimized power allocation only; no TCI branch.
# -----------------------------------------------------------------------------


def optimized_power_allocation(
    u_arg: np.ndarray,
    delta_arg: np.ndarray,
    p_arg: float,
    tol: float = 1e-5,
    max_iters: int = 60,
) -> np.ndarray:
    """Algorithm 2 / Proposition 1 power allocation solved by bisection.

    This minimizes || |delta| - v ||^2 subject to sum(u * v^2) <= P and v >= 0.
    Returned p is u * v^2.  The implementation intentionally avoids the TCI
    truncation rule.
    """
    u = np.asarray(u_arg, dtype=np.float64)
    delta = np.asarray(delta_arg, dtype=np.float64)
    rho = delta * delta
    unconstrained = u * rho

    if float(np.sum(unconstrained)) <= p_arg:
        return unconstrained.astype(np.float32, copy=False)

    def power_for(c_val: float) -> float:
        denom = 1.0 + c_val * u
        return float(np.sum(unconstrained / (denom * denom)))

    c_low = 0.0
    c_high = 1.0
    while power_for(c_high) > p_arg:
        c_high *= 2.0
        if c_high > 1e30:
            break

    p_out = unconstrained
    for _ in range(max_iters):
        c_mid = 0.5 * (c_low + c_high)
        denom = 1.0 + c_mid * u
        p_out = unconstrained / (denom * denom)
        diff = p_arg - float(np.sum(p_out))
        if abs(diff) <= tol:
            break
        if diff > 0.0:
            c_high = c_mid
        else:
            c_low = c_mid

    return p_out.astype(np.float32, copy=False)


def simulate_ota_client_received_signal(
    delta_flat: np.ndarray,
    dataset_weight: float,
    distance_m: float,
    rho_ref: float,
    cfg: SimConfig,
    round_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate one device's OTA contribution h*x and ideal weighted delta.

    Returns:
        rx_noiseless_padded: h*x summed at the BS before receiver noise, padded
            to N*F values.  This is what the Flower client sends to the custom
            server strategy for simulated OTA summation.
        ideal_weighted_delta: w_k * delta_flat, length D, used only for
            distortion logging.
    """
    delta = np.asarray(delta_flat, dtype=np.float32)
    d_model = int(delta.size)
    f_sub = int(cfg.num_subchannels)
    n_symbols = int(math.ceil(d_model / f_sub))
    padded_size = n_symbols * f_sub
    pad = padded_size - d_model
    if pad > 0:
        delta_padded = np.pad(delta, (0, pad), mode="constant")
    else:
        delta_padded = delta

    rng = np.random.default_rng(round_seed)
    variance = float(distance_m ** (-cfg.path_loss_exp))
    std = math.sqrt(variance / 2.0)
    h_real = rng.normal(0.0, std, size=padded_size)
    h_imag = rng.normal(0.0, std, size=padded_size)
    h_abs_sq = h_real * h_real + h_imag * h_imag + cfg.channel_eps

    rho_safe = max(float(rho_ref), cfg.rho_eps)
    coeff = (dataset_weight * dataset_weight) * (
        cfg.power_scaling_linear * cfg.noise_power_mw / rho_safe
    )

    rx = np.zeros(padded_size, dtype=np.float32)
    for start in range(0, padded_size, f_sub):
        stop = start + f_sub
        delta_chunk = delta_padded[start:stop]
        h2_chunk = h_abs_sq[start:stop]
        g_chunk = 1.0 / h2_chunk
        u_chunk = coeff * g_chunk
        p_chunk = optimized_power_allocation(
            u_chunk, delta_chunk, cfg.max_symbol_power_mw, cfg.power_tol, cfg.power_max_iters
        )
        # After phase inversion, h*x = |h| * sign(delta) * sqrt(p), real-valued.
        rx[start:stop] = (
            np.sqrt(h2_chunk).astype(np.float32)
            * np.sign(delta_chunk).astype(np.float32)
            * np.sqrt(np.maximum(p_chunk, 0.0)).astype(np.float32)
        )

    ideal_weighted_delta = (dataset_weight * delta).astype(np.float32, copy=False)
    return rx, ideal_weighted_delta


# -----------------------------------------------------------------------------
# Flower client and custom OTA aggregation strategy
# -----------------------------------------------------------------------------


class OtaCifarClient(fl.client.NumPyClient):
    """Flower client that trains locally and returns a simulated OTA contribution."""

    def __init__(
        self,
        cid: str,
        trainset: Dataset,
        distance_m: float,
        cfg: SimConfig,
        total_examples_fn: Callable[[], int],
    ) -> None:
        self.cid = int(cid)
        self.trainset = trainset
        self.distance_m = float(distance_m)
        self.cfg = cfg
        self.total_examples_fn = total_examples_fn
        self.device = resolve_device(cfg.device)
        self.model = Cifar10CNN().to(self.device)
        self.trainloader = DataLoader(
            trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_model_parameters(self.model)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        server_round = int(config["server_round"])
        rho_ref = float(config["rho_ref"])
        global_examples = int(config["global_examples"])

        set_model_parameters(self.model, parameters, self.device)
        before = clone_parameters(parameters)

        train_local_model(self.model, self.trainloader, self.cfg, self.device)
        after = get_model_parameters(self.model)

        delta_flat = flatten_parameters(after) - flatten_parameters(before)
        weight = len(self.trainset) / max(global_examples, 1)
        round_seed = self.cfg.seed + 1_000_003 * server_round + self.cid
        rx, ideal = simulate_ota_client_received_signal(
            delta_flat=delta_flat,
            dataset_weight=weight,
            distance_m=self.distance_m,
            rho_ref=rho_ref,
            cfg=self.cfg,
            round_seed=round_seed,
        )

        if self.cfg.track_distortion:
            response: NDArrays = [rx, ideal]
        else:
            response = [rx]

        metrics: Dict[str, Scalar] = {
            "client_examples": int(len(self.trainset)),
            "distance_m": float(self.distance_m),
        }
        return response, len(self.trainset), metrics


class BsDatasetAssistedOtaStrategy(FedAvg):
    """Flower strategy implementing Algorithm 3 with optimized OTA aggregation."""

    def __init__(
        self,
        initial_parameters_ndarrays: NDArrays,
        bs_dataset: Dataset,
        testset: Dataset,
        client_sizes: Sequence[int],
        client_distances_m: Sequence[float],
        cfg: SimConfig,
        output_dir: Path,
        experiment_name: str,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.initial_parameters_ndarrays = clone_parameters(initial_parameters_ndarrays)
        self.shapes = [tuple(arr.shape) for arr in initial_parameters_ndarrays]
        self.d_model = int(sum(int(np.prod(s)) for s in self.shapes))
        self.n_symbols = int(math.ceil(self.d_model / cfg.num_subchannels))
        self.padded_size = self.n_symbols * cfg.num_subchannels

        self.bs_dataset = bs_dataset
        self.bs_loader = DataLoader(
            bs_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        self.bs_examples = int(len(bs_dataset))
        self.client_sizes = [int(x) for x in client_sizes]
        self.global_examples = int(self.bs_examples + sum(self.client_sizes))
        self.w0 = self.bs_examples / max(self.global_examples, 1)
        self.client_distances_m = np.asarray(client_distances_m, dtype=np.float64)

        self.bs_model = Cifar10CNN().to(self.device)
        self.eval_model = Cifar10CNN().to(self.device)
        self.testloader = DataLoader(
            testset,
            batch_size=256,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        self.theta_prime_ndarrays: Optional[NDArrays] = None
        self.rho_ref: float = 0.0
        self.last_distortion: float = float("nan")
        self.history: List[Dict[str, float]] = []
        self.rng = np.random.default_rng(cfg.seed + 77)

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=cfg.num_devices,
            min_available_clients=cfg.num_devices,
            min_evaluate_clients=0,
            initial_parameters=ndarrays_to_parameters(self.initial_parameters_ndarrays),
            accept_failures=False,
        )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Run BS initial update, then send theta'_t and rho_ref to all clients."""
        current_params = parameters_to_ndarrays(parameters)
        set_model_parameters(self.bs_model, current_params, self.device)
        train_local_model(self.bs_model, self.bs_loader, self.cfg, self.device)
        bs_updated = get_model_parameters(self.bs_model)

        current_flat = flatten_parameters(current_params)
        bs_updated_flat = flatten_parameters(bs_updated)
        delta0 = bs_updated_flat - current_flat
        theta_prime_flat = current_flat + self.w0 * delta0
        self.theta_prime_ndarrays = unflatten_parameters(theta_prime_flat, self.shapes)
        self.rho_ref = float(np.linalg.norm(delta0.astype(np.float64)) ** 2 / max(self.d_model, 1))

        fit_config: Dict[str, Scalar] = {
            "server_round": int(server_round),
            "rho_ref": float(self.rho_ref),
            "global_examples": int(self.global_examples),
            "num_subchannels": int(self.cfg.num_subchannels),
        }
        fit_ins = FitIns(ndarrays_to_parameters(self.theta_prime_ndarrays), fit_config)
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Metrics]:
        if failures:
            raise RuntimeError(f"Flower fit failures in round {server_round}: {failures}")
        if not results:
            return None, {}
        if self.theta_prime_ndarrays is None:
            raise RuntimeError("theta_prime was not initialized before aggregate_fit")

        rx_sum = np.zeros(self.padded_size, dtype=np.float64)
        ideal_sum = np.zeros(self.d_model, dtype=np.float64) if self.cfg.track_distortion else None

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            rx = arrays[0].reshape(-1)
            if rx.size != self.padded_size:
                raise ValueError(f"Client returned rx size {rx.size}, expected {self.padded_size}")
            rx_sum += rx.astype(np.float64, copy=False)
            if self.cfg.track_distortion:
                if len(arrays) < 2:
                    raise ValueError("track_distortion=True but client did not return ideal delta")
                ideal = arrays[1].reshape(-1)
                if ideal.size != self.d_model:
                    raise ValueError(f"Client returned ideal size {ideal.size}, expected {self.d_model}")
                ideal_sum += ideal.astype(np.float64, copy=False)

        scale = math.sqrt(max(self.rho_ref, self.cfg.rho_eps) / (
            self.cfg.power_scaling_linear * self.cfg.noise_power_mw
        ))

        actual_noiseless = scale * rx_sum[: self.d_model]
        if ideal_sum is not None:
            self.last_distortion = float(np.linalg.norm(actual_noiseless - ideal_sum) ** 2)
        else:
            self.last_distortion = float("nan")

        # Receiver noise z_t is complex CN(0, sigma_z^2 I); decoding uses real part.
        noise_std = math.sqrt(self.cfg.noise_power_mw / 2.0)
        noise_real = self.rng.normal(0.0, noise_std, size=self.padded_size)
        delta_hat_padded = scale * (rx_sum + noise_real)
        delta_hat_flat = delta_hat_padded[: self.d_model].astype(np.float32, copy=False)

        theta_prime_flat = flatten_parameters(self.theta_prime_ndarrays)
        next_flat = theta_prime_flat + delta_hat_flat
        next_parameters = unflatten_parameters(next_flat, self.shapes)

        metrics: Metrics = {
            "rho_ref": float(self.rho_ref),
            "distortion": float(self.last_distortion),
            "Nt": int(server_round * self.n_symbols),
            "N_symbols_per_round": int(self.n_symbols),
        }
        return ndarrays_to_parameters(next_parameters), metrics

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Metrics]]:
        params = parameters_to_ndarrays(parameters)
        set_model_parameters(self.eval_model, params, self.device)
        loss, accuracy = evaluate_model(self.eval_model, self.testloader, self.device)
        nt = int(server_round * self.n_symbols)
        row = {
            "round": float(server_round),
            "Nt": float(nt),
            "accuracy": float(accuracy),
            "loss": float(loss),
            "distortion": float(self.last_distortion),
            "rho_ref": float(self.rho_ref),
            "m0": float(self.bs_examples),
        }
        self.history.append(row)
        print(
            f"[{self.experiment_name}] round={server_round:04d} Nt={nt:07d} "
            f"acc={accuracy:.4f} loss={loss:.4f} distortion={self.last_distortion:.4e}"
        )
        return float(loss), {
            "accuracy": float(accuracy),
            "Nt": int(nt),
            "distortion": float(self.last_distortion),
            "rho_ref": float(self.rho_ref),
        }

    def save_history_csv(self) -> Path:
        csv_path = self.output_dir / f"{self.experiment_name}_history.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["round", "Nt", "accuracy", "loss", "distortion", "rho_ref", "m0"]
            )
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)
        return csv_path

    def save_model(self, parameters: Optional[Parameters] = None) -> Path:
        model_path = self.output_dir / f"{self.experiment_name}_model.pt"
        if parameters is not None:
            params = parameters_to_ndarrays(parameters)
        elif self.history:
            # The latest parameters are managed by Flower; this fallback saves the current eval model.
            params = get_model_parameters(self.eval_model)
        else:
            params = self.initial_parameters_ndarrays
        set_model_parameters(self.eval_model, params, self.device)
        torch.save(self.eval_model.state_dict(), model_path)
        return model_path


# -----------------------------------------------------------------------------
# Plotting and experiment runner
# -----------------------------------------------------------------------------


def plot_histories(history_paths: Sequence[Path], output_dir: Path) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plot creation because matplotlib could not be imported: {exc}")
        return None

    histories: Dict[str, Dict[str, List[float]]] = {}
    for path in history_paths:
        key = path.stem.replace("_history", "")
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        histories[key] = {
            "Nt": [float(r["Nt"]) for r in rows if float(r["round"]) > 0],
            "accuracy": [float(r["accuracy"]) for r in rows if float(r["round"]) > 0],
            "distortion": [float(r["distortion"]) for r in rows if float(r["round"]) > 0],
        }

    if not histories:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 8.0), sharex=True)
    for name, data in histories.items():
        label = name.replace("m0_", r"$|M_0|=$")
        ax1.plot(data["Nt"], data["accuracy"], label=f"Alg.2, {label}")
        ax2.plot(data["Nt"], data["distortion"], label=f"Alg.2, {label}")

    ax1.set_ylabel("Accuracy for test dataset")
    ax1.set_title("(a) Test accuracy")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.set_xlabel("Number of symbol transmissions, Nt")
    ax2.set_ylabel(r"2-Norm of Aggregated Distortion, $||\xi_t||^2$")
    ax2.set_title("(b) Aggregated update distortion")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    fig.tight_layout()
    out_path = output_dir / "fig1_cifar10_ota_flower.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def run_one_m0_experiment(
    m0: int,
    full_bs_dataset: Subset,
    client_datasets: Sequence[Subset],
    testset: Dataset,
    client_distances_m: np.ndarray,
    initial_parameters: NDArrays,
    cfg: SimConfig,
    rounds: int,
    output_dir: Path,
    client_cpus: float,
    client_gpus: float,
) -> Tuple[Path, BsDatasetAssistedOtaStrategy]:
    bs_subset = Subset(full_bs_dataset.dataset, list(full_bs_dataset.indices[:m0]))
    client_sizes = [len(ds) for ds in client_datasets]
    experiment_name = f"m0_{m0}"
    strategy = BsDatasetAssistedOtaStrategy(
        initial_parameters_ndarrays=initial_parameters,
        bs_dataset=bs_subset,
        testset=testset,
        client_sizes=client_sizes,
        client_distances_m=client_distances_m,
        cfg=cfg,
        output_dir=output_dir,
        experiment_name=experiment_name,
    )

    def total_examples() -> int:
        return len(bs_subset) + sum(client_sizes)

    def client_fn(cid: str) -> fl.client.Client:
        cid_int = int(cid)
        return OtaCifarClient(
            cid=cid,
            trainset=client_datasets[cid_int],
            distance_m=float(client_distances_m[cid_int]),
            cfg=cfg,
            total_examples_fn=total_examples,
        ).to_client()

    print(
        f"\n=== Flower OTA-FL / CIFAR-10 / |M0|={m0} / "
        f"K={cfg.num_devices} / D={strategy.d_model} / F={cfg.num_subchannels} / "
        f"N={strategy.n_symbols} ==="
    )
    print(f"Global examples: {strategy.global_examples}, BS weight w0={strategy.w0:.6f}")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_devices,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": client_cpus, "num_gpus": client_gpus},
    )
    _ = history  # Flower's History is still printed by Flower; custom CSV is saved below.

    csv_path = strategy.save_history_csv()
    strategy.save_model()
    print(f"Saved history: {csv_path}")
    return csv_path, strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BS dataset-assisted broadband OTA-FL on CIFAR-10 using Flower"
    )
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR-10 data directory")
    parser.add_argument("--output-dir", type=str, default="./outputs/cifar10_ota_flower")
    parser.add_argument("--rounds", type=int, default=1000, help="Number of Flower/FL rounds")
    parser.add_argument("--m0-values", type=int, nargs="+", default=[1600, 160, 20])
    parser.add_argument("--num-devices", type=int, default=300)
    parser.add_argument("--coverage-m", type=float, default=550.0)
    parser.add_argument("--mean-client-size", type=int, default=160)
    parser.add_argument("--min-client-size", type=int, default=10)
    parser.add_argument("--classes-per-client", type=int, default=3)
    parser.add_argument("--F", dest="num_subchannels", type=int, default=1024)
    parser.add_argument("--P-dbm", dest="max_symbol_power_dbm", type=float, default=20.0)
    parser.add_argument("--sigma-z2-dbm", dest="noise_power_dbm", type=float, default=-50.0)
    parser.add_argument("--gamma-db", dest="power_scaling_db", type=float, default=-10.0)
    parser.add_argument("--alpha", dest="path_loss_exp", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--client-cpus", type=float, default=1.0)
    parser.add_argument("--client-gpus", type=float, default=0.0)
    parser.add_argument("--no-distortion", action="store_true", help="Do not return ideal deltas for distortion logging")
    parser.add_argument("--plot", action="store_true", help="Save a Fig. 1-style accuracy/distortion plot")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Override settings for a quick syntax/runtime check: K=5, M0=20, rounds=1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke_test:
        args.rounds = 1
        args.num_devices = 5
        args.m0_values = [20]
        args.mean_client_size = 20
        args.local_epochs = 1
        args.batch_size = 10
        args.output_dir = str(Path(args.output_dir) / "smoke_test")

    cfg = SimConfig(
        num_devices=args.num_devices,
        coverage_m=args.coverage_m,
        max_symbol_power_dbm=args.max_symbol_power_dbm,
        noise_power_dbm=args.noise_power_dbm,
        power_scaling_db=args.power_scaling_db,
        num_subchannels=args.num_subchannels,
        path_loss_exp=args.path_loss_exp,
        mean_client_size=args.mean_client_size,
        min_client_size=args.min_client_size,
        classes_per_client=args.classes_per_client,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        lr=args.lr,
        optimizer=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        track_distortion=not args.no_distortion,
        device=args.device,
    )

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = Cifar10CNN().to(device)
    d_model = count_trainable_params(base_model)
    if d_model != 307_498:
        raise RuntimeError(f"Expected CIFAR-10 CNN to have D=307,498 parameters, got {d_model}")
    initial_parameters = get_model_parameters(base_model)

    print(f"Using device: {device}")
    print(f"Flower version: {fl.__version__}")
    print(f"CIFAR-10 CNN trainable parameters D={d_model:,}")
    print(
        f"N=ceil(D/F)={math.ceil(d_model / cfg.num_subchannels)} OFDM symbols per update round "
        f"with F={cfg.num_subchannels}"
    )

    trainset, testset = load_cifar10_datasets(args.data_dir)
    max_m0 = max(args.m0_values)
    full_bs_dataset, client_datasets = partition_cifar10_non_iid(trainset, cfg, max_m0=max_m0)

    rng = np.random.default_rng(cfg.seed + 999)
    client_distances = np.clip(cfg.coverage_m * np.sqrt(rng.random(cfg.num_devices)), 10.0, cfg.coverage_m)

    # Save a compact summary of the data split.
    split_path = output_dir / "data_split_summary.csv"
    with split_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entity", "num_examples", "class_counts_0_to_9"])
        writer.writerow(["BS_max", len(full_bs_dataset), label_distribution(full_bs_dataset).tolist()])
        for cid, ds in enumerate(client_datasets):
            writer.writerow([f"client_{cid}", len(ds), label_distribution(ds).tolist()])
    print(f"Saved data split summary: {split_path}")

    history_paths: List[Path] = []
    for m0 in args.m0_values:
        csv_path, _ = run_one_m0_experiment(
            m0=m0,
            full_bs_dataset=full_bs_dataset,
            client_datasets=client_datasets,
            testset=testset,
            client_distances_m=client_distances,
            initial_parameters=initial_parameters,
            cfg=cfg,
            rounds=args.rounds,
            output_dir=output_dir,
            client_cpus=args.client_cpus,
            client_gpus=args.client_gpus,
        )
        history_paths.append(csv_path)

    if args.plot:
        plot_path = plot_histories(history_paths, output_dir)
        if plot_path is not None:
            print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
