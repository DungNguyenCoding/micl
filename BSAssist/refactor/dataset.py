"""CIFAR-10 dataset loading and BS/device partitioning for OTA-FL."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10

from config import SimConfig


def load_cifar10(data_dir: str) -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    testset = CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return trainset, testset


def partition_cifar10_non_iid(
    trainset: Dataset,
    cfg: SimConfig,
    max_m0: int,
) -> Tuple[Subset, List[Subset]]:
    """Create BS IID dataset and non-IID/imbalanced edge-device datasets.

    - BS receives an IID subset of size max(m0_values).
    - Every edge device receives samples from exactly cfg.classes_per_client
      classes out of 10.
    - Edge-device local dataset sizes are Poisson-like and imbalanced.
    - No BS sample is reused by edge devices.
    """
    rng = np.random.default_rng(cfg.seed)
    all_indices = np.arange(len(trainset))
    rng.shuffle(all_indices)

    if max_m0 >= len(trainset):
        raise ValueError("max_m0 must be smaller than the CIFAR-10 training size")

    bs_indices = all_indices[:max_m0]
    remaining_indices = all_indices[max_m0:]
    targets = np.asarray(getattr(trainset, "targets"), dtype=np.int64)

    class_bins: Dict[int, List[int]] = {label: [] for label in range(10)}
    for idx in remaining_indices:
        class_bins[int(targets[idx])].append(int(idx))
    for label in range(10):
        rng.shuffle(class_bins[label])

    # Replacement pools are only used if a class is depleted. BS overlap is still avoided.
    class_replacement_pools: Dict[int, np.ndarray] = {
        label: np.asarray(
            [idx for idx in remaining_indices if int(targets[idx]) == label], dtype=np.int64
        )
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
            take = min(int(quota), len(available))
            if take > 0:
                client_indices.extend(available[:take])
                del available[:take]

            deficit = int(quota) - take
            if deficit > 0:
                pool = class_replacement_pools[label]
                if len(pool) > 0:
                    client_indices.extend(rng.choice(pool, size=deficit, replace=True).tolist())

        rng.shuffle(client_indices)
        client_datasets.append(Subset(trainset, client_indices))

    return Subset(trainset, bs_indices.tolist()), client_datasets


def sample_client_distances(cfg: SimConfig) -> np.ndarray:
    """Uniform device placement inside the BS coverage disk."""
    rng = np.random.default_rng(cfg.seed + 999)
    return np.clip(cfg.coverage_m * np.sqrt(rng.random(cfg.num_devices)), 10.0, cfg.coverage_m)


def label_distribution(subset: Subset, num_classes: int = 10) -> np.ndarray:
    targets = np.asarray(getattr(subset.dataset, "targets"), dtype=np.int64)
    labels = targets[np.asarray(subset.indices, dtype=np.int64)]
    return np.bincount(labels, minlength=num_classes)


def make_device_groups(num_devices: int, num_groups: int) -> List[List[int]]:
    """Split K simulated devices across fewer Flower virtual clients.

    Each Flower client acts as a worker that trains/simulates several edge
    devices. The physical simulation still contains num_devices devices.
    """
    if num_groups <= 0:
        num_groups = num_devices
    num_groups = min(num_groups, num_devices)
    base = num_devices // num_groups
    rem = num_devices % num_groups
    groups: List[List[int]] = []
    cursor = 0
    for group_id in range(num_groups):
        size = base + (1 if group_id < rem else 0)
        groups.append(list(range(cursor, cursor + size)))
        cursor += size
    return groups


def save_split_summary(path: Path, full_bs_dataset: Subset, client_datasets: Sequence[Subset]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entity", "num_examples", "class_counts_0_to_9"])
        writer.writerow(["BS_max", len(full_bs_dataset), label_distribution(full_bs_dataset).tolist()])
        for cid, ds in enumerate(client_datasets):
            writer.writerow([f"client_{cid}", len(ds), label_distribution(ds).tolist()])
