"""MNIST loading and scarce/non-IID client partition generation."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import DataConfig, TrainingConfig


MNIST_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def ensure_mnist(root: str | Path) -> None:
    """Download MNIST once before Ray workers start."""
    root = str(root)
    datasets.MNIST(root=root, train=True, download=True, transform=MNIST_TRANSFORM)
    datasets.MNIST(root=root, train=False, download=True, transform=MNIST_TRANSFORM)


def _training_targets(root: str | Path) -> np.ndarray:
    dataset = datasets.MNIST(root=str(root), train=True, download=False)
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        return targets.cpu().numpy().astype(np.int64)
    return np.asarray(targets, dtype=np.int64)


def partition_filename(
    partition_dir: str | Path,
    seed: int,
    num_clients: int,
    labels_per_client: int,
    mean_samples: float,
) -> Path:
    safe_mean = str(mean_samples).replace(".", "p")
    return Path(partition_dir) / (
        f"mnist_seed{seed}_k{num_clients}_l{labels_per_client}_m{safe_mean}.json"
    )


def prepare_partitions(
    data_cfg: DataConfig,
    seed: int,
    partition_dir: str | Path,
    force: bool = False,
) -> Path:
    """Create a deterministic partition file shared by all methods in a run.

    Each client receives a Poisson-distributed number of examples and data from
    ``labels_per_client`` labels. Distances are sampled uniformly in area in a
    circular cell, i.e. r = R*sqrt(U).
    """
    partition_dir = Path(partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)
    path = partition_filename(
        partition_dir,
        seed,
        data_cfg.num_clients,
        data_cfg.labels_per_client,
        data_cfg.mean_samples_per_client,
    )
    if path.exists() and not force:
        return path

    ensure_mnist(data_cfg.root)
    targets = _training_targets(data_cfg.root)
    rng = np.random.default_rng(seed)

    pools: Dict[int, np.ndarray] = {}
    offsets: Dict[int, int] = {}
    for label in range(10):
        indices = np.flatnonzero(targets == label).astype(np.int64)
        rng.shuffle(indices)
        pools[label] = indices
        offsets[label] = 0

    clients: Dict[str, Dict[str, object]] = {}
    for client_id in range(data_cfg.num_clients):
        n_samples = int(rng.poisson(data_cfg.mean_samples_per_client))
        n_samples = max(data_cfg.min_samples_per_client, n_samples)

        if data_cfg.labels_per_client == 10:
            client_labels = list(range(10))
        else:
            client_labels = sorted(
                int(v)
                for v in rng.choice(
                    10, size=data_cfg.labels_per_client, replace=False
                ).tolist()
            )

        base = n_samples // len(client_labels)
        remainder = n_samples % len(client_labels)
        counts = [base + (1 if i < remainder else 0) for i in range(len(client_labels))]
        client_indices: List[int] = []

        for label, count in zip(client_labels, counts):
            if count == 0:
                continue
            pool = pools[label]
            start = offsets[label]
            end = start + count
            if end <= len(pool):
                selected = pool[start:end]
                offsets[label] = end
            else:
                # The paper-scale configuration never exhausts MNIST, but this
                # branch keeps custom large simulations usable.
                first = pool[start:]
                remaining = count - len(first)
                reshuffled = pool.copy()
                rng.shuffle(reshuffled)
                pools[label] = reshuffled
                second = reshuffled[:remaining]
                offsets[label] = remaining
                selected = np.concatenate([first, second])
            client_indices.extend(int(v) for v in selected.tolist())

        rng.shuffle(client_indices)
        distance = float(data_cfg.bs_radius_m * math.sqrt(rng.uniform(0.0, 1.0)))
        distance = max(1.0, distance)
        clients[str(client_id)] = {
            "indices": client_indices,
            "labels": client_labels,
            "distance_m": distance,
            "num_examples": len(client_indices),
        }

    payload = {
        "seed": seed,
        "num_clients": data_cfg.num_clients,
        "labels_per_client": data_cfg.labels_per_client,
        "mean_samples_per_client": data_cfg.mean_samples_per_client,
        "bs_radius_m": data_cfg.bs_radius_m,
        "clients": clients,
    }

    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(tmp_name, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    return path


def load_partition_metadata(partition_path: str | Path, client_id: int) -> Dict[str, object]:
    with Path(partition_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        return payload["clients"][str(client_id)]
    except KeyError as exc:
        raise KeyError(f"Client {client_id} not found in {partition_path}") from exc


def load_client_loader(
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    partition_path: str | Path,
    client_id: int,
    shuffle_seed: int,
    pin_memory: bool | None = None,
) -> Tuple[DataLoader, Dict[str, object]]:
    metadata = load_partition_metadata(partition_path, client_id)
    dataset = datasets.MNIST(
        root=data_cfg.root,
        train=True,
        download=False,
        transform=MNIST_TRANSFORM,
    )
    subset = Subset(dataset, [int(v) for v in metadata["indices"]])
    generator = torch.Generator()
    generator.manual_seed(int(shuffle_seed))
    effective_pin_memory = (
        data_cfg.pin_memory if pin_memory is None else bool(pin_memory)
    )
    loader = DataLoader(
        subset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=data_cfg.num_workers,
        pin_memory=effective_pin_memory,
        drop_last=False,
    )
    return loader, metadata


def load_test_loader(
    data_cfg: DataConfig,
    batch_size: int = 512,
    pin_memory: bool | None = None,
) -> DataLoader:
    dataset = datasets.MNIST(
        root=data_cfg.root,
        train=False,
        download=False,
        transform=MNIST_TRANSFORM,
    )
    effective_pin_memory = (
        data_cfg.pin_memory if pin_memory is None else bool(pin_memory)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=effective_pin_memory,
        drop_last=False,
    )


def client_sizes(partition_path: str | Path) -> List[int]:
    with Path(partition_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [int(payload["clients"][str(i)]["num_examples"]) for i in range(payload["num_clients"])]
