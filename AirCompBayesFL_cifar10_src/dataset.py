"""MNIST loading and scarce/non-IID client partition generation."""

from __future__ import annotations

import json
import math
import os
import tempfile
from functools import lru_cache
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


def _root_cache_key(root: str | Path) -> str:
    """Return one stable cache key for a dataset root."""
    return str(Path(root).resolve())


@lru_cache(maxsize=4)
def _cached_mnist(root_key: str, train: bool) -> datasets.MNIST:
    """Load each MNIST split once per Python process.

    The native-Windows local backend calls ``load_client_loader`` for every
    client and every physical phase. Reconstructing ``datasets.MNIST`` on each
    call repeatedly allocates the full ~47 MB training image tensor and can
    eventually exhaust/fragment host RAM during long two-phase runs.

    The dataset object is immutable for our usage; client-specific subsets and
    shuffle order remain separate, so caching does not change the experiment.
    Ray workers naturally get independent process-local caches.
    """
    return datasets.MNIST(
        root=root_key,
        train=bool(train),
        download=False,
        transform=MNIST_TRANSFORM,
    )


def clear_dataset_cache() -> None:
    """Clear process-local dataset objects (mainly useful for tests)."""
    _cached_mnist.cache_clear()


def ensure_mnist(root: str | Path) -> None:
    """Download MNIST once before Ray workers start."""
    root = str(root)
    datasets.MNIST(root=root, train=True, download=True, transform=MNIST_TRANSFORM)
    datasets.MNIST(root=root, train=False, download=True, transform=MNIST_TRANSFORM)


def _training_targets(root: str | Path) -> np.ndarray:
    dataset = _cached_mnist(_root_cache_key(root), True)
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
    label_pairing_mode: str = "uniform",
) -> Path:
    safe_mean = str(mean_samples).replace(".", "p")

    pairing_mode = str(
        label_pairing_mode
    ).strip().lower()

    suffix = (
        ""
        if pairing_mode == "uniform"
        else f"_{pairing_mode}"
    )

    return Path(partition_dir) / (
        f"mnist_seed{seed}_k{num_clients}_l{labels_per_client}_"
        f"m{safe_mean}{suffix}.json"
    )


def _sample_client_labels(
    rng: np.random.Generator,
    labels_per_client: int,
    label_pairing_mode: str,
) -> List[int]:
    """Sample one client's class-label set.

    uniform:
        Original behavior.

    random_nonadjacent:
        Uniform random draw over unordered two-class pairs satisfying
        abs(label_a - label_b) > 1.

        Therefore:
            (0, 1), (1, 2), ..., (8, 9) are forbidden.
            (0, 9) is allowed.
    """

    labels_per_client = int(
        labels_per_client
    )

    mode = str(
        label_pairing_mode
    ).strip().lower()

    if mode == "random_nonadjacent":

        if labels_per_client != 2:
            raise ValueError(
                "random_nonadjacent pairing requires "
                "labels_per_client=2"
            )

        allowed_pairs = [
            (a, b)
            for a in range(10)
            for b in range(a + 1, 10)
            if abs(a - b) > 1
        ]

        pair = allowed_pairs[
            int(
                rng.integers(
                    0,
                    len(allowed_pairs),
                )
            )
        ]

        return [
            int(pair[0]),
            int(pair[1]),
        ]

    if mode != "uniform":
        raise ValueError(
            f"Unknown label pairing mode: {mode!r}"
        )

    if labels_per_client == 10:
        return list(range(10))

    return sorted(
        int(v)
        for v in rng.choice(
            10,
            size=labels_per_client,
            replace=False,
        ).tolist()
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
        data_cfg.label_pairing_mode,
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

        client_labels = _sample_client_labels(
            rng,
            data_cfg.labels_per_client,
            data_cfg.label_pairing_mode,
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
        "label_pairing_mode": data_cfg.label_pairing_mode,
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
    dataset = _cached_mnist(_root_cache_key(data_cfg.root), True)
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
    dataset = _cached_mnist(_root_cache_key(data_cfg.root), False)
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


# ============================================================
# AIRCOMP_RAY_SAFE_CIFAR_DATASET
#
# main_cifar10.py sets AIRCOMP_DATASET=cifar10.
# Local execution already receives the normal in-process
# override. Ray actors are fresh Python processes, so they
# need to install the same override when dataset.py imports.
# ============================================================

import os as _aircomp_os

if (
    _aircomp_os.environ
    .get("AIRCOMP_DATASET", "mnist")
    .strip()
    .lower()
    == "cifar10"
):
    from cifar10_support import (
        CIFAR10AsMNIST as _AirCompCIFAR10AsMNIST,
    )

    datasets.MNIST = _AirCompCIFAR10AsMNIST

