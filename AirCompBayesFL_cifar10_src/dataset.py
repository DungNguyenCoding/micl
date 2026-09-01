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


def _float_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def partition_filename(
    partition_dir: str | Path,
    seed: int,
    num_clients: int,
    labels_per_client: int,
    mean_samples: float,
    label_pairing_mode: str = "uniform",
    partition_mode: str = "legacy",
    dirichlet_alpha: float = 0.1,
) -> Path:
    """Return a stable filename without changing legacy cache names."""
    mode = str(partition_mode).strip().lower()
    safe_mean = _float_token(mean_samples)

    if mode == "dirichlet":
        dataset_name = (
            os.environ.get("AIRCOMP_DATASET", "mnist").strip().lower()
            or "mnist"
        )
        safe_alpha = _float_token(dirichlet_alpha)
        return Path(partition_dir) / (
            f"{dataset_name}_seed{seed}_k{num_clients}_"
            f"dirichlet_a{safe_alpha}_m{safe_mean}.json"
        )

    pairing = str(label_pairing_mode).strip().lower()
    suffix = "" if pairing == "uniform" else f"_{pairing}"
    return Path(partition_dir) / (
        f"mnist_seed{seed}_k{num_clients}_l{labels_per_client}_"
        f"m{safe_mean}{suffix}.json"
    )


def _sample_client_labels(
    rng: np.random.Generator,
    labels_per_client: int,
    label_pairing_mode: str,
) -> List[int]:
    """Sample labels for the original fixed-label partition mode."""
    labels_per_client = int(labels_per_client)
    mode = str(label_pairing_mode).strip().lower()

    if mode == "random_nonadjacent":
        if labels_per_client != 2:
            raise ValueError(
                "random_nonadjacent pairing requires labels_per_client=2"
            )
        allowed_pairs = [
            (a, b)
            for a in range(10)
            for b in range(a + 1, 10)
            if abs(a - b) > 1
        ]
        pair = allowed_pairs[int(rng.integers(0, len(allowed_pairs)))]
        return [int(pair[0]), int(pair[1])]

    if mode != "uniform":
        raise ValueError(f"Unknown label pairing mode: {mode!r}")

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


def _build_label_pools(
    targets: np.ndarray,
    rng: np.random.Generator,
) -> tuple[Dict[int, np.ndarray], Dict[int, int]]:
    pools: Dict[int, np.ndarray] = {}
    offsets: Dict[int, int] = {}
    for label in range(10):
        indices = np.flatnonzero(targets == label).astype(np.int64)
        rng.shuffle(indices)
        pools[label] = indices
        offsets[label] = 0
    return pools, offsets


def _take_from_pool(
    *,
    label: int,
    count: int,
    pools: Dict[int, np.ndarray],
    offsets: Dict[int, int],
) -> np.ndarray:
    """Take unique examples from one label pool without replacement."""
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    pool = pools[int(label)]
    start = int(offsets[int(label)])
    end = start + int(count)
    if end > len(pool):
        raise RuntimeError(
            f"Partition requested {end} examples from class {label}, "
            f"but only {len(pool)} are available without replacement. "
            "Reduce mean_samples_per_client or use a larger dataset."
        )
    offsets[int(label)] = end
    return pool[start:end]


def prepare_partitions(
    data_cfg: DataConfig,
    seed: int,
    partition_dir: str | Path,
    force: bool = False,
) -> Path:
    """Create one deterministic partition file shared by all methods.

    ``legacy`` preserves the original scarce/non-IID partition rule.

    ``dirichlet`` preserves the existing data-scarcity budget: each client
    first receives a Poisson-distributed total sample count with mean
    ``mean_samples_per_client``.  Its ten-class mixture is then sampled as

        pi_k ~ Dirichlet(alpha * 1_10)
        n_k,* ~ Multinomial(n_k, pi_k)

    where ``alpha=data_cfg.dirichlet_alpha``.  Examples are allocated without
    replacement, so the same image never belongs to two clients.
    """
    partition_dir = Path(partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)

    pairing_mode = str(getattr(data_cfg, "label_pairing_mode", "uniform"))
    partition_mode = str(getattr(data_cfg, "partition_mode", "legacy")).strip().lower()
    dirichlet_alpha = float(getattr(data_cfg, "dirichlet_alpha", 0.1))

    path = partition_filename(
        partition_dir,
        seed,
        data_cfg.num_clients,
        data_cfg.labels_per_client,
        data_cfg.mean_samples_per_client,
        pairing_mode,
        partition_mode,
        dirichlet_alpha,
    )
    if path.exists() and not force:
        return path

    ensure_mnist(data_cfg.root)
    targets = _training_targets(data_cfg.root)
    rng = np.random.default_rng(seed)
    pools, offsets = _build_label_pools(targets, rng)

    clients: Dict[str, Dict[str, object]] = {}

    for client_id in range(data_cfg.num_clients):
        n_samples = int(rng.poisson(data_cfg.mean_samples_per_client))
        n_samples = max(data_cfg.min_samples_per_client, n_samples)

        if partition_mode == "dirichlet":
            proportions = rng.dirichlet(
                np.full(10, dirichlet_alpha, dtype=np.float64)
            )
            counts = rng.multinomial(n_samples, proportions).astype(np.int64)
            client_labels = [int(i) for i in np.flatnonzero(counts > 0)]
        elif partition_mode == "legacy":
            client_labels = _sample_client_labels(
                rng,
                data_cfg.labels_per_client,
                pairing_mode,
            )
            base = n_samples // len(client_labels)
            remainder = n_samples % len(client_labels)
            counts = np.zeros(10, dtype=np.int64)
            for i, label in enumerate(client_labels):
                counts[int(label)] = base + (1 if i < remainder else 0)
        else:
            raise ValueError(f"Unknown partition_mode: {partition_mode!r}")

        client_indices: List[int] = []
        for label in range(10):
            selected = _take_from_pool(
                label=label,
                count=int(counts[label]),
                pools=pools,
                offsets=offsets,
            )
            client_indices.extend(int(v) for v in selected.tolist())

        rng.shuffle(client_indices)
        distance = float(data_cfg.bs_radius_m * math.sqrt(rng.uniform(0.0, 1.0)))
        distance = max(1.0, distance)

        clients[str(client_id)] = {
            "indices": client_indices,
            "labels": client_labels,
            "class_counts": {
                str(label): int(counts[label])
                for label in range(10)
                if int(counts[label]) > 0
            },
            "distance_m": distance,
            "num_examples": len(client_indices),
        }

    all_indices = [
        int(index)
        for client in clients.values()
        for index in client["indices"]
    ]
    if len(all_indices) != len(set(all_indices)):
        raise RuntimeError("Partition contains duplicate training indices across clients")

    payload = {
        "seed": seed,
        "num_clients": data_cfg.num_clients,
        "partition_mode": partition_mode,
        "labels_per_client": data_cfg.labels_per_client,
        "label_pairing_mode": pairing_mode,
        "mean_samples_per_client": data_cfg.mean_samples_per_client,
        "dirichlet_alpha": dirichlet_alpha if partition_mode == "dirichlet" else None,
        "dirichlet_allocation": (
            "client_label_proportions" if partition_mode == "dirichlet" else None
        ),
        "bs_radius_m": data_cfg.bs_radius_m,
        "dataset_train_size": int(len(targets)),
        "total_selected_examples": int(len(all_indices)),
        "unique_selected_examples": int(len(set(all_indices))),
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

