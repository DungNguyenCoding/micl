"""Deterministic non-IID partition generators and persisted manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


@dataclass
class PartitionResult:
    indices: list[np.ndarray]
    metadata: Dict[str, Any]


def _hash_partitions(parts: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for cid, arr in enumerate(parts):
        digest.update(np.asarray([cid, len(arr)], dtype=np.int64).tobytes())
        digest.update(np.asarray(arr, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _adjust_total_sizes(
    sizes: np.ndarray,
    target_total: int,
    rng: np.random.RandomState,
    min_size: int,
) -> np.ndarray:
    sizes = sizes.astype(np.int64, copy=True)
    delta = int(target_total) - int(sizes.sum())
    if delta > 0:
        candidates = np.arange(len(sizes))
        for idx in rng.choice(candidates, size=delta, replace=True):
            sizes[idx] += 1
    elif delta < 0:
        for _ in range(-delta):
            candidates = np.flatnonzero(sizes > min_size)
            if len(candidates) == 0:
                raise ValueError("Cannot reduce client sizes to target_total without violating min_size")
            idx = int(rng.choice(candidates))
            sizes[idx] -= 1
    return sizes


def build_sparse_dirichlet_indices(
    labels: np.ndarray,
    *,
    num_clients: int,
    num_classes: int,
    alpha: float,
    avg_samples_per_client: float,
    classes_per_client: int,
    min_samples_per_client: int,
    seed: int,
    target_total_samples: int | None = None,
) -> PartitionResult:
    """Sparse, unbalanced partition used by the CIFAR-10 experiments.

    Each client receives a Poisson-sized local set and exactly `classes_per_client`
    active classes. One sample is reserved for every active class, then the
    remaining count is drawn from a Dirichlet allocation over those classes.
    Samples are assigned without replacement from class-specific pools.
    """
    if classes_per_client > num_classes:
        raise ValueError("classes_per_client cannot exceed num_classes")
    rng = np.random.RandomState(seed)
    sizes = np.maximum(rng.poisson(avg_samples_per_client, size=num_clients), min_samples_per_client)
    sizes = np.maximum(sizes, classes_per_client)
    if target_total_samples is not None:
        sizes = _adjust_total_sizes(sizes, int(target_total_samples), rng, max(min_samples_per_client, classes_per_client))

    pools: dict[int, np.ndarray] = {}
    cursor = {c: 0 for c in range(num_classes)}
    for c in range(num_classes):
        arr = np.flatnonzero(labels == c).astype(np.int64)
        rng.shuffle(arr)
        pools[c] = arr

    partitions: list[np.ndarray] = []
    class_draws_exhausted = 0
    for client_size in sizes:
        active = rng.choice(num_classes, size=classes_per_client, replace=False)
        remaining = int(client_size) - classes_per_client
        proportions = rng.dirichlet(np.full(classes_per_client, alpha, dtype=np.float64))
        extra = rng.multinomial(remaining, proportions) if remaining > 0 else np.zeros(classes_per_client, dtype=int)
        counts = extra + 1

        client_chunks: list[np.ndarray] = []
        shortage = 0
        for c, count in zip(active, counts):
            c = int(c)
            start = cursor[c]
            end = min(start + int(count), len(pools[c]))
            if end > start:
                client_chunks.append(pools[c][start:end])
                cursor[c] = end
            shortage += int(count) - (end - start)

        # With CIFAR-10's 50k pool and ~10k samples used this should stay zero.
        if shortage:
            class_draws_exhausted += shortage
            available_classes = [c for c in range(num_classes) if cursor[c] < len(pools[c])]
            while shortage and available_classes:
                c = int(rng.choice(available_classes))
                take = min(shortage, len(pools[c]) - cursor[c])
                client_chunks.append(pools[c][cursor[c] : cursor[c] + take])
                cursor[c] += take
                shortage -= take
                available_classes = [j for j in available_classes if cursor[j] < len(pools[j])]
            if shortage:
                raise RuntimeError("Dataset exhausted while constructing sparse partition")

        client_indices = np.concatenate(client_chunks).astype(np.int64)
        rng.shuffle(client_indices)
        partitions.append(client_indices)

    realized_classes = [len(np.unique(labels[idx])) for idx in partitions]
    realized_sizes = np.asarray([len(idx) for idx in partitions], dtype=np.int64)
    metadata = {
        "type": "sparse_dirichlet",
        "seed": int(seed),
        "num_clients": int(num_clients),
        "num_classes": int(num_classes),
        "dirichlet_alpha": float(alpha),
        "avg_samples_per_client_config": float(avg_samples_per_client),
        "classes_per_client_config": int(classes_per_client),
        "total_samples_used": int(realized_sizes.sum()),
        "mean_size": float(realized_sizes.mean()),
        "min_size": int(realized_sizes.min()),
        "max_size": int(realized_sizes.max()),
        "mean_classes_per_client": float(np.mean(realized_classes)),
        "empty_client_backfills": 0,
        "num_empty_clients_after_backfill": int(np.sum(realized_sizes == 0)),
        "class_draws_exhausted": int(class_draws_exhausted),
    }
    metadata["sha256"] = _hash_partitions(partitions)
    return PartitionResult(partitions, metadata)


def build_mnist_dirichlet_lognormal_indices(
    labels: np.ndarray,
    *,
    num_clients: int,
    num_classes: int,
    alpha: float,
    lognormal_sigma: float,
    min_samples_per_client: int,
    seed: int,
) -> PartitionResult:
    """Use all MNIST data with non-IID labels and unbalanced client sizes."""
    rng = np.random.RandomState(seed)
    size_bias = rng.lognormal(mean=0.0, sigma=lognormal_sigma, size=num_clients)
    size_bias = size_bias / size_bias.mean()

    partitions: list[list[int]] = [[] for _ in range(num_clients)]
    concentration = np.maximum(alpha * size_bias, 1e-6)
    for c in range(num_classes):
        class_idx = np.flatnonzero(labels == c).astype(np.int64)
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(concentration)
        counts = rng.multinomial(len(class_idx), proportions)
        cursor = 0
        for cid, count in enumerate(counts):
            if count:
                partitions[cid].extend(class_idx[cursor : cursor + count].tolist())
                cursor += count

    # Backfill very small clients by moving samples from the largest clients.
    backfills = 0
    sizes = np.asarray([len(p) for p in partitions], dtype=np.int64)
    for cid in np.argsort(sizes):
        while len(partitions[int(cid)]) < min_samples_per_client:
            donor = int(np.argmax([len(p) for p in partitions]))
            if donor == int(cid) or len(partitions[donor]) <= min_samples_per_client:
                raise RuntimeError("Unable to satisfy MNIST min_samples_per_client")
            move_pos = int(rng.randint(0, len(partitions[donor])))
            partitions[int(cid)].append(partitions[donor].pop(move_pos))
            backfills += 1

    arrays = [np.asarray(p, dtype=np.int64) for p in partitions]
    for arr in arrays:
        rng.shuffle(arr)
    realized_sizes = np.asarray([len(p) for p in arrays], dtype=np.int64)
    realized_classes = [len(np.unique(labels[p])) for p in arrays]
    metadata = {
        "type": "dirichlet_lognormal",
        "seed": int(seed),
        "num_clients": int(num_clients),
        "num_classes": int(num_classes),
        "dirichlet_alpha": float(alpha),
        "lognormal_sigma": float(lognormal_sigma),
        "total_samples_used": int(realized_sizes.sum()),
        "mean_size": float(realized_sizes.mean()),
        "min_size": int(realized_sizes.min()),
        "max_size": int(realized_sizes.max()),
        "mean_classes_per_client": float(np.mean(realized_classes)),
        "empty_client_backfills": int(backfills),
        "num_empty_clients_after_backfill": int(np.sum(realized_sizes == 0)),
        "class_draws_exhausted": 0,
    }
    metadata["sha256"] = _hash_partitions(arrays)
    return PartitionResult(arrays, metadata)


def save_partition(result: PartitionResult, npz_path: Path, metadata_path: Path) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"client_{cid:04d}": arr for cid, arr in enumerate(result.indices)}
    np.savez_compressed(npz_path, **payload)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(result.metadata, handle, indent=2, sort_keys=True)


def load_partition(npz_path: str | Path) -> list[np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        keys = sorted(data.files)
        return [np.asarray(data[key], dtype=np.int64) for key in keys]
