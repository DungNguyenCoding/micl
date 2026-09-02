"""MNIST/CIFAR-10 loading and deterministic scarce client partitions."""

from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import DataConfig, TrainingConfig


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _root_key(root: str | Path) -> str:
    return str(Path(root).resolve())


def _transform(data_cfg: DataConfig, train: bool):
    dataset = str(data_cfg.dataset).lower()
    if dataset == "mnist":
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
        )

    if dataset == "cifar10":
        steps = []
        if train and bool(data_cfg.augment):
            if int(data_cfg.crop_padding) > 0:
                steps.append(
                    transforms.RandomCrop(32, padding=int(data_cfg.crop_padding))
                )
            if bool(data_cfg.random_flip):
                steps.append(transforms.RandomHorizontalFlip())
        steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
        return transforms.Compose(steps)

    raise ValueError(f"Unsupported dataset: {data_cfg.dataset!r}")


@lru_cache(maxsize=16)
def _cached_dataset(
    dataset_name: str,
    root_key: str,
    train: bool,
    augment: bool,
    crop_padding: int,
    random_flip: bool,
):
    cfg = DataConfig(
        dataset=dataset_name,
        root=root_key,
        augment=augment,
        crop_padding=crop_padding,
        random_flip=random_flip,
    )
    transform = _transform(cfg, bool(train))
    if dataset_name == "mnist":
        return datasets.MNIST(
            root=root_key,
            train=bool(train),
            download=False,
            transform=transform,
        )
    if dataset_name == "cifar10":
        return datasets.CIFAR10(
            root=root_key,
            train=bool(train),
            download=False,
            transform=transform,
        )
    raise ValueError(dataset_name)


def clear_dataset_cache() -> None:
    _cached_dataset.cache_clear()


def ensure_dataset(data_cfg: DataConfig) -> None:
    root = str(data_cfg.root)
    if data_cfg.dataset == "mnist":
        datasets.MNIST(root=root, train=True, download=True, transform=_transform(data_cfg, True))
        datasets.MNIST(root=root, train=False, download=True, transform=_transform(data_cfg, False))
    elif data_cfg.dataset == "cifar10":
        datasets.CIFAR10(root=root, train=True, download=True, transform=_transform(data_cfg, True))
        datasets.CIFAR10(root=root, train=False, download=True, transform=_transform(data_cfg, False))
    else:
        raise ValueError(data_cfg.dataset)


def _dataset(data_cfg: DataConfig, train: bool):
    return _cached_dataset(
        str(data_cfg.dataset),
        _root_key(data_cfg.root),
        bool(train),
        bool(data_cfg.augment if train else False),
        int(data_cfg.crop_padding),
        bool(data_cfg.random_flip),
    )


def _training_targets(data_cfg: DataConfig) -> np.ndarray:
    ds = _dataset(data_cfg, True)
    targets = ds.targets
    if isinstance(targets, torch.Tensor):
        return targets.detach().cpu().numpy().astype(np.int64)
    return np.asarray(targets, dtype=np.int64)


def _float_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def partition_filename(data_cfg: DataConfig, seed: int, partition_dir: str | Path) -> Path:
    root = Path(partition_dir)
    if data_cfg.partition == "single_label":
        return root / (
            f"{data_cfg.dataset}_seed{seed}_k{data_cfg.num_clients}_"
            f"l{data_cfg.labels_per_client}_m{_float_token(data_cfg.avg_samples_per_client)}.json"
        )
    return root / (
        f"{data_cfg.dataset}_seed{seed}_k{data_cfg.num_clients}_"
        f"sparse_dirichlet_a{_float_token(data_cfg.dirichlet_alpha)}_"
        f"m{_float_token(data_cfg.avg_samples_per_client)}_"
        f"c{data_cfg.sparse_classes_per_client}.json"
    )


def _atomic_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(tmp_name, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _build_pools(targets: np.ndarray, rng) -> tuple[Dict[int, np.ndarray], Dict[int, int]]:
    pools: Dict[int, np.ndarray] = {}
    offsets: Dict[int, int] = {}
    for label in range(10):
        indices = np.flatnonzero(targets == label).astype(np.int64)
        rng.shuffle(indices)
        pools[label] = indices
        offsets[label] = 0
    return pools, offsets


def _take_unique(
    label: int,
    count: int,
    pools: Dict[int, np.ndarray],
    offsets: Dict[int, int],
) -> np.ndarray:
    start = int(offsets[label])
    end = start + int(count)
    pool = pools[label]
    if end > len(pool):
        raise RuntimeError(
            f"Class {label} exhausted: requested through {end}, available {len(pool)}"
        )
    offsets[label] = end
    return pool[start:end]


def _prepare_single_label(
    data_cfg: DataConfig,
    seed: int,
    targets: np.ndarray,
) -> Dict[str, object]:
    """v1.6.1-style Poisson scarce partition for MNIST."""
    rng = np.random.default_rng(int(seed))
    pools, offsets = _build_pools(targets, rng)
    clients: Dict[str, Dict[str, object]] = {}

    for client_id in range(int(data_cfg.num_clients)):
        n_samples = max(
            int(data_cfg.min_samples_per_client),
            int(rng.poisson(float(data_cfg.avg_samples_per_client))),
        )
        labels = sorted(
            int(v)
            for v in rng.choice(
                10,
                size=int(data_cfg.labels_per_client),
                replace=False,
            ).tolist()
        )
        base = n_samples // len(labels)
        rem = n_samples % len(labels)
        indices: List[int] = []
        counts: Dict[str, int] = {}
        for i, label in enumerate(labels):
            count = base + (1 if i < rem else 0)
            picked = _take_unique(label, count, pools, offsets)
            indices.extend(int(v) for v in picked.tolist())
            if count:
                counts[str(label)] = int(count)
        rng.shuffle(indices)
        clients[str(client_id)] = {
            "indices": indices,
            "labels": labels,
            "class_counts": counts,
            "num_examples": len(indices),
        }

    sizes = np.asarray([int(c["num_examples"]) for c in clients.values()])
    all_indices = [int(i) for c in clients.values() for i in c["indices"]]
    return {
        "seed": int(seed),
        "dataset": data_cfg.dataset,
        "partition": "single_label",
        "num_clients": int(data_cfg.num_clients),
        "labels_per_client": int(data_cfg.labels_per_client),
        "avg_samples_per_client": float(data_cfg.avg_samples_per_client),
        "total_samples_used": int(len(all_indices)),
        "mean_size": float(sizes.mean()),
        "min_size": int(sizes.min()),
        "max_size": int(sizes.max()),
        "mean_classes_per_client": float(
            np.mean([len(c["labels"]) for c in clients.values()])
        ),
        "empty_client_backfills": 0,
        "num_empty_clients_after_backfill": 0,
        "class_draws_exhausted": 0,
        "unique_samples_used": int(len(set(all_indices))),
        "clients": clients,
    }


def _prepare_sparse_dirichlet(
    data_cfg: DataConfig,
    seed: int,
    targets: np.ndarray,
) -> Dict[str, object]:
    """Sparse-support Dirichlet partition used by the CIFAR-10 baseline.

    For the requested seed-0 configuration (K=100, mean=100, support=4):

      size_k = 1 + Poisson(99)

    gives exactly total=10046, mean=100.46, min=79, max=127.  Each client
    uniformly chooses four distinct class labels, samples a Dirichlet(alpha)
    mixture on those four labels, and receives at least one example from each
    selected class.  Therefore the realized mean number of classes/client is
    exactly 4.0 while alpha controls within-client skew.
    """
    rng = np.random.RandomState(int(seed))
    mean = float(data_cfg.avg_samples_per_client)
    minimum = int(data_cfg.min_samples_per_client)
    support_size = int(data_cfg.sparse_classes_per_client)

    if mean >= 1.0 and minimum == 1:
        sizes = 1 + rng.poisson(max(0.0, mean - 1.0), size=int(data_cfg.num_clients))
    else:
        sizes = np.maximum(
            minimum,
            rng.poisson(mean, size=int(data_cfg.num_clients)),
        )
    sizes = sizes.astype(np.int64)

    # Build pools only after the size draw so the requested seed-0 size
    # statistics are invariant to dataset index shuffling.
    pools, offsets = _build_pools(targets, rng)
    clients: Dict[str, Dict[str, object]] = {}
    class_draws_exhausted = 0

    for client_id, n_samples_raw in enumerate(sizes.tolist()):
        n_samples = max(int(n_samples_raw), support_size)
        support = np.sort(
            rng.choice(10, size=support_size, replace=False).astype(np.int64)
        )
        proportions = rng.dirichlet(
            np.full(support_size, float(data_cfg.dirichlet_alpha), dtype=np.float64)
        )
        # Guarantee all support classes are represented.
        counts_support = np.ones(support_size, dtype=np.int64)
        counts_support += rng.multinomial(n_samples - support_size, proportions)

        indices: List[int] = []
        class_counts: Dict[str, int] = {}
        for label, count in zip(support.tolist(), counts_support.tolist()):
            try:
                picked = _take_unique(int(label), int(count), pools, offsets)
            except RuntimeError:
                class_draws_exhausted += 1
                raise
            indices.extend(int(v) for v in picked.tolist())
            class_counts[str(int(label))] = int(count)
        rng.shuffle(indices)
        clients[str(client_id)] = {
            "indices": indices,
            "labels": [int(v) for v in support.tolist()],
            "class_counts": class_counts,
            "num_examples": len(indices),
        }

    all_indices = [int(i) for c in clients.values() for i in c["indices"]]
    if len(all_indices) != len(set(all_indices)):
        raise RuntimeError("Sparse Dirichlet partition contains duplicate indices")

    realised_sizes = np.asarray(
        [int(c["num_examples"]) for c in clients.values()], dtype=np.int64
    )
    active = np.asarray([len(c["labels"]) for c in clients.values()], dtype=np.int64)
    empty = int(np.count_nonzero(realised_sizes == 0))

    return {
        "seed": int(seed),
        "dataset": data_cfg.dataset,
        "partition": "sparse_dirichlet",
        "num_clients": int(data_cfg.num_clients),
        "dirichlet_alpha": float(data_cfg.dirichlet_alpha),
        "sparse_classes_per_client": support_size,
        "avg_samples_per_client": mean,
        "total_samples_used": int(realised_sizes.sum()),
        "mean_size": float(realised_sizes.mean()),
        "min_size": int(realised_sizes.min()),
        "max_size": int(realised_sizes.max()),
        "mean_classes_per_client": float(active.mean()),
        "empty_client_backfills": 0,
        "num_empty_clients_after_backfill": empty,
        "class_draws_exhausted": int(class_draws_exhausted),
        "unique_samples_used": int(len(set(all_indices))),
        "clients": clients,
    }


def prepare_partitions(
    data_cfg: DataConfig,
    seed: int,
    partition_dir: str | Path,
    force: bool = False,
) -> Path:
    root = Path(partition_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = partition_filename(data_cfg, int(seed), root)
    if path.exists() and not force:
        return path

    ensure_dataset(data_cfg)
    targets = _training_targets(data_cfg)
    if data_cfg.partition == "single_label":
        payload = _prepare_single_label(data_cfg, int(seed), targets)
    elif data_cfg.partition == "sparse_dirichlet":
        payload = _prepare_sparse_dirichlet(data_cfg, int(seed), targets)
    else:
        raise ValueError(data_cfg.partition)
    _atomic_json(path, payload)
    return path


def load_partition(partition_path: str | Path) -> Dict[str, object]:
    with Path(partition_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_partition_metadata(partition_path: str | Path, client_id: int) -> Dict[str, object]:
    payload = load_partition(partition_path)
    try:
        metadata = dict(payload["clients"][str(int(client_id))])
    except KeyError as exc:
        raise KeyError(f"Client {client_id} not found in {partition_path}") from exc
    metadata["partition_mean_size"] = float(payload["mean_size"])
    metadata["partition_total_samples"] = int(payload["total_samples_used"])
    return metadata


def load_client_loader(
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    partition_path: str | Path,
    client_id: int,
    shuffle_seed: int,
    pin_memory: bool | None = None,
) -> Tuple[DataLoader, Dict[str, object]]:
    metadata = load_partition_metadata(partition_path, int(client_id))
    ds = _dataset(data_cfg, True)
    subset = Subset(ds, [int(v) for v in metadata["indices"]])
    generator = torch.Generator()
    generator.manual_seed(int(shuffle_seed))
    use_pin = bool(data_cfg.pin_memory if pin_memory is None else pin_memory)
    loader = DataLoader(
        subset,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        generator=generator,
        num_workers=int(data_cfg.num_workers),
        pin_memory=use_pin,
        drop_last=False,
    )
    return loader, metadata


def load_test_loader(
    data_cfg: DataConfig,
    batch_size: int = 512,
    pin_memory: bool | None = None,
) -> DataLoader:
    ds = _dataset(data_cfg, False)
    use_pin = bool(data_cfg.pin_memory if pin_memory is None else pin_memory)
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(data_cfg.num_workers),
        pin_memory=use_pin,
        drop_last=False,
    )


def client_sizes(partition_path: str | Path) -> List[int]:
    payload = load_partition(partition_path)
    return [
        int(payload["clients"][str(i)]["num_examples"])
        for i in range(int(payload["num_clients"]))
    ]
