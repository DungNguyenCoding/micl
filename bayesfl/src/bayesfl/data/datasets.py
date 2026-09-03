"""Torchvision dataset loading and partition preparation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from bayesfl.config import ExperimentConfig
from .partition import (
    build_mnist_dirichlet_lognormal_indices,
    build_paper_dirichlet_indices,
    build_sparse_dirichlet_indices,
    load_partition,
    save_partition,
)
from .transforms import build_transform


def _dataset_class(name: str):
    if name == "mnist":
        return datasets.MNIST
    if name == "cifar10":
        return datasets.CIFAR10
    raise ValueError(name)


def _raw_training_dataset(cfg: ExperimentConfig, download: bool):
    cls = _dataset_class(cfg.data.dataset)
    return cls(root=cfg.data.root, train=True, download=download)


def _labels_from_dataset(dataset) -> np.ndarray:
    targets = dataset.targets
    if torch.is_tensor(targets):
        return targets.cpu().numpy().astype(np.int64)
    return np.asarray(targets, dtype=np.int64)


def partition_stem(cfg: ExperimentConfig) -> str:
    p = cfg.data.partition
    if cfg.data.dataset == "cifar10":
        kind = str(p.get("type", "sparse_dirichlet")).lower()
        alpha = p.get("dirichlet_alpha", 0.1)
        if kind == "paper_dirichlet":
            return (
                f"cifar10_paper_dirichlet_a{alpha}_"
                f"n{cfg.federation.num_clients}_seed{cfg.runtime.seed}"
            )
        avg = p.get("avg_samples_per_client", 100)
        target = p.get("target_total_samples")
        target_tag = f"_t{target}" if target is not None else ""
        return (
            f"cifar10_sparse_dirichlet_a{alpha}_"
            f"c{p.get('classes_per_client', 4)}_m{avg}{target_tag}_"
            f"n{cfg.federation.num_clients}_seed{cfg.runtime.seed}"
        )
    return (
        f"mnist_dirichlet_lognormal_a{p.get('dirichlet_alpha', 0.3)}_"
        f"n{cfg.federation.num_clients}_seed{cfg.runtime.seed}"
    )


def prepare_partition(cfg: ExperimentConfig) -> tuple[Path, dict]:
    """Download training data once and persist deterministic client indices."""
    out_root = Path(cfg.output.outputs_dir).resolve() / "partitions"
    stem = partition_stem(cfg)
    npz_path = out_root / f"{stem}.npz"
    metadata_path = out_root / f"{stem}.json"
    if npz_path.exists() and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            return npz_path, json.load(handle)

    raw = _raw_training_dataset(cfg, download=True)
    labels = _labels_from_dataset(raw)
    part = cfg.data.partition
    if cfg.data.dataset == "cifar10":
        kind = str(part.get("type", "sparse_dirichlet")).lower()
        if kind == "paper_dirichlet":
            result = build_paper_dirichlet_indices(
                labels,
                num_clients=cfg.federation.num_clients,
                num_classes=cfg.data.num_classes,
                alpha=float(part.get("dirichlet_alpha", 0.01)),
                seed=cfg.runtime.seed,
            )
        elif kind == "sparse_dirichlet":
            result = build_sparse_dirichlet_indices(
                labels,
                num_clients=cfg.federation.num_clients,
                num_classes=cfg.data.num_classes,
                alpha=float(part.get("dirichlet_alpha", 0.1)),
                avg_samples_per_client=float(part.get("avg_samples_per_client", 100)),
                classes_per_client=int(part.get("classes_per_client", 4)),
                min_samples_per_client=int(part.get("min_samples_per_client", 1)),
                seed=cfg.runtime.seed,
                target_total_samples=part.get("target_total_samples"),
            )
        else:
            raise ValueError(f"Unknown CIFAR partition type: {kind}")
    else:
        result = build_mnist_dirichlet_lognormal_indices(
            labels,
            num_clients=cfg.federation.num_clients,
            num_classes=cfg.data.num_classes,
            alpha=float(part.get("dirichlet_alpha", 0.3)),
            lognormal_sigma=float(part.get("lognormal_sigma", 0.5)),
            min_samples_per_client=int(part.get("min_samples_per_client", 100)),
            seed=cfg.runtime.seed,
        )
    save_partition(result, npz_path, metadata_path)
    return npz_path, result.metadata


@lru_cache(maxsize=8)
def _cached_transformed_dataset(
    dataset_name: str,
    root: str,
    train: bool,
    augment: bool,
    crop_padding: int,
    crop_fill: int,
    random_flip: bool,
    autoaugment_policy: str,
    cutout_holes: int,
    cutout_length: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
):
    from bayesfl.config import DataConfig

    cfg = DataConfig(
        dataset=dataset_name,
        root=root,
        augment=augment,
        crop_padding=crop_padding,
        crop_fill=crop_fill,
        random_flip=random_flip,
        autoaugment_policy=autoaugment_policy,
        cutout_holes=cutout_holes,
        cutout_length=cutout_length,
        normalize_mean=list(mean),
        normalize_std=list(std),
    )
    cls = _dataset_class(dataset_name)
    return cls(root=root, train=train, download=False, transform=build_transform(cfg, train=train))


def _dataset_args(cfg: ExperimentConfig, *, train: bool):
    return (
        cfg.data.dataset,
        str(Path(cfg.data.root).resolve()),
        train,
        cfg.data.augment if train else False,
        cfg.data.crop_padding,
        cfg.data.crop_fill,
        cfg.data.random_flip if train else False,
        cfg.data.autoaugment_policy if train else "none",
        cfg.data.cutout_holes if train else 0,
        cfg.data.cutout_length,
        tuple(cfg.data.normalize_mean),
        tuple(cfg.data.normalize_std),
    )


def load_client_loader(
    cfg: ExperimentConfig,
    partition_path: str | Path,
    client_id: int,
    *,
    shuffle_seed: int | None = None,
) -> tuple[DataLoader, int]:
    parts = load_partition(partition_path)
    if client_id < 0 or client_id >= len(parts):
        raise IndexError(f"client_id {client_id} outside [0, {len(parts)})")
    dataset = _cached_transformed_dataset(*_dataset_args(cfg, train=True))
    subset = Subset(dataset, parts[client_id].tolist())
    generator = torch.Generator()
    if shuffle_seed is None:
        shuffle_seed = cfg.runtime.seed + 100_003 * (client_id + 1)
    generator.manual_seed(int(shuffle_seed))
    loader = DataLoader(
        subset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )
    return loader, len(subset)


def load_test_loader(cfg: ExperimentConfig, batch_size: int = 512) -> DataLoader:
    dataset = _cached_transformed_dataset(*_dataset_args(cfg, train=False))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
