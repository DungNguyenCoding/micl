"""Dataset loading, client partitioning, and virtual-client grouping.

This module is intentionally independent from Flower so partitioning can be
unit-tested without starting a federated simulation.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from config import RunConfig


@dataclass
class DataBundle:
    """Container returned by :func:`load_federated_data`."""

    trainsets: List[Dataset]
    valsets: List[Dataset]
    testloader: DataLoader
    input_shape: Tuple[int, int, int]
    num_classes: int
    client_sizes: List[int]
    label_counts: np.ndarray
    device_groups: List[List[int]]
    device_positions: np.ndarray  # columns: device_id, group_id, radius_m, angle_rad, x_m, y_m


def load_federated_data(cfg: RunConfig) -> DataBundle:
    """Load MNIST/CIFAR-10 and partition it into physical-device datasets.

    Args:
        cfg: Runtime configuration.

    Returns:
        A :class:`DataBundle` containing one train subset per physical device,
        a central test loader, and metadata for logging/plotting.
    """
    trainset, testset, input_shape, num_classes = _load_base_dataset(cfg.dataset, cfg.data_dir)
    if cfg.class_balance:
        trainset = _class_balance_dataset(trainset, num_classes, cfg.seed)

    targets = _get_targets(trainset)
    indices_per_client = partition_indices(
        targets=targets,
        num_clients=cfg.num_devices,
        iid=cfg.iid,
        balanced=cfg.balanced,
        noniid_alpha=cfg.noniid_alpha,
        unbalanced_alpha=cfg.unbalanced_alpha,
        min_size=cfg.min_client_examples,
        seed=cfg.seed,
    )

    trainsets: List[Dataset] = []
    valsets: List[Dataset] = []
    split_gen = torch.Generator().manual_seed(cfg.seed + 13)
    for indices in indices_per_client:
        subset = Subset(trainset, list(map(int, indices)))
        if cfg.val_ratio > 0:
            val_len = int(len(subset) * cfg.val_ratio)
            train_len = len(subset) - val_len
            if train_len < 1:
                raise ValueError("A client has no training examples after val split")
            train_subset, val_subset = random_split(subset, [train_len, val_len], generator=split_gen)
        else:
            train_subset, val_subset = subset, Subset(trainset, [])
        trainsets.append(train_subset)
        valsets.append(val_subset)

    label_counts = client_label_counts(indices_per_client, targets, num_classes)
    client_sizes = [len(ds) for ds in trainsets]
    device_groups = make_device_groups(cfg.num_devices, cfg.num_virtual_clients)
    device_positions = sample_device_positions(
        num_devices=cfg.num_devices,
        device_groups=device_groups,
        radius_m=cfg.area_radius_m,
        seed=cfg.seed,
    )

    testloader = DataLoader(
        testset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    return DataBundle(
        trainsets=trainsets,
        valsets=valsets,
        testloader=testloader,
        input_shape=input_shape,
        num_classes=num_classes,
        client_sizes=client_sizes,
        label_counts=label_counts,
        device_groups=device_groups,
        device_positions=device_positions,
    )


def _load_base_dataset(dataset_name: str, data_dir: str) -> tuple[Dataset, Dataset, tuple[int, int, int], int]:
    """Download/load a torchvision dataset."""
    name = dataset_name.lower()
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = datasets.MNIST(str(data_path), train=True, download=True, transform=transform)
        testset = datasets.MNIST(str(data_path), train=False, download=True, transform=transform)
        return trainset, testset, (1, 28, 28), 10

    if name == "cifar10":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        trainset = datasets.CIFAR10(str(data_path), train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(str(data_path), train=False, download=True, transform=transform_test)
        return trainset, testset, (3, 32, 32), 10

    raise ValueError(f"Unsupported dataset {dataset_name!r}")


def partition_indices(
    targets: torch.Tensor,
    num_clients: int,
    iid: bool,
    balanced: bool,
    noniid_alpha: float,
    unbalanced_alpha: float,
    min_size: int,
    seed: int,
) -> List[List[int]]:
    """Create physical-client index lists.

    ``iid=True`` shuffles the full dataset before splitting. ``iid=False`` uses a
    per-client Dirichlet label distribution. ``balanced`` controls sample counts.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if min_size < 1:
        raise ValueError("min_size must be at least 1")

    total = int(targets.numel())
    if total < num_clients * min_size:
        raise ValueError("Dataset too small for requested num_clients/min_size")

    lengths = _balanced_lengths(total, num_clients) if balanced else _dirichlet_lengths(
        total, num_clients, alpha=unbalanced_alpha, min_size=min_size, seed=seed
    )

    rng = np.random.default_rng(seed)
    if iid:
        shuffled = rng.permutation(total).astype(np.int64).tolist()
        return _split_list_by_lengths(shuffled, lengths)

    return _label_skew_split_by_lengths(
        targets=targets,
        lengths=lengths,
        alpha=noniid_alpha,
        seed=seed,
    )


def _label_skew_split_by_lengths(
    targets: torch.Tensor,
    lengths: Sequence[int],
    alpha: float,
    seed: int,
) -> List[List[int]]:
    """Sample without replacement using one Dirichlet label profile per client."""
    if alpha <= 0:
        raise ValueError("noniid_alpha must be positive")
    rng = np.random.default_rng(seed)
    labels = sorted(int(x) for x in torch.unique(targets).tolist())
    num_classes = len(labels)

    available: dict[int, list[int]] = {}
    for label in labels:
        idxs = torch.where(targets == label)[0].cpu().numpy().astype(np.int64)
        rng.shuffle(idxs)
        available[label] = idxs.tolist()

    partitions: List[List[int]] = []
    for _cid, length in enumerate(lengths):
        probs = rng.dirichlet(np.full(num_classes, alpha, dtype=np.float64))
        client_indices: List[int] = []
        for _ in range(int(length)):
            remaining_labels = [label for label in labels if available[label]]
            if not remaining_labels:
                break
            remaining_pos = np.array([labels.index(label) for label in remaining_labels], dtype=np.int64)
            p = probs[remaining_pos].astype(np.float64)
            if not np.isfinite(p).all() or p.sum() <= 0:
                p = np.ones_like(p, dtype=np.float64) / len(p)
            else:
                p = p / p.sum()
            chosen_label = int(rng.choice(np.array(remaining_labels), p=p))
            client_indices.append(int(available[chosen_label].pop()))
        partitions.append(client_indices)

    # If the last few requests were blocked by class shortages, distribute any
    # remaining examples round-robin so no data are silently dropped.
    leftovers: list[int] = []
    for label in labels:
        leftovers.extend(available[label])
    rng.shuffle(leftovers)
    cursor = 0
    while cursor < len(leftovers):
        for part in partitions:
            if cursor >= len(leftovers):
                break
            part.append(int(leftovers[cursor]))
            cursor += 1

    for part in partitions:
        rng.shuffle(part)
    return partitions


def _balanced_lengths(total: int, num_clients: int) -> List[int]:
    base = total // num_clients
    rem = total % num_clients
    return [base + (1 if idx < rem else 0) for idx in range(num_clients)]


def _dirichlet_lengths(total: int, num_clients: int, alpha: float, min_size: int, seed: int) -> List[int]:
    if alpha <= 0:
        raise ValueError("unbalanced_alpha must be positive")
    base = np.full(num_clients, int(min_size), dtype=np.int64)
    remaining = total - int(base.sum())
    rng = np.random.default_rng(seed + 101)
    probs = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
    base += rng.multinomial(remaining, probs)
    return base.astype(int).tolist()


def _split_list_by_lengths(values: Sequence[int], lengths: Sequence[int]) -> List[List[int]]:
    out: List[List[int]] = []
    cursor = 0
    for length in lengths:
        out.append(list(values[cursor : cursor + int(length)]))
        cursor += int(length)
    return out


def _get_targets(dataset: Dataset) -> torch.Tensor:
    """Return labels from torchvision datasets or nested subsets."""
    if hasattr(dataset, "targets"):
        raw = getattr(dataset, "targets")
        return raw.clone().long() if isinstance(raw, torch.Tensor) else torch.as_tensor(raw, dtype=torch.long)
    if isinstance(dataset, Subset):
        parent_targets = _get_targets(dataset.dataset)
        return parent_targets[torch.as_tensor(dataset.indices, dtype=torch.long)].long()
    raise AttributeError("Dataset must expose targets, or be a Subset of one that does")


def _class_balance_dataset(dataset: Dataset, num_classes: int, seed: int) -> Dataset:
    """Keep an equal number of examples per class before client partitioning."""
    targets = _get_targets(dataset)
    rng = np.random.default_rng(seed + 17)
    class_indices = []
    min_count = min(int((targets == c).sum().item()) for c in range(num_classes))
    for c in range(num_classes):
        idxs = torch.where(targets == c)[0].cpu().numpy().astype(np.int64)
        rng.shuffle(idxs)
        class_indices.extend(idxs[:min_count].tolist())
    rng.shuffle(class_indices)
    balanced = Subset(dataset, class_indices)
    balanced.targets = targets[torch.as_tensor(class_indices, dtype=torch.long)].clone()  # type: ignore[attr-defined]
    return balanced


def client_label_counts(indices_per_client: Sequence[Sequence[int]], targets: torch.Tensor, num_classes: int) -> np.ndarray:
    """Return a matrix of shape ``num_clients x num_classes``."""
    counts = np.zeros((len(indices_per_client), num_classes), dtype=np.int64)
    targets_np = targets.cpu().numpy()
    for cid, indices in enumerate(indices_per_client):
        if len(indices) == 0:
            continue
        labels = targets_np[np.asarray(indices, dtype=np.int64)]
        counts[cid] = np.bincount(labels, minlength=num_classes)[:num_classes]
    return counts


def make_device_groups(num_devices: int, num_virtual_clients: int) -> List[List[int]]:
    """Map many physical devices to fewer Flower virtual clients.

    Consecutive chunks make it easy to inspect logs: virtual client 0 owns the
    first chunk, virtual client 1 owns the next, and so on.
    """
    if num_virtual_clients <= 0 or num_devices <= 0:
        raise ValueError("num_devices and num_virtual_clients must be positive")
    groups: List[List[int]] = []
    for gid in range(num_virtual_clients):
        start = math.floor(gid * num_devices / num_virtual_clients)
        end = math.floor((gid + 1) * num_devices / num_virtual_clients)
        groups.append(list(range(start, end)))
    return groups


def group_id_for_device(device_id: int, groups: Sequence[Sequence[int]]) -> int:
    """Return the group id for a physical device."""
    for gid, members in enumerate(groups):
        if int(device_id) in set(map(int, members)):
            return gid
    raise ValueError(f"Device {device_id} is not assigned to any group")


def sample_device_positions(
    num_devices: int,
    device_groups: Sequence[Sequence[int]],
    radius_m: float,
    seed: int,
) -> np.ndarray:
    """Create deterministic polar coordinates for later radar plotting.

    The training code does not use these synthetic positions yet. They are saved
    so a future wireless-aware selector can replace the random selector without
    changing the output schema.
    """
    rng = np.random.default_rng(seed + 2026)
    rows = []
    gid_lookup = {did: gid for gid, members in enumerate(device_groups) for did in members}
    for did in range(num_devices):
        # Uniform over disk area: radius = R * sqrt(U)
        r = float(radius_m) * float(np.sqrt(rng.random()))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        rows.append([did, gid_lookup[did], r, theta, r * np.cos(theta), r * np.sin(theta)])
    return np.asarray(rows, dtype=np.float64)


def save_device_summary(path: str | Path, bundle: DataBundle) -> None:
    """Save physical-device metadata to CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    num_classes = bundle.label_counts.shape[1]
    with output.open("w", newline="") as f:
        fieldnames = [
            "device_id",
            "virtual_client_id",
            "num_examples",
            "radius_m",
            "angle_rad",
            "x_m",
            "y_m",
        ] + [f"label_{c}" for c in range(num_classes)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in bundle.device_positions:
            did = int(row[0])
            values = {
                "device_id": did,
                "virtual_client_id": int(row[1]),
                "num_examples": int(bundle.client_sizes[did]),
                "radius_m": float(row[2]),
                "angle_rad": float(row[3]),
                "x_m": float(row[4]),
                "y_m": float(row[5]),
            }
            for c in range(num_classes):
                values[f"label_{c}"] = int(bundle.label_counts[did, c])
            writer.writerow(values)
