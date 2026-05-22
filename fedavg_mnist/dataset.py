"""MNIST dataset utilities for federated learning.

This version fixes the original `_balance_classes` bug and adds optional
unbalanced client partitioning for future FL experiments.

Supported modes:
    iid=True,  unbalanced=False  -> random IID clients, roughly equal sizes
    iid=False, unbalanced=False  -> sorted-label shard non-IID clients, equal sizes
    iid=True,  unbalanced=True   -> random IID clients, unequal sizes
    iid=False, unbalanced=True   -> sorted-label shard non-IID clients, unequal sizes

Paper-like FedAvg MNIST non-IID setup:
    load_datasets(num_clients=100, iid=False, balance=True, batch_size=10,
                  unbalanced=False)

Realistic harder FL setup:
    load_datasets(num_clients=30, iid=False, balance=False, batch_size=10,
                  unbalanced=True, unbalanced_alpha=0.5)
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision.datasets import MNIST


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def load_datasets(  # pylint: disable=too-many-arguments
    num_clients: int = 10,
    iid: Optional[bool] = True,
    balance: Optional[bool] = True,
    val_ratio: float = 0.1,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,
    unbalanced: Optional[bool] = False,
    unbalanced_alpha: float = 0.5,
    num_shards: Optional[int] = None,
    min_shards_per_client: int = 1,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """Create train/validation/test dataloaders for federated MNIST.

    Parameters
    ----------
    num_clients : int, optional
        Number of FL clients/devices.
    iid : bool, optional
        If True, randomly split examples across clients.
        If False, sort examples by label and assign label-skewed shards.
    balance : bool, optional
        If True, first class-balance MNIST by keeping the same number of
        examples from every digit. This fixes the original Flower baseline bug
        where only digits 0 and 1 could remain after balancing.
    val_ratio : float, optional
        Fraction of each client's local data used for validation.
    batch_size : int, optional
        Local train/validation/test batch size.
    seed : int, optional
        Random seed for reproducibility.
    unbalanced : bool, optional
        If True, create clients with different numbers of local samples.
        If False, create equal-size client partitions as much as possible.
    unbalanced_alpha : float, optional
        Dirichlet concentration used for unbalanced client sizes.
        Smaller values create stronger imbalance.
    num_shards : int, optional
        Number of label-sorted shards used in non-IID partitioning.
        If None, defaults to 2 * num_clients for balanced non-IID and
        max(2 * num_clients, num_clients) for unbalanced non-IID.
    min_shards_per_client : int, optional
        Minimum number of shards per client in unbalanced non-IID mode.

    Returns
    -------
    Tuple[List[DataLoader], List[DataLoader], DataLoader]
        Train dataloaders, validation dataloaders, and centralized test dataloader.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if batch_size is None or batch_size <= 0:
        raise ValueError("batch_size must be positive")

    datasets, testset = _partition_data(
        num_clients=num_clients,
        iid=bool(iid),
        balance=bool(balance),
        seed=seed,
        unbalanced=bool(unbalanced),
        unbalanced_alpha=unbalanced_alpha,
        num_shards=num_shards,
        min_shards_per_client=min_shards_per_client,
    )

    trainloaders: List[DataLoader] = []
    valloaders: List[DataLoader] = []
    split_generator = torch.Generator().manual_seed(seed if seed is not None else 42)

    for local_dataset in datasets:
        len_val = int(len(local_dataset) * val_ratio)
        len_train = len(local_dataset) - len_val

        if len_train <= 0:
            raise ValueError(
                "A client has no training samples after train/val split. "
                "Decrease val_ratio or reduce client imbalance."
            )

        if len_val > 0:
            ds_train, ds_val = random_split(
                local_dataset,
                [len_train, len_val],
                generator=split_generator,
            )
        else:
            ds_train = local_dataset
            ds_val = Subset(local_dataset, [])

        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size, shuffle=False))

    return trainloaders, valloaders, DataLoader(testset, batch_size=batch_size, shuffle=False)


# -----------------------------------------------------------------------------
# Dataset download
# -----------------------------------------------------------------------------
def _download_data() -> Tuple[Dataset, Dataset]:
    """Download, normalize, and return the MNIST train/test datasets."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = MNIST("./dataset", train=True, download=True, transform=transform)
    testset = MNIST("./dataset", train=False, download=True, transform=transform)
    return trainset, testset


# -----------------------------------------------------------------------------
# Partitioning
# -----------------------------------------------------------------------------
def _partition_data(  # pylint: disable=too-many-arguments
    num_clients: int = 10,
    iid: Optional[bool] = True,
    balance: Optional[bool] = True,
    seed: Optional[int] = 42,
    unbalanced: Optional[bool] = False,
    unbalanced_alpha: float = 0.5,
    num_shards: Optional[int] = None,
    min_shards_per_client: int = 1,
) -> Tuple[List[Dataset], Dataset]:
    """Split MNIST into IID/non-IID and balanced/unbalanced client datasets."""
    trainset, testset = _download_data()

    # `balance=True` fixes class imbalance before partitioning.  The original
    # Flower baseline intended this behavior, but its implementation used wrong
    # class offsets and could accidentally keep only digits 0 and 1.
    if balance:
        trainset = _balance_classes(trainset, seed)

    if iid:
        if unbalanced:
            datasets = _make_unbalanced_iid_partitions(
                trainset=trainset,
                num_clients=num_clients,
                alpha=unbalanced_alpha,
                seed=seed,
            )
        else:
            datasets = _make_balanced_iid_partitions(
                trainset=trainset,
                num_clients=num_clients,
                seed=seed,
            )
    else:
        if unbalanced:
            datasets = _make_unbalanced_noniid_partitions(
                trainset=trainset,
                num_clients=num_clients,
                alpha=unbalanced_alpha,
                seed=seed,
                num_shards=num_shards,
                min_shards_per_client=min_shards_per_client,
            )
        else:
            datasets = _make_balanced_noniid_partitions(
                trainset=trainset,
                num_clients=num_clients,
                seed=seed,
                num_shards=num_shards,
            )

    return datasets, testset


def _make_balanced_iid_partitions(
    trainset: Dataset,
    num_clients: int,
    seed: Optional[int],
) -> List[Dataset]:
    """Random IID split with approximately equal client sizes."""
    lengths = _nearly_equal_lengths(len(trainset), num_clients)
    return list(random_split(trainset, lengths, torch.Generator().manual_seed(seed or 42)))


def _make_unbalanced_iid_partitions(
    trainset: Dataset,
    num_clients: int,
    alpha: float,
    seed: Optional[int],
) -> List[Dataset]:
    """Random IID split with unequal client sizes."""
    lengths = _dirichlet_lengths(
        total_size=len(trainset),
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
        min_size=1,
    )
    generator = torch.Generator().manual_seed(seed or 42)
    return list(random_split(trainset, lengths, generator))


def _make_balanced_noniid_partitions(
    trainset: Dataset,
    num_clients: int,
    seed: Optional[int],
    num_shards: Optional[int] = None,
) -> List[Dataset]:
    """Sorted-label shard split with equal number of shards per client.

    This is the MNIST pathological non-IID split used by the FedAvg paper:
    sort by label, create 2 shards per client, and assign 2 random shards to
    each client. For num_clients=100 and an unbalanced=False paper-like run,
    this gives 200 shards and 2 shards/client.
    """
    if num_shards is None:
        num_shards = 2 * num_clients
    if num_shards % num_clients != 0:
        raise ValueError("For balanced non-IID, num_shards must be divisible by num_clients")

    targets = _get_targets(trainset)
    sorted_indices = torch.argsort(targets)
    shards = _split_indices_evenly(sorted_indices, num_shards)

    shard_order = torch.randperm(num_shards, generator=torch.Generator().manual_seed(seed or 42))
    shards_per_client = num_shards // num_clients

    datasets: List[Dataset] = []
    for cid in range(num_clients):
        chosen = shard_order[cid * shards_per_client : (cid + 1) * shards_per_client]
        client_indices = torch.cat([shards[int(sid)] for sid in chosen]).tolist()
        datasets.append(Subset(trainset, client_indices))

    return datasets


def _make_unbalanced_noniid_partitions(  # pylint: disable=too-many-arguments
    trainset: Dataset,
    num_clients: int,
    alpha: float,
    seed: Optional[int],
    num_shards: Optional[int] = None,
    min_shards_per_client: int = 1,
) -> List[Dataset]:
    """Sorted-label shard split with unequal number of shards per client.

    Lower `alpha` creates stronger imbalance. Since shards are sorted by label,
    this mode creates both label skew and sample-count skew.
    """
    if alpha <= 0:
        raise ValueError("unbalanced_alpha must be positive")
    if min_shards_per_client < 1:
        raise ValueError("min_shards_per_client must be at least 1")

    if num_shards is None:
        num_shards = 2 * num_clients
    if num_shards < num_clients * min_shards_per_client:
        raise ValueError(
            "num_shards must be >= num_clients * min_shards_per_client "
            "so every client receives data"
        )

    rng = np.random.default_rng(seed if seed is not None else 42)
    targets = _get_targets(trainset)
    sorted_indices = torch.argsort(targets)
    shards = _split_indices_evenly(sorted_indices, num_shards)

    # Assign variable number of shards per client.
    shard_counts = np.full(num_clients, min_shards_per_client, dtype=np.int64)
    remaining = num_shards - int(shard_counts.sum())
    if remaining > 0:
        probs = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        shard_counts += rng.multinomial(remaining, probs)

    shard_order = rng.permutation(num_shards)
    datasets: List[Dataset] = []
    cursor = 0
    for count in shard_counts:
        assigned = shard_order[cursor : cursor + int(count)]
        cursor += int(count)
        client_indices = torch.cat([shards[int(sid)] for sid in assigned]).tolist()
        datasets.append(Subset(trainset, client_indices))

    return datasets


# -----------------------------------------------------------------------------
# Class balancing and helpers
# -----------------------------------------------------------------------------
def _balance_classes(
    trainset: Dataset,
    seed: Optional[int] = 42,
) -> Dataset:
    """Balance MNIST by keeping the same number of examples from every digit.

    This implementation fixes the original bug.  The old function used raw class
    counts as slice offsets, which could select mostly the first few classes.
    Here, each digit is handled independently using `torch.where(targets == digit)`.
    """
    targets = _get_targets(trainset)
    class_counts = torch.bincount(targets, minlength=10)
    smallest = int(class_counts.min().item())

    generator = torch.Generator().manual_seed(seed or 42)
    selected_indices = []

    for digit in range(10):
        digit_indices = torch.where(targets == digit)[0]
        shuffled = digit_indices[torch.randperm(len(digit_indices), generator=generator)]
        selected_indices.append(shuffled[:smallest])

    idxs = torch.cat(selected_indices)
    idxs = idxs[torch.randperm(len(idxs), generator=generator)]

    balanced = Subset(trainset, idxs.tolist())
    # Several helper functions need a `targets` attribute.  Attach the balanced
    # targets directly to the Subset for compatibility with the original code.
    balanced.targets = targets[idxs].clone()  # type: ignore[attr-defined]
    return balanced


def _get_targets(dataset: Dataset) -> torch.Tensor:
    """Return labels from MNIST/Subset/ConcatDataset-like objects."""
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if isinstance(targets, torch.Tensor):
            return targets.long()
        return torch.as_tensor(targets, dtype=torch.long)

    if isinstance(dataset, Subset):
        parent_targets = _get_targets(dataset.dataset)
        return parent_targets[torch.as_tensor(dataset.indices, dtype=torch.long)].long()

    if isinstance(dataset, ConcatDataset):
        return torch.cat([_get_targets(ds) for ds in dataset.datasets]).long()

    raise AttributeError(
        "Could not find targets for dataset. Expected MNIST, Subset with targets, "
        "or ConcatDataset."
    )


def _nearly_equal_lengths(total_size: int, num_clients: int) -> List[int]:
    """Create lengths that sum to total_size and differ by at most one."""
    base = total_size // num_clients
    remainder = total_size % num_clients
    return [base + (1 if i < remainder else 0) for i in range(num_clients)]


def _dirichlet_lengths(
    total_size: int,
    num_clients: int,
    alpha: float,
    seed: Optional[int],
    min_size: int = 1,
) -> List[int]:
    """Sample unbalanced positive client lengths that sum to total_size."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if total_size < num_clients * min_size:
        raise ValueError("total_size too small for requested minimum client size")

    rng = np.random.default_rng(seed if seed is not None else 42)
    base = np.full(num_clients, min_size, dtype=np.int64)
    remaining = total_size - int(base.sum())
    probs = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
    base += rng.multinomial(remaining, probs)
    return base.astype(int).tolist()


def _split_indices_evenly(indices: Sequence[int] | torch.Tensor, num_parts: int) -> List[torch.Tensor]:
    """Split an index tensor into nearly equal consecutive parts."""
    if not isinstance(indices, torch.Tensor):
        indices = torch.as_tensor(indices, dtype=torch.long)
    lengths = _nearly_equal_lengths(len(indices), num_parts)
    return list(torch.split(indices, lengths))
