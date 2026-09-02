"""Shared Flower strategy helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

from bayesfl.logging_utils import CsvRecorder


def normalized_example_weights(example_counts: Sequence[int]) -> list[float]:
    total = float(sum(example_counts))
    if total <= 0:
        raise ValueError("Cannot aggregate zero examples")
    return [count / total for count in example_counts]


def weighted_average_arrays(
    clients: Sequence[Sequence[np.ndarray]],
    weights: Sequence[float],
) -> list[np.ndarray]:
    if not clients:
        raise ValueError("No client arrays to aggregate")
    n_arrays = len(clients[0])
    if any(len(c) != n_arrays for c in clients):
        raise ValueError("Client parameter counts do not match")
    out: list[np.ndarray] = []
    for idx in range(n_arrays):
        acc = np.zeros_like(clients[0][idx], dtype=np.float64)
        for client, weight in zip(clients, weights):
            acc += float(weight) * np.asarray(client[idx], dtype=np.float64)
        out.append(acc.astype(clients[0][idx].dtype, copy=False))
    return out


def weighted_metrics(
    metrics_and_counts: Sequence[tuple[int, Dict[str, object]]],
) -> Dict[str, float]:
    total = float(sum(count for count, _ in metrics_and_counts))
    if total <= 0:
        return {}
    keys = set()
    for _, metrics in metrics_and_counts:
        keys.update(k for k, v in metrics.items() if isinstance(v, (int, float, np.number)))
    out: Dict[str, float] = {}
    for key in sorted(keys):
        numerator = 0.0
        denom = 0.0
        for count, metrics in metrics_and_counts:
            value = metrics.get(key)
            if isinstance(value, (int, float, np.number)):
                numerator += count * float(value)
                denom += count
        if denom:
            out[key] = numerator / denom
    return out
