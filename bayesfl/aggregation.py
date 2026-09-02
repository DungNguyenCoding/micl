"""Plain sample-size-weighted federated averaging, without wireless effects."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def normalized_weights(num_examples: Sequence[int]) -> np.ndarray:
    values = np.asarray(num_examples, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("num_examples cannot be empty")
    if np.any(values < 0):
        raise ValueError("num_examples cannot be negative")
    total = float(values.sum())
    if total <= 0:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return values / total


def weighted_average(vectors: Sequence[np.ndarray], num_examples: Sequence[int]) -> np.ndarray:
    if not vectors:
        raise ValueError("vectors cannot be empty")
    weights = normalized_weights(num_examples)
    reference_shape = np.asarray(vectors[0]).shape
    result = np.zeros(np.asarray(vectors[0]).size, dtype=np.float64)
    for weight, vector in zip(weights, vectors):
        value = np.asarray(vector, dtype=np.float64).reshape(-1)
        if value.size != result.size:
            raise ValueError("All vectors must have the same dimension")
        result += float(weight) * value
    return result.reshape(reference_shape).astype(np.float32)
