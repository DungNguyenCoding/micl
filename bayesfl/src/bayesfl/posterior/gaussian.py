"""Diagonal-Gaussian math used by BBB aggregation and FOLA."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def softplus_np(x: np.ndarray) -> np.ndarray:
    # Stable softplus: log(1 + exp(x)).
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def inverse_softplus_np(y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y = np.maximum(np.asarray(y), eps)
    # For large y, inverse-softplus is approximately y.
    return np.where(y > 20.0, y, np.log(np.expm1(y)))


def gaussian_product(
    means: Sequence[np.ndarray],
    precisions: Sequence[np.ndarray],
    weights: Sequence[float],
    *,
    precision_min: float = 1e-12,
    precision_max: float = 1e12,
) -> tuple[np.ndarray, np.ndarray]:
    """Tempered product of diagonal Gaussians.

    q_global(theta) is proportional to prod_k q_k(theta) ** weight_k.
    """
    if not means or len(means) != len(precisions) or len(means) != len(weights):
        raise ValueError("means, precisions, and weights must have the same nonzero length")
    denom = np.zeros_like(precisions[0], dtype=np.float64)
    numer = np.zeros_like(means[0], dtype=np.float64)
    for mean, precision, weight in zip(means, precisions, weights):
        p = np.clip(np.asarray(precision, dtype=np.float64), precision_min, precision_max)
        w = float(weight)
        denom += w * p
        numer += w * p * np.asarray(mean, dtype=np.float64)
    denom = np.clip(denom, precision_min, precision_max)
    mean_global = numer / denom
    return mean_global.astype(means[0].dtype, copy=False), denom.astype(precisions[0].dtype, copy=False)


def moment_match_gaussians(
    means: Sequence[np.ndarray],
    variances: Sequence[np.ndarray],
    weights: Sequence[float],
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a weighted Gaussian mixture by matching first two moments."""
    mean = np.zeros_like(means[0], dtype=np.float64)
    second = np.zeros_like(means[0], dtype=np.float64)
    for mu, var, weight in zip(means, variances, weights):
        w = float(weight)
        mu64 = np.asarray(mu, dtype=np.float64)
        mean += w * mu64
        second += w * (np.asarray(var, dtype=np.float64) + mu64 * mu64)
    var = np.maximum(second - mean * mean, eps)
    return mean.astype(means[0].dtype), var.astype(variances[0].dtype)


def fola_local_precision(
    fisher: np.ndarray,
    global_precision: np.ndarray,
    server_round: int,
    *,
    precision_min: float = 1e-8,
    precision_max: float = 1e8,
) -> np.ndarray:
    """Eq. (24) from the online-Laplace FL paper, diagonal form."""
    if server_round < 1:
        raise ValueError("server_round must be >= 1")
    r = float(server_round)
    local = (np.asarray(fisher) / r) + ((r - 1.0) / r) * np.asarray(global_precision)
    return np.clip(local, precision_min, precision_max)


def apply_precision_variance_floor(
    local_precision: np.ndarray,
    global_precision: np.ndarray,
    ratio: float,
    *,
    precision_min: float = 1e-8,
    precision_max: float = 1e8,
) -> tuple[np.ndarray, float]:
    """Enforce sigma_local >= ratio * sigma_global via a precision ceiling."""
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    local = np.clip(np.asarray(local_precision), precision_min, precision_max)
    global_p = np.clip(np.asarray(global_precision), precision_min, precision_max)
    ceiling = global_p / (ratio * ratio)
    mask = local > ceiling
    floored = np.minimum(local, ceiling)
    fraction = float(np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0
    return np.clip(floored, precision_min, precision_max), fraction
