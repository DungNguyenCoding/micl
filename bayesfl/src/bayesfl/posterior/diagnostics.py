"""Compact posterior and model-update diagnostics for client metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from torch import nn

from .gaussian import softplus_np


def update_norms(
    updated: Sequence[np.ndarray],
    reference: Sequence[np.ndarray],
) -> dict[str, float]:
    """Return L2/RMS size of an update without materializing one giant vector."""
    if len(updated) != len(reference):
        raise ValueError("updated/reference parameter counts differ")
    sq_sum = 0.0
    max_abs = 0.0
    numel = 0
    for new, old in zip(updated, reference):
        delta = np.asarray(new, dtype=np.float64) - np.asarray(old, dtype=np.float64)
        sq_sum += float(np.sum(delta * delta))
        if delta.size:
            max_abs = max(max_abs, float(np.max(np.abs(delta))))
        numel += int(delta.size)
    l2 = float(np.sqrt(sq_sum))
    return {
        "update_l2": l2,
        "update_rms": float(np.sqrt(sq_sum / max(1, numel))),
        "update_max_abs": max_abs,
    }


def bbb_posterior_summary(model: nn.Module) -> dict[str, float]:
    """Element-weighted global summary of a client's variational posterior."""
    mu_abs_sum = 0.0
    sigma_sum = 0.0
    snr_sum = 0.0
    sigma_min = float("inf")
    sigma_max = 0.0
    numel = 0
    params = dict(model.named_parameters())
    for name, param in params.items():
        if "mu_" not in name:
            continue
        rho_name = name.replace("mu_", "rho_", 1)
        rho = params.get(rho_name)
        if rho is None:
            continue
        mu = param.detach().cpu().numpy().astype(np.float64, copy=False)
        sigma = softplus_np(rho.detach().cpu().numpy().astype(np.float64, copy=False))
        count = int(mu.size)
        if not count:
            continue
        mu_abs_sum += float(np.sum(np.abs(mu)))
        sigma_sum += float(np.sum(sigma))
        snr_sum += float(np.sum(np.abs(mu) / np.maximum(sigma, 1e-12)))
        sigma_min = min(sigma_min, float(np.min(sigma)))
        sigma_max = max(sigma_max, float(np.max(sigma)))
        numel += count
    if numel == 0:
        return {}
    return {
        "posterior_mu_abs_mean": mu_abs_sum / numel,
        "posterior_sigma_mean": sigma_sum / numel,
        "posterior_sigma_min": sigma_min,
        "posterior_sigma_max": sigma_max,
        "posterior_snr_mean": snr_sum / numel,
    }


def fola_posterior_summary(
    means: Sequence[np.ndarray],
    precisions: Sequence[np.ndarray],
    *,
    precision_min: float,
) -> dict[str, float]:
    """Element-weighted summary of a diagonal Laplace posterior."""
    if len(means) != len(precisions):
        raise ValueError("mean/precision parameter counts differ")
    mean_abs_sum = 0.0
    precision_sum = 0.0
    sigma_sum = 0.0
    precision_min_seen = float("inf")
    precision_max_seen = 0.0
    numel = 0
    for mean, precision in zip(means, precisions):
        mean64 = np.asarray(mean, dtype=np.float64)
        p = np.maximum(np.asarray(precision, dtype=np.float64), precision_min)
        count = int(mean64.size)
        if not count:
            continue
        mean_abs_sum += float(np.sum(np.abs(mean64)))
        precision_sum += float(np.sum(p))
        sigma_sum += float(np.sum(1.0 / np.sqrt(p)))
        precision_min_seen = min(precision_min_seen, float(np.min(p)))
        precision_max_seen = max(precision_max_seen, float(np.max(p)))
        numel += count
    if numel == 0:
        return {}
    return {
        "posterior_mean_abs_mean": mean_abs_sum / numel,
        "posterior_precision_mean": precision_sum / numel,
        "posterior_precision_min": precision_min_seen,
        "posterior_precision_max": precision_max_seen,
        "posterior_sigma_mean": sigma_sum / numel,
    }
