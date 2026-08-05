"""Server-side deterministic and two-phase Bayesian aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from aircomp import AirCompStats, aggregate_updates, combine_stats
from config import ModelConfig, WirelessConfig


@dataclass
class AggregationResult:
    parameters: list[np.ndarray]
    aircomp_stats: AirCompStats


def normalized_weights(num_examples: Sequence[int]) -> np.ndarray:
    values = np.asarray(num_examples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("num_examples must be a non-empty one-dimensional sequence")
    if np.any(values < 0):
        raise ValueError("num_examples cannot contain negative values")
    total = float(values.sum())
    if total <= 0:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return values / total


def aggregate_deterministic(
    current_model: np.ndarray,
    local_models: Sequence[np.ndarray],
    weights: np.ndarray,
    channels: np.ndarray,
    wireless_cfg: WirelessConfig,
    rng: np.random.Generator,
) -> AggregationResult:
    current = np.asarray(current_model, dtype=np.float32).reshape(-1)
    updates = [
        np.asarray(local, dtype=np.float32).reshape(-1) - current
        for local in local_models
    ]
    aggregate, stats = aggregate_updates(updates, weights, channels, wireless_cfg, rng)
    return AggregationResult([current + aggregate], stats)


def aggregate_scaffold(
    current_model: np.ndarray,
    current_control: np.ndarray,
    local_models: Sequence[np.ndarray],
    local_control_deltas: Sequence[np.ndarray],
    weights: np.ndarray,
    channels: np.ndarray,
    wireless_cfg: WirelessConfig,
    rng: np.random.Generator,
) -> AggregationResult:
    current_model = np.asarray(current_model, dtype=np.float32).reshape(-1)
    current_control = np.asarray(current_control, dtype=np.float32).reshape(-1)
    model_updates = [
        np.asarray(local, dtype=np.float32).reshape(-1) - current_model
        for local in local_models
    ]
    control_updates = [
        np.asarray(delta, dtype=np.float32).reshape(-1)
        for delta in local_control_deltas
    ]
    model_aggregate, model_stats = aggregate_updates(
        model_updates, weights, channels, wireless_cfg, rng
    )
    control_aggregate, control_stats = aggregate_updates(
        control_updates, weights, channels, wireless_cfg, rng
    )
    return AggregationResult(
        [current_model + model_aggregate, current_control + control_aggregate],
        combine_stats(model_stats, control_stats),
    )


def aggregate_gaussian_precision_phase(
    current_precision: np.ndarray,
    local_precisions: Sequence[np.ndarray],
    weights: np.ndarray,
    channels: np.ndarray,
    wireless_cfg: WirelessConfig,
    model_cfg: ModelConfig,
    rng: np.random.Generator,
) -> AggregationResult:
    """Aggregate Delta-rho and update rho_{t+1}; Eqs. (26)-(32)."""
    current = np.asarray(current_precision, dtype=np.float32).reshape(-1)
    updates = [
        np.asarray(local, dtype=np.float32).reshape(-1) - current
        for local in local_precisions
    ]
    aggregate, stats = aggregate_updates(updates, weights, channels, wireless_cfg, rng)
    next_precision = np.clip(
        current + aggregate,
        float(model_cfg.min_precision),
        float(model_cfg.max_precision),
    ).astype(np.float32)
    if not np.all(np.isfinite(next_precision)):
        raise FloatingPointError("Aggregated global precision contains non-finite values")
    return AggregationResult([next_precision], stats)


def aggregate_gaussian_natural_mean_phase(
    current_mean: np.ndarray,
    local_nus: Sequence[np.ndarray],
    weights: np.ndarray,
    channels: np.ndarray,
    wireless_cfg: WirelessConfig,
    rng: np.random.Generator,
) -> AggregationResult:
    """Aggregate Delta-nu and update mu_{t+1}; Eqs. (36)-(37).

    The global phase-2 coordinate at the beginning of a logical round is
    ``nu_t = mu_t``. Therefore every client transmits ``nu_{t,k} - mu_t`` and
    the ideal update produces ``mu_{t+1} = sum_k pi_k nu_{t,k}``.
    """
    current = np.asarray(current_mean, dtype=np.float32).reshape(-1)
    updates = [
        np.asarray(local_nu, dtype=np.float32).reshape(-1) - current
        for local_nu in local_nus
    ]
    aggregate, stats = aggregate_updates(updates, weights, channels, wireless_cfg, rng)
    next_mean = (current + aggregate).astype(np.float32)
    if not np.all(np.isfinite(next_mean)):
        raise FloatingPointError("Aggregated global mean contains non-finite values")
    return AggregationResult([next_mean], stats)


def aggregate_gaussian_natural_parameters(*args, **kwargs):
    """Removed v1.2 API retained only to produce a clear migration error."""
    del args, kwargs
    raise RuntimeError(
        "aggregate_gaussian_natural_parameters was removed in v1.3.0. "
        "Use aggregate_gaussian_precision_phase, broadcast rho_{t+1}, then "
        "aggregate_gaussian_natural_mean_phase."
    )
