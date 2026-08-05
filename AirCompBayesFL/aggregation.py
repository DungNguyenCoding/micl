"""Server-side deterministic and Gaussian posterior aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

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
    updates = [np.asarray(local, dtype=np.float32).reshape(-1) - current for local in local_models]
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


def aggregate_gaussian_natural_parameters(
    current_mean: np.ndarray,
    current_precision: np.ndarray,
    local_means: Sequence[np.ndarray],
    local_precisions: Sequence[np.ndarray],
    weights: np.ndarray,
    channels: np.ndarray,
    wireless_cfg: WirelessConfig,
    model_cfg: ModelConfig,
    rng: np.random.Generator,
) -> AggregationResult:
    """AirComp aggregation of Gaussian precision and precision-weighted mean.

    This implements the two sufficient statistics in Eqs. (18)-(19):
    precision and precision*mean. Updates relative to the previous global
    natural parameters are transmitted to match the iterative FL formulation.
    """
    current_mean = np.asarray(current_mean, dtype=np.float32).reshape(-1)
    current_precision = np.asarray(current_precision, dtype=np.float32).reshape(-1)

    precision_updates = [
        np.asarray(local_precision, dtype=np.float32).reshape(-1) - current_precision
        for local_precision in local_precisions
    ]
    aggregate_precision_update, precision_stats = aggregate_updates(
        precision_updates, weights, channels, wireless_cfg, rng
    )
    next_precision = np.clip(
        current_precision + aggregate_precision_update,
        model_cfg.min_precision,
        model_cfg.max_precision,
    ).astype(np.float32)

    current_natural_mean = current_precision * current_mean
    natural_mean_updates = [
        np.asarray(local_precision, dtype=np.float32).reshape(-1)
        * np.asarray(local_mean, dtype=np.float32).reshape(-1)
        - current_natural_mean
        for local_mean, local_precision in zip(local_means, local_precisions)
    ]
    aggregate_natural_mean_update, mean_stats = aggregate_updates(
        natural_mean_updates, weights, channels, wireless_cfg, rng
    )
    next_natural_mean = current_natural_mean + aggregate_natural_mean_update
    next_mean = (next_natural_mean / next_precision).astype(np.float32)
    return AggregationResult(
        [next_mean, next_precision],
        combine_stats(precision_stats, mean_stats),
    )
