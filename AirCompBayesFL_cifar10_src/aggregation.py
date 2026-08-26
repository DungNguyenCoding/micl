"""Server-side deterministic and two-phase Bayesian aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from aircomp import (
    AirCompStats,
    aggregate_updates_hong2023,
    aggregate_updates_proposed,
    combine_stats,
)
from config import ModelConfig, WirelessConfig


@dataclass
class AggregationResult:
    parameters: list[np.ndarray]
    aircomp_stats: AirCompStats
    diagnostics: Mapping[str, float] = field(default_factory=dict)


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
    """Aggregate FedAvg/FedProx local model *updates* over AirComp.

    Each client first computes the d-dimensional local update

        Delta-w_{t,k} = w_{t,k} - w_t.

    The Hong-2023 reference-[13] amplitude-alignment scaling is applied to
    these Delta-w vectors, while sharing the same KKT magnitude optimizer used
    by the Bayesian Delta-rho/Delta-nu phases. The server then performs the
    additive update

        w_{t+1} = w_t + AirComp({Delta-w_{t,k}}).

    This follows the reference paper's local-update transmission and avoids
    the systematic repeated shrinkage that occurs when an attenuated absolute
    model is used as a replacement state every round. Communication accounting
    is unchanged: FedAvg/FedProx still transmit d real values per round.
    """
    if str(wireless_cfg.deterministic_payload_mode).strip().lower() != "update":
        raise ValueError(
            "FedAvg/FedProx require deterministic_payload_mode=update"
        )

    current = np.asarray(current_model, dtype=np.float32).reshape(-1)
    updates = [
        np.asarray(local, dtype=np.float32).reshape(-1) - current
        for local in local_models
    ]
    if not updates:
        raise ValueError("At least one local model is required")

    normalized = np.asarray(weights, dtype=np.float64).reshape(-1)
    normalized = normalized / max(float(np.sum(normalized)), 1.0e-30)
    ideal_update = np.sum(
        np.stack([
            weight * np.asarray(update, dtype=np.float64)
            for weight, update in zip(normalized, updates)
        ]),
        axis=0,
    )

    received_update, stats = aggregate_updates_hong2023(
        updates, normalized, channels, wireless_cfg, rng
    )
    received_update64 = np.asarray(received_update, dtype=np.float64)
    next_model = (
        current.astype(np.float64) + received_update64
    ).astype(np.float32)

    diagnostics = {
        "ideal_model_update_l2": float(np.linalg.vector_norm(ideal_update)),
        "received_model_update_l2": float(
            np.linalg.vector_norm(received_update64)
        ),
        "global_model_update_l2": float(
            np.linalg.vector_norm(
                next_model.astype(np.float64) - current.astype(np.float64)
            )
        ),
    }
    return AggregationResult([next_model], stats, diagnostics)


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
    model_aggregate, model_stats = aggregate_updates_hong2023(
        model_updates, weights, channels, wireless_cfg, rng
    )
    control_aggregate, control_stats = aggregate_updates_hong2023(
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
    *,
    sparse_missing_is_silent: bool = False,
) -> AggregationResult:
    """Aggregate Delta-rho and update rho_{t+1}; Eqs. (26)-(32)."""
    # Precision uses a float64 master representation.  The direct rho update in
    # Eq. (25) can be smaller than one float32 ULP when rho is large.
    current = np.asarray(current_precision, dtype=np.float64).reshape(-1)
    updates = [
        np.asarray(local, dtype=np.float64).reshape(-1) - current
        for local in local_precisions
    ]
    active_mask = None
    if sparse_missing_is_silent:
        active_mask = np.any(
            np.stack([np.asarray(update) != 0.0 for update in updates]),
            axis=0,
        )
    aggregate, stats = aggregate_updates_proposed(
        updates,
        weights,
        channels,
        wireless_cfg,
        rng,
        output_dtype=np.float64,
        active_coordinate_mask=active_mask,
    )
    next_precision = np.clip(
        current + aggregate,
        float(model_cfg.min_precision),
        float(model_cfg.max_precision),
    ).astype(np.float64)
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
    *,
    sparse_missing_is_silent: bool = False,
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
    active_mask = None
    if sparse_missing_is_silent:
        active_mask = np.any(
            np.stack([np.asarray(update) != 0.0 for update in updates]),
            axis=0,
        )
    aggregate, stats = aggregate_updates_proposed(
        updates,
        weights,
        channels,
        wireless_cfg,
        rng,
        active_coordinate_mask=active_mask,
    )
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
