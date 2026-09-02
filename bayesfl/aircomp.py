"""Over-the-air aggregation and optimized transmit-power control.

v1.5.0 deliberately separates the *power-scale normalization* used by the
Bayesian method in the target 2025 paper from the conventional-FL benchmark
normalization borrowed from Hong, Park, and Choi (IEEE TWC 2023, reference
[13] of the target paper).

Both paths still share the same KKT/QCQP magnitude optimizer
``_optimal_magnitude``:

    v = |Delta|                                if feasible
    v = |Delta| / (1 + lambda * u)            otherwise.

What differs is the definition of ``u`` and the receiver de-scaling:

* Proposed 2025 path: target-paper Eqs. (27), (28), (31).
* Deterministic benchmark path: Hong-2023 Eqs. (8), (10), (19), (20).

This distinction is important because Hong-2023 includes ``sigma_z^2`` in the
transmit-power scale and divides it back out at the receiver.  v1.4.x used the
Proposed normalization for every method and therefore did not faithfully
implement the benchmark power-control reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from config import WirelessConfig
from wireless import db_to_linear, dbm_to_watts, sample_complex_noise


@dataclass
class AirCompStats:
    nmse: float
    distortion_nmse: float
    clipped_fraction: float
    average_symbol_power_watts: float
    maximum_symbol_power_watts: float
    noise_l2: float
    ideal_l2: float
    received_l2: float
    # Kept under the historical name ``delta_bar`` for CSV compatibility.
    # Proposed: weighted local update-power normalization (target Eq. 27).
    # Hong-2023 benchmark: rho_ref used by Eqs. (8) and (10).
    delta_bar: float
    retained_magnitude_ratio: float
    distorted_to_ideal_norm_ratio: float

    @classmethod
    def zero(cls) -> "AirCompStats":
        return cls(
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        )


def _require_paper_reference_mode(wireless_cfg: WirelessConfig) -> None:
    mode = str(wireless_cfg.power_control_mode).strip().lower()
    if mode != "paper_reference_kkt":
        raise ValueError(
            "AirComp aggregation requires "
            "wireless.power_control_mode=paper_reference_kkt"
        )


def _normalize_inputs(
    updates: Sequence[np.ndarray],
    weights: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, int, np.ndarray]:
    if not updates:
        raise ValueError("At least one client update is required")

    vectors = [np.asarray(update, dtype=np.float64).reshape(-1) for update in updates]
    dimension = vectors[0].size
    if any(vector.size != dimension for vector in vectors):
        raise ValueError("All client updates must have the same dimension")

    normalized_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if normalized_weights.size != len(vectors):
        raise ValueError("weights and updates have inconsistent lengths")
    total_weight = float(np.sum(normalized_weights))
    if total_weight <= 0.0:
        normalized_weights = np.full(
            len(vectors), 1.0 / len(vectors), dtype=np.float64
        )
    else:
        normalized_weights = normalized_weights / total_weight

    ideal = np.sum(
        np.stack([
            weight * vector
            for weight, vector in zip(normalized_weights, vectors)
        ]),
        axis=0,
    )
    return vectors, normalized_weights, dimension, ideal


def _validate_channels(
    channels: np.ndarray,
    num_clients: int,
    num_subchannels: int,
) -> np.ndarray:
    channels = np.asarray(channels, dtype=np.complex128)
    expected = (num_clients, num_subchannels)
    if channels.shape != expected:
        raise ValueError(f"channels has shape {channels.shape}, expected {expected}")
    return channels


def _optimal_magnitude(
    absolute_update: np.ndarray,
    u: np.ndarray,
    power_budget_watts: float,
    bisection_steps: int,
) -> Tuple[np.ndarray, float, bool]:
    """Solve the shared KKT magnitude problem.

    This is Hong-2023 Eq. (20) and the target 2025 paper Eq. (43).  A scalar
    bisection finds the multiplier that makes ``u^T(v o v) = P`` when the
    unconstrained update is infeasible.
    """
    absolute_update = np.asarray(absolute_update, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    unconstrained_power = float(np.sum(u * absolute_update * absolute_update))
    if unconstrained_power <= power_budget_watts:
        return absolute_update, unconstrained_power, False

    def used_power(multiplier: float) -> float:
        scaled = absolute_update / (1.0 + multiplier * u)
        return float(np.sum(u * scaled * scaled))

    low = 0.0
    high = 1.0
    while used_power(high) > power_budget_watts and high < 1.0e30:
        high *= 2.0

    for _ in range(int(bisection_steps)):
        mid = 0.5 * (low + high)
        if used_power(mid) > power_budget_watts:
            low = mid
        else:
            high = mid

    multiplier = high
    scaled = absolute_update / (1.0 + multiplier * u)
    return scaled, float(np.sum(u * scaled * scaled)), True


def _stats_from_vectors(
    *,
    ideal: np.ndarray,
    received: np.ndarray,
    distorted_without_noise: np.ndarray,
    noise_vector: np.ndarray,
    symbol_powers: list[float],
    clipped_values: int,
    total_values: int,
    reference_power: float,
    original_magnitude_sum: float,
    retained_magnitude_sum: float,
) -> AirCompStats:
    ideal_error = received - ideal
    distortion_error = distorted_without_noise - ideal
    ideal_energy = float(np.dot(ideal, ideal))
    denominator = max(ideal_energy, 1.0e-30)
    return AirCompStats(
        nmse=float(np.dot(ideal_error, ideal_error) / denominator),
        distortion_nmse=float(np.dot(distortion_error, distortion_error) / denominator),
        clipped_fraction=float(clipped_values / max(1, total_values)),
        average_symbol_power_watts=(
            float(np.mean(symbol_powers)) if symbol_powers else 0.0
        ),
        maximum_symbol_power_watts=(
            float(np.max(symbol_powers)) if symbol_powers else 0.0
        ),
        noise_l2=float(np.linalg.norm(noise_vector)),
        ideal_l2=float(np.sqrt(ideal_energy)),
        received_l2=float(np.linalg.norm(received)),
        delta_bar=float(reference_power),
        retained_magnitude_ratio=float(
            retained_magnitude_sum / max(original_magnitude_sum, 1.0e-30)
        ),
        distorted_to_ideal_norm_ratio=float(
            np.linalg.norm(distorted_without_noise)
            / max(np.linalg.norm(ideal), 1.0e-30)
        ),
    )


def aggregate_updates_proposed(
    updates: Sequence[np.ndarray],
    weights: np.ndarray,
    channels: np.ndarray,
    wireless_cfg: WirelessConfig,
    rng: np.random.Generator,
    *,
    output_dtype: np.dtype | type = np.float32,
    active_coordinate_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, AirCompStats]:
    """Aggregate Proposed Delta-rho/Delta-nu using the target-paper scaling.

    The power normalization is the weighted average local update power

        delta_bar = sum_k pi_k ||Delta_k||^2 / d,

    and

        u_k = pi_k^2 * gamma / delta_bar * |h_k|^{-2}.

    This is the behavior already validated for the Proposed method in v1.4.x;
    v1.5.0 keeps it unchanged.
    """
    _require_paper_reference_mode(wireless_cfg)
    vectors, weights, dimension, ideal = _normalize_inputs(updates, weights)
    output_dtype = np.dtype(output_dtype)
    coordinate_mask = None
    if active_coordinate_mask is not None:
        coordinate_mask = np.asarray(active_coordinate_mask, dtype=bool).reshape(-1)
        if coordinate_mask.shape != (dimension,):
            raise ValueError(
                f"active_coordinate_mask has shape {coordinate_mask.shape}; "
                f"expected {(dimension,)}"
            )
    if not wireless_cfg.enabled:
        return ideal.astype(output_dtype), AirCompStats.zero()

    num_subchannels = int(wireless_cfg.num_subchannels)
    channels = _validate_channels(channels, len(vectors), num_subchannels)

    delta_bar = float(
        sum(
            weight * float(np.dot(vector, vector)) / max(1, dimension)
            for weight, vector in zip(weights, vectors)
        )
    )
    if delta_bar <= 1.0e-30:
        return np.zeros(dimension, dtype=output_dtype), AirCompStats.zero()

    gamma = db_to_linear(wireless_cfg.gamma_db)
    power_budget = dbm_to_watts(wireless_cfg.power_dbm)
    noise_power = dbm_to_watts(wireless_cfg.noise_dbm)
    channel_power = np.maximum(
        np.abs(channels) ** 2, float(wireless_cfg.min_channel_power)
    )
    channel_inversion = 1.0 / channel_power

    received = np.zeros(dimension, dtype=np.float64)
    distorted_without_noise = np.zeros(dimension, dtype=np.float64)
    noise_vector = np.zeros(dimension, dtype=np.float64)
    clipped_values = 0
    total_values = 0
    symbol_powers: List[float] = []
    original_magnitude_sum = 0.0
    retained_magnitude_sum = 0.0

    for start in range(0, dimension, num_subchannels):
        stop = min(start + num_subchannels, dimension)
        active = stop - start
        aggregate_chunk = np.zeros(num_subchannels, dtype=np.float64)

        for client_index, (weight, vector) in enumerate(zip(weights, vectors)):
            chunk = np.zeros(num_subchannels, dtype=np.float64)
            chunk[:active] = vector[start:stop]
            absolute = np.abs(chunk)
            u = (
                (weight * weight)
                * gamma
                / delta_bar
                * channel_inversion[client_index]
            )
            magnitude, used_power, clipped = _optimal_magnitude(
                absolute,
                u,
                power_budget,
                wireless_cfg.bisection_steps,
            )
            aggregate_chunk += weight * np.sign(chunk) * magnitude
            symbol_powers.append(used_power)
            original_active = absolute[:active]
            retained_active = magnitude[:active]
            original_magnitude_sum += float(np.sum(original_active))
            retained_magnitude_sum += float(np.sum(retained_active))
            if clipped:
                materially_reduced = (
                    (original_active > 1.0e-12)
                    & (retained_active < original_active * (1.0 - 1.0e-6))
                )
                clipped_values += int(np.count_nonzero(materially_reduced))
            total_values += int(np.count_nonzero(original_active > 1.0e-12))

        # Target-paper receiver reconstruction leaves the physical channel
        # noise multiplied by sqrt(delta_bar/gamma).
        noise = sample_complex_noise(
            (num_subchannels,), noise_power, rng
        ).real * np.sqrt(delta_bar / gamma)
        if coordinate_mask is not None:
            # Missing sparse coordinates carry no posterior evidence and are
            # not allocated a sparse payload slot.  Suppress receiver noise on
            # coordinates selected by no client so the server truly leaves
            # those global posterior coordinates unchanged.
            noise[:active] *= coordinate_mask[start:stop].astype(np.float64)
        distorted_without_noise[start:stop] = aggregate_chunk[:active]
        noise_vector[start:stop] = noise[:active]
        received[start:stop] = aggregate_chunk[:active] + noise[:active]

    stats = _stats_from_vectors(
        ideal=ideal,
        received=received,
        distorted_without_noise=distorted_without_noise,
        noise_vector=noise_vector,
        symbol_powers=symbol_powers,
        clipped_values=clipped_values,
        total_values=total_values,
        reference_power=delta_bar,
        original_magnitude_sum=original_magnitude_sum,
        retained_magnitude_sum=retained_magnitude_sum,
    )
    return received.astype(output_dtype), stats


def _hong2023_reference_power(
    vectors: Sequence[np.ndarray],
    weights: np.ndarray,
    ideal: np.ndarray,
    mode: str,
) -> float:
    """Return rho_ref for the Hong-2023 benchmark adaptation.

    Hong-2023 obtains rho_ref from a BS-local update.  The target 2025 paper
    borrows the optimized power-allocation method for FedAvg/FedProx/SCAFFOLD
    but does not include a BS dataset or state how rho_ref is instantiated.

    ``coordinated_aggregate`` is the source-motivated default: Hong-2023
    Remark 6 explains that conventional AirComp coordinates power using the
    statistics of the *aggregated update*, while its BS-local rho_ref is used
    only as an effective estimate that avoids those reports.  In simulation we
    can compute that scalar exactly after the local updates are available.

    ``weighted_local`` is retained as a documented sensitivity option and
    equals the v1.4.x weighted average of local update powers.
    """
    dimension = max(1, ideal.size)
    normalized = str(mode).strip().lower()
    if normalized == "coordinated_aggregate":
        return float(np.dot(ideal, ideal) / dimension)
    if normalized == "weighted_local":
        return float(
            sum(
                weight * float(np.dot(vector, vector)) / dimension
                for weight, vector in zip(weights, vectors)
            )
        )
    raise ValueError(
        "deterministic_reference_power_mode must be "
        "coordinated_aggregate or weighted_local"
    )


def aggregate_updates_hong2023(
    updates: Sequence[np.ndarray],
    weights: np.ndarray,
    channels: np.ndarray,
    wireless_cfg: WirelessConfig,
    rng: np.random.Generator,
    *,
    output_dtype: np.dtype | type = np.float32,
) -> Tuple[np.ndarray, AirCompStats]:
    """Aggregate a conventional-FL update using Hong-2023 Eqs. (8)-(10),(20).

    The deterministic clients transmit local model *updates*, not absolute
    replacement models.  For each OFDM group the Hong-2023 power coefficient
    is

        u_k = w_k^2 * gamma * sigma_z^2 / rho_ref * |h_k|^{-2}.

    The KKT magnitude solver is exactly the same helper used by the Proposed
    path.  The BS then applies the receiver scale

        sqrt(rho_ref / (gamma * sigma_z^2)),

    which produces the desired weighted update plus de-scaled channel noise.
    """
    _require_paper_reference_mode(wireless_cfg)
    vectors, weights, dimension, ideal = _normalize_inputs(updates, weights)
    output_dtype = np.dtype(output_dtype)
    if not wireless_cfg.enabled:
        return ideal.astype(output_dtype), AirCompStats.zero()

    num_subchannels = int(wireless_cfg.num_subchannels)
    channels = _validate_channels(channels, len(vectors), num_subchannels)

    rho_ref = _hong2023_reference_power(
        vectors,
        weights,
        ideal,
        wireless_cfg.deterministic_reference_power_mode,
    )
    if rho_ref <= 1.0e-30:
        return np.zeros(dimension, dtype=output_dtype), AirCompStats.zero()

    gamma = db_to_linear(wireless_cfg.gamma_db)
    power_budget = dbm_to_watts(wireless_cfg.power_dbm)
    noise_power = dbm_to_watts(wireless_cfg.noise_dbm)
    safe_noise_power = max(noise_power, np.finfo(np.float64).tiny)
    channel_power = np.maximum(
        np.abs(channels) ** 2, float(wireless_cfg.min_channel_power)
    )
    channel_inversion = 1.0 / channel_power

    received = np.zeros(dimension, dtype=np.float64)
    distorted_without_noise = np.zeros(dimension, dtype=np.float64)
    noise_vector = np.zeros(dimension, dtype=np.float64)
    clipped_values = 0
    total_values = 0
    symbol_powers: List[float] = []
    original_magnitude_sum = 0.0
    retained_magnitude_sum = 0.0

    # Eq. (10) receiver scaling.  Keeping it outside the client loop makes the
    # simulator algebraically equivalent to constructing Eq. (9), summing in
    # Eq. (4)/(5), and finally applying Eq. (10).
    receiver_scale = np.sqrt(rho_ref / (gamma * safe_noise_power))

    for start in range(0, dimension, num_subchannels):
        stop = min(start + num_subchannels, dimension)
        active = stop - start
        aggregate_chunk = np.zeros(num_subchannels, dtype=np.float64)

        for client_index, (weight, vector) in enumerate(zip(weights, vectors)):
            chunk = np.zeros(num_subchannels, dtype=np.float64)
            chunk[:active] = vector[start:stop]
            absolute = np.abs(chunk)

            # Hong-2023 Eq. (8): p = u o (v o v).
            u = (
                (weight * weight)
                * gamma
                * noise_power
                / rho_ref
                * channel_inversion[client_index]
            )
            magnitude, used_power, clipped = _optimal_magnitude(
                absolute,
                u,
                power_budget,
                wireless_cfg.bisection_steps,
            )

            # Eq. (9) channel-phase inversion followed by Eq. (10) yields this
            # weighted signed magnitude in the recovered update domain.
            aggregate_chunk += weight * np.sign(chunk) * magnitude
            symbol_powers.append(used_power)
            original_active = absolute[:active]
            retained_active = magnitude[:active]
            original_magnitude_sum += float(np.sum(original_active))
            retained_magnitude_sum += float(np.sum(retained_active))
            if clipped:
                materially_reduced = (
                    (original_active > 1.0e-12)
                    & (retained_active < original_active * (1.0 - 1.0e-6))
                )
                clipped_values += int(np.count_nonzero(materially_reduced))
            total_values += int(np.count_nonzero(original_active > 1.0e-12))

        physical_noise = sample_complex_noise(
            (num_subchannels,), noise_power, rng
        ).real
        recovered_noise = physical_noise * receiver_scale
        distorted_without_noise[start:stop] = aggregate_chunk[:active]
        noise_vector[start:stop] = recovered_noise[:active]
        received[start:stop] = aggregate_chunk[:active] + recovered_noise[:active]

    stats = _stats_from_vectors(
        ideal=ideal,
        received=received,
        distorted_without_noise=distorted_without_noise,
        noise_vector=noise_vector,
        symbol_powers=symbol_powers,
        clipped_values=clipped_values,
        total_values=total_values,
        reference_power=rho_ref,
        original_magnitude_sum=original_magnitude_sum,
        retained_magnitude_sum=retained_magnitude_sum,
    )
    return received.astype(output_dtype), stats


# Backward-compatible API name.  Historically ``aggregate_updates`` was the
# target-Proposed normalization.  Keep that meaning so external analysis code
# does not silently change; deterministic server paths explicitly call the
# Hong-2023 function in v1.5.0.
def aggregate_updates(*args, **kwargs):
    return aggregate_updates_proposed(*args, **kwargs)


def combine_stats(*stats_items: AirCompStats) -> AirCompStats:
    items = [item for item in stats_items if item is not None]
    if not items:
        return AirCompStats.zero()
    return AirCompStats(
        nmse=float(np.mean([item.nmse for item in items])),
        distortion_nmse=float(np.mean([item.distortion_nmse for item in items])),
        clipped_fraction=float(np.mean([item.clipped_fraction for item in items])),
        average_symbol_power_watts=float(
            np.mean([item.average_symbol_power_watts for item in items])
        ),
        maximum_symbol_power_watts=float(
            np.max([item.maximum_symbol_power_watts for item in items])
        ),
        noise_l2=float(np.sqrt(sum(item.noise_l2**2 for item in items))),
        ideal_l2=float(np.sqrt(sum(item.ideal_l2**2 for item in items))),
        received_l2=float(np.sqrt(sum(item.received_l2**2 for item in items))),
        delta_bar=float(np.mean([item.delta_bar for item in items])),
        retained_magnitude_ratio=float(
            np.mean([item.retained_magnitude_ratio for item in items])
        ),
        distorted_to_ideal_norm_ratio=float(
            np.mean([item.distorted_to_ideal_norm_ratio for item in items])
        ),
    )
