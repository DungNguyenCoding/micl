"""Distribution/statistic-level over-the-air aggregation and power control."""

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
    delta_bar: float

    @classmethod
    def zero(cls) -> "AirCompStats":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _optimal_magnitude(
    absolute_update: np.ndarray,
    u: np.ndarray,
    power_budget_watts: float,
    bisection_steps: int,
) -> Tuple[np.ndarray, float, bool]:
    """Solve Eq. (42)-(43) by scalar bisection on the KKT multiplier."""
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


def aggregate_updates(
    updates: Sequence[np.ndarray],
    weights: np.ndarray,
    channels: np.ndarray,
    wireless_cfg: WirelessConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, AirCompStats]:
    """Aggregate real update vectors using the paper's optimized AirComp rule.

    Parameters
    ----------
    updates:
        One real vector per participating client.
    weights:
        pi_k values; normalized internally.
    channels:
        Complex channel matrix with shape [K, F]. The same block-fading
        realization is reused for every OFDM vector in this aggregation phase.
    """
    if not updates:
        raise ValueError("At least one client update is required")

    vectors = [np.asarray(update, dtype=np.float64).reshape(-1) for update in updates]
    dimension = vectors[0].size
    if any(vector.size != dimension for vector in vectors):
        raise ValueError("All client updates must have the same dimension")

    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size != len(vectors):
        raise ValueError("weights and updates have inconsistent lengths")
    weights = weights / max(float(np.sum(weights)), 1.0e-30)

    ideal = np.sum(
        np.stack([weight * vector for weight, vector in zip(weights, vectors)]),
        axis=0,
    )
    if not wireless_cfg.enabled:
        return ideal.astype(np.float32), AirCompStats.zero()

    num_subchannels = int(wireless_cfg.num_subchannels)
    if channels.shape != (len(vectors), num_subchannels):
        raise ValueError(
            f"channels has shape {channels.shape}, expected "
            f"({len(vectors)}, {num_subchannels})"
        )

    delta_bar = float(
        sum(
            weight * float(np.dot(vector, vector)) / max(1, dimension)
            for weight, vector in zip(weights, vectors)
        )
    )
    if delta_bar <= 1.0e-30:
        return np.zeros(dimension, dtype=np.float32), AirCompStats.zero()

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
            if clipped:
                clipped_values += int(np.count_nonzero(magnitude[:active] < absolute[:active]))
            total_values += active

        noise = sample_complex_noise(
            (num_subchannels,), noise_power, rng
        ).real * np.sqrt(delta_bar / gamma)
        distorted_without_noise[start:stop] = aggregate_chunk[:active]
        noise_vector[start:stop] = noise[:active]
        received[start:stop] = aggregate_chunk[:active] + noise[:active]

    ideal_error = received - ideal
    distortion_error = distorted_without_noise - ideal
    ideal_energy = float(np.dot(ideal, ideal))
    denominator = max(ideal_energy, 1.0e-30)
    stats = AirCompStats(
        nmse=float(np.dot(ideal_error, ideal_error) / denominator),
        distortion_nmse=float(np.dot(distortion_error, distortion_error) / denominator),
        clipped_fraction=float(clipped_values / max(1, total_values)),
        average_symbol_power_watts=float(np.mean(symbol_powers)) if symbol_powers else 0.0,
        maximum_symbol_power_watts=float(np.max(symbol_powers)) if symbol_powers else 0.0,
        noise_l2=float(np.linalg.norm(noise_vector)),
        ideal_l2=float(np.sqrt(ideal_energy)),
        received_l2=float(np.linalg.norm(received)),
        delta_bar=delta_bar,
    )
    return received.astype(np.float32), stats


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
    )
