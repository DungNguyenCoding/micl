"""Optimized power allocation and superimposed OTA update-report simulation."""

from __future__ import annotations

import math

import numpy as np

from config import SimConfig


def optimized_power_allocation(
    u_arg: np.ndarray,
    delta_arg: np.ndarray,
    p_arg: float,
    tol: float = 1e-5,
    max_iters: int = 60,
) -> np.ndarray:
    """Algorithm 2 power allocation; no TCI branch is implemented."""
    u = np.asarray(u_arg, dtype=np.float64)
    delta = np.asarray(delta_arg, dtype=np.float64)
    rho = delta * delta
    unconstrained = u * rho

    if float(np.sum(unconstrained)) <= p_arg:
        return unconstrained.astype(np.float32, copy=False)

    def power_for(c_val: float) -> float:
        denom = 1.0 + c_val * u
        return float(np.sum(unconstrained / (denom * denom)))

    c_low = 0.0
    c_high = 1.0
    while power_for(c_high) > p_arg:
        c_high *= 2.0
        if c_high > 1e30:
            break

    p_out = unconstrained
    for _ in range(max_iters):
        c_mid = 0.5 * (c_low + c_high)
        denom = 1.0 + c_mid * u
        p_out = unconstrained / (denom * denom)
        diff = p_arg - float(np.sum(p_out))
        if abs(diff) <= tol:
            break
        if diff > 0.0:
            c_high = c_mid
        else:
            c_low = c_mid

    return p_out.astype(np.float32, copy=False)


def simulate_device_ota_contribution(
    delta_flat: np.ndarray,
    dataset_weight: float,
    distance_m: float,
    rho_ref: float,
    cfg: SimConfig,
    round_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one device's noiseless received signal and ideal weighted delta."""
    delta = np.asarray(delta_flat, dtype=np.float32)
    d_model = int(delta.size)
    f_sub = int(cfg.num_subchannels)
    n_symbols = int(math.ceil(d_model / f_sub))
    padded_size = n_symbols * f_sub
    pad = padded_size - d_model
    delta_padded = np.pad(delta, (0, pad), mode="constant") if pad > 0 else delta

    rng = np.random.default_rng(round_seed)
    variance = float(distance_m ** (-cfg.path_loss_exp))
    std = math.sqrt(variance / 2.0)
    h_real = rng.normal(0.0, std, size=padded_size)
    h_imag = rng.normal(0.0, std, size=padded_size)
    h_abs_sq = h_real * h_real + h_imag * h_imag + cfg.channel_eps

    rho_safe = max(float(rho_ref), cfg.rho_eps)
    coeff = (dataset_weight * dataset_weight) * (
        cfg.power_scaling_linear * cfg.noise_power_mw / rho_safe
    )

    rx = np.zeros(padded_size, dtype=np.float32)
    for start in range(0, padded_size, f_sub):
        stop = start + f_sub
        delta_chunk = delta_padded[start:stop]
        h2_chunk = h_abs_sq[start:stop]
        u_chunk = coeff / h2_chunk
        p_chunk = optimized_power_allocation(
            u_chunk, delta_chunk, cfg.max_symbol_power_mw, cfg.power_tol, cfg.power_max_iters
        )
        # Phase inversion makes the received contribution real-valued:
        # h*x = |h| * sign(delta) * sqrt(p).
        rx[start:stop] = (
            np.sqrt(h2_chunk).astype(np.float32)
            * np.sign(delta_chunk).astype(np.float32)
            * np.sqrt(np.maximum(p_chunk, 0.0)).astype(np.float32)
        )

    ideal_weighted_delta = (dataset_weight * delta).astype(np.float32, copy=False)
    return rx, ideal_weighted_delta
