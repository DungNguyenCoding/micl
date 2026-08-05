"""Wireless channel utilities for the AirComp simulator."""

from __future__ import annotations

import numpy as np


def dbm_to_watts(dbm: float) -> float:
    return 10.0 ** ((float(dbm) - 30.0) / 10.0)


def db_to_linear(db: float) -> float:
    return 10.0 ** (float(db) / 10.0)


def sample_rayleigh_channels(
    distances_m: np.ndarray,
    num_subchannels: int,
    path_loss_exponent: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample h_k ~ CN(0, r_k^{-alpha} I_F)."""
    distances = np.maximum(np.asarray(distances_m, dtype=np.float64), 1.0)
    variances = distances ** (-float(path_loss_exponent))
    scale = np.sqrt(variances[:, None] / 2.0)
    real = rng.standard_normal((len(distances), num_subchannels))
    imag = rng.standard_normal((len(distances), num_subchannels))
    return (real + 1j * imag) * scale


def sample_complex_noise(
    shape: tuple[int, ...],
    noise_power_watts: float,
    rng: np.random.Generator,
) -> np.ndarray:
    scale = np.sqrt(max(0.0, noise_power_watts) / 2.0)
    return scale * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )
