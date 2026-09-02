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
    path_loss_reference_m: float = 1000.0,
) -> np.ndarray:
    """Sample block-Rayleigh channels with an explicit distance reference.

    The paper writes ``h_k ~ CN(0, r_k^{-alpha} I)``.  A power law requires a
    dimensionless distance ratio, while the dataset module stores distances in
    metres.  We therefore use

        variance_k = (distance_m / path_loss_reference_m) ** (-alpha).

    ``path_loss_reference_m=1000`` is equivalent to expressing distance in km.
    Set it to ``1`` to recover the legacy raw-metre implementation.

    The accessible paper does not state the numerical reference used in the
    authors' private simulator, so this parameter is exposed in YAML rather
    than silently hard-coded.
    """
    distances = np.maximum(np.asarray(distances_m, dtype=np.float64), 1.0e-12)
    reference = float(path_loss_reference_m)
    if reference <= 0.0:
        raise ValueError("path_loss_reference_m must be positive")
    normalized_distances = np.maximum(distances / reference, 1.0e-12)
    variances = normalized_distances ** (-float(path_loss_exponent))
    scale = np.sqrt(variances[:, None] / 2.0)
    real = rng.standard_normal((len(distances), int(num_subchannels)))
    imag = rng.standard_normal((len(distances), int(num_subchannels)))
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
