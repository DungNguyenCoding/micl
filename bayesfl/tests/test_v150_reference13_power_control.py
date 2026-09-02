import numpy as np
import pytest

import aircomp
from aircomp import (
    _hong2023_reference_power,
    aggregate_updates_hong2023,
    aggregate_updates_proposed,
)
from config import SimulationConfig, WirelessConfig
from wireless import db_to_linear, dbm_to_watts


def test_reference_power_default_is_coordinated_aggregate():
    cfg = SimulationConfig()
    assert cfg.wireless.deterministic_reference_power_mode == "coordinated_aggregate"
    cfg.validate()


def test_coordinated_aggregate_reference_power_is_power_of_weighted_update():
    vectors = [
        np.asarray([2.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 2.0], dtype=np.float64),
    ]
    weights = np.asarray([0.25, 0.75], dtype=np.float64)
    ideal = weights[0] * vectors[0] + weights[1] * vectors[1]
    expected = float(np.dot(ideal, ideal) / ideal.size)
    actual = _hong2023_reference_power(
        vectors, weights, ideal, "coordinated_aggregate"
    )
    assert actual == pytest.approx(expected)


def test_weighted_local_reference_power_matches_legacy_delta_bar_definition():
    vectors = [
        np.asarray([2.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 2.0], dtype=np.float64),
    ]
    weights = np.asarray([0.25, 0.75], dtype=np.float64)
    ideal = weights[0] * vectors[0] + weights[1] * vectors[1]
    expected = sum(w * np.dot(v, v) / 2.0 for w, v in zip(weights, vectors))
    actual = _hong2023_reference_power(vectors, weights, ideal, "weighted_local")
    assert actual == pytest.approx(expected)


def test_hong2023_eq8_symbol_power_includes_noise_power(monkeypatch):
    # One client, one OFDM group.  Suppress random channel noise so the test
    # isolates Eq. (8) transmit power exactly.
    monkeypatch.setattr(
        aircomp,
        "sample_complex_noise",
        lambda shape, noise_power_watts, rng: np.zeros(shape, dtype=np.complex128),
    )
    update = np.asarray([2.0, -3.0], dtype=np.float32)
    weights = np.asarray([1.0], dtype=np.float64)
    channels = np.asarray([[2.0 + 0j, 1.0 + 0j]], dtype=np.complex128)
    cfg = WirelessConfig(
        enabled=True,
        power_dbm=30.0,  # 1 W, safely unconstrained here
        noise_dbm=0.0,   # sigma_z^2 = 1 mW
        num_subchannels=2,
        gamma_db=10.0,
        deterministic_reference_power_mode="coordinated_aggregate",
    )
    result, stats = aggregate_updates_hong2023(
        [update], weights, channels, cfg, np.random.default_rng(1)
    )

    rho_ref = float(np.dot(update, update) / update.size)
    gamma = db_to_linear(cfg.gamma_db)
    sigma2 = dbm_to_watts(cfg.noise_dbm)
    g = 1.0 / (np.abs(channels[0]) ** 2)
    u = gamma * sigma2 / rho_ref * g
    expected_power = float(np.sum(u * (np.abs(update) ** 2)))

    assert stats.delta_bar == pytest.approx(rho_ref)
    assert stats.maximum_symbol_power_watts == pytest.approx(expected_power)
    np.testing.assert_allclose(result, update, rtol=1e-6, atol=1e-6)


def test_hong2023_eq10_receiver_scale(monkeypatch):
    # Return physical noise equal to sqrt(sigma_z^2) in the real component.
    def fake_noise(shape, noise_power_watts, rng):
        del rng
        return np.full(
            shape,
            np.sqrt(noise_power_watts) + 0j,
            dtype=np.complex128,
        )

    monkeypatch.setattr(aircomp, "sample_complex_noise", fake_noise)
    update = np.asarray([1.0, 1.0], dtype=np.float32)
    cfg = WirelessConfig(
        enabled=True,
        power_dbm=100.0,
        noise_dbm=-30.0,
        num_subchannels=2,
        gamma_db=10.0,
        deterministic_reference_power_mode="coordinated_aggregate",
    )
    result, _ = aggregate_updates_hong2023(
        [update],
        np.ones(1),
        np.ones((1, 2), dtype=np.complex128),
        cfg,
        np.random.default_rng(2),
    )
    rho_ref = 1.0
    gamma = db_to_linear(cfg.gamma_db)
    # sqrt(sigma2) * sqrt(rho_ref/(gamma*sigma2)) = sqrt(rho_ref/gamma)
    expected_noise = np.sqrt(rho_ref / gamma)
    np.testing.assert_allclose(
        result,
        update + expected_noise,
        rtol=1e-6,
        atol=1e-6,
    )


def test_proposed_and_hong2023_paths_share_kkt_helper(monkeypatch):
    calls = []
    original = aircomp._optimal_magnitude

    def wrapped(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(aircomp, "_optimal_magnitude", wrapped)
    cfg = WirelessConfig(
        enabled=True,
        power_dbm=100.0,
        noise_dbm=-300.0,
        num_subchannels=2,
        gamma_db=10.0,
    )
    updates = [np.asarray([0.1, -0.2], dtype=np.float32)]
    channels = np.ones((1, 2), dtype=np.complex128)
    aggregate_updates_proposed(
        updates, np.ones(1), channels, cfg, np.random.default_rng(3)
    )
    aggregate_updates_hong2023(
        updates, np.ones(1), channels, cfg, np.random.default_rng(4)
    )
    assert len(calls) == 2
