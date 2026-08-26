import numpy as np
import pytest

from aggregation import aggregate_deterministic
from config import WirelessConfig


def test_fedavg_no_wireless_update_path_equals_weighted_model_average():
    current = np.array([10.0, -10.0], dtype=np.float32)
    local = [
        np.array([1.0, 3.0], dtype=np.float32),
        np.array([5.0, 7.0], dtype=np.float32),
    ]
    weights = np.array([0.25, 0.75], dtype=np.float64)
    cfg = WirelessConfig(
        enabled=False, num_subchannels=2, deterministic_payload_mode="update"
    )
    channels = np.ones((2, 2), dtype=np.complex128)
    result = aggregate_deterministic(
        current, local, weights, channels, cfg, np.random.default_rng(0)
    )
    expected = weights[0] * local[0] + weights[1] * local[1]
    np.testing.assert_allclose(result.parameters[0], expected, rtol=0, atol=1e-6)


def test_deterministic_payload_mode_must_be_update():
    cfg = WirelessConfig(enabled=False, deterministic_payload_mode="model")
    with pytest.raises(ValueError, match="deterministic_payload_mode=update"):
        aggregate_deterministic(
            np.zeros(2, dtype=np.float32),
            [np.ones(2, dtype=np.float32)],
            np.ones(1, dtype=np.float64),
            np.ones((1, 1024), dtype=np.complex128),
            cfg,
            np.random.default_rng(0),
        )


def test_deterministic_update_diagnostics_are_exact_without_wireless():
    current = np.array([1.0, 2.0], dtype=np.float32)
    local = [
        np.array([2.0, 4.0], dtype=np.float32),
        np.array([0.0, 3.0], dtype=np.float32),
    ]
    weights = np.array([0.5, 0.5], dtype=np.float64)
    cfg = WirelessConfig(
        enabled=False, num_subchannels=2, deterministic_payload_mode="update"
    )
    result = aggregate_deterministic(
        current, local, weights, np.ones((2, 2), dtype=np.complex128), cfg,
        np.random.default_rng(0),
    )
    ideal_update = 0.5 * (local[0] - current) + 0.5 * (local[1] - current)
    expected_norm = float(np.linalg.vector_norm(ideal_update.astype(np.float64)))
    assert result.diagnostics["ideal_model_update_l2"] == pytest.approx(expected_norm)
    assert result.diagnostics["received_model_update_l2"] == pytest.approx(expected_norm)
    assert result.diagnostics["global_model_update_l2"] == pytest.approx(expected_norm)


def test_deterministic_wireless_result_is_applied_additively(monkeypatch):
    import aggregation as aggregation_module
    from aircomp import AirCompStats

    current = np.array([10.0, -10.0], dtype=np.float32)
    local = [np.array([11.0, -8.0], dtype=np.float32)]
    cfg = WirelessConfig(
        enabled=True, num_subchannels=2, deterministic_payload_mode="update"
    )

    captured = {}

    def fake_aggregate_updates_hong2023(updates, weights, channels, wireless_cfg, rng):
        del weights, channels, wireless_cfg, rng
        captured["payload"] = np.asarray(updates[0]).copy()
        return np.array([0.25, -0.5], dtype=np.float32), AirCompStats.zero()

    monkeypatch.setattr(
        aggregation_module,
        "aggregate_updates_hong2023",
        fake_aggregate_updates_hong2023,
    )
    result = aggregate_deterministic(
        current, local, np.ones(1), np.ones((1, 2), dtype=np.complex128), cfg,
        np.random.default_rng(0),
    )

    np.testing.assert_allclose(captured["payload"], local[0] - current)
    np.testing.assert_allclose(
        result.parameters[0], current + np.array([0.25, -0.5], dtype=np.float32)
    )
