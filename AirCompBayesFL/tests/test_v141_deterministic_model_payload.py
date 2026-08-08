import numpy as np
import pytest

from aggregation import aggregate_deterministic
from config import WirelessConfig


def test_fedavg_no_wireless_is_weighted_average_of_local_models():
    current = np.array([10.0, -10.0], dtype=np.float32)
    local = [
        np.array([1.0, 3.0], dtype=np.float32),
        np.array([5.0, 7.0], dtype=np.float32),
    ]
    weights = np.array([0.25, 0.75], dtype=np.float64)
    cfg = WirelessConfig(enabled=False, num_subchannels=2)
    channels = np.ones((2, 2), dtype=np.complex128)
    result = aggregate_deterministic(
        current, local, weights, channels, cfg, np.random.default_rng(0)
    )
    expected = weights[0] * local[0] + weights[1] * local[1]
    np.testing.assert_allclose(result.parameters[0], expected, rtol=0, atol=1e-7)


def test_deterministic_payload_mode_must_be_model():
    cfg = WirelessConfig(enabled=False, deterministic_payload_mode="update")
    with pytest.raises(ValueError, match="deterministic_payload_mode=model"):
        aggregate_deterministic(
            np.zeros(2, dtype=np.float32),
            [np.ones(2, dtype=np.float32)],
            np.ones(1, dtype=np.float64),
            np.ones((1, 1024), dtype=np.complex128),
            cfg,
            np.random.default_rng(0),
        )
