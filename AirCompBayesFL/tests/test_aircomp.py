import numpy as np

from aircomp import aggregate_updates
from config import WirelessConfig


def test_aircomp_matches_weighted_sum_when_unconstrained_and_noiseless():
    updates = [
        np.asarray([1.0, -2.0, 0.5], dtype=np.float32),
        np.asarray([-1.0, 4.0, 1.5], dtype=np.float32),
    ]
    weights = np.asarray([0.25, 0.75], dtype=np.float64)
    channels = np.ones((2, 4), dtype=np.complex128)
    config = WirelessConfig(
        enabled=True,
        power_dbm=100.0,
        noise_dbm=-300.0,
        num_subchannels=4,
        path_loss_exponent=4.0,
        gamma_db=10.0,
    )
    actual, stats = aggregate_updates(
        updates, weights, channels, config, np.random.default_rng(7)
    )
    expected = weights[0] * updates[0] + weights[1] * updates[1]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert stats.clipped_fraction == 0.0
    assert stats.retained_magnitude_ratio == 1.0
    assert abs(stats.distorted_to_ideal_norm_ratio - 1.0) < 1.0e-6
