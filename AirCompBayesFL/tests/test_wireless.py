import numpy as np

from wireless import sample_rayleigh_channels


def test_reference_distance_controls_channel_scale():
    # Reuse the same Gaussian samples; changing the reference from 1 m to
    # 1000 m with alpha=4 multiplies channel amplitudes by 1000**2.
    distances = np.asarray([100.0], dtype=np.float64)
    legacy = sample_rayleigh_channels(
        distances, 32, 4.0, np.random.default_rng(123), 1.0
    )
    normalized = sample_rayleigh_channels(
        distances, 32, 4.0, np.random.default_rng(123), 1000.0
    )
    np.testing.assert_allclose(normalized, legacy * 1.0e6, rtol=1e-12, atol=1e-12)
