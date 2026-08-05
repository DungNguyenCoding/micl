import numpy as np

from aggregation import aggregate_gaussian_natural_parameters
from config import ModelConfig, WirelessConfig


def test_gaussian_conflation_without_wireless_distortion():
    current_mean = np.asarray([0.0, 0.0], dtype=np.float32)
    current_precision = np.asarray([1.0, 1.0], dtype=np.float32)
    local_means = [
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([3.0, 4.0], dtype=np.float32),
    ]
    local_precisions = [
        np.asarray([2.0, 4.0], dtype=np.float32),
        np.asarray([6.0, 8.0], dtype=np.float32),
    ]
    weights = np.asarray([0.25, 0.75], dtype=np.float64)
    channels = np.ones((2, 4), dtype=np.complex128)
    result = aggregate_gaussian_natural_parameters(
        current_mean,
        current_precision,
        local_means,
        local_precisions,
        weights,
        channels,
        WirelessConfig(enabled=False, num_subchannels=4),
        ModelConfig(),
        np.random.default_rng(1),
    )
    expected_precision = weights[0] * local_precisions[0] + weights[1] * local_precisions[1]
    expected_mean = (
        weights[0] * local_precisions[0] * local_means[0]
        + weights[1] * local_precisions[1] * local_means[1]
    ) / expected_precision
    np.testing.assert_allclose(result.parameters[1], expected_precision)
    np.testing.assert_allclose(result.parameters[0], expected_mean)
