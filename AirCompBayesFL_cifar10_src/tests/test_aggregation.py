import numpy as np

from aggregation import (
    aggregate_gaussian_natural_mean_phase,
    aggregate_gaussian_precision_phase,
)
from bayesian_protocol import initialize_local_nu
from config import ModelConfig, WirelessConfig


def test_two_phase_conflation_without_wireless_distortion():
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
    wireless = WirelessConfig(enabled=False, num_subchannels=4)

    precision_result = aggregate_gaussian_precision_phase(
        current_precision=current_precision,
        local_precisions=local_precisions,
        weights=weights,
        channels=channels,
        wireless_cfg=wireless,
        model_cfg=ModelConfig(),
        rng=np.random.default_rng(1),
    )
    next_precision = precision_result.parameters[0]
    expected_precision = (
        weights[0] * local_precisions[0] + weights[1] * local_precisions[1]
    )
    np.testing.assert_allclose(next_precision, expected_precision)

    local_nus = [
        local_precisions[index] / next_precision * local_means[index]
        for index in range(2)
    ]
    mean_result = aggregate_gaussian_natural_mean_phase(
        current_mean=current_mean,
        local_nus=local_nus,
        weights=weights,
        channels=channels,
        wireless_cfg=wireless,
        rng=np.random.default_rng(2),
    )
    expected_mean = (
        weights[0] * local_precisions[0] * local_means[0]
        + weights[1] * local_precisions[1] * local_means[1]
    ) / expected_precision
    np.testing.assert_allclose(mean_result.parameters[0], expected_mean)


def test_phase2_initial_nus_average_back_to_old_global_mean():
    mean = np.asarray([2.0, -1.0], dtype=np.float32)
    local_precisions = [
        np.asarray([2.0, 8.0], dtype=np.float32),
        np.asarray([6.0, 4.0], dtype=np.float32),
    ]
    weights = np.asarray([0.25, 0.75], dtype=np.float64)
    next_precision = sum(
        weight * precision
        for weight, precision in zip(weights, local_precisions)
    )
    initialized = [
        initialize_local_nu(mean, precision, next_precision)
        for precision in local_precisions
    ]
    weighted = sum(weight * nu for weight, nu in zip(weights, initialized))
    np.testing.assert_allclose(weighted, mean, rtol=1e-6, atol=1e-6)


def test_precision_phase_preserves_updates_below_float32_ulp():
    current = np.full(8, 400.0, dtype=np.float64)
    # 1e-6 is far below one float32 ULP at 400 (~3.05e-5).
    local = [current + 1.0e-6, current - 2.0e-6]
    weights = np.asarray([0.25, 0.75], dtype=np.float64)
    channels = np.ones((2, 8), dtype=np.complex128)
    wireless = WirelessConfig(enabled=False, num_subchannels=8)

    result = aggregate_gaussian_precision_phase(
        current_precision=current,
        local_precisions=local,
        weights=weights,
        channels=channels,
        wireless_cfg=wireless,
        model_cfg=ModelConfig(),
        rng=np.random.default_rng(9),
    )
    expected = current + weights[0] * 1.0e-6 - weights[1] * 2.0e-6
    assert result.parameters[0].dtype == np.float64
    np.testing.assert_allclose(result.parameters[0], expected, rtol=0.0, atol=1e-12)
