import numpy as np

from bayesian_protocol import (
    NATURAL_MEAN_PHASE,
    PRECISION_PHASE,
    implied_local_mean,
    initialize_local_nu,
    phase_context,
    physical_round_count,
)


def test_proposed_uses_two_physical_rounds_per_logical_round():
    assert physical_round_count("proposed", 5) == 10
    assert phase_context("proposed", 1).logical_round == 1
    assert phase_context("proposed", 1).phase == PRECISION_PHASE
    assert phase_context("proposed", 2).logical_round == 1
    assert phase_context("proposed", 2).phase == NATURAL_MEAN_PHASE
    assert phase_context("proposed", 9).logical_round == 5
    assert phase_context("proposed", 10).logical_round == 5


def test_nu_coordinate_round_trip_at_initialization():
    mean = np.asarray([1.5, -2.0, 0.25], dtype=np.float32)
    local_precision = np.asarray([4.0, 2.0, 8.0], dtype=np.float32)
    next_global_precision = np.asarray([3.0, 5.0, 6.0], dtype=np.float32)
    nu = initialize_local_nu(mean, local_precision, next_global_precision)
    recovered = implied_local_mean(nu, local_precision, next_global_precision)
    np.testing.assert_allclose(recovered, mean, rtol=1e-6, atol=1e-6)
