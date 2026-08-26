import numpy as np

from aircomp import aggregate_updates_proposed
from config import SimulationConfig, WirelessConfig
from experiments import experiment_conditions
from sparse_posterior import (
    bayesian_update_snr_score,
    kept_coordinate_count,
    select_sparse_mask,
)


def test_update_snr_matches_requested_formula():
    local_mu = np.array([2.0, 1.0, -1.0])
    global_mu = np.array([1.0, 1.5, -1.5])
    sigma = np.array([0.5, 0.25, 1.0])
    score = bayesian_update_snr_score(local_mu, global_mu, sigma, epsilon=1e-12)
    expected = np.abs(local_mu - global_mu) / (sigma + 1e-12)
    np.testing.assert_allclose(score, expected)


def test_bayesian_and_random_keep_same_number_of_coordinates():
    scores = np.linspace(0.0, 1.0, 100)
    bayes = select_sparse_mask(
        selection="bayesian",
        keep_ratio=0.25,
        min_keep=1,
        bayesian_scores=scores,
        random_seed=7,
    )
    random = select_sparse_mask(
        selection="random",
        keep_ratio=0.25,
        min_keep=1,
        bayesian_scores=scores,
        random_seed=7,
    )
    assert bayes.kept == random.kept == 25
    assert bayes.total == random.total == 100
    assert not np.array_equal(bayes.mask, random.mask)


def test_keep_counts_match_requested_ratios_for_paper_dimension():
    d = 62_346
    assert kept_coordinate_count(d, 1.0) == d
    assert kept_coordinate_count(d, 0.75) == int(np.ceil(0.75 * d))
    assert kept_coordinate_count(d, 0.02) == int(np.ceil(0.02 * d))


def test_sparse_experiment_has_12_paired_proposed_conditions_without_keep100():
    cfg = SimulationConfig()
    conditions = experiment_conditions("sparse", cfg, methods_override=["proposed"])
    assert len(conditions) == 12
    assert {c.sparse_selection for c in conditions} == {"bayesian", "random"}
    assert {c.sparse_keep_ratio for c in conditions} == {
        0.75,
        0.5,
        0.25,
        0.1,
        0.05,
        0.02,
    }
    assert all(c.sparse_keep_ratio < 1.0 for c in conditions)
    assert all(c.methods == ("proposed",) for c in conditions)


def test_sparse_active_mask_suppresses_noise_on_missing_coordinates():
    cfg = WirelessConfig(
        enabled=True,
        power_dbm=60.0,
        noise_dbm=-20.0,
        num_subchannels=3,
        gamma_db=10.0,
        bisection_steps=20,
    )
    updates = [np.array([1.0, 0.0, 0.0], dtype=np.float64)]
    weights = np.array([1.0], dtype=np.float64)
    channels = np.ones((1, 3), dtype=np.complex128)
    received, _ = aggregate_updates_proposed(
        updates,
        weights,
        channels,
        cfg,
        np.random.default_rng(123),
        output_dtype=np.float64,
        active_coordinate_mask=np.array([True, False, False]),
    )
    assert received[1] == 0.0
    assert received[2] == 0.0


def test_paper_default_stays_sparse_disabled():
    cfg = SimulationConfig()
    assert cfg.sparse.enabled is False
    fig2 = experiment_conditions("fig2", cfg, methods_override=["proposed"])
    assert len(fig2) == 1
    assert fig2[0].sparse_selection == ""
    assert fig2[0].sparse_keep_ratio == 1.0
