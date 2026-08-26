import pandas as pd

from utils import (
    _shared_dense_keep100_reliability,
    _shared_dense_keep100_rows,
    _sparse_target_round,
)


def _sparse_metrics():
    return pd.DataFrame(
        [
            {
                "run_id": "sparse_bayesian_keep75_proposed_rep00_seed12025",
                "experiment": "sparse",
                "condition": "bayesian_keep75",
                "method": "proposed",
                "seed": 12025,
                "round": 0,
                "accuracy": 0.1,
                "ece": 0.2,
                "channel_uses_cumulative": 0,
            },
            {
                "run_id": "sparse_bayesian_keep75_proposed_rep00_seed12025",
                "experiment": "sparse",
                "condition": "bayesian_keep75",
                "method": "proposed",
                "seed": 12025,
                "round": 120,
                "accuracy": 0.8,
                "ece": 0.05,
                "channel_uses_cumulative": 11_222_280,
            },
        ]
    )


def _dense_metrics():
    return pd.DataFrame(
        [
            {
                "run_id": "fig2_default_proposed_rep00_seed12025",
                "experiment": "fig2",
                "condition": "default",
                "method": "proposed",
                "seed": 12025,
                "round": 0,
                "accuracy": 0.1,
                "ece": 0.2,
                "channel_uses_cumulative": 0,
            },
            {
                "run_id": "fig2_default_proposed_rep00_seed12025",
                "experiment": "fig2",
                "condition": "default",
                "method": "proposed",
                "seed": 12025,
                "round": 120,
                "accuracy": 0.8426,
                "ece": 0.019,
                "channel_uses_cumulative": 14_963_040,
            },
            {
                "run_id": "fig2_default_proposed_rep00_seed12025",
                "experiment": "fig2",
                "condition": "default",
                "method": "proposed",
                "seed": 12025,
                "round": 240,
                "accuracy": 0.92,
                "ece": 0.029,
                "channel_uses_cumulative": 29_926_080,
            },
        ]
    )


def test_sparse_target_round_comes_from_sparse_run():
    assert _sparse_target_round(_sparse_metrics()) == 120


def test_dense_keep100_reuses_fig2_only_through_sparse_target_round():
    rows = _shared_dense_keep100_rows(
        _sparse_metrics(), _dense_metrics(), target_round=120
    )
    assert set(rows["condition"]) == {"bayesian_keep100", "random_keep100"}
    assert set(rows["sparse_selection"]) == {"bayesian", "random"}
    assert set(rows["sparse_keep_ratio"]) == {1.0}
    assert rows["round"].max() == 120
    assert 240 not in set(rows["round"])
    final = rows[rows["round"] == 120]
    assert len(final) == 2
    assert set(final["accuracy"]) == {0.8426}
    assert set(final["channel_uses_cumulative"]) == {14_963_040}


def test_dense_keep100_reliability_reuses_exact_target_round():
    dense_rel = pd.DataFrame(
        [
            {
                "run_id": "fig2_default_proposed_rep00_seed12025",
                "experiment": "fig2",
                "condition": "default",
                "method": "proposed",
                "seed": 12025,
                "round": 120,
                "bin": 0,
                "lower": 0.0,
                "upper": 0.1,
                "count": 10,
                "confidence": 0.05,
                "accuracy": 0.1,
            },
            {
                "run_id": "fig2_default_proposed_rep00_seed12025",
                "experiment": "fig2",
                "condition": "default",
                "method": "proposed",
                "seed": 12025,
                "round": 240,
                "bin": 0,
                "lower": 0.0,
                "upper": 0.1,
                "count": 20,
                "confidence": 0.05,
                "accuracy": 0.2,
            },
        ]
    )
    rows = _shared_dense_keep100_reliability(
        _sparse_metrics(), dense_rel, target_round=120
    )
    assert len(rows) == 2
    assert set(rows["condition"]) == {"bayesian_keep100", "random_keep100"}
    assert set(rows["round"]) == {120}
    assert set(rows["count"]) == {10}
