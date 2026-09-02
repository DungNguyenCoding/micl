import math

import pytest

from config import TrainingConfig
from training_schedule import learning_rate_for_round


def test_constant_scheduler_preserves_legacy_learning_rate():
    cfg = TrainingConfig(learning_rate=0.1, lr_scheduler="constant")
    assert learning_rate_for_round(cfg, 1, 80) == pytest.approx(0.1)
    assert learning_rate_for_round(cfg, 40, 80) == pytest.approx(0.1)
    assert learning_rate_for_round(cfg, 80, 80) == pytest.approx(0.1)


def test_cosine_scheduler_hits_exact_endpoints():
    cfg = TrainingConfig(
        learning_rate=0.05,
        lr_scheduler="cosine",
        min_learning_rate=1.0e-4,
    )
    assert learning_rate_for_round(cfg, 1, 80) == pytest.approx(0.05)
    assert learning_rate_for_round(cfg, 80, 80) == pytest.approx(1.0e-4)


def test_cosine_scheduler_matches_formula_mid_run():
    cfg = TrainingConfig(
        learning_rate=0.05,
        lr_scheduler="cosine",
        min_learning_rate=1.0e-4,
    )
    logical_round = 40
    total = 80
    progress = (logical_round - 1) / (total - 1)
    expected = 1.0e-4 + (0.05 - 1.0e-4) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
    assert learning_rate_for_round(cfg, logical_round, total) == pytest.approx(expected)


def test_proposed_two_physical_phases_share_logical_round_lr():
    from bayesian_protocol import phase_context

    cfg = TrainingConfig(
        learning_rate=0.05,
        lr_scheduler="cosine",
        min_learning_rate=1.0e-4,
    )
    precision = phase_context("proposed", 19)
    natural_mean = phase_context("proposed", 20)
    assert precision.logical_round == natural_mean.logical_round == 10
    lr_precision = learning_rate_for_round(cfg, precision.logical_round, 80)
    lr_natural = learning_rate_for_round(cfg, natural_mean.logical_round, 80)
    assert lr_precision == pytest.approx(lr_natural)
