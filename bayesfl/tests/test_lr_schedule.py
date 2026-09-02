from bayesfl.config import TrainingConfig, round_learning_rate


def test_cifar_cosine_schedule_matches_spec():
    cfg = TrainingConfig(lr=0.05, lr_min=0.0001, lr_schedule="cosine", lr_decay_rounds=400)
    expected = {
        1: 0.05000,
        50: 0.04817,
        100: 0.04280,
        150: 0.03471,
        200: 0.02515,
        250: 0.01557,
        300: 0.00744,
        400: 0.00010,
    }
    for rnd, target in expected.items():
        assert round_learning_rate(cfg, rnd) == pytest.approx(target, abs=5e-6)


import pytest
