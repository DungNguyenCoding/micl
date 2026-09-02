import pytest

from config import SimulationConfig
from training_schedule import learning_rate_for_round


def test_cifar_fixed_horizon_cosine_values():
    cfg = SimulationConfig.profile("cifar10")
    expected = {
        1: 0.05000,
        50: 0.04816602697700931,
        100: 0.0427961882392544,
        150: 0.034711269477100314,
        200: 0.02514822372711288,
        250: 0.015570150227176525,
        300: 0.007442447387434369,
        400: 0.0001,
    }
    for round_number, value in expected.items():
        assert learning_rate_for_round(cfg.training, round_number) == pytest.approx(value)
