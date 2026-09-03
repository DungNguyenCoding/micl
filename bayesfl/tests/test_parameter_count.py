from pathlib import Path

import pytest
from torch import nn

from bayesfl.config import load_config
from bayesfl.models.factory import build_model, count_bayesian_random_variables


ROOT = Path(__file__).resolve().parents[1]


def test_cifar_resnet56_stochastic_parameter_dimension_without_bayesian_torch():
    """ResNet-56 stochastic dimension counts Conv/Linear weights and biases only."""
    cfg = load_config(ROOT / "scripts/configs/fedavg_cifar10.yaml")
    assert cfg.model.name == "resnet56_gn8"
    model = build_model(cfg)
    total = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total += module.weight.numel()
            if module.bias is not None:
                total += module.bias.numel()
    assert total == 851_514


def test_cifar_bayesian_dimension_exact():
    pytest.importorskip("bayesian_torch")
    cfg = load_config(ROOT / "scripts/configs/bbb_cifar10.yaml")
    assert cfg.model.name == "resnet56_gn8"
    model = build_model(cfg)
    assert count_bayesian_random_variables(model) == 851_514
