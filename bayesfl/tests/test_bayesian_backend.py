import pytest
import torch
import torch.nn as nn

from bayes_vi import BayesianVITrainer
from bayesian_backend import (
    create_bayesian_trainer,
    normalize_bayesian_backend,
)
from bayesian_torch_vi import (
    BayesianTorchVITrainer,
)
from config import (
    ModelConfig,
    TrainingConfig,
)
from serialization import ParameterLayout


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(
            4,
            2,
        )

    def forward(self, x):
        return self.fc(x)


def _make(
    backend,
):
    model = TinyNet()
    layout = ParameterLayout(
        model
    )

    model_cfg = ModelConfig(
        num_classes=2,
    )

    train_cfg = TrainingConfig(
        bayesian_backend=backend,
    )

    return create_bayesian_trainer(
        model=model,
        layout=layout,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        device=torch.device("cpu"),
        learning_rate=0.01,
    )


def test_default_backend_is_pyro():
    cfg = TrainingConfig()

    assert (
        cfg.bayesian_backend
        == "pyro"
    )


def test_factory_returns_pyro():
    trainer = _make(
        "pyro"
    )

    assert isinstance(
        trainer,
        BayesianVITrainer,
    )


def test_factory_returns_bayesian_torch():
    trainer = _make(
        "bayesian_torch"
    )

    assert isinstance(
        trainer,
        BayesianTorchVITrainer,
    )


def test_invalid_backend_rejected():
    with pytest.raises(
        ValueError
    ):
        normalize_bayesian_backend(
            "not-a-backend"
        )
