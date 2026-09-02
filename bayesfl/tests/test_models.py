from pathlib import Path

import torch

from bayesfl.config import load_config
from bayesfl.models.factory import build_model


ROOT = Path(__file__).resolve().parents[1]


def test_mnist_deterministic_forward():
    cfg = load_config(ROOT / "scripts/configs/fedavg_mnist.yaml")
    model = build_model(cfg)
    out = model(torch.randn(2, 1, 28, 28))
    assert out.shape == (2, 10)


def test_cifar_deterministic_forward():
    cfg = load_config(ROOT / "scripts/configs/fedavg_cifar10.yaml")
    model = build_model(cfg)
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)
