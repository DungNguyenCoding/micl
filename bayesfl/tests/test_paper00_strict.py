"""Tests for opt-in Paper00 strict FedAvg mode.

Run: PYTHONPATH="$PWD/src" python -m pytest -q tests/test_paper00_strict.py
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from bayesfl.config import load_config
from bayesfl.experiment_state import config_fingerprint
from bayesfl.training.deterministic import train_fedavg

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "scripts/configs/mnist_fig5_n20_large_b_sweep_seed0/"
            "run_fig5probe_mnist_oldmlp_fixed1c_s2710_n20_fedavg_b128_e5_lr0p01_r300_seed0.yaml")


def cfg_paper00():
    cfg = load_config(TEMPLATE)
    cfg.training.paper00_strict = True
    cfg.training.lr = 0.1
    cfg.training.lr_schedule = "constant"
    cfg.training.momentum = 0.0
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip_norm = None
    cfg.validate()
    return cfg


def test_legacy_resume_fingerprint_is_unchanged():
    cfg = load_config(TEMPLATE)
    assert cfg.training.paper00_strict is False
    legacy_dict = copy.deepcopy(cfg.to_dict())
    legacy_dict["training"].pop("rounds")
    legacy_dict["training"].pop("paper00_strict")
    legacy_dict["communication"].pop("max_communication_bytes")
    legacy_dict["communication"].pop("downlink_mode", None)
    old_expected = hashlib.sha256(json.dumps(legacy_dict, sort_keys=True).encode()).hexdigest()
    assert config_fingerprint(cfg) == old_expected


def test_strict_changes_fingerprint():
    a = load_config(TEMPLATE)
    b = cfg_paper00()
    # Toggle just strict mode, with the same other parameters:
    c = copy.deepcopy(b)
    c.training.paper00_strict = False
    assert config_fingerprint(b) != config_fingerprint(c)


@pytest.mark.parametrize("field,value", [
    ("momentum", 0.9),
    ("weight_decay", 1e-4),
    ("grad_clip_norm", 10.0),
    ("lr_schedule", "cosine"),
])
def test_strict_rejects_extra_optimization(field, value):
    cfg = cfg_paper00()
    setattr(cfg.training, field, value)
    with pytest.raises(ValueError):
        cfg.validate()


def test_strict_requires_fedavg():
    cfg = cfg_paper00()
    cfg.method = "fola"
    with pytest.raises(ValueError, match="only for FedAvg"):
        cfg.validate()


def test_strict_plain_sgd_matches_manual_update():
    torch.manual_seed(17)
    cfg = cfg_paper00()
    cfg.training.local_epochs = 1
    cfg.training.batch_size = 4
    cfg.training.lr = 0.12
    x = torch.randn(4, 3)
    y = torch.tensor([1, 1, 1, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    a = torch.nn.Linear(3, 2)
    b = copy.deepcopy(a)
    expected_opt = torch.optim.SGD(b.parameters(), lr=0.12)
    expected_opt.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(b(x), y).backward()
    expected_opt.step()
    result = train_fedavg(a, loader, cfg, server_round=1, device=torch.device("cpu"))
    assert result["local_steps"] == 1
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p, q, atol=1e-8, rtol=0)


def test_legacy_clipping_still_available():
    cfg = load_config(TEMPLATE)
    assert cfg.training.paper00_strict is False
    cfg.training.grad_clip_norm = 1e-5
    cfg.training.lr = 0.1
    cfg.training.local_epochs = 1
    cfg.training.batch_size = 4
    torch.manual_seed(13)
    x = torch.randn(4, 3)
    y = torch.ones(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(x,y), batch_size=4)
    a = torch.nn.Linear(3,2)
    b = copy.deepcopy(a)
    opt = torch.optim.SGD(b.parameters(), lr=0.1, momentum=cfg.training.momentum,
                          weight_decay=cfg.training.weight_decay)
    opt.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(b(x), y).backward()
    torch.nn.utils.clip_grad_norm_(b.parameters(), 1e-5)
    opt.step()
    train_fedavg(a, loader, cfg, server_round=1, device=torch.device("cpu"))
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p, q, atol=1e-8, rtol=0)
