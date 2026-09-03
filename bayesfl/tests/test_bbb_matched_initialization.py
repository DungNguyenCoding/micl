from copy import deepcopy
from pathlib import Path

import pytest
import torch

from bayesfl.config import load_config
from bayesfl.models.factory import initialize_model


ROOT = Path(__file__).resolve().parents[1]


def _deterministic_name(name: str) -> str:
    if name.endswith(".mu_kernel"):
        return name[:-len(".mu_kernel")] + ".weight"
    if name.endswith(".mu_weight"):
        return name[:-len(".mu_weight")] + ".weight"
    if name.endswith(".mu_bias"):
        return name[:-len(".mu_bias")] + ".bias"
    return name


def test_bbb_matched_initialization_matches_fedavg():
    pytest.importorskip("bayesian_torch")

    cfg = load_config(
        ROOT / "scripts/configs/bbb_cifar10.yaml"
    )

    cfg.runtime.seed = 123
    cfg.bbb.match_deterministic_init = True

    bbb_model = initialize_model(cfg)

    deterministic_cfg = deepcopy(cfg)
    deterministic_cfg.method = "fedavg"

    deterministic_model = initialize_model(
        deterministic_cfg
    )

    det = dict(
        deterministic_model.named_parameters()
    )

    checked = 0

    for name, param in bbb_model.named_parameters():
        if "rho_" in name:
            continue

        det_name = _deterministic_name(name)

        assert det_name in det

        torch.testing.assert_close(
            param.detach().cpu(),
            det[det_name].detach().cpu(),
            rtol=0,
            atol=0,
        )

        checked += 1

    assert checked > 0
