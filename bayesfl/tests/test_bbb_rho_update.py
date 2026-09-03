from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from bayesfl.config import load_config
from bayesfl.models.factory import build_model, count_bayesian_random_variables
from bayesfl.posterior.scale_mixture import bbb_complexity_cost
from bayesfl.training.bbb import _make_optimizer


ROOT = Path(__file__).resolve().parents[1]


def test_bbb_rho_gets_gradient_and_updates():
    pytest.importorskip("bayesian_torch")
    torch.manual_seed(0)

    cfg = load_config(ROOT / "scripts/configs/smoke_bbb_mnist.yaml")
    model = build_model(cfg)
    model.train()

    rho_params = [(name, p) for name, p in model.named_parameters() if "rho_" in name]
    assert rho_params
    before = {name: p.detach().clone() for name, p in rho_params}

    optimizer = _make_optimizer(model, cfg, lr=1e-3)
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))

    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    task_loss = F.cross_entropy(logits, y)
    complexity = bbb_complexity_cost(
        model,
        pi=cfg.bbb.prior_pi,
        sigma1=cfg.bbb.prior_sigma1,
        sigma2=cfg.bbb.prior_sigma2,
    )
    d = count_bayesian_random_variables(model)
    loss = task_loss + cfg.resolved_kl_weight(d) * complexity
    loss.backward()

    grad_sq = 0.0
    for _, p in rho_params:
        if p.grad is not None:
            grad_sq += float(p.grad.detach().pow(2).sum())
    assert grad_sq > 0.0

    optimizer.step()
    changed = any(not torch.equal(before[name], p.detach()) for name, p in rho_params)
    assert changed
