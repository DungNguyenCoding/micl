import pytest

pytest.importorskip("bayesian_torch")

import torch

from bayesian_torch_backend import BayesianTorchStateAdapter, build_initial_states
from bayesian_training import resolved_base_kl_weight, resolved_kl_weight
from config import SimulationConfig
from models import build_model
from serialization import ParameterLayout


def test_cifar_native_bayesian_scope_and_kl_resolution():
    cfg = SimulationConfig.profile("cifar10")
    model = build_model("cifar10", "resnet56_gn", 10)
    layout = ParameterLayout(model)
    adapter = BayesianTorchStateAdapter(model, layout, cfg.variational)
    assert adapter.bayesian_dimension == 851_514
    assert adapter.deterministic_dimension == 4_256
    assert adapter.state_dimension == 1_707_284
    assert resolved_base_kl_weight(cfg.variational, adapter.bayesian_dimension) == pytest.approx(
        1.0 / 851_514
    )


def test_kl_schedule_off_makes_warmup_inert_and_size_scaling_active():
    cfg = SimulationConfig.profile("cifar10")
    base, warmup, client = resolved_kl_weight(
        cfg.variational,
        851_514,
        server_round=1,
        client_size=127,
        average_client_size=100.46,
    )
    assert base == pytest.approx(1.0 / 851_514)
    assert warmup == 1.0
    assert client == pytest.approx(base * 127.0 / 100.46)


def test_variance_floor_enforces_sigma_ratio():
    cfg = SimulationConfig.profile("mnist")
    model = build_model("mnist", "paper_cnn", 10)
    layout = ParameterLayout(model)
    adapter = BayesianTorchStateAdapter(model, layout, cfg.variational)
    global_rho = adapter.rho_vector().detach().clone()
    with torch.no_grad():
        # Force local sigma far below half the global sigma.
        for binding in adapter.bindings:
            if binding.kind == "bayesian":
                module = adapter._module(binding.module_path)
                getattr(module, str(binding.rho_name)).fill_(-20.0)
    fraction = adapter.apply_variance_floor(global_rho, 0.5)
    assert fraction > 0.99
    local_sigma = adapter.sigma_vector().detach()
    global_sigma = torch.nn.functional.softplus(global_rho)
    assert torch.all(local_sigma >= 0.5 * global_sigma - 1e-7)


def test_matched_initial_mean_is_same_shape_as_fedavg_model():
    cfg = SimulationConfig.profile("cifar10")
    state, mean, bd, dd = build_initial_states(
        dataset="cifar10",
        model_cfg=cfg.model,
        variational_cfg=cfg.variational,
        seed=0,
    )
    assert mean.shape == (855_770,)
    assert state.shape == (1_707_284,)
    assert bd == 851_514
    assert dd == 4_256
