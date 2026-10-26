"""Model-registry and local-training regression tests for the three MNIST candidates."""

from pathlib import Path
import copy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from bayesfl.config import load_config
from bayesfl.models.factory import build_model, initialize_model
from bayesfl.training.deterministic import train_fedavg
from bayesfl.training.fola import train_fola


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "mlp_784_200_200_10": 199_210,
    "mnist_cnn_16_32_128": 80_202,
    "mlp_784_256x3_10": 335_114,
}


def cfg_for(name, method="fedavg"):
    cfg = copy.deepcopy(load_config(ROOT / "scripts/configs/fedavg_mnist.yaml"))
    cfg.model.name = name
    cfg.method = method
    cfg.runtime.seed = 42
    cfg.training.local_epochs = 1
    cfg.training.batch_size = 2
    cfg.training.lr_schedule = "constant"
    cfg.training.momentum = 0.0
    cfg.fola.mode = "paper_reference"
    cfg.fola.prior_lambda = 0.01
    cfg.validate()
    return cfg


@pytest.mark.parametrize("name,n_params", EXPECTED.items())
def test_new_mnist_model_dimensions_parameters_and_gradients(name, n_params):
    cfg = cfg_for(name)
    model = build_model(cfg)
    parameters = list(model.named_parameters())
    assert sum(p.numel() for _, p in parameters) == n_params
    assert set(n for n, _ in parameters) == set(n for n, _ in model.state_dict().items())
    assert not list(model.buffers()), "No non-aggregated buffers are expected"
    x = torch.randn(2, 1, 28, 28)
    y = torch.tensor([1, 7])
    logits = model(x)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for _, p in parameters)


@pytest.mark.parametrize("name", EXPECTED)
def test_new_mnist_models_identical_fola_fedavg_initialization(name):
    avg = initialize_model(cfg_for(name, "fedavg"))
    fola = initialize_model(cfg_for(name, "fola"))
    pa, pf = dict(avg.named_parameters()), dict(fola.named_parameters())
    assert pa.keys() == pf.keys()
    assert all(torch.equal(pa[k], pf[k]) for k in pa)


@pytest.mark.parametrize("name", EXPECTED)
def test_new_mnist_models_one_local_step_each_method(name):
    # One-class/client synthetic data, exactly one minibatch per epoch.
    rng = torch.Generator().manual_seed(2)
    x = torch.randn(2, 1, 28, 28, generator=rng)
    y = torch.tensor([3, 3])
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

    avgcfg = cfg_for(name, "fedavg")
    avg = initialize_model(avgcfg)
    out_avg = train_fedavg(avg, loader, avgcfg, server_round=1, device=torch.device("cpu"))
    assert np.isfinite(out_avg["train_loss"])
    assert out_avg["local_steps"] == 1

    fcfg = cfg_for(name, "fola")
    fola = initialize_model(fcfg)
    params = list(fola.parameters())
    means = [p.detach().numpy().astype(np.float32, copy=True) for p in params]
    precisions = [np.ones_like(a, dtype=np.float32) for a in means]
    updated, out_fola = train_fola(
        fola, loader, fcfg, server_round=1, device=torch.device("cpu"),
        global_mean_arrays=means, global_precision_arrays=precisions,
        client_size=2, average_client_size=2.0,
    )
    assert out_fola["local_steps"] == 1
    assert np.isfinite(out_fola["train_loss"])
    assert len(updated) == len(precisions)
    assert all(u.shape == old.shape and np.isfinite(u).all() and (u >= 1.0).all()
               for u, old in zip(updated, precisions))
