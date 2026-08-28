import numpy as np
import torch

from bayesian_torch_adapter import (
    BayesianTorchParameterAdapter,
)
from cifar10_support import (
    CIFAR10ResidualCNN,
    CIFAR10_PARAMETER_COUNT,
)
from serialization import ParameterLayout


def _build():
    model = CIFAR10ResidualCNN(
        num_classes=10
    )

    layout = ParameterLayout(
        model
    )

    adapter = (
        BayesianTorchParameterAdapter(
            model,
            layout,
        )
    )

    return (
        model,
        layout,
        adapter,
    )


def test_bayesian_torch_adapter_covers_cifar_layout():
    _, layout, adapter = _build()

    assert (
        layout.total_numel
        == CIFAR10_PARAMETER_COUNT
        == 78_042
    )

    assert (
        adapter.coordinate_count()
        == 78_042
    )

    # GroupNorm must not be silently left deterministic.
    assert (
        adapter.coordinate_count(
            "groupnorm"
        )
        > 0
    )


def test_bayesian_torch_posterior_round_trip():
    model, layout, adapter = _build()

    mean = (
        layout
        .flatten_model(model)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    precision = np.full(
        layout.total_numel,
        10_000.0,
        dtype=np.float64,
    )

    adapter.set_prior(
        mean,
        precision,
    )

    adapter.set_posterior_mean(
        mean
    )

    adapter.set_posterior_precision(
        precision
    )

    recovered_mean = (
        adapter
        .posterior_mean_vector()
        .detach()
        .cpu()
        .numpy()
    )

    recovered_precision = (
        adapter
        .posterior_precision_vector()
        .detach()
        .cpu()
        .numpy()
    )

    np.testing.assert_allclose(
        recovered_mean,
        mean,
        rtol=0.0,
        atol=1.0e-7,
    )

    np.testing.assert_allclose(
        recovered_precision,
        precision,
        rtol=2.0e-5,
        atol=2.0e-2,
    )


def test_bayesian_torch_forward_and_kl():
    torch.manual_seed(123)

    model, layout, adapter = _build()

    mean = (
        layout
        .flatten_model(model)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    precision = np.full(
        layout.total_numel,
        10_000.0,
        dtype=np.float64,
    )

    adapter.set_prior(
        mean,
        precision,
    )

    adapter.set_posterior_mean(
        mean
    )

    adapter.set_posterior_precision(
        precision
    )

    adapter.model.eval()

    x = torch.randn(
        2,
        3,
        32,
        32,
    )

    y = adapter.model(x)

    assert y.shape == (
        2,
        10,
    )

    kl0 = (
        adapter
        .total_kl_sum()
        .detach()
        .item()
    )

    # Identical prior/posterior should have numerically tiny KL.
    assert abs(kl0) < 0.1

    # Perturb one scale tensor and confirm that both precision and KL move.
    before = (
        adapter
        .posterior_precision_vector()
        .detach()
        .clone()
    )

    parameter = (
        adapter
        .scale_parameters()[0]
    )

    with torch.no_grad():
        parameter.add_(0.01)

    after = (
        adapter
        .posterior_precision_vector()
        .detach()
    )

    assert not torch.equal(
        before,
        after,
    )

    kl1 = (
        adapter
        .total_kl_sum()
        .detach()
        .item()
    )

    assert kl1 > kl0
