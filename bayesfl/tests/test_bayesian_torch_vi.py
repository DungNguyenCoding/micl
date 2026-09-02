import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from bayesian_torch_vi import (
    BayesianTorchVITrainer,
)
from config import (
    ModelConfig,
    TrainingConfig,
)
from serialization import ParameterLayout


class TinyBayesianTestNet(
    nn.Module
):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            3,
            4,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.gn = nn.GroupNorm(
            2,
            4,
        )

        self.fc = nn.Linear(
            4,
            2,
        )

    def forward(
        self,
        x,
    ):
        x = F.relu(
            self.gn(
                self.conv(
                    x
                )
            )
        )

        x = F.adaptive_avg_pool2d(
            x,
            1,
        )

        x = torch.flatten(
            x,
            1,
        )

        return self.fc(
            x
        )


def _loader():
    generator = torch.Generator()
    generator.manual_seed(
        123
    )

    x = torch.randn(
        6,
        3,
        8,
        8,
        generator=generator,
    )

    y = torch.tensor(
        [
            0,
            1,
            0,
            1,
            0,
            1,
        ],
        dtype=torch.long,
    )

    return DataLoader(
        TensorDataset(
            x,
            y,
        ),
        batch_size=6,
        shuffle=False,
    )


def _configs(
    learning_rate,
):
    model_cfg = ModelConfig(
        num_classes=2,
        initial_prior_std=0.1,
    )

    train_cfg = TrainingConfig(
        local_epochs=1,
        batch_size=6,
        learning_rate=learning_rate,
        optimizer="sgd",
        momentum=0.0,
        weight_decay=0.0,
        kl_weight=2.0e-5,
        mc_train_samples=2,
        gradient_clip_norm=10.0,
    )

    return (
        model_cfg,
        train_cfg,
    )


def _trainer(
    learning_rate,
):
    torch.manual_seed(
        321
    )

    model = TinyBayesianTestNet()

    layout = ParameterLayout(
        model
    )

    model_cfg, train_cfg = (
        _configs(
            learning_rate
        )
    )

    trainer = (
        BayesianTorchVITrainer(
            model=model,
            layout=layout,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            device=torch.device(
                "cpu"
            ),
            learning_rate=learning_rate,
        )
    )

    return (
        model,
        layout,
        trainer,
    )


def test_phase1_zero_lr_has_no_fake_precision_delta():
    model, layout, trainer = (
        _trainer(
            0.0
        )
    )

    mean = (
        layout
        .flatten_model(
            model
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    precision = np.full(
        layout.total_numel,
        100.0,
        dtype=np.float64,
    )

    result = (
        trainer
        .train_precision_phase(
            global_mean=mean,
            global_precision=precision,
            loader=_loader(),
            seed=101,
        )
    )

    np.testing.assert_allclose(
        result.precision,
        precision,
        rtol=0.0,
        atol=1.0e-12,
    )

    assert (
        result.precision_delta_l2
        == 0.0
    )

    assert (
        result.precision_delta_max_abs
        == 0.0
    )

    assert (
        result.precision_changed_fraction
        == 0.0
    )

    assert (
        result.local_steps
        == 1
    )


def test_phase1_outputs_valid_precision_and_gradient():
    model, layout, trainer = (
        _trainer(
            0.01
        )
    )

    mean = (
        layout
        .flatten_model(
            model
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    precision = np.full(
        layout.total_numel,
        100.0,
        dtype=np.float64,
    )

    result = (
        trainer
        .train_precision_phase(
            global_mean=mean,
            global_precision=precision,
            loader=_loader(),
            seed=202,
        )
    )

    assert (
        result.precision.shape
        == (
            layout.total_numel,
        )
    )

    assert (
        result.precision.dtype
        == np.float64
    )

    assert np.all(
        np.isfinite(
            result.precision
        )
    )

    assert np.all(
        result.precision
        > 0
    )

    assert np.isfinite(
        result.average_loss
    )

    assert (
        result.local_steps
        == 1
    )

    assert (
        result.applied_gradient_l2_mean
        >= 0.0
    )

    assert (
        result.applied_gradient_max_abs
        >= 0.0
    )


def test_phase2_zero_lr_preserves_coordinate_transform():
    model, layout, trainer = (
        _trainer(
            0.0
        )
    )

    mean = (
        layout
        .flatten_model(
            model
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    prior_precision = np.full(
        layout.total_numel,
        100.0,
        dtype=np.float64,
    )

    local_precision = np.full(
        layout.total_numel,
        120.0,
        dtype=np.float64,
    )

    next_precision = np.full(
        layout.total_numel,
        110.0,
        dtype=np.float64,
    )

    result = (
        trainer
        .train_natural_mean_phase(
            global_mean=mean,
            prior_global_precision=prior_precision,
            next_global_precision=next_precision,
            local_precision=local_precision,
            loader=_loader(),
            seed=303,
        )
    )

    np.testing.assert_allclose(
        result.implied_mean,
        mean,
        rtol=0.0,
        atol=1.0e-7,
    )

    reconstructed_mean = (
        next_precision
        / local_precision
        * result.nu.astype(
            np.float64
        )
    )

    np.testing.assert_allclose(
        reconstructed_mean,
        result.implied_mean,
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    assert (
        result.local_steps
        == 1
    )


def test_phase2_training_outputs_finite_state():
    model, layout, trainer = (
        _trainer(
            0.01
        )
    )

    mean = (
        layout
        .flatten_model(
            model
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    prior_precision = np.full(
        layout.total_numel,
        100.0,
        dtype=np.float64,
    )

    local_precision = np.full(
        layout.total_numel,
        105.0,
        dtype=np.float64,
    )

    next_precision = np.full(
        layout.total_numel,
        103.0,
        dtype=np.float64,
    )

    result = (
        trainer
        .train_natural_mean_phase(
            global_mean=mean,
            prior_global_precision=prior_precision,
            next_global_precision=next_precision,
            local_precision=local_precision,
            loader=_loader(),
            seed=404,
        )
    )

    assert (
        result.nu.shape
        == (
            layout.total_numel,
        )
    )

    assert (
        result.implied_mean.shape
        == (
            layout.total_numel,
        )
    )

    assert np.all(
        np.isfinite(
            result.nu
        )
    )

    assert np.all(
        np.isfinite(
            result.implied_mean
        )
    )

    reconstructed_mean = (
        next_precision
        / local_precision
        * result.nu.astype(
            np.float64
        )
    )

    np.testing.assert_allclose(
        reconstructed_mean,
        result.implied_mean,
        rtol=2.0e-6,
        atol=2.0e-7,
    )

    assert np.isfinite(
        result.average_loss
    )

    assert (
        result.local_steps
        == 1
    )
