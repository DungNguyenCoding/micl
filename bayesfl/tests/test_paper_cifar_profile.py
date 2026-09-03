from pathlib import Path

import numpy as np

from bayesfl.config import load_config, round_learning_rate
from bayesfl.data.partition import build_paper_dirichlet_indices
from bayesfl.models.factory import build_model


ROOT = Path(__file__).resolve().parents[1]


def test_main_cifar_configs_use_paper_environment_resnet56_and_kept_optimizer():
    for method in ("fedavg", "fola", "bbb"):
        cfg = load_config(ROOT / "scripts" / "configs" / f"{method}_cifar10.yaml")
        assert cfg.data.dataset == "cifar10"
        assert cfg.data.partition["type"] == "paper_dirichlet"
        assert cfg.data.partition["dirichlet_alpha"] == 0.01
        assert cfg.federation.num_clients == 20
        assert cfg.federation.clients_per_round == 20

        # Explicit user override: retain the project's ResNet-56 + GN8.
        assert cfg.model.name == "resnet56_gn8"
        assert cfg.model.group_norm_groups == 8

        # Paper-style CIFAR augmentation/environment.
        assert cfg.data.augment is True
        assert cfg.data.crop_padding == 4
        assert cfg.data.crop_fill == 128
        assert cfg.data.random_flip is True
        assert cfg.data.autoaugment_policy == "cifar10"
        assert cfg.data.cutout_holes == 1
        assert cfg.data.cutout_length == 16
        assert cfg.data.normalize_mean == [0.5, 0.5, 0.5]
        assert cfg.data.normalize_std == [0.5, 0.5, 0.5]

        # Explicit user-preserved local training/schedule.
        assert cfg.training.optimizer == "sgd"
        assert cfg.training.lr == 0.05
        assert cfg.training.momentum == 0.9
        assert cfg.training.weight_decay == 0.0
        assert cfg.training.batch_size == 128
        assert cfg.training.local_epochs == 10
        assert cfg.training.rounds == 300
        assert cfg.training.lr_schedule == "cosine"
        assert cfg.training.lr_min == 0.0001
        assert cfg.training.lr_decay_rounds == 400

        # Explicit user-preserved BBB/variational settings are carried in all
        # CIFAR configs so method switches remain reproducible.
        assert cfg.bbb.prior_type == "standard_normal"
        assert cfg.bbb.prior_mean == 0.0
        assert cfg.bbb.prior_sigma == 1.0
        assert cfg.bbb.posterior_mu_init == 0.0
        assert cfg.bbb.posterior_rho_init == -3.0
        assert cfg.bbb.kl_weight is None
        # With ResNet-56, null KL now truly resolves as 1/actual_d.
        assert cfg.bbb.kl_reference_dimension is None
        assert cfg.bbb.kl_weight_schedule is False
        assert cfg.bbb.kl_warmup_rounds == 20
        assert cfg.bbb.lambda_scale_by_size is True
        assert cfg.bbb.mc_train == 2
        assert cfg.bbb.mc_eval == 5
        assert cfg.bbb.variance_floor_ratio == 0.5

    fola = load_config(ROOT / "scripts" / "configs" / "fola_cifar10.yaml")
    assert fola.fola.mode == "paper_reference"
    assert fola.fola.initial_precision == 0.0
    assert fola.fola.aggregation_epsilon == 1e-5
    assert fola.fola.paper_mean_only_eval is True


def test_requested_lr_schedule_values():
    cfg = load_config(ROOT / "scripts/configs/fedavg_cifar10.yaml")
    expected = {
        1: 0.05000,
        50: 0.04816602697700931,
        100: 0.0427961882392544,
        150: 0.034711269477100314,
        200: 0.02514822372711288,
        250: 0.015570150227176525,
        300: 0.007442447387434369,
        400: 0.0001,
    }
    for rnd, value in expected.items():
        assert abs(round_learning_rate(cfg.training, rnd) - value) < 1e-12


def test_paper_dirichlet_partition_uses_full_cifar_scale_without_class_cap():
    labels = np.repeat(np.arange(10), 5000)
    result = build_paper_dirichlet_indices(
        labels,
        num_clients=20,
        num_classes=10,
        alpha=0.01,
        seed=0,
    )
    md = result.metadata
    assert len(result.indices) == 20
    assert md["type"] == "paper_dirichlet"
    assert md["total_unique_samples_used"] > 49_800
    assert md["total_unassigned_samples"] < 200
    assert md["mean_size"] > 2_400


def test_resnet56_dimension_and_null_kl_uses_actual_dimension():
    cfg = load_config(ROOT / "scripts/configs/fedavg_cifar10.yaml")
    model = build_model(cfg)
    stochastic_d = 0
    from torch import nn
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            stochastic_d += module.weight.numel()
            if module.bias is not None:
                stochastic_d += module.bias.numel()
    assert stochastic_d == 851_514

    bbb_cfg = load_config(ROOT / "scripts/configs/bbb_cifar10.yaml")
    assert bbb_cfg.bbb.kl_reference_dimension is None
    assert abs(bbb_cfg.resolved_kl_weight(stochastic_d) - (1.0 / 851_514.0)) < 1e-18
