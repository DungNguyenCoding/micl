import torch.nn as nn

from config import SimulationConfig
from models import (
    CIFAR10_RESNET56_GN_BAYESIAN_PARAMETER_COUNT,
    CIFAR10_RESNET56_GN_GROUPNORM_PARAMETER_COUNT,
    CIFAR10_RESNET56_GN_PARAMETER_COUNT,
    MNIST_PAPER_CNN_PARAMETER_COUNT,
    build_model,
    count_parameters,
)


def test_mnist_profile_matches_legacy_local_hyperparameters():
    cfg = SimulationConfig.profile("mnist")
    assert cfg.data.dataset == "mnist"
    assert cfg.model.name == "paper_cnn"
    assert cfg.data.num_clients == 40
    assert cfg.data.labels_per_client == 1
    assert cfg.data.avg_samples_per_client == 10
    assert cfg.training.learning_rate == 0.1
    assert cfg.training.batch_size == 10
    assert cfg.training.local_epochs == 3
    assert cfg.training.momentum == 0.0
    assert cfg.training.lr_scheduler == "constant"


def test_cifar_profile_matches_requested_hyperparameters():
    cfg = SimulationConfig.profile("cifar10")
    assert cfg.data.num_clients == 100
    assert cfg.federation.client_fraction == 1.0
    assert cfg.data.partition == "sparse_dirichlet"
    assert cfg.data.dirichlet_alpha == 0.1
    assert cfg.data.avg_samples_per_client == 100
    assert cfg.data.augment is False
    assert cfg.training.learning_rate == 0.05
    assert cfg.training.momentum == 0.9
    assert cfg.training.batch_size == 128
    assert cfg.training.local_epochs == 10
    assert cfg.training.num_rounds == 300
    assert cfg.training.lr_scheduler == "cosine"
    assert cfg.training.min_learning_rate == 0.0001
    assert cfg.training.lr_decay_rounds == 400
    assert cfg.variational.mc_train == 2
    assert cfg.variational.mc_eval == 5
    assert cfg.variational.kl_weight is None
    assert cfg.variational.kl_weight_schedule is False
    assert cfg.variational.variance_floor_ratio == 0.5


def test_client_fraction_count():
    cfg = SimulationConfig.profile("cifar10")
    cfg.federation.client_fraction = 0.1
    cfg.validate()
    assert cfg.participating_clients() == 10


def test_model_dimensions():
    mnist = build_model("mnist", "paper_cnn", 10)
    assert count_parameters(mnist) == MNIST_PAPER_CNN_PARAMETER_COUNT == 62_346

    cifar = build_model("cifar10", "resnet56_gn", 10)
    assert count_parameters(cifar) == CIFAR10_RESNET56_GN_PARAMETER_COUNT == 855_770
    gn = sum(
        p.numel()
        for module in cifar.modules()
        if isinstance(module, nn.GroupNorm)
        for p in module.parameters(recurse=False)
    )
    assert gn == CIFAR10_RESNET56_GN_GROUPNORM_PARAMETER_COUNT == 4_256
    assert (
        CIFAR10_RESNET56_GN_PARAMETER_COUNT - gn
        == CIFAR10_RESNET56_GN_BAYESIAN_PARAMETER_COUNT
        == 851_514
    )
