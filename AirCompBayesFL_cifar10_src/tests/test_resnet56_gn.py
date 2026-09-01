import pytest
import torch
from torch import nn

from resnet56_gn import (
    CIFAR10ResNet56GN,
    CIFARResNetBasicBlock,
    CIFAR10_RESNET56_GN_PARAMETER_COUNT,
)
from serialization import ParameterLayout


def test_resnet56_gn_structure_and_size():
    model = CIFAR10ResNet56GN()
    assert sum(p.numel() for p in model.parameters()) == 855_770
    assert CIFAR10_RESNET56_GN_PARAMETER_COUNT == 855_770
    assert sum(isinstance(m, CIFARResNetBasicBlock) for m in model.modules()) == 27
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert sum(isinstance(m, nn.GroupNorm) for m in model.modules()) > 0
    y = model(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_resnet56_gn_bayesian_torch_adapter_dimension():
    adapter_module = pytest.importorskip("bayesian_torch_adapter")
    model = CIFAR10ResNet56GN()
    layout = ParameterLayout(model)
    adapter = adapter_module.BayesianTorchParameterAdapter(model, layout)
    assert layout.total_numel == 855_770
    assert adapter.coordinate_count() == 855_770
    assert adapter.coordinate_count("groupnorm") > 0
