import numpy as np
import torch
from torch import nn

from bayesfl.posterior.packing import ParameterLayout, model_to_ndarrays, ndarrays_to_model, pack_fola, unpack_fola


def test_model_pack_roundtrip():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    arrays = model_to_ndarrays(model)
    changed = [a + 1.0 for a in arrays]
    ndarrays_to_model(model, changed)
    restored = model_to_ndarrays(model)
    for a, b in zip(changed, restored):
        assert np.allclose(a, b)


def test_fola_pack_roundtrip():
    model = nn.Linear(3, 2)
    layout = ParameterLayout.from_model(model)
    means = model_to_ndarrays(model)
    precisions = [np.ones_like(a) for a in means]
    packed = pack_fola(means, precisions)
    m2, p2 = unpack_fola(packed, layout)
    assert len(m2) == len(means)
    assert all(np.allclose(a, b) for a, b in zip(precisions, p2))
