import numpy as np

from bayesfl.posterior.gaussian import gaussian_product, fola_local_precision


def test_gaussian_product_one_dimensional():
    means = [np.array([0.0], dtype=np.float32), np.array([2.0], dtype=np.float32)]
    precisions = [np.array([1.0], dtype=np.float32), np.array([3.0], dtype=np.float32)]
    mean, precision = gaussian_product(means, precisions, [0.5, 0.5])
    assert np.allclose(precision, [2.0])
    assert np.allclose(mean, [1.5])


def test_fola_round_formula():
    fisher = np.array([4.0, 8.0], dtype=np.float32)
    prior = np.array([2.0, 2.0], dtype=np.float32)
    out = fola_local_precision(fisher, prior, 2)
    assert np.allclose(out, [3.0, 5.0])
