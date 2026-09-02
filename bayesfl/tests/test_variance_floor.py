import numpy as np

from bayesfl.posterior.gaussian import apply_precision_variance_floor


def test_precision_variance_floor():
    local = np.array([10.0, 2.0])
    global_p = np.array([1.0, 1.0])
    out, fraction = apply_precision_variance_floor(local, global_p, ratio=0.5)
    # sigma_local >= 0.5 sigma_global -> local precision <= 4*global precision.
    assert np.allclose(out, [4.0, 2.0])
    assert fraction == 0.5
