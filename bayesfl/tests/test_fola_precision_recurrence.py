import numpy as np

from bayesfl.posterior.gaussian import fola_local_precision


def test_fola_round_one_keeps_unit_prior_precision():
    fisher = np.array([0.0, 2.0], dtype=np.float32)
    global_precision = np.array([1.0, 1.0], dtype=np.float32)
    out = fola_local_precision(
        fisher,
        global_precision,
        server_round=1,
        initial_precision=1.0,
    )
    assert np.allclose(out, [1.0, 3.0])


def test_fola_recurrence_matches_prior_plus_average_fisher():
    prior = 1.0
    p0 = np.array([prior], dtype=np.float32)
    f1 = np.array([2.0], dtype=np.float32)
    f2 = np.array([4.0], dtype=np.float32)
    f3 = np.array([8.0], dtype=np.float32)

    p1 = fola_local_precision(f1, p0, 1, initial_precision=prior)
    p2 = fola_local_precision(f2, p1, 2, initial_precision=prior)
    p3 = fola_local_precision(f3, p2, 3, initial_precision=prior)

    assert np.allclose(p1, prior + f1)
    assert np.allclose(p2, prior + (f1 + f2) / 2.0)
    assert np.allclose(p3, prior + (f1 + f2 + f3) / 3.0)
