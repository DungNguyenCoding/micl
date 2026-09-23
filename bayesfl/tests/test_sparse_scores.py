import random
import numpy as np
import pytest
import torch
from bayesfl.posterior.sparse import diagonal_kl, dedicated_rng, keep_count, score_variance, select_mask


def test_argument_order_and_ranking():
    np.testing.assert_allclose(diagonal_kl([0], [1], [1], [.01]), [97.197414907], atol=1e-9)
    np.testing.assert_allclose(diagonal_kl([1], [.01], [0], [1]), [2.307585093], atol=1e-9)
    g = diagonal_kl([0, 0], [1, 1], [0, 3], [.01, 1])
    l = diagonal_kl([0, 3], [.01, 1], [0, 0], [1, 1])
    assert np.flatnonzero(select_mask(g, 2, 1)).tolist() == [0]
    assert np.flatnonzero(select_mask(l, 2, 1)).tolist() == [1]


def test_identical_and_equal_variance():
    assert np.array_equal(diagonal_kl([1, 2], [.1, 2], [1, 2], [.1, 2]), [0, 0])
    np.testing.assert_allclose(diagonal_kl([0, 1], [1, 4], [3, 5], [1, 4]),
                               diagonal_kl([3, 5], [1, 4], [0, 1], [1, 4]))


@pytest.mark.parametrize('d,r,m', [(4,.5,2), (9,.5,5), (1,.001,1), (17,1,17), (545810,.1,54581)])
def test_exact_ceiling(d, r, m):
    assert keep_count(d, r) == m


def test_partial_selection_matches_lexsort_with_ties():
    rng = np.random.default_rng(91)
    for d in (1, 8, 107):
        scores = rng.integers(0, 6, d).astype(float)
        for m in range(d + 1):
            expected = np.zeros(d, dtype=bool)
            expected[np.lexsort((np.arange(d), -scores))[:m]] = True
            assert np.array_equal(select_mask(scores, d, m), expected)


def test_random_stream_isolation():
    np.random.seed(42); random.seed(42); torch.manual_seed(42)
    before = np.random.get_state(), random.getstate(), torch.get_rng_state().clone()
    a = select_mask(None, 100, 19, rng=dedicated_rng(7, 2, 11))
    b = select_mask(None, 100, 19, rng=dedicated_rng(7, 2, 11))
    assert a.sum() == 19 and np.array_equal(a, b)
    after = np.random.get_state(), random.getstate(), torch.get_rng_state()
    assert before[0][0] == after[0][0] and np.array_equal(before[0][1], after[0][1])
    assert before[0][2:] == after[0][2:] and before[1] == after[1]
    assert torch.equal(before[2], after[2])


@pytest.mark.parametrize('v', [0, -1, np.nan, np.inf])
def test_invalid_variance_rejected(v):
    with pytest.raises((ValueError, FloatingPointError)):
        diagonal_kl([0], [v], [1], [1])


def test_score_only_floor_preserves_raw_state():
    p = np.array([0, 1e-12, 2], np.float32); before = p.copy()
    with pytest.raises(ValueError):
        score_variance(p, 'strict', 1e-8)
    v, fraction = score_variance(p, 'floor_for_score', 1e-8)
    np.testing.assert_allclose(v, [1e8, 1e8, .5])
    assert fraction == 2/3 and np.array_equal(p, before)
