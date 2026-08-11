from experiments import paired_realization_seed


def test_matching_conditions_share_seed_for_same_realization():
    base = 12025
    seeds = [paired_realization_seed(base, 0) for _ in range(3)]
    assert seeds == [12025, 12025, 12025]


def test_replications_advance_seed_once_not_per_condition():
    base = 12025
    assert [paired_realization_seed(base, r) for r in range(3)] == [12025, 12026, 12027]


def test_negative_realization_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="realization must be non-negative"):
        paired_realization_seed(12025, -1)
