"""Sparse posterior-evidence helpers for the optional Proposed-method ablation.

This module is deliberately independent from the paper-reproduction path.  It
implements the user's Bayesian update-SNR score

    S_i = |mu_k,i - mu_G,i| / (sigma_k,i + eps)

and a same-budget random top-k control.  The mask is shared by the two Proposed
communication phases within one logical round so the selected posterior
evidence coordinates are consistent for Delta-rho and Delta-nu.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SparseMaskInfo:
    mask: np.ndarray
    kept: int
    total: int
    threshold: float
    score_mean: float
    selected_score_mean: float
    dropped_score_mean: float

    @property
    def keep_ratio_actual(self) -> float:
        return float(self.kept / max(1, self.total))


def kept_coordinate_count(dimension: int, keep_ratio: float, min_keep: int = 1) -> int:
    dimension = int(dimension)
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    ratio = float(keep_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0, 1]")
    return min(dimension, max(int(min_keep), int(np.ceil(ratio * dimension))))


def bayesian_update_snr_score(
    local_mean: np.ndarray,
    global_mean: np.ndarray,
    local_sigma: np.ndarray,
    epsilon: float = 1.0e-12,
) -> np.ndarray:
    """Return |mu_k - mu_G| / (sigma_k + epsilon) coordinate-wise."""
    local_mean = np.asarray(local_mean, dtype=np.float64).reshape(-1)
    global_mean = np.asarray(global_mean, dtype=np.float64).reshape(-1)
    local_sigma = np.asarray(local_sigma, dtype=np.float64).reshape(-1)
    if not (local_mean.size == global_mean.size == local_sigma.size):
        raise ValueError("local_mean/global_mean/local_sigma must have equal size")
    eps = float(epsilon)
    if eps <= 0.0:
        raise ValueError("epsilon must be positive")
    return np.abs(local_mean - global_mean) / (local_sigma + eps)


def _topk_mask(scores: np.ndarray, keep: int) -> tuple[np.ndarray, float]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    keep = int(keep)
    if keep <= 0 or keep > values.size:
        raise ValueError("keep must be in [1, number of scores]")
    if keep == values.size:
        return np.ones(values.size, dtype=bool), float(np.min(values))
    # argpartition avoids sorting all 62k coordinates every client/round.
    selected = np.argpartition(values, values.size - keep)[-keep:]
    mask = np.zeros(values.size, dtype=bool)
    mask[selected] = True
    threshold = float(np.min(values[selected]))
    return mask, threshold


def select_sparse_mask(
    *,
    selection: str,
    keep_ratio: float,
    min_keep: int,
    bayesian_scores: np.ndarray | None,
    random_seed: int,
) -> SparseMaskInfo:
    """Select Bayesian top-k or random k coordinates under the same budget."""
    normalized = str(selection).strip().lower()
    if bayesian_scores is None:
        if normalized == "bayesian":
            raise ValueError("Bayesian selection requires bayesian_scores")
        # Random selection still needs the model dimension. The caller should
        # provide a dummy zero score vector in random mode.
        raise ValueError("bayesian_scores must be supplied to define dimension")

    importance = np.asarray(bayesian_scores, dtype=np.float64).reshape(-1)
    keep = kept_coordinate_count(importance.size, keep_ratio, min_keep)

    if normalized == "bayesian":
        selection_scores = importance
    elif normalized == "random":
        rng = np.random.default_rng(int(random_seed) % (2**32 - 1))
        selection_scores = rng.random(importance.size, dtype=np.float64)
    else:
        raise ValueError("selection must be bayesian or random")

    mask, threshold = _topk_mask(selection_scores, keep)
    selected_imp = importance[mask]
    dropped_imp = importance[~mask]
    return SparseMaskInfo(
        mask=mask,
        kept=int(mask.sum()),
        total=int(mask.size),
        threshold=float(threshold),
        score_mean=float(np.mean(importance)) if importance.size else 0.0,
        selected_score_mean=(
            float(np.mean(selected_imp)) if selected_imp.size else 0.0
        ),
        dropped_score_mean=(
            float(np.mean(dropped_imp)) if dropped_imp.size else 0.0
        ),
    )


def full_mask_info(dimension: int) -> SparseMaskInfo:
    mask = np.ones(int(dimension), dtype=bool)
    return SparseMaskInfo(
        mask=mask,
        kept=int(dimension),
        total=int(dimension),
        threshold=0.0,
        score_mean=0.0,
        selected_score_mean=0.0,
        dropped_score_mean=0.0,
    )
