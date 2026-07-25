"""Bayesian pruning and sparse communication helpers.

This module implements post-hoc SNR pruning and sparse Bayesian
communication.  Sparse communication supports two selection modes:

* ``bayesian``: choose top-k coordinates by the requested Bayesian score
  (SNR, update-SNR, precision-update, or coordinate KL).
* ``random``: choose k coordinates uniformly at random under the same keep
  ratio.  The random scores are used only for selection; the Bayesian
  importance scores are still logged for selected-vs-dropped analysis.

The random mode is an ablation baseline.  It keeps the communication budget
identical to Bayesian sparse communication while changing only which
coordinates are transmitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

EPS = 1.0e-12
SparseMetric = Literal["snr", "update_snr", "precision_update", "kl"]
SparseSelection = Literal["bayesian", "random"]


@dataclass
class SparsePack:
    indices: np.ndarray
    first_values: np.ndarray
    second_values: np.ndarray
    count_values: np.ndarray
    threshold: float
    # Bayesian importance score statistics over all coordinates.  These remain
    # comparable between bayesian and random selection.
    score_mean: float
    score_p50: float
    score_p90: float
    # Actual selection-score statistics over all coordinates.  In random mode,
    # these are uniform random scores; in bayesian mode, they equal score_*.
    selection_score_mean: float
    selection_score_p50: float
    selection_score_p90: float
    # Bayesian importance score split by selected vs dropped coordinates.
    selected_score_mean: float
    selected_score_p50: float
    selected_score_p90: float
    dropped_score_mean: float
    dropped_score_p50: float
    dropped_score_p90: float
    # Selection score split by selected vs dropped coordinates.  Mainly useful
    # for verifying that random mode did use a random selector.
    selected_selection_score_mean: float
    selected_selection_score_p50: float
    selected_selection_score_p90: float
    dropped_selection_score_mean: float
    dropped_selection_score_p50: float
    dropped_selection_score_p90: float
    total_params: int
    sent_params: int
    selection_method: str = "bayesian"

    @property
    def compression_ratio(self) -> float:
        return float(self.sent_params / max(self.total_params, 1))


def finite_stats(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(x.mean()),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
    }


def weight_snr(mu: np.ndarray, sigma: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Bayes-by-Backprop weight SNR, |mu| / sigma."""
    return np.abs(np.asarray(mu, dtype=np.float64)) / (np.asarray(sigma, dtype=np.float64) + eps)


def update_snr(local_mu: np.ndarray, global_mu: np.ndarray, local_sigma: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Federated adaptation: |local_mu - global_mu| / local_sigma."""
    return np.abs(np.asarray(local_mu, dtype=np.float64) - np.asarray(global_mu, dtype=np.float64)) / (
        np.asarray(local_sigma, dtype=np.float64) + eps
    )


def precision_update_score(local_mu: np.ndarray, global_mu: np.ndarray, local_precision: np.ndarray) -> np.ndarray:
    """OLA/FOLA score: |delta_mu| * precision."""
    return np.abs(np.asarray(local_mu, dtype=np.float64) - np.asarray(global_mu, dtype=np.float64)) * np.asarray(
        local_precision, dtype=np.float64
    )


def diag_gaussian_kl_score(
    local_mu: np.ndarray,
    local_sigma: np.ndarray,
    global_mu: np.ndarray,
    global_sigma: np.ndarray,
    min_sigma: float = 1.0e-8,
) -> np.ndarray:
    """Per-coordinate KL[q_local || q_global] for diagonal Gaussians."""
    lm = np.asarray(local_mu, dtype=np.float64)
    ls = np.maximum(np.asarray(local_sigma, dtype=np.float64), min_sigma)
    gm = np.asarray(global_mu, dtype=np.float64)
    gs = np.maximum(np.asarray(global_sigma, dtype=np.float64), min_sigma)
    return 0.5 * ((ls**2 + (lm - gm) ** 2) / (gs**2) - 1.0 + 2.0 * (np.log(gs) - np.log(ls)))


def score_for_sparse_metric(
    metric: str,
    local_mu: np.ndarray,
    global_mu: np.ndarray,
    local_sigma: np.ndarray | None = None,
    local_precision: np.ndarray | None = None,
    global_sigma: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Bayesian importance scores for sparse communication."""
    metric = str(metric).lower()
    if metric == "snr":
        if local_sigma is None:
            raise ValueError("metric='snr' requires local_sigma")
        return weight_snr(local_mu, local_sigma)
    if metric == "update_snr":
        if local_sigma is None:
            raise ValueError("metric='update_snr' requires local_sigma")
        return update_snr(local_mu, global_mu, local_sigma)
    if metric == "precision_update":
        if local_precision is None:
            raise ValueError("metric='precision_update' requires local_precision")
        return precision_update_score(local_mu, global_mu, local_precision)
    if metric == "kl":
        if local_sigma is None or global_sigma is None:
            raise ValueError("metric='kl' requires local_sigma and global_sigma")
        return diag_gaussian_kl_score(local_mu, local_sigma, global_mu, global_sigma)
    raise ValueError(f"Unsupported sparse_metric={metric!r}")


def random_sparse_score(num_params: int, seed: int) -> np.ndarray:
    """Return deterministic U(0,1) scores for random sparse ablations."""
    # NumPy accepts 32-bit-ish seeds most portably.
    safe_seed = int(seed) % (2**32 - 1)
    rng = np.random.default_rng(safe_seed)
    return rng.random(int(num_params), dtype=np.float64)


def score_for_sparse_selection(selection: str, bayesian_scores: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Return scores actually used for top-k selection.

    ``bayesian_scores`` should always be the scientific/importance score.  In
    ``random`` mode, random scores are returned for mask selection while
    ``bayesian_scores`` should still be logged as reference importance.
    """
    selection = str(selection or "bayesian").lower()
    bayes = np.asarray(bayesian_scores, dtype=np.float64).reshape(-1)
    if selection == "bayesian":
        return bayes
    if selection == "random":
        if seed is None:
            seed = 0
        return random_sparse_score(bayes.size, int(seed))
    raise ValueError(f"Unsupported sparse_selection={selection!r}")


def topk_mask(scores: np.ndarray, ratio: float, min_keep: int = 1) -> tuple[np.ndarray, float]:
    """Return a boolean mask keeping the largest scores."""
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = int(score.size)
    if n == 0:
        return np.zeros(0, dtype=bool), float("nan")
    keep = int(np.ceil(float(ratio) * n))
    keep = max(int(min_keep), keep)
    keep = min(n, keep)
    if keep >= n:
        mask = np.ones(n, dtype=bool)
        threshold = float(np.nanmin(score))
        return mask, threshold
    safe_score = np.where(np.isfinite(score), score, -np.inf)
    kth = n - keep
    threshold = float(np.partition(safe_score, kth)[kth])
    mask = safe_score >= threshold
    if int(mask.sum()) > keep:
        idx = np.argsort(safe_score)[-keep:]
        new_mask = np.zeros(n, dtype=bool)
        new_mask[idx] = True
        mask = new_mask
    return mask, threshold


def pack_sparse_contribution(
    first_dense: np.ndarray,
    second_dense: np.ndarray,
    count_dense: np.ndarray,
    scores: np.ndarray,
    ratio: float,
    min_keep: int = 1,
    *,
    importance_scores: np.ndarray | None = None,
    selection_method: str = "bayesian",
) -> SparsePack:
    """Pack dense contribution arrays into sparse index/value arrays.

    Parameters
    ----------
    scores:
        Scores used to choose the top-k mask.  In random mode, these are random
        scores.
    importance_scores:
        Bayesian importance scores to log and compare.  If omitted, ``scores``
        are used.  For random ablation runs, pass the Bayesian metric scores
        here so selected-vs-dropped statistics remain meaningful.
    """
    f = np.asarray(first_dense, dtype=np.float64).reshape(-1)
    s = np.asarray(second_dense, dtype=np.float64).reshape(-1)
    c = np.asarray(count_dense, dtype=np.float64).reshape(-1)
    selection_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    importance = selection_score if importance_scores is None else np.asarray(importance_scores, dtype=np.float64).reshape(-1)
    if not (f.size == s.size == c.size == selection_score.size == importance.size):
        raise ValueError("first/second/count/score arrays must have the same flat size")
    mask, threshold = topk_mask(selection_score, ratio=ratio, min_keep=min_keep)
    idx = np.where(mask)[0].astype(np.int64, copy=False)
    imp_all = finite_stats(importance)
    sel_all = finite_stats(selection_score)
    imp_selected = finite_stats(importance[mask])
    imp_dropped = finite_stats(importance[~mask])
    sel_selected = finite_stats(selection_score[mask])
    sel_dropped = finite_stats(selection_score[~mask])
    return SparsePack(
        indices=idx,
        first_values=f[idx].astype(np.float32, copy=False),
        second_values=s[idx].astype(np.float32, copy=False),
        count_values=c[idx].astype(np.float32, copy=False),
        threshold=threshold,
        score_mean=imp_all["mean"],
        score_p50=imp_all["p50"],
        score_p90=imp_all["p90"],
        selection_score_mean=sel_all["mean"],
        selection_score_p50=sel_all["p50"],
        selection_score_p90=sel_all["p90"],
        selected_score_mean=imp_selected["mean"],
        selected_score_p50=imp_selected["p50"],
        selected_score_p90=imp_selected["p90"],
        dropped_score_mean=imp_dropped["mean"],
        dropped_score_p50=imp_dropped["p50"],
        dropped_score_p90=imp_dropped["p90"],
        selected_selection_score_mean=sel_selected["mean"],
        selected_selection_score_p50=sel_selected["p50"],
        selected_selection_score_p90=sel_selected["p90"],
        dropped_selection_score_mean=sel_dropped["mean"],
        dropped_selection_score_p50=sel_dropped["p50"],
        dropped_selection_score_p90=sel_dropped["p90"],
        total_params=int(selection_score.size),
        sent_params=int(idx.size),
        selection_method=str(selection_method),
    )


def unpack_sparse_to_dense(
    indices: np.ndarray,
    first_values: np.ndarray,
    second_values: np.ndarray,
    count_values: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack sparse index/value arrays into dense first/second/count arrays."""
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    first = np.zeros(int(size), dtype=np.float64)
    second = np.zeros(int(size), dtype=np.float64)
    count = np.zeros(int(size), dtype=np.float64)
    if idx.size == 0:
        return first, second, count
    first[idx] += np.asarray(first_values, dtype=np.float64).reshape(-1)
    second[idx] += np.asarray(second_values, dtype=np.float64).reshape(-1)
    count[idx] += np.asarray(count_values, dtype=np.float64).reshape(-1)
    return first, second, count


def _empty_sparse_metrics(metric: str, selection: str, ratio: float, warmup_rounds: int) -> dict[str, float | int | bool | str]:
    return {
        "sparse_comm_enabled": bool(False),
        "sparse_metric": str(metric),
        "sparse_selection_method": str(selection),
        "sparse_ratio": float(ratio),
        "sparse_warmup_rounds": int(warmup_rounds),
        "sparse_num_params_total": float("nan"),
        "sparse_num_params_sent": float("nan"),
        "sparse_compression_ratio": float("nan"),
        "sparse_threshold": float("nan"),
        "sparse_score_mean": float("nan"),
        "sparse_score_p50": float("nan"),
        "sparse_score_p90": float("nan"),
        "sparse_selection_score_mean": float("nan"),
        "sparse_selection_score_p50": float("nan"),
        "sparse_selection_score_p90": float("nan"),
        "sparse_selected_score_mean": float("nan"),
        "sparse_selected_score_p50": float("nan"),
        "sparse_selected_score_p90": float("nan"),
        "sparse_dropped_score_mean": float("nan"),
        "sparse_dropped_score_p50": float("nan"),
        "sparse_dropped_score_p90": float("nan"),
        "sparse_selected_selection_score_mean": float("nan"),
        "sparse_selected_selection_score_p50": float("nan"),
        "sparse_selected_selection_score_p90": float("nan"),
        "sparse_dropped_selection_score_mean": float("nan"),
        "sparse_dropped_selection_score_p50": float("nan"),
        "sparse_dropped_selection_score_p90": float("nan"),
        "sparse_sent_update_l2": float("nan"),
        "sparse_dropped_update_l2": float("nan"),
        "sparse_sent_update_fraction_l2": float("nan"),
        "sparse_dropped_update_fraction_l2": float("nan"),
        "sparse_update_energy_retention": float("nan"),
        "sparse_dense_bytes": float("nan"),
        "sparse_sent_bytes": float("nan"),
        "sparse_index_bytes": float("nan"),
        "sparse_value_bytes": float("nan"),
        "sparse_byte_saving_ratio": float("nan"),
    }


def sparse_row_metrics(
    enabled: bool,
    metric: str,
    ratio: float,
    warmup_rounds: int,
    pack: SparsePack | None,
    update: np.ndarray | None = None,
    mask_indices: np.ndarray | None = None,
    selection: str = "bayesian",
) -> dict[str, float | int | bool | str]:
    """Metrics suitable for client_train_metrics.csv and sparse_comm_metrics.csv."""
    if not enabled or pack is None:
        return _empty_sparse_metrics(metric, selection, ratio, warmup_rounds)

    sent_l2 = dropped_l2 = frac_l2 = dropped_frac_l2 = float("nan")
    if update is not None and mask_indices is not None:
        upd = np.asarray(update, dtype=np.float64).reshape(-1)
        mask = np.zeros_like(upd, dtype=bool)
        idx = np.asarray(mask_indices, dtype=np.int64).reshape(-1)
        idx = idx[(idx >= 0) & (idx < upd.size)]
        mask[idx] = True
        sent_l2 = float(np.linalg.norm(upd[mask]))
        dropped_l2 = float(np.linalg.norm(upd[~mask]))
        denom = float(np.linalg.norm(upd))
        frac_l2 = float(sent_l2 / denom) if denom > 0 else float("nan")
        dropped_frac_l2 = float(dropped_l2 / denom) if denom > 0 else float("nan")

    # Dense Bayesian communication transmits two float32 arrays.  Sparse
    # product aggregation transmits int64 indices plus three float32 arrays
    # (precision/evidence, precision*mean/evidence, count evidence).  For
    # keep100 controls we report optimized dense bytes because a real
    # implementation would skip index transmission when all coordinates are sent.
    dense_bytes = int(pack.total_params * 2 * 4)
    if int(pack.sent_params) >= int(pack.total_params):
        index_bytes = 0
        value_bytes = dense_bytes
        sent_bytes = dense_bytes
    else:
        index_bytes = int(pack.sent_params * 8)
        value_bytes = int(pack.sent_params * 3 * 4)
        sent_bytes = int(index_bytes + value_bytes)

    return {
        "sparse_comm_enabled": bool(True),
        "sparse_metric": str(metric),
        "sparse_selection_method": str(selection),
        "sparse_ratio": float(ratio),
        "sparse_warmup_rounds": int(warmup_rounds),
        "sparse_num_params_total": int(pack.total_params),
        "sparse_num_params_sent": int(pack.sent_params),
        "sparse_compression_ratio": float(pack.compression_ratio),
        "sparse_threshold": float(pack.threshold),
        "sparse_score_mean": float(pack.score_mean),
        "sparse_score_p50": float(pack.score_p50),
        "sparse_score_p90": float(pack.score_p90),
        "sparse_selection_score_mean": float(pack.selection_score_mean),
        "sparse_selection_score_p50": float(pack.selection_score_p50),
        "sparse_selection_score_p90": float(pack.selection_score_p90),
        "sparse_selected_score_mean": float(pack.selected_score_mean),
        "sparse_selected_score_p50": float(pack.selected_score_p50),
        "sparse_selected_score_p90": float(pack.selected_score_p90),
        "sparse_dropped_score_mean": float(pack.dropped_score_mean),
        "sparse_dropped_score_p50": float(pack.dropped_score_p50),
        "sparse_dropped_score_p90": float(pack.dropped_score_p90),
        "sparse_selected_selection_score_mean": float(pack.selected_selection_score_mean),
        "sparse_selected_selection_score_p50": float(pack.selected_selection_score_p50),
        "sparse_selected_selection_score_p90": float(pack.selected_selection_score_p90),
        "sparse_dropped_selection_score_mean": float(pack.dropped_selection_score_mean),
        "sparse_dropped_selection_score_p50": float(pack.dropped_selection_score_p50),
        "sparse_dropped_selection_score_p90": float(pack.dropped_selection_score_p90),
        "sparse_sent_update_l2": sent_l2,
        "sparse_dropped_update_l2": dropped_l2,
        "sparse_sent_update_fraction_l2": frac_l2,
        "sparse_dropped_update_fraction_l2": dropped_frac_l2,
        "sparse_update_energy_retention": frac_l2,
        "sparse_dense_bytes": dense_bytes,
        "sparse_sent_bytes": sent_bytes,
        "sparse_index_bytes": index_bytes,
        "sparse_value_bytes": value_bytes,
        "sparse_byte_saving_ratio": float(1.0 - sent_bytes / max(dense_bytes, 1)),
    }
