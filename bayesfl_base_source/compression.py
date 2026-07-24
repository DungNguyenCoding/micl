"""Bayesian pruning and sparse communication helpers.

This module implements the first three compression experiments derived from
Bayes-by-Backprop SNR pruning:

1. Post-hoc pruning by posterior SNR.
2. Sparse VI communication by update SNR / KL / weight SNR.
3. Sparse OLA/FOLA communication by precision-weighted update score.

The sparse communication implementation supports actual sparse Flower payloads:
clients return index/value arrays instead of full dense vectors when enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

EPS = 1.0e-12
SparseMetric = Literal["snr", "update_snr", "precision_update", "kl"]


@dataclass
class SparsePack:
    indices: np.ndarray
    first_values: np.ndarray
    second_values: np.ndarray
    count_values: np.ndarray
    threshold: float
    score_mean: float
    score_p50: float
    score_p90: float
    total_params: int
    sent_params: int

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




def random_sparse_score(num_params: int, seed: int | None = None) -> np.ndarray:
    """Generate random scores for random top-k sparse communication ablation.

    The existing top-k packing code is reused; only the ranking score changes.
    This keeps communication budget identical to Bayesian sparse selection.
    """
    rng = np.random.default_rng(seed)
    return rng.random(int(num_params)).astype(np.float64)

def score_for_sparse_metric(
    metric: str,
    local_mu: np.ndarray,
    global_mu: np.ndarray,
    local_sigma: np.ndarray | None = None,
    local_precision: np.ndarray | None = None,
    global_sigma: np.ndarray | None = None,
) -> np.ndarray:
    """Compute importance scores for sparse Bayesian communication."""
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
    # kth largest threshold
    kth = n - keep
    threshold = float(np.partition(safe_score, kth)[kth])
    mask = safe_score >= threshold
    # Resolve ties to exactly `keep` if needed.
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
) -> SparsePack:
    """Pack dense contribution arrays into sparse index/value arrays."""
    f = np.asarray(first_dense, dtype=np.float64).reshape(-1)
    s = np.asarray(second_dense, dtype=np.float64).reshape(-1)
    c = np.asarray(count_dense, dtype=np.float64).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not (f.size == s.size == c.size == score.size):
        raise ValueError("first/second/count/score arrays must have the same flat size")
    mask, threshold = topk_mask(score, ratio=ratio, min_keep=min_keep)
    idx = np.where(mask)[0].astype(np.int64, copy=False)
    st = finite_stats(score)
    return SparsePack(
        indices=idx,
        first_values=f[idx].astype(np.float32, copy=False),
        second_values=s[idx].astype(np.float32, copy=False),
        count_values=c[idx].astype(np.float32, copy=False),
        threshold=threshold,
        score_mean=st["mean"],
        score_p50=st["p50"],
        score_p90=st["p90"],
        total_params=int(score.size),
        sent_params=int(idx.size),
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


def sparse_row_metrics(
    enabled: bool,
    metric: str,
    ratio: float,
    warmup_rounds: int,
    pack: SparsePack | None,
    update: np.ndarray | None = None,
    mask_indices: np.ndarray | None = None,
) -> dict[str, float | int | bool | str]:
    """Metrics suitable for client_train_metrics.csv and sparse_comm_metrics.csv."""
    if not enabled or pack is None:
        return {
            "sparse_comm_enabled": bool(False),
            "sparse_metric": str(metric),
            "sparse_ratio": float(ratio),
            "sparse_warmup_rounds": int(warmup_rounds),
            "sparse_num_params_total": float("nan"),
            "sparse_num_params_sent": float("nan"),
            "sparse_compression_ratio": float("nan"),
            "sparse_threshold": float("nan"),
            "sparse_score_mean": float("nan"),
            "sparse_score_p50": float("nan"),
            "sparse_score_p90": float("nan"),
            "sparse_sent_update_l2": float("nan"),
            "sparse_dropped_update_l2": float("nan"),
            "sparse_sent_update_fraction_l2": float("nan"),
        }
    sent_l2 = dropped_l2 = frac_l2 = float("nan")
    if update is not None and mask_indices is not None:
        upd = np.asarray(update, dtype=np.float64).reshape(-1)
        mask = np.zeros_like(upd, dtype=bool)
        mask[np.asarray(mask_indices, dtype=np.int64)] = True
        sent_l2 = float(np.linalg.norm(upd[mask]))
        dropped_l2 = float(np.linalg.norm(upd[~mask]))
        denom = float(np.linalg.norm(upd))
        frac_l2 = float(sent_l2 / denom) if denom > 0 else float("nan")
    return {
        "sparse_comm_enabled": bool(True),
        "sparse_metric": str(metric),
        "sparse_ratio": float(ratio),
        "sparse_warmup_rounds": int(warmup_rounds),
        "sparse_num_params_total": int(pack.total_params),
        "sparse_num_params_sent": int(pack.sent_params),
        "sparse_compression_ratio": float(pack.compression_ratio),
        "sparse_threshold": float(pack.threshold),
        "sparse_score_mean": float(pack.score_mean),
        "sparse_score_p50": float(pack.score_p50),
        "sparse_score_p90": float(pack.score_p90),
        "sparse_sent_update_l2": sent_l2,
        "sparse_dropped_update_l2": dropped_l2,
        "sparse_sent_update_fraction_l2": frac_l2,
    }
