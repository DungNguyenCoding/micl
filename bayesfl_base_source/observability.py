"""Metric, uncertainty, calibration, and CSV utilities for Bayes-FL runs.

This module keeps training code readable by centralizing the experiment
observability logic. It intentionally uses only stdlib CSV plus NumPy/Torch, so
it does not depend on Pandas during training or plotting.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import model
from config import RunConfig

SCHEMA_VERSION = "bayesfl_observability_v1"
NAN = float("nan")
EPS = 1.0e-12


def nan() -> float:
    return float("nan")


def is_finite_number(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def float_or_nan(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return nan()


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else nan()


def weighted_mean(rows: Sequence[Mapping[str, Any]], value_key: str, weight_key: str = "num_examples") -> float:
    total = 0.0
    weight_sum = 0.0
    for row in rows:
        value = float_or_nan(row.get(value_key, nan()))
        weight = float_or_nan(row.get(weight_key, 1.0))
        if np.isfinite(value) and np.isfinite(weight) and weight > 0:
            total += value * weight
            weight_sum += weight
    return float(total / weight_sum) if weight_sum > 0 else nan()


def percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return nan()
    return float(np.percentile(arr, q))


def array_stats(arr: np.ndarray | torch.Tensor | None, prefix: str, include_abs: bool = False) -> Dict[str, float]:
    """Return scalar summary statistics for a flat numeric array."""
    out: Dict[str, float] = {}
    if arr is None:
        for suffix in ["mean", "std", "min", "p10", "p25", "p50", "p75", "p90", "p95", "max"]:
            out[f"{prefix}_{suffix}"] = nan()
        if include_abs:
            for suffix in ["mean", "std", "p50", "p90"]:
                out[f"{prefix}_abs_{suffix}"] = nan()
            out[f"{prefix}_l2"] = nan()
        return out
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return array_stats(None, prefix, include_abs=include_abs)
    out.update(
        {
            f"{prefix}_mean": float(x.mean()),
            f"{prefix}_std": float(x.std()),
            f"{prefix}_min": float(x.min()),
            f"{prefix}_p10": percentile(x, 10),
            f"{prefix}_p25": percentile(x, 25),
            f"{prefix}_p50": percentile(x, 50),
            f"{prefix}_p75": percentile(x, 75),
            f"{prefix}_p90": percentile(x, 90),
            f"{prefix}_p95": percentile(x, 95),
            f"{prefix}_max": float(x.max()),
        }
    )
    if include_abs:
        ax = np.abs(x)
        out.update(
            {
                f"{prefix}_abs_mean": float(ax.mean()),
                f"{prefix}_abs_std": float(ax.std()),
                f"{prefix}_abs_p50": percentile(ax, 50),
                f"{prefix}_abs_p90": percentile(ax, 90),
                f"{prefix}_l2": float(np.linalg.norm(x)),
            }
        )
    return out


def finite_or_empty(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else ""
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> Path:
    """Write CSV with a stable header, allowing rows to contain extra keys."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    if fieldnames is not None:
        keys.extend([k for k in fieldnames if k not in keys])
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: finite_or_empty(row.get(k, "")) for k in keys})
    return output


# ---------------------------------------------------------------------------
# CSV schemas
# ---------------------------------------------------------------------------
METRICS_FIELDS: List[str] = [
    "schema_version", "run_id", "round", "method", "dataset", "model", "iid", "balanced",
    "noniid_alpha", "unbalanced_alpha", "num_devices", "num_virtual_clients", "client_fraction",
    "selected_count", "selected_examples", "total_examples", "local_epochs", "batch_size", "lr", "seed",
    "eval_every", "heavy_eval_every", "eval_mc_samples",
    "round_time_sec", "fit_time_sec", "aggregate_time_sec", "eval_time_sec", "num_fit_failures",
    "active_physical_devices", "active_virtual_clients",
    # Backward-compatible aliases used by older plotting commands
    "accuracy", "loss", "train_loss",
    # Global deterministic/probabilistic performance
    "global_accuracy", "global_error_rate", "global_loss", "global_nll", "global_brier", "global_ece", "global_mce",
    "global_mean_confidence", "global_mean_entropy", "global_num_eval_examples",
    # Deterministic posterior-mean evaluation. For FedAvg this equals global_*;
    # for VI/OLA this evaluates theta=mu without posterior sampling.
    "global_mean_accuracy", "global_mean_loss", "global_mean_nll", "global_mean_brier", "global_mean_ece", "global_mean_mce",
    "global_mean_prediction_confidence", "global_mean_prediction_entropy",
    # Monte Carlo posterior-predictive evaluation with posterior_sample_scale.
    "global_mc_accuracy", "global_mc_loss", "global_mc_nll", "global_mc_brier", "global_mc_ece", "global_mc_mce",
    "global_mc_mean_confidence", "global_mc_mean_entropy", "global_mc_posterior_sample_scale",
    # Bayesian predictive uncertainty
    "global_mc_samples", "global_predictive_entropy", "global_expected_entropy", "global_mutual_information",
    "global_aleatoric_uncertainty", "global_epistemic_uncertainty", "global_predictive_variance_mean", "global_predictive_variance_std",
    # Local summaries
    "local_eval_count", "local_accuracy_weighted", "local_accuracy_mean", "local_accuracy_std", "local_accuracy_min",
    "local_accuracy_p10", "local_accuracy_p25", "local_accuracy_p50", "local_accuracy_p75", "local_accuracy_p90", "local_accuracy_max",
    "local_loss_weighted", "local_loss_mean", "local_loss_std", "local_loss_min", "local_loss_p50", "local_loss_max",
    "local_ece_mean", "local_ece_std", "local_nll_mean", "local_brier_mean",
    "local_forgetting_proxy_mean", "local_forgetting_proxy_std", "local_forgetting_proxy_weighted",
    "local_global_loss_gap_mean", "local_global_acc_gap_mean",
    # Client/update drift summaries
    "client_drift_from_global_l2_mean", "client_drift_from_global_l2_std", "client_drift_from_global_cosine_mean",
    "client_update_l2_mean", "client_update_l2_std", "client_update_l2_min", "client_update_l2_max", "client_update_cosine_mean",
    # Training decomposition
    "train_loss_mean", "train_loss_std", "task_loss_mean", "task_loss_std", "prior_loss_mean", "prior_loss_std",
    "regularization_loss_mean", "regularization_loss_std",
    # VI-specific
    "vi_elbo_loss_mean", "vi_elbo_loss_std", "vi_kl_loss_mean", "vi_kl_loss_std", "vi_kl_loss_per_example_mean", "vi_kl_loss_per_example_std",
    "vi_kl_per_param_mean", "vi_likelihood_loss_mean", "vi_likelihood_loss_std", "vi_complexity_cost_mean", "vi_effective_lr_mean",
    "vi_scale_mean", "vi_scale_std", "vi_scale_p50", "vi_scale_p90", "vi_scale_max",
    "vi_prior_scale", "vi_min_scale", "vi_particles", "vi_aggregation_mode",
    # OLA-specific
    "ola_prior_lambda", "ola_prior_loss_mean", "ola_prior_loss_std", "ola_prior_loss_raw_mean", "ola_prior_loss_raw_std",
    "ola_regularization_loss_raw_mean", "ola_task_loss_mean", "ola_task_loss_std", "ola_prior_task_ratio",
    "ola_fisher_mean", "ola_fisher_std", "ola_fisher_min", "ola_fisher_p10", "ola_fisher_p50", "ola_fisher_p90", "ola_fisher_max",
    "ola_precision_mean", "ola_precision_std", "ola_precision_min", "ola_precision_p10", "ola_precision_p50", "ola_precision_p90", "ola_precision_max",
    "ola_sigma_mean", "ola_sigma_std", "ola_sigma_p50", "ola_sigma_p90", "ola_sigma_max", "ola_gamma", "ola_online_weight_fisher", "ola_online_weight_prior",
    # Global posterior
    "posterior_available", "posterior_num_params", "posterior_mu_l2", "posterior_mu_abs_mean", "posterior_mu_abs_std",
    "posterior_mu_abs_p50", "posterior_mu_abs_p90",
    "posterior_sigma_mean", "posterior_sigma_std", "posterior_sigma_min", "posterior_sigma_p10", "posterior_sigma_p25",
    "posterior_sigma_p50", "posterior_sigma_p75", "posterior_sigma_p90", "posterior_sigma_p95", "posterior_sigma_max",
    "posterior_precision_mean", "posterior_precision_std", "posterior_precision_min", "posterior_precision_p50", "posterior_precision_p90", "posterior_precision_max",
    "posterior_var_trace", "posterior_logdet_diag", "posterior_entropy_diag_gaussian",
    # SNR summaries
    "posterior_snr_raw_mean", "posterior_snr_raw_std", "posterior_snr_raw_min", "posterior_snr_raw_p10", "posterior_snr_raw_p25",
    "posterior_snr_raw_p50", "posterior_snr_raw_p75", "posterior_snr_raw_p90", "posterior_snr_raw_p95", "posterior_snr_raw_max",
    "posterior_snr_db_mean", "posterior_snr_db_std", "posterior_snr_db_p10", "posterior_snr_db_p50", "posterior_snr_db_p90",
    "posterior_snr_frac_lt_0_5", "posterior_snr_frac_lt_1", "posterior_snr_frac_lt_2", "posterior_snr_frac_lt_5",
    "posterior_snr_frac_gt_1", "posterior_snr_frac_gt_2", "posterior_snr_frac_gt_5",
    "effective_params_snr_gt_1", "effective_params_snr_gt_2", "effective_params_snr_gt_5",
    # Aggregation diagnostics
    "aggregation_delta_l2", "aggregation_delta_linf", "aggregation_delta_cosine_to_mean_client_update",
    "aggregation_weight_entropy", "aggregation_weight_max", "aggregation_weight_min", "aggregation_error_proxy",
    "aggregation_energy_before", "aggregation_energy_after", "posterior_product_precision_mean", "posterior_product_precision_std",
    "posterior_product_mu_norm", "posterior_product_sigma_mean",
    # Sparse Bayesian communication metrics
    "sparse_comm_enabled", "sparse_metric", "sparse_ratio", "sparse_warmup_rounds", "sparse_min_keep",
    "sparse_updated_params", "sparse_total_params", "sparse_updated_ratio",
    "sparse_num_params_sent_mean", "sparse_num_params_sent_std", "sparse_compression_ratio_mean", "sparse_compression_ratio_std",
    "sparse_threshold_mean", "sparse_threshold_std", "sparse_score_mean_avg", "sparse_score_p50_avg", "sparse_score_p90_avg",
    "sparse_score_mean_mean", "sparse_score_p50_mean", "sparse_score_p90_mean",
    "communication_dense_params", "communication_sent_params_mean", "communication_sent_params_total", "communication_compression_ratio",
    "communication_dense_bytes", "communication_sparse_bytes", "communication_index_bytes", "communication_value_bytes", "communication_saving_ratio",
    # Data/selection and future wireless summaries
    "selected_label_entropy_mean", "selected_label_entropy_std", "selected_label_entropy_min", "selected_label_entropy_max",
    "selected_kl_to_global_label_mean", "selected_kl_to_global_label_std", "selected_num_examples_mean", "selected_num_examples_std",
    "selected_num_examples_min", "selected_num_examples_max", "wireless_policy", "wireless_selected_channel_snr_db_mean",
    "wireless_selected_channel_snr_db_std", "wireless_selected_pathloss_db_mean", "wireless_selected_rate_mbps_mean",
    "wireless_selected_delay_ms_mean", "wireless_selected_energy_j_mean", "wireless_outage_count", "ota_noise_power", "ota_distortion", "ota_mse",
    "digital_packet_error_rate", "posterior_snapshot_path", "prediction_snapshot_path", "snr_histogram_path", "calibration_bins_path",
]

CLIENT_TRAIN_FIELDS = [
    "schema_version", "run_id", "round", "method", "physical_client_id", "virtual_client_id", "num_examples", "num_batches", "local_epochs", "batch_size",
    "train_loss", "task_loss", "prior_loss", "regularization_loss", "accuracy_local_train_estimate", "loss_local_train_estimate",
    "update_l2_norm", "update_linf_norm", "update_cosine_to_global", "drift_from_global_before_l2", "drift_from_global_after_l2",
    "label_entropy", "kl_to_global_label_distribution",
    "vi_elbo_loss", "vi_kl_loss", "vi_kl_loss_per_example", "vi_kl_per_param", "vi_likelihood_loss", "vi_complexity_cost", "vi_effective_lr",
    "vi_loc_l2", "vi_scale_mean", "vi_scale_p50", "vi_scale_p90", "vi_scale_max",
    "vi_snr_raw_mean", "vi_snr_raw_p50", "vi_snr_raw_p90",
    "ola_task_loss", "ola_prior_loss", "ola_prior_loss_raw", "ola_regularization_loss_raw", "ola_fisher_mean", "ola_fisher_p50", "ola_fisher_p90", "ola_fisher_max", "ola_precision_mean",
    "ola_precision_p50", "ola_precision_p90", "ola_sigma_mean", "ola_sigma_p50", "ola_sigma_p90", "ola_snr_raw_mean", "ola_snr_raw_p50", "ola_snr_raw_p90",
    "sparse_comm_enabled", "sparse_metric", "sparse_ratio", "sparse_warmup_rounds", "sparse_num_params_total", "sparse_num_params_sent",
    "sparse_compression_ratio", "sparse_threshold", "sparse_score_mean", "sparse_score_p50", "sparse_score_p90",
    "sparse_sent_update_l2", "sparse_dropped_update_l2", "sparse_sent_update_fraction_l2",
    "channel_snr_db", "pathloss_db", "rate_mbps", "delay_ms", "energy_j", "ota_contribution_norm", "digital_payload_bytes", "communication_success",
]

CLIENT_EVAL_FIELDS = [
    "schema_version", "run_id", "round", "method", "physical_client_id", "virtual_client_id", "eval_scope", "eval_dataset", "num_eval_examples",
    "local_accuracy", "local_loss", "local_nll", "local_brier", "local_ece", "local_mce", "local_mean_confidence", "local_mean_entropy",
    "local_predictive_entropy", "local_expected_entropy", "local_mutual_information", "local_mc_samples", "global_accuracy_reference", "global_loss_reference",
    "local_global_accuracy_gap", "local_global_loss_gap", "local_forgetting_proxy", "num_examples_train", "label_entropy", "kl_to_global_label_distribution",
]

POSTERIOR_SUMMARY_FIELDS = [
    "schema_version", "run_id", "round", "method", "scope", "physical_client_id", "virtual_client_id", "layer_name", "param_name", "num_params",
    "mu_mean", "mu_std", "mu_abs_mean", "mu_abs_p50", "mu_abs_p90", "mu_l2", "sigma_mean", "sigma_std", "sigma_min", "sigma_p10", "sigma_p50", "sigma_p90", "sigma_p95", "sigma_max",
    "variance_mean", "variance_sum", "precision_mean", "precision_std", "precision_min", "precision_p50", "precision_p90", "precision_max",
    "snr_raw_mean", "snr_raw_std", "snr_raw_p10", "snr_raw_p50", "snr_raw_p75", "snr_raw_p90", "snr_raw_p95",
    "snr_db_mean", "snr_db_std", "snr_db_p50", "snr_db_p90", "snr_frac_lt_1", "snr_frac_lt_2", "snr_frac_gt_1", "snr_frac_gt_2",
    "effective_params_snr_gt_1", "effective_params_snr_gt_2",
]

SNR_HISTOGRAM_FIELDS = [
    "schema_version", "run_id", "round", "method", "scope", "physical_client_id", "virtual_client_id", "layer_name", "value_space", "bin_id", "bin_left", "bin_right", "bin_center", "count", "density", "cdf", "total_count",
]

CALIBRATION_BIN_FIELDS = [
    "schema_version", "run_id", "round", "method", "eval_scope", "physical_client_id", "virtual_client_id", "bin_id", "bin_left", "bin_right", "bin_count", "bin_accuracy", "bin_confidence", "bin_gap", "ece_contribution", "nll_mean", "brier_mean",
]

SELECTION_SUMMARY_FIELDS = [
    "schema_version", "run_id", "round", "method", "selection_policy", "selected_count", "available_count", "selected_fraction", "selected_examples", "selected_examples_fraction",
    "selected_label_entropy_mean", "selected_label_entropy_std", "selected_kl_to_global_label_mean", "selected_distance_m_mean", "selected_distance_m_std",
    "selected_channel_snr_db_mean", "selected_channel_snr_db_std", "selection_score_mean", "selection_score_std",
]

SELECTED_CLIENT_FIELDS = [
    "schema_version", "run_id", "round", "physical_client_id", "virtual_client_id", "selected", "selected_count", "selection_policy", "selection_score", "selection_probability", "selection_reason",
    "num_examples", "label_entropy", "dominant_label", "kl_to_global_label_distribution", "distance_m", "angle_rad", "channel_gain", "channel_snr_db", "pathloss_db", "rate_mbps", "delay_ms", "energy_j", "outage",
]

AGGREGATION_FIELDS = [
    "schema_version", "run_id", "round", "method", "aggregation_mode", "num_results_received", "num_failures", "total_selected_examples", "aggregation_weight_entropy",
    "aggregation_weight_min", "aggregation_weight_max", "global_before_l2", "global_after_l2", "aggregation_delta_l2", "aggregation_delta_linf", "aggregation_delta_cosine",
    "client_update_l2_mean", "client_update_l2_std", "client_update_l2_min", "client_update_l2_max", "client_update_cosine_mean", "client_update_cosine_std",
    "pairwise_update_cosine_mean", "pairwise_update_cosine_std", "update_conflict_fraction", "fedavg_equivalent_delta_l2", "bayes_product_delta_l2", "bayes_vs_fedavg_delta_l2",
    "aggregation_energy_before", "aggregation_energy_after", "aggregation_error_proxy",
]

COMMUNICATION_FIELDS = [
    "schema_version", "run_id", "round", "physical_client_id", "virtual_client_id", "selected", "selection_policy", "distance_m", "angle_rad", "channel_gain", "channel_snr_db", "pathloss_db", "noise_power", "tx_power", "rate_mbps", "delay_ms", "energy_j", "outage", "analog_ota_enabled", "ota_noise_power", "ota_distortion", "ota_mse", "ota_contribution_norm", "digital_enabled", "packet_error_rate", "payload_bytes", "communication_success",
]

SPARSE_COMM_FIELDS = [
    "schema_version", "run_id", "round", "method", "physical_client_id", "virtual_client_id", "num_examples",
    "sparse_comm_enabled", "sparse_metric", "sparse_ratio", "sparse_warmup_rounds",
    "sparse_num_params_total", "sparse_num_params_sent", "sparse_compression_ratio", "sparse_threshold",
    "sparse_score_mean", "sparse_score_p50", "sparse_score_p90",
    "sparse_sent_update_l2", "sparse_dropped_update_l2", "sparse_sent_update_fraction_l2",
    "update_l2_norm", "label_entropy", "kl_to_global_label_distribution",
]

RUN_SUMMARY_FIELDS = [
    "schema_version", "run_id", "method", "dataset", "model", "iid", "balanced", "noniid_alpha", "unbalanced_alpha", "num_devices", "num_virtual_clients", "client_fraction", "num_rounds", "local_epochs", "batch_size", "lr", "seed", "final_global_accuracy", "final_global_loss", "final_global_nll", "final_global_ece", "final_local_accuracy_weighted", "best_global_accuracy", "best_global_accuracy_round", "best_global_ece", "best_global_ece_round", "final_posterior_sigma_mean", "final_posterior_snr_raw_p50", "final_posterior_snr_frac_gt_1", "total_time_sec", "mean_round_time_sec", "final_model_path",
]


PRUNING_EVAL_FIELDS = [
    "schema_version", "run_id", "round", "method", "layer_name", "threshold_type", "threshold_raw", "threshold_db",
    "prune_fraction", "kept_fraction", "num_params_total", "num_params_kept", "num_params_pruned",
    "accuracy_after_prune", "loss_after_prune", "nll_after_prune", "ece_after_prune", "brier_after_prune",
    "mean_confidence_after_prune", "mean_entropy_after_prune", "posterior_snapshot_path",
]

def base_round_row(cfg: RunConfig, run_id: str, round_idx: int) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "round": int(round_idx),
        "method": cfg.method,
        "dataset": cfg.dataset,
        "model": cfg.model,
        "iid": bool(cfg.iid),
        "balanced": bool(cfg.balanced),
        "noniid_alpha": float(cfg.noniid_alpha),
        "unbalanced_alpha": float(cfg.unbalanced_alpha),
        "num_devices": int(cfg.num_devices),
        "num_virtual_clients": int(cfg.num_virtual_clients),
        "client_fraction": float(cfg.client_fraction),
        "local_epochs": int(cfg.local_epochs),
        "batch_size": int(cfg.batch_size),
        "lr": float(cfg.lr),
        "seed": int(cfg.seed),
        "eval_every": int(cfg.eval_every),
        "heavy_eval_every": int(cfg.heavy_eval_every),
        "eval_mc_samples": int(cfg.eval_mc_samples),
        "sparse_comm_enabled": bool(cfg.sparse_comm),
        "sparse_metric": str(cfg.sparse_metric),
        "sparse_ratio": float(cfg.sparse_ratio),
        "sparse_warmup_rounds": int(cfg.sparse_warmup_rounds),
        "sparse_min_keep": int(cfg.sparse_min_keep),
        "posterior_sample_scale": float(cfg.posterior_sample_scale),
        "vi_prior_scale": float(cfg.vi_prior_scale),
        "vi_min_scale": float(cfg.vi_min_scale),
        "vi_particles": int(cfg.vi_particles),
        "vi_aggregation_mode": cfg.bayes_aggregation,
        "ola_prior_lambda": float(cfg.ola_prior_lambda),
        "ola_gamma": float(cfg.precision_init),
        "wireless_policy": cfg.selector,
    }


# ---------------------------------------------------------------------------
# Dataset heterogeneity helpers
# ---------------------------------------------------------------------------
def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def kl_to_global(counts: np.ndarray, global_probs: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    q = np.asarray(global_probs, dtype=np.float64)
    q = np.maximum(q, EPS)
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))


def label_metadata(label_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.asarray(label_counts, dtype=np.float64)
    global_counts = counts.sum(axis=0)
    global_probs = global_counts / max(global_counts.sum(), 1.0)
    ent = np.asarray([entropy_from_counts(row) for row in counts], dtype=np.float64)
    kl = np.asarray([kl_to_global(row, global_probs) for row in counts], dtype=np.float64)
    dominant = np.argmax(counts, axis=1).astype(np.int64) if counts.size else np.asarray([], dtype=np.int64)
    dominant_frac = counts.max(axis=1) / np.maximum(counts.sum(axis=1), 1.0) if counts.size else np.asarray([], dtype=np.float64)
    return ent, kl, dominant, dominant_frac


# ---------------------------------------------------------------------------
# Posterior summaries, SNR, and snapshots
# ---------------------------------------------------------------------------
def posterior_arrays(cfg: RunConfig, payload: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return ``mu, sigma, precision`` for FedAvg/VI/OLA payloads."""
    mu = np.asarray(payload[0], dtype=np.float32).reshape(-1)
    if cfg.method == "fedavg" or len(payload) < 2:
        return mu, None, None
    aux = np.asarray(payload[1], dtype=np.float32).reshape(-1)
    if cfg.method == "vi":
        sigma = np.maximum(aux, float(cfg.vi_min_scale))
        precision = 1.0 / np.maximum(sigma * sigma, float(cfg.vi_min_scale) ** 2)
        return mu, sigma.astype(np.float32), precision.astype(np.float32)
    if cfg.method == "ola":
        precision = np.maximum(aux, float(cfg.precision_floor))
        sigma = np.sqrt(1.0 / precision)
        return mu, sigma.astype(np.float32), precision.astype(np.float32)
    return mu, None, None


def snr_values(mu: np.ndarray, sigma: np.ndarray | None) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if sigma is None:
        return None, None
    snr_raw = np.abs(np.asarray(mu, dtype=np.float64)) / (np.asarray(sigma, dtype=np.float64) + EPS)
    snr_db = 20.0 * np.log10(snr_raw + EPS)
    return snr_raw, snr_db


def posterior_global_metrics(cfg: RunConfig, payload: Sequence[np.ndarray]) -> Dict[str, Any]:
    mu, sigma, precision = posterior_arrays(cfg, payload)
    out: Dict[str, Any] = {
        "posterior_available": 0 if sigma is None else 1,
        "posterior_num_params": int(mu.size),
        "posterior_mu_l2": float(np.linalg.norm(mu.astype(np.float64))),
    }
    abs_mu = np.abs(mu.astype(np.float64))
    out.update(
        {
            "posterior_mu_abs_mean": float(abs_mu.mean()) if abs_mu.size else nan(),
            "posterior_mu_abs_std": float(abs_mu.std()) if abs_mu.size else nan(),
            "posterior_mu_abs_p50": percentile(abs_mu, 50),
            "posterior_mu_abs_p90": percentile(abs_mu, 90),
        }
    )
    out.update(array_stats(sigma, "posterior_sigma"))
    out.update(array_stats(precision, "posterior_precision"))
    if sigma is not None:
        var = np.square(sigma.astype(np.float64))
        out["posterior_var_trace"] = float(var.sum())
        out["posterior_logdet_diag"] = float(np.log(var + EPS).sum())
        out["posterior_entropy_diag_gaussian"] = float(0.5 * np.log(2.0 * np.pi * np.e * var + EPS).sum())
    else:
        out["posterior_var_trace"] = nan()
        out["posterior_logdet_diag"] = nan()
        out["posterior_entropy_diag_gaussian"] = nan()

    snr_raw, snr_db = snr_values(mu, sigma)
    out.update(array_stats(snr_raw, "posterior_snr_raw"))
    out.update(array_stats(snr_db, "posterior_snr_db"))
    if snr_raw is not None and snr_raw.size:
        out.update(
            {
                "posterior_snr_frac_lt_0_5": float(np.mean(snr_raw < 0.5)),
                "posterior_snr_frac_lt_1": float(np.mean(snr_raw < 1.0)),
                "posterior_snr_frac_lt_2": float(np.mean(snr_raw < 2.0)),
                "posterior_snr_frac_lt_5": float(np.mean(snr_raw < 5.0)),
                "posterior_snr_frac_gt_1": float(np.mean(snr_raw > 1.0)),
                "posterior_snr_frac_gt_2": float(np.mean(snr_raw > 2.0)),
                "posterior_snr_frac_gt_5": float(np.mean(snr_raw > 5.0)),
                "effective_params_snr_gt_1": int(np.sum(snr_raw > 1.0)),
                "effective_params_snr_gt_2": int(np.sum(snr_raw > 2.0)),
                "effective_params_snr_gt_5": int(np.sum(snr_raw > 5.0)),
            }
        )
    return out


def posterior_summary_rows(
    cfg: RunConfig,
    run_id: str,
    round_idx: int,
    payload: Sequence[np.ndarray],
    param_meta: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    mu, sigma, precision = posterior_arrays(cfg, payload)
    rows: List[Dict[str, Any]] = []

    def make_row(layer: str, start: int, end: int) -> Dict[str, Any]:
        mu_part = mu[start:end]
        sigma_part = None if sigma is None else sigma[start:end]
        precision_part = None if precision is None else precision[start:end]
        snr_raw, snr_db = snr_values(mu_part, sigma_part)
        row: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "round": int(round_idx),
            "method": cfg.method,
            "scope": "global",
            "physical_client_id": "",
            "virtual_client_id": "",
            "layer_name": layer,
            "param_name": layer,
            "num_params": int(end - start),
            "mu_mean": float(mu_part.mean()) if mu_part.size else nan(),
            "mu_std": float(mu_part.std()) if mu_part.size else nan(),
            "mu_abs_mean": float(np.abs(mu_part).mean()) if mu_part.size else nan(),
            "mu_abs_p50": percentile(np.abs(mu_part), 50),
            "mu_abs_p90": percentile(np.abs(mu_part), 90),
            "mu_l2": float(np.linalg.norm(mu_part.astype(np.float64))) if mu_part.size else nan(),
        }
        row.update({k.replace("posterior_sigma_", "sigma_"): v for k, v in array_stats(sigma_part, "posterior_sigma").items()})
        if sigma_part is not None:
            var = np.square(sigma_part.astype(np.float64))
            row["variance_mean"] = float(var.mean())
            row["variance_sum"] = float(var.sum())
        else:
            row["variance_mean"] = nan()
            row["variance_sum"] = nan()
        row.update({k.replace("posterior_precision_", "precision_"): v for k, v in array_stats(precision_part, "posterior_precision").items()})
        row.update({k.replace("posterior_snr_raw_", "snr_raw_"): v for k, v in array_stats(snr_raw, "posterior_snr_raw").items()})
        row.update({k.replace("posterior_snr_db_", "snr_db_"): v for k, v in array_stats(snr_db, "posterior_snr_db").items()})
        if snr_raw is not None and snr_raw.size:
            row["snr_frac_lt_1"] = float(np.mean(snr_raw < 1.0))
            row["snr_frac_lt_2"] = float(np.mean(snr_raw < 2.0))
            row["snr_frac_gt_1"] = float(np.mean(snr_raw > 1.0))
            row["snr_frac_gt_2"] = float(np.mean(snr_raw > 2.0))
            row["effective_params_snr_gt_1"] = int(np.sum(snr_raw > 1.0))
            row["effective_params_snr_gt_2"] = int(np.sum(snr_raw > 2.0))
        return row

    rows.append(make_row("all", 0, int(mu.size)))
    for item in param_meta:
        rows.append(make_row(str(item["name"]), int(item["start"]), int(item["end"])))
    return rows


def snr_histogram_rows(
    cfg: RunConfig,
    run_id: str,
    round_idx: int,
    payload: Sequence[np.ndarray],
    param_meta: Sequence[Mapping[str, Any]],
    bins: int,
) -> List[Dict[str, Any]]:
    mu, sigma, _precision = posterior_arrays(cfg, payload)
    if sigma is None:
        return []
    rows: List[Dict[str, Any]] = []

    def add_hist(layer_name: str, start: int, end: int, value_space: str, values: np.ndarray) -> None:
        vals = values[np.isfinite(values)]
        if vals.size == 0:
            return
        counts, edges = np.histogram(vals, bins=max(int(bins), 2))
        total = int(counts.sum())
        cum = np.cumsum(counts)
        widths = np.diff(edges)
        for idx, count in enumerate(counts):
            width = float(widths[idx]) if widths[idx] > 0 else 1.0
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "round": int(round_idx),
                    "method": cfg.method,
                    "scope": "global",
                    "physical_client_id": "",
                    "virtual_client_id": "",
                    "layer_name": layer_name,
                    "value_space": value_space,
                    "bin_id": int(idx),
                    "bin_left": float(edges[idx]),
                    "bin_right": float(edges[idx + 1]),
                    "bin_center": float(0.5 * (edges[idx] + edges[idx + 1])),
                    "count": int(count),
                    "density": float(count / max(total * width, EPS)),
                    "cdf": float(cum[idx] / max(total, 1)),
                    "total_count": int(total),
                }
            )

    all_snr_raw, all_snr_db = snr_values(mu, sigma)
    assert all_snr_raw is not None and all_snr_db is not None
    add_hist("all", 0, int(mu.size), "raw", all_snr_raw)
    add_hist("all", 0, int(mu.size), "db", all_snr_db)
    for item in param_meta:
        s, e = int(item["start"]), int(item["end"])
        layer_raw, layer_db = snr_values(mu[s:e], sigma[s:e])
        if layer_raw is not None and layer_db is not None:
            add_hist(str(item["name"]), s, e, "raw", layer_raw)
            add_hist(str(item["name"]), s, e, "db", layer_db)
    return rows


# ---------------------------------------------------------------------------
# Evaluation and calibration
# ---------------------------------------------------------------------------
def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs.clamp_min(EPS)), dim=-1)


@torch.no_grad()
def evaluate_payload(
    cfg: RunConfig,
    payload: Sequence[np.ndarray],
    input_shape: Sequence[int],
    num_classes: int,
    dataloader: DataLoader,
    device: torch.device,
    mc_samples: int,
    eval_scope: str,
    run_id: str,
    round_idx: int,
    posterior_sample_scale: float | None = None,
    physical_client_id: int | str = "",
    virtual_client_id: int | str = "",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate payload with optional posterior Monte Carlo prediction."""
    mu, sigma, _precision = posterior_arrays(cfg, payload)
    sample_scale = float(cfg.posterior_sample_scale if posterior_sample_scale is None else posterior_sample_scale)
    # Keep deterministic mean evaluation as an explicit option.
    # If sample_scale == 0, all posterior samples collapse to theta=mu.
    samples = max(1, int(mc_samples)) if sigma is not None else 1
    bins = max(1, int(cfg.calibration_bins))
    bin_count = np.zeros(bins, dtype=np.int64)
    bin_correct = np.zeros(bins, dtype=np.float64)
    bin_conf = np.zeros(bins, dtype=np.float64)
    bin_nll = np.zeros(bins, dtype=np.float64)
    bin_brier = np.zeros(bins, dtype=np.float64)

    total = 0
    correct = 0
    nll_sum = 0.0
    brier_sum = 0.0
    conf_sum = 0.0
    entropy_sum = 0.0
    expected_entropy_sum = 0.0
    pred_var_sum = 0.0
    pred_var_sq_sum = 0.0
    pred_var_count = 0

    eval_model = model.build_model(cfg, input_shape, num_classes).to(device)
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        probs_samples: List[torch.Tensor] = []
        entropy_samples: List[torch.Tensor] = []
        for s in range(samples):
            if sigma is not None and samples > 1 and sample_scale > 0.0:
                eps = np.random.normal(0.0, 1.0, size=mu.shape).astype(np.float32)
                flat = mu + sample_scale * sigma * eps
            else:
                flat = mu
            model.set_flat_parameters(eval_model, flat, device)
            logits = eval_model(x)
            probs = F.softmax(logits, dim=1)
            probs_samples.append(probs)
            entropy_samples.append(_entropy_from_probs(probs))
        stacked = torch.stack(probs_samples, dim=0)  # S x B x C
        probs_mean = stacked.mean(dim=0)
        if samples > 1:
            ent_samples = torch.stack(entropy_samples, dim=0)
            expected_entropy = ent_samples.mean(dim=0)
            pred_var = stacked.var(dim=0, unbiased=False).mean(dim=1)
            pred_var_sum += float(pred_var.sum().detach().cpu())
            pred_var_sq_sum += float(torch.sum(pred_var.pow(2)).detach().cpu())
            pred_var_count += int(pred_var.numel())
        else:
            expected_entropy = _entropy_from_probs(probs_mean)

        confidence, predicted = torch.max(probs_mean, dim=1)
        correct_tensor = (predicted == y)
        nll = -torch.log(probs_mean[torch.arange(y.numel(), device=device), y].clamp_min(EPS))
        one_hot = F.one_hot(y, num_classes=int(num_classes)).float()
        brier = torch.sum((probs_mean - one_hot).pow(2), dim=1)
        entropy = _entropy_from_probs(probs_mean)

        batch_total = int(y.numel())
        total += batch_total
        correct += int(correct_tensor.sum().item())
        nll_sum += float(nll.sum().detach().cpu())
        brier_sum += float(brier.sum().detach().cpu())
        conf_sum += float(confidence.sum().detach().cpu())
        entropy_sum += float(entropy.sum().detach().cpu())
        expected_entropy_sum += float(expected_entropy.sum().detach().cpu())

        conf_np = confidence.detach().cpu().numpy()
        corr_np = correct_tensor.detach().cpu().numpy().astype(np.float64)
        nll_np = nll.detach().cpu().numpy()
        brier_np = brier.detach().cpu().numpy()
        ids = np.minimum((conf_np * bins).astype(np.int64), bins - 1)
        for bid in range(bins):
            mask = ids == bid
            if np.any(mask):
                c = int(mask.sum())
                bin_count[bid] += c
                bin_correct[bid] += float(corr_np[mask].sum())
                bin_conf[bid] += float(conf_np[mask].sum())
                bin_nll[bid] += float(nll_np[mask].sum())
                bin_brier[bid] += float(brier_np[mask].sum())

    accuracy = correct / max(total, 1)
    nll_mean = nll_sum / max(total, 1)
    brier_mean = brier_sum / max(total, 1)
    conf_mean = conf_sum / max(total, 1)
    entropy_mean = entropy_sum / max(total, 1)
    expected_entropy_mean = expected_entropy_sum / max(total, 1)
    mi = entropy_mean - expected_entropy_mean if samples > 1 else nan()
    if pred_var_count > 0:
        pred_var_mean = pred_var_sum / pred_var_count
        pred_var_std = math.sqrt(max(pred_var_sq_sum / pred_var_count - pred_var_mean**2, 0.0))
    else:
        pred_var_mean = nan()
        pred_var_std = nan()

    ece = 0.0
    mce = 0.0
    calibration_rows: List[Dict[str, Any]] = []
    for bid in range(bins):
        count = int(bin_count[bid])
        if count > 0:
            bin_acc = float(bin_correct[bid] / count)
            bin_c = float(bin_conf[bid] / count)
            gap = abs(bin_acc - bin_c)
            contrib = gap * count / max(total, 1)
            ece += contrib
            mce = max(mce, gap)
            bin_nll_mean = float(bin_nll[bid] / count)
            bin_brier_mean = float(bin_brier[bid] / count)
        else:
            bin_acc = bin_c = gap = contrib = bin_nll_mean = bin_brier_mean = nan()
        calibration_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "round": int(round_idx),
                "method": cfg.method,
                "eval_scope": eval_scope,
                "physical_client_id": physical_client_id,
                "virtual_client_id": virtual_client_id,
                "bin_id": int(bid),
                "bin_left": float(bid / bins),
                "bin_right": float((bid + 1) / bins),
                "bin_count": count,
                "bin_accuracy": bin_acc,
                "bin_confidence": bin_c,
                "bin_gap": gap,
                "ece_contribution": contrib,
                "nll_mean": bin_nll_mean,
                "brier_mean": bin_brier_mean,
            }
        )

    metrics = {
        "accuracy": float(accuracy),
        "error_rate": float(1.0 - accuracy),
        "loss": float(nll_mean),
        "nll": float(nll_mean),
        "brier": float(brier_mean),
        "ece": float(ece),
        "mce": float(mce),
        "mean_confidence": float(conf_mean),
        "mean_entropy": float(entropy_mean),
        "num_eval_examples": int(total),
        "mc_samples": int(samples),
        "posterior_sample_scale": float(sample_scale) if sigma is not None else nan(),
        "predictive_entropy": float(entropy_mean),
        "expected_entropy": float(expected_entropy_mean) if samples > 1 else nan(),
        "mutual_information": float(mi),
        "aleatoric_uncertainty": float(expected_entropy_mean) if samples > 1 else nan(),
        "epistemic_uncertainty": float(mi),
        "predictive_variance_mean": float(pred_var_mean),
        "predictive_variance_std": float(pred_var_std),
    }
    return metrics, calibration_rows


def prefixed(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def summarize_client_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not rows:
        return out

    def values(key: str) -> np.ndarray:
        vals = [float_or_nan(r.get(key, nan())) for r in rows]
        arr = np.asarray(vals, dtype=np.float64)
        return arr[np.isfinite(arr)]

    for key, out_prefix in [
        ("train_loss", "train_loss"), ("task_loss", "task_loss"), ("prior_loss", "prior_loss"), ("regularization_loss", "regularization_loss"),
        ("update_l2_norm", "client_update_l2"), ("update_cosine_to_global", "client_update_cosine"),
        ("drift_from_global_after_l2", "client_drift_from_global_l2"),
        ("drift_from_global_after_cosine", "client_drift_from_global_cosine"),
        ("vi_elbo_loss", "vi_elbo_loss"), ("vi_kl_loss", "vi_kl_loss"), ("vi_kl_loss_per_example", "vi_kl_loss_per_example"),
        ("vi_kl_per_param", "vi_kl_per_param"), ("vi_likelihood_loss", "vi_likelihood_loss"),
        ("vi_complexity_cost", "vi_complexity_cost"), ("vi_effective_lr", "vi_effective_lr"),
        ("ola_prior_loss", "ola_prior_loss"), ("ola_prior_loss_raw", "ola_prior_loss_raw"), ("ola_regularization_loss_raw", "ola_regularization_loss_raw"),
        ("ola_task_loss", "ola_task_loss"), ("ola_fisher_mean", "ola_fisher"),
        ("ola_precision_mean", "ola_precision"), ("ola_sigma_mean", "ola_sigma"),
        ("sparse_num_params_sent", "sparse_num_params_sent"),
        ("sparse_compression_ratio", "sparse_compression_ratio"),
        ("sparse_threshold", "sparse_threshold"),
        ("sparse_score_mean", "sparse_score_mean"),
        ("sparse_score_p50", "sparse_score_p50"),
        ("sparse_score_p90", "sparse_score_p90"),
    ]:
        arr = values(key)
        if arr.size:
            out[f"{out_prefix}_mean"] = float(arr.mean())
            out[f"{out_prefix}_std"] = float(arr.std())
            out[f"{out_prefix}_min"] = float(arr.min())
            out[f"{out_prefix}_p10"] = percentile(arr, 10)
            out[f"{out_prefix}_p50"] = percentile(arr, 50)
            out[f"{out_prefix}_p90"] = percentile(arr, 90)
            out[f"{out_prefix}_max"] = float(arr.max())
    # Friendlier aliases for round-level sparse score summaries. The older
    # automatic names remain in the CSV for backward compatibility.
    for src, dst in [
        ("sparse_score_mean_mean", "sparse_score_mean_avg"),
        ("sparse_score_p50_mean", "sparse_score_p50_avg"),
        ("sparse_score_p90_mean", "sparse_score_p90_avg"),
    ]:
        if src in out:
            out[dst] = out[src]
    if "ola_prior_loss_mean" in out and "ola_task_loss_mean" in out:
        denom = float_or_nan(out.get("ola_task_loss_mean", nan()))
        num = float_or_nan(out.get("ola_prior_loss_mean", nan()))
        out["ola_prior_task_ratio"] = float(num / denom) if np.isfinite(num) and np.isfinite(denom) and abs(denom) > EPS else nan()
    return out


def summarize_eval_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"local_eval_count": int(len(rows))}
    if not rows:
        return out

    def vals(key: str) -> np.ndarray:
        arr = np.asarray([float_or_nan(r.get(key, nan())) for r in rows], dtype=np.float64)
        return arr[np.isfinite(arr)]

    acc = vals("local_accuracy")
    loss = vals("local_loss")
    if acc.size:
        out.update(
            {
                "local_accuracy_mean": float(acc.mean()),
                "local_accuracy_std": float(acc.std()),
                "local_accuracy_min": float(acc.min()),
                "local_accuracy_p10": percentile(acc, 10),
                "local_accuracy_p25": percentile(acc, 25),
                "local_accuracy_p50": percentile(acc, 50),
                "local_accuracy_p75": percentile(acc, 75),
                "local_accuracy_p90": percentile(acc, 90),
                "local_accuracy_max": float(acc.max()),
                "local_accuracy_weighted": weighted_mean(rows, "local_accuracy", "num_examples_train"),
            }
        )
    if loss.size:
        out.update(
            {
                "local_loss_mean": float(loss.mean()),
                "local_loss_std": float(loss.std()),
                "local_loss_min": float(loss.min()),
                "local_loss_p50": percentile(loss, 50),
                "local_loss_max": float(loss.max()),
                "local_loss_weighted": weighted_mean(rows, "local_loss", "num_examples_train"),
            }
        )
    for key in ["local_ece", "local_nll", "local_brier", "local_forgetting_proxy", "local_global_accuracy_gap", "local_global_loss_gap"]:
        arr = vals(key)
        if arr.size:
            out[f"{key}_mean"] = float(arr.mean())
            out[f"{key}_std"] = float(arr.std())
            if key == "local_forgetting_proxy":
                out["local_forgetting_proxy_weighted"] = weighted_mean(rows, key, "num_examples_train")
    # Normalize names to the metrics.csv schema.
    if "local_nll_mean" not in out and vals("local_nll").size:
        out["local_nll_mean"] = float(vals("local_nll").mean())
    if "local_brier_mean" not in out and vals("local_brier").size:
        out["local_brier_mean"] = float(vals("local_brier").mean())
    if "local_ece_mean" not in out and vals("local_ece").size:
        out["local_ece_mean"] = float(vals("local_ece").mean())
        out["local_ece_std"] = float(vals("local_ece").std())
    return out


def aggregation_weight_stats(weights: Sequence[float]) -> Dict[str, Any]:
    w = np.asarray(weights, dtype=np.float64)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return {"aggregation_weight_entropy": nan(), "aggregation_weight_min": nan(), "aggregation_weight_max": nan()}
    p = w / w.sum()
    return {
        "aggregation_weight_entropy": float(-(p * np.log(p + EPS)).sum()),
        "aggregation_weight_min": float(p.min()),
        "aggregation_weight_max": float(p.max()),
    }


def vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return nan()
    return float(np.dot(a, b) / denom)
