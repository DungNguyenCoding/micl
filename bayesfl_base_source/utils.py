"""Offline plotting utilities for Bayesian FL runs.

Training writes CSV/LOG/PT files only. Run this module separately when you
want PNG plots. This version intentionally avoids pandas so it is more robust
against NumPy/Pandas binary-version issues.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dataset
import model
import observability as obs
from config import RunConfig, str2bool


Rows = List[Dict[str, str]]


def _read_csv_rows(csv_path: str | Path) -> Rows:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _column_as_float(rows: Rows, column: str, csv_path: str | Path) -> list[float]:
    if not rows:
        return []
    if column not in rows[0]:
        raise ValueError(
            f"Column {column!r} not found in {csv_path}. "
            f"Available columns: {list(rows[0].keys())}"
        )
    values: list[float] = []
    for row in rows:
        value = row.get(column, "")
        if value != "":
            values.append(float(value))
    return values


def _save_figure(fig: plt.Figure, out_path: Path, dpi: int = 180) -> Path:
    """Save without tight_layout because some broken NumPy installs crash there."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _resolve_history_path(run_spec: str) -> tuple[str, Path]:
    """Parse either label=run_path or run_path.

    run_path can be a run directory containing metrics.csv, or a direct path to
    metrics.csv.
    """
    if "=" in run_spec:
        label, raw_path = run_spec.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
    else:
        path = Path(run_spec.strip())
        label = path.parent.name if path.name == "metrics.csv" else path.name

    if not label:
        raise ValueError(f"Empty label in run spec: {run_spec!r}")

    history_path = path if path.name == "metrics.csv" else path / "metrics.csv"
    if not history_path.exists():
        raise FileNotFoundError(
            f"Could not find metrics.csv for run {label!r}. Checked: {history_path}"
        )
    return label, history_path


def _load_metric_series(history_csv: str | Path, metric: str) -> tuple[list[float], list[float]]:
    rows = _read_csv_rows(history_csv)
    rounds = _column_as_float(rows, "round", history_csv)
    values = _column_as_float(rows, metric, history_csv)
    if len(rounds) != len(values):
        min_len = min(len(rounds), len(values))
        rounds = rounds[:min_len]
        values = values[:min_len]
    return rounds, values


def plot_metric(history_csv: str | Path, metric: str, output_dir: str | Path) -> Path:
    rounds, values = _load_metric_series(history_csv, metric)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, values, marker="o")
    ax.set_xlabel("Round")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} / round")
    ax.grid(True, alpha=0.3)

    out_path = out_dir / f"{metric}_round.png"
    return _save_figure(fig, out_path)


def plot_mixed_metrics(
    run_specs: Sequence[str],
    metrics: Sequence[str],
    output_dir: str | Path,
    filename_prefix: str = "mix",
) -> list[Path]:
    """Plot one overlay PNG per metric for selected experiment folders."""
    if not run_specs:
        raise ValueError("At least one --runs entry is required")
    if not metrics:
        raise ValueError("At least one --metrics entry is required")

    runs = [_resolve_history_path(spec) for spec in run_specs]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plotted = 0

        for label, history_path in runs:
            try:
                rounds, values = _load_metric_series(history_path, metric)
            except ValueError as exc:
                print(f"[skip] {label}: {exc}")
                continue

            if not rounds or not values:
                print(f"[skip] {label}: no data for metric {metric!r}")
                continue

            ax.plot(rounds, values, marker="o", markersize=3, linewidth=1.5, label=label)
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            raise ValueError(f"No runs contained metric {metric!r}")

        ax.set_xlabel("Round")
        ax.set_ylabel(metric)
        ax.set_title(f"Mixed {metric} / round")
        ax.grid(True, alpha=0.3)
        ax.legend()

        out_path = out_dir / f"{filename_prefix}_{metric}_round.png"
        output_paths.append(_save_figure(fig, out_path))

    return output_paths


def plot_active_clients(selection_csv: str | Path, output_dir: str | Path) -> Path:
    """Plot selected-client count per round.

    Accepts either selection_summary.csv (one row per round) or
    selected_clients.csv (one row per selected device per round).
    """
    rows = _read_csv_rows(selection_csv)
    grouped: dict[float, list[float]] = {}
    if rows and "selected_count" in rows[0]:
        for row in rows:
            if row.get("round", "") == "":
                continue
            rnd = float(row["round"])
            val = float(row.get("selected_count", "1") or 1)
            grouped.setdefault(rnd, []).append(val)
        rounds = sorted(grouped)
        selected = [max(grouped[r]) for r in rounds]
    else:
        for row in rows:
            if row.get("round", "") == "":
                continue
            rnd = float(row["round"])
            grouped.setdefault(rnd, []).append(1.0)
        rounds = sorted(grouped)
        selected = [sum(grouped[r]) for r in rounds]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, selected, marker="o")
    ax.set_xlabel("Round")
    ax.set_ylabel("Selected physical clients")
    ax.set_title("Selected clients / round")
    ax.grid(True, alpha=0.3)

    out_path = out_dir / "selected_clients_round.png"
    return _save_figure(fig, out_path)


def plot_device_radar(device_summary_csv: str | Path, output_dir: str | Path) -> Path:
    rows = _read_csv_rows(device_summary_csv)
    angle_rad = _column_as_float(rows, "angle_rad", device_summary_csv)
    radius_m = _column_as_float(rows, "radius_m", device_summary_csv)
    virtual_client_id = _column_as_float(rows, "virtual_client_id", device_summary_csv)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    scatter = ax.scatter(angle_rad, radius_m, c=virtual_client_id, s=18, alpha=0.75)
    ax.set_title("Physical devices around central server")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    fig.colorbar(scatter, ax=ax, pad=0.1, label="Flower virtual client")

    out_path = out_dir / "device_distribution_radar.png"
    return _save_figure(fig, out_path)



def _filter_rows(rows: Rows, **filters: object) -> Rows:
    out: Rows = []
    for row in rows:
        keep = True
        for key, expected in filters.items():
            if expected is None:
                continue
            if str(row.get(key, "")) != str(expected):
                keep = False
                break
        if keep:
            out.append(row)
    return out


def plot_snr_histogram(
    snr_csv: str | Path,
    output_dir: str | Path,
    round_idx: int | None = None,
    layer_name: str = "all",
    value_space: str = "db",
) -> list[Path]:
    """Plot SNR density and CDF from snr_histograms.csv."""
    rows = _read_csv_rows(snr_csv)
    if round_idx is None:
        available = [int(float(r["round"])) for r in rows if r.get("round", "") != ""]
        if not available:
            raise ValueError(f"No round values found in {snr_csv}")
        round_idx = max(available)
    rows = _filter_rows(rows, round=round_idx, layer_name=layer_name, value_space=value_space)
    if not rows:
        raise ValueError(f"No SNR rows for round={round_idx}, layer={layer_name}, value_space={value_space}")
    rows = sorted(rows, key=lambda r: int(float(r["bin_id"])))
    centers = [float(r["bin_center"]) for r in rows]
    density = [float(r["density"]) for r in rows]
    cdf = [float(r["cdf"]) for r in rows]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"round_{round_idx:04d}_{layer_name}_{value_space}"
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(centers, density)
    ax.set_xlabel(f"SNR ({value_space})")
    ax.set_ylabel("Density")
    ax.set_title(f"SNR density / {layer_name} / round {round_idx}")
    ax.grid(True, alpha=0.3)
    paths.append(_save_figure(fig, out_dir / f"snr_density_{suffix}.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(centers, cdf)
    ax.set_xlabel(f"SNR ({value_space})")
    ax.set_ylabel("CDF")
    ax.set_title(f"SNR CDF / {layer_name} / round {round_idx}")
    ax.grid(True, alpha=0.3)
    paths.append(_save_figure(fig, out_dir / f"snr_cdf_{suffix}.png"))
    return paths


def plot_calibration(calibration_csv: str | Path, output_dir: str | Path, round_idx: int | None = None, eval_scope: str = "global_test") -> Path:
    """Plot a reliability diagram from calibration_bins.csv."""
    rows = _read_csv_rows(calibration_csv)
    if round_idx is None:
        available = [int(float(r["round"])) for r in rows if r.get("round", "") != ""]
        if not available:
            raise ValueError(f"No round values found in {calibration_csv}")
        round_idx = max(available)
    rows = _filter_rows(rows, round=round_idx, eval_scope=eval_scope)
    if not rows:
        raise ValueError(f"No calibration rows for round={round_idx}, eval_scope={eval_scope}")
    rows = sorted(rows, key=lambda r: int(float(r["bin_id"])))
    centers = [(float(r["bin_left"]) + float(r["bin_right"])) / 2.0 for r in rows]
    acc = [float(r["bin_accuracy"]) if r.get("bin_accuracy", "") != "" else 0.0 for r in rows]
    conf = [float(r["bin_confidence"]) if r.get("bin_confidence", "") != "" else 0.0 for r in rows]
    width = (float(rows[0]["bin_right"]) - float(rows[0]["bin_left"])) if rows else 0.05

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(centers, acc, width=width * 0.9, alpha=0.7, label="accuracy")
    ax.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    ax.plot(centers, conf, marker="o", linewidth=1.5, label="confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram / round {round_idx}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_figure(fig, out_dir / f"calibration_round_{round_idx:04d}_{eval_scope}.png")


# -----------------------------------------------------------------------------
# Method-characteristic plotting helpers
# -----------------------------------------------------------------------------

def _to_float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        val = float(value)
    except Exception:
        return None
    # NaN guard without importing numpy
    if val != val:
        return None
    return val


def _series_for_metric(rows: Rows, metric: str) -> tuple[list[float], list[float]]:
    rounds: list[float] = []
    values: list[float] = []
    if not rows or metric not in rows[0]:
        return rounds, values
    for row in rows:
        rnd = _to_float_or_none(row.get("round"))
        val = _to_float_or_none(row.get(metric))
        if rnd is None or val is None:
            continue
        rounds.append(rnd)
        values.append(val)
    return rounds, values


def _plot_series_group(
    rows: Rows,
    metrics: Sequence[str],
    output_dir: str | Path,
    filename: str,
    title: str,
    ylabel: str | None = None,
) -> Path | None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = 0
    for metric in metrics:
        rounds, values = _series_for_metric(rows, metric)
        if not rounds or not values:
            print(f"[skip] characteristic metric not available: {metric}")
            continue
        ax.plot(rounds, values, marker="o", markersize=2.5, linewidth=1.4, label=metric)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(f"[skip] no characteristic data for {filename}")
        return None

    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel or "Value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return _save_figure(fig, out_dir / filename)


def _plot_computed_metric(
    rows: Rows,
    output_dir: str | Path,
    filename: str,
    title: str,
    ylabel: str,
    series: Sequence[tuple[str, str, str, str]],
) -> Path | None:
    """Plot computed series.

    Each series tuple is (label, numerator_metric, denominator_metric, operation).
    Supported operations: gap, ratio, diff, abs_gap.
    For gap/diff, value = numerator - denominator.
    For ratio, value = numerator / denominator.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = 0

    for label, metric_a, metric_b, op in series:
        rounds: list[float] = []
        values: list[float] = []
        if not rows or metric_a not in rows[0] or metric_b not in rows[0]:
            print(f"[skip] computed metric requires {metric_a} and {metric_b}")
            continue
        for row in rows:
            rnd = _to_float_or_none(row.get("round"))
            a = _to_float_or_none(row.get(metric_a))
            b = _to_float_or_none(row.get(metric_b))
            if rnd is None or a is None or b is None:
                continue
            if op in {"gap", "diff"}:
                val = a - b
            elif op == "abs_gap":
                val = abs(a - b)
            elif op == "ratio":
                if abs(b) < 1e-12:
                    continue
                val = a / b
            else:
                raise ValueError(f"Unknown computed op: {op}")
            rounds.append(rnd)
            values.append(val)

        if not rounds:
            print(f"[skip] no data for computed series {label}")
            continue
        ax.plot(rounds, values, marker="o", markersize=2.5, linewidth=1.4, label=label)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(f"[skip] no computed characteristic data for {filename}")
        return None

    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return _save_figure(fig, out_dir / filename)


def _infer_round(rows: Rows, preferred_round: int | None) -> int | None:
    if preferred_round is not None:
        return int(preferred_round)
    rounds = [int(float(r.get("round", "nan"))) for r in rows if _to_float_or_none(r.get("round")) is not None]
    return max(rounds) if rounds else None


def _posterior_rows_for_round(posterior_csv: str | Path, round_idx: int | None, scope: str = "global") -> tuple[Rows, int]:
    rows = _read_csv_rows(posterior_csv)
    available = sorted(
        {int(float(r["round"])) for r in rows if r.get("round", "") != "" and (not scope or r.get("scope", "") == scope)}
    )
    if not available:
        available = sorted({int(float(r["round"])) for r in rows if r.get("round", "") != ""})
    if not available:
        raise ValueError(f"No round values found in {posterior_csv}")

    if round_idx is None:
        chosen_round = max(available)
    elif int(round_idx) in available:
        chosen_round = int(round_idx)
    else:
        # posterior_summary is usually written only at save_posterior_every rounds.
        # For requested best-accuracy/ECE rounds, use the nearest saved posterior round.
        chosen_round = min(available, key=lambda r: (abs(r - int(round_idx)), r > int(round_idx)))
        print(f"[info] posterior round {round_idx} not available in {posterior_csv}; using nearest saved round {chosen_round}")

    filtered = _filter_rows(rows, round=chosen_round, scope=scope)
    if not filtered:
        filtered = _filter_rows(rows, round=chosen_round)
    return filtered, int(chosen_round)


def plot_posterior_layer_metric(
    posterior_csv: str | Path,
    output_dir: str | Path,
    metric: str,
    round_idx: int | None = None,
    scope: str = "global",
    filename_prefix: str = "posterior_layer",
) -> Path | None:
    """Bar plot for one posterior_summary.csv metric across layers."""
    rows, round_idx = _posterior_rows_for_round(posterior_csv, round_idx, scope)
    usable = []
    for row in rows:
        layer = row.get("layer_name", "")
        if not layer or layer == "all":
            continue
        val = _to_float_or_none(row.get(metric))
        if val is None:
            continue
        usable.append((layer, val))

    if not usable:
        print(f"[skip] no layer posterior metric {metric} for round={round_idx}")
        return None

    # Keep layer order from CSV and shorten labels for readability.
    labels = [x[0] for x in usable]
    values = [x[1] for x in usable]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = list(range(len(labels)))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"Layer-wise {metric} / round {round_idx}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=0.35, left=0.12, right=0.98, top=0.90)
    return _save_figure(fig, out_dir / f"{filename_prefix}_{metric}_round_{round_idx:04d}.png")


def _client_rows_for_round(client_csv: str | Path, round_idx: int | None) -> tuple[Rows, int]:
    rows = _read_csv_rows(client_csv)
    if round_idx is None:
        available = [int(float(r["round"])) for r in rows if r.get("round", "") != ""]
        if not available:
            raise ValueError(f"No round values found in {client_csv}")
        round_idx = max(available)
    filtered = _filter_rows(rows, round=round_idx)
    return filtered, int(round_idx)


def plot_client_boxplots(
    client_csv: str | Path,
    output_dir: str | Path,
    metrics: Sequence[str],
    round_idx: int | None = None,
    filename_prefix: str = "client_distribution",
) -> list[Path]:
    """Create boxplots over client-level metrics for one round."""
    rows, round_idx = _client_rows_for_round(client_csv, round_idx)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for metric in metrics:
        values = []
        if rows and metric not in rows[0]:
            print(f"[skip] client metric not available: {metric}")
            continue
        for row in rows:
            val = _to_float_or_none(row.get(metric))
            if val is not None:
                values.append(val)
        if not values:
            print(f"[skip] no client values for metric {metric} at round {round_idx}")
            continue
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.boxplot(values, vert=True, showmeans=True)
        ax.set_xticks([1])
        ax.set_xticklabels([metric], rotation=15, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"Client distribution / {metric} / round {round_idx}")
        ax.grid(True, axis="y", alpha=0.3)
        fig.subplots_adjust(bottom=0.22, left=0.18, right=0.95, top=0.88)
        paths.append(_save_figure(fig, out_dir / f"{filename_prefix}_{metric}_round_{round_idx:04d}.png"))
    return paths


def plot_ola_characteristics(
    run_dir: str | Path,
    output_dir: str | Path,
    final_round: int | None = None,
    best_round: int | None = None,
    best_ece_round: int | None = None,
) -> list[Path]:
    """Generate OLA-specific plots from one OLA run directory."""
    run_dir = Path(run_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv_rows(run_dir / "metrics.csv")
    final_round = _infer_round(rows, final_round)
    paths: list[Path] = []

    plot_specs = [
        ("ola_mean_vs_mc_accuracy.png", "OLA posterior mean vs MC accuracy", "Accuracy", ["global_accuracy", "global_mean_accuracy", "global_mc_accuracy"]),
        ("ola_mean_vs_mc_loss.png", "OLA posterior mean vs MC loss", "Loss", ["global_loss", "global_mean_loss", "global_mc_loss"]),
        ("ola_uncertainty_decomposition.png", "OLA predictive uncertainty decomposition", "Uncertainty", ["global_predictive_entropy", "global_expected_entropy", "global_mutual_information", "global_epistemic_uncertainty", "global_aleatoric_uncertainty"]),
        ("ola_loss_decomposition.png", "OLA task/prior/train loss decomposition", "Loss", ["ola_task_loss_mean", "ola_prior_loss_mean", "train_loss"]),
        ("ola_fisher_precision.png", "OLA Fisher and precision summaries", "Value", ["ola_fisher_mean", "ola_precision_mean", "posterior_precision_mean", "posterior_product_precision_mean"]),
        ("ola_sigma_uncertainty.png", "OLA sigma / posterior uncertainty", "Sigma", ["ola_sigma_mean", "ola_sigma_p50", "ola_sigma_p90", "posterior_sigma_mean", "posterior_sigma_p50", "posterior_sigma_p90"]),
        ("ola_snr_summary.png", "OLA posterior SNR summaries", "SNR", ["posterior_snr_raw_mean", "posterior_snr_raw_p50", "posterior_snr_raw_p90", "posterior_snr_frac_gt_1", "posterior_snr_frac_lt_1"]),
        ("ola_global_local_accuracy.png", "OLA global vs local accuracy", "Accuracy", ["global_accuracy", "local_accuracy_weighted", "local_accuracy_mean"]),
        ("ola_calibration_confidence_entropy.png", "OLA calibration, confidence, entropy", "Value", ["global_ece", "global_mean_ece", "global_mc_ece", "global_mean_confidence", "global_mean_entropy"]),
    ]
    for filename, title, ylabel, metrics in plot_specs:
        p = _plot_series_group(rows, metrics, out_dir, filename, title, ylabel)
        if p is not None:
            paths.append(p)

    computed_specs = [
        ("ola_mean_mc_accuracy_gap.png", "OLA posterior mean - MC accuracy gap", "Accuracy gap", [("mean_acc - mc_acc", "global_mean_accuracy", "global_mc_accuracy", "gap")]),
        ("ola_mean_mc_loss_gap.png", "OLA posterior mean - MC loss gap", "Loss gap", [("mean_loss - mc_loss", "global_mean_loss", "global_mc_loss", "gap")]),
        ("ola_local_global_accuracy_gap.png", "OLA local - global accuracy gap", "Accuracy gap", [("local_weighted - global", "local_accuracy_weighted", "global_accuracy", "gap")]),
        ("ola_prior_task_ratio.png", "OLA prior/task loss ratio", "Ratio", [("prior_loss / task_loss", "ola_prior_loss_mean", "ola_task_loss_mean", "ratio")]),
    ]
    for filename, title, ylabel, series in computed_specs:
        p = _plot_computed_metric(rows, out_dir, filename, title, ylabel, series)
        if p is not None:
            paths.append(p)

    # Layer-wise posterior characteristics.
    posterior_csv = run_dir / "posterior_summary.csv"
    if posterior_csv.exists():
        for rnd_name, rnd in [("final", final_round), ("best_acc", best_round), ("best_ece", best_ece_round)]:
            if rnd is None:
                continue
            layer_dir = out_dir / f"layer_{rnd_name}"
            for metric in ["sigma_mean", "precision_mean", "snr_raw_p50", "snr_raw_p90"]:
                p = plot_posterior_layer_metric(posterior_csv, layer_dir, metric, rnd, filename_prefix=f"ola_layer_{rnd_name}")
                if p is not None:
                    paths.append(p)

    # Client-level distributions.
    client_csv = run_dir / "client_train_metrics.csv"
    if client_csv.exists():
        for rnd_name, rnd in [("final", final_round), ("best_acc", best_round)]:
            if rnd is None:
                continue
            paths.extend(
                plot_client_boxplots(
                    client_csv,
                    out_dir / f"client_{rnd_name}",
                    ["task_loss", "prior_loss", "update_l2_norm", "ola_fisher_mean", "ola_precision_mean", "ola_sigma_mean", "ola_snr_raw_p50"],
                    rnd,
                    filename_prefix=f"ola_client_{rnd_name}",
                )
            )
    return paths


def plot_vi_characteristics(
    run_dir: str | Path,
    output_dir: str | Path,
    final_round: int | None = None,
    best_round: int | None = None,
    best_ece_round: int | None = None,
) -> list[Path]:
    """Generate VI-specific plots from one VI run directory."""
    run_dir = Path(run_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv_rows(run_dir / "metrics.csv")
    final_round = _infer_round(rows, final_round)
    paths: list[Path] = []

    plot_specs = [
        ("vi_mean_vs_mc_accuracy.png", "VI posterior mean vs MC accuracy", "Accuracy", ["global_accuracy", "global_mean_accuracy", "global_mc_accuracy"]),
        ("vi_mean_vs_mc_loss.png", "VI posterior mean vs MC loss", "Loss", ["global_loss", "global_mean_loss", "global_mc_loss"]),
        ("vi_uncertainty_decomposition.png", "VI predictive uncertainty decomposition", "Uncertainty", ["global_predictive_entropy", "global_expected_entropy", "global_mutual_information", "global_epistemic_uncertainty", "global_aleatoric_uncertainty"]),
        ("vi_elbo_decomposition.png", "VI ELBO/KL/likelihood decomposition", "Loss", ["vi_elbo_loss_mean", "vi_kl_loss_mean", "vi_likelihood_loss_mean", "vi_complexity_cost_mean", "train_loss"]),
        ("vi_scale_uncertainty.png", "VI scale / posterior uncertainty", "Scale/Sigma", ["vi_scale_mean", "vi_scale_p50", "vi_scale_p90", "posterior_sigma_mean", "posterior_sigma_p50", "posterior_sigma_p90"]),
        ("vi_snr_summary.png", "VI posterior SNR summaries", "SNR", ["posterior_snr_raw_mean", "posterior_snr_raw_p50", "posterior_snr_raw_p90", "posterior_snr_frac_gt_1", "posterior_snr_frac_lt_1"]),
        ("vi_global_local_accuracy.png", "VI global vs local accuracy", "Accuracy", ["global_accuracy", "local_accuracy_weighted", "local_accuracy_mean"]),
        ("vi_calibration_confidence_entropy.png", "VI calibration, confidence, entropy", "Value", ["global_ece", "global_mean_ece", "global_mc_ece", "global_mean_confidence", "global_mean_entropy"]),
    ]
    for filename, title, ylabel, metrics in plot_specs:
        p = _plot_series_group(rows, metrics, out_dir, filename, title, ylabel)
        if p is not None:
            paths.append(p)

    computed_specs = [
        ("vi_mean_mc_accuracy_gap.png", "VI posterior mean - MC accuracy gap", "Accuracy gap", [("mean_acc - mc_acc", "global_mean_accuracy", "global_mc_accuracy", "gap")]),
        ("vi_mean_mc_loss_gap.png", "VI posterior mean - MC loss gap", "Loss gap", [("mean_loss - mc_loss", "global_mean_loss", "global_mc_loss", "gap")]),
        ("vi_local_global_accuracy_gap.png", "VI local - global accuracy gap", "Accuracy gap", [("local_weighted - global", "local_accuracy_weighted", "global_accuracy", "gap")]),
        ("vi_kl_likelihood_ratio.png", "VI KL/likelihood loss ratio", "Ratio", [("KL / likelihood", "vi_kl_loss_mean", "vi_likelihood_loss_mean", "ratio")]),
    ]
    for filename, title, ylabel, series in computed_specs:
        p = _plot_computed_metric(rows, out_dir, filename, title, ylabel, series)
        if p is not None:
            paths.append(p)

    posterior_csv = run_dir / "posterior_summary.csv"
    if posterior_csv.exists():
        for rnd_name, rnd in [("final", final_round), ("best_acc", best_round), ("best_ece", best_ece_round)]:
            if rnd is None:
                continue
            layer_dir = out_dir / f"layer_{rnd_name}"
            for metric in ["sigma_mean", "precision_mean", "snr_raw_p50", "snr_raw_p90"]:
                p = plot_posterior_layer_metric(posterior_csv, layer_dir, metric, rnd, filename_prefix=f"vi_layer_{rnd_name}")
                if p is not None:
                    paths.append(p)

    client_csv = run_dir / "client_train_metrics.csv"
    if client_csv.exists():
        for rnd_name, rnd in [("final", final_round), ("best_acc", best_round)]:
            if rnd is None:
                continue
            paths.extend(
                plot_client_boxplots(
                    client_csv,
                    out_dir / f"client_{rnd_name}",
                    ["train_loss", "update_l2_norm", "vi_elbo_loss", "vi_kl_loss", "vi_likelihood_loss", "vi_scale_mean", "vi_snr_raw_p50"],
                    rnd,
                    filename_prefix=f"vi_client_{rnd_name}",
                )
            )
    return paths


def plot_characteristics(
    run_dir: str | Path,
    method: str,
    output_dir: str | Path,
    final_round: int | None = None,
    best_round: int | None = None,
    best_ece_round: int | None = None,
) -> list[Path]:
    method = method.lower().strip()
    if method == "ola":
        return plot_ola_characteristics(run_dir, output_dir, final_round, best_round, best_ece_round)
    if method == "vi":
        return plot_vi_characteristics(run_dir, output_dir, final_round, best_round, best_ece_round)
    raise ValueError("--method for characteristics must be 'ola' or 'vi'")



# ---------------------------------------------------------------------------
# BBB-style post-hoc SNR pruning for Bayesian FL
# ---------------------------------------------------------------------------

def _parse_config_csv(run_dir: str | Path) -> RunConfig:
    """Load a RunConfig from config.csv with best-effort type restoration."""
    cfg = RunConfig()
    path = Path(run_dir) / "config.csv"
    if not path.exists():
        return cfg
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = row.get("key", "")
            value = row.get("value", "")
            if not key or not hasattr(cfg, key):
                continue
            current = getattr(cfg, key)
            if value == "":
                continue
            try:
                if isinstance(current, bool):
                    parsed = str2bool(value)
                elif isinstance(current, int) and not isinstance(current, bool):
                    parsed = int(float(value))
                elif isinstance(current, float):
                    parsed = float(value)
                else:
                    parsed = value
                setattr(cfg, key, parsed)
            except Exception:
                setattr(cfg, key, value)
    return cfg


def _load_posterior_snapshot(run_dir: str | Path, round_arg: str | int | None = "final") -> tuple[Path, dict]:
    snap_dir = Path(run_dir) / "posterior_snapshots"
    if str(round_arg or "final").lower() == "final":
        candidates = [snap_dir / "final.pt", Path(run_dir) / "final_model.pt"]
        for path in candidates:
            if path.exists():
                return path, torch.load(path, map_location="cpu")
        raise FileNotFoundError(f"No final posterior snapshot found under {snap_dir}")
    r = int(round_arg)
    path = snap_dir / f"round_{r:04d}.pt"
    if not path.exists():
        # Choose nearest saved snapshot.
        snapshots = sorted(snap_dir.glob("round_*.pt"))
        if not snapshots:
            raise FileNotFoundError(f"No round snapshots found under {snap_dir}")
        rounds = []
        for pth in snapshots:
            try:
                rounds.append((abs(int(pth.stem.split("_")[-1]) - r), pth))
            except Exception:
                pass
        if not rounds:
            raise FileNotFoundError(f"No valid round snapshots found under {snap_dir}")
        path = sorted(rounds, key=lambda x: x[0])[0][1]
        print(f"[info] requested pruning round {r} not available; using nearest snapshot {path.name}")
    return path, torch.load(path, map_location="cpu")


def _payload_from_checkpoint(cfg: RunConfig, ckpt: dict) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return mu/sigma/precision arrays from final_model.pt or posterior snapshot."""
    if "global" in ckpt:
        glob = ckpt["global"]
        mu = glob.get("mu_flat")
        sigma = glob.get("sigma_flat")
        precision = glob.get("precision_flat")
        mu_np = mu.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(mu) else np.asarray(mu, dtype=np.float32)
        sigma_np = None if sigma is None else (sigma.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(sigma) else np.asarray(sigma, dtype=np.float32))
        precision_np = None if precision is None else (precision.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(precision) else np.asarray(precision, dtype=np.float32))
        return mu_np, sigma_np, precision_np
    payload = ckpt.get("payload")
    if payload is None:
        raise ValueError("Checkpoint does not contain posterior payload")
    arrs = [np.asarray(x, dtype=np.float32) for x in payload]
    mu, sigma, precision = obs.posterior_arrays(cfg, arrs)
    return mu.astype(np.float32), None if sigma is None else sigma.astype(np.float32), None if precision is None else precision.astype(np.float32)


def run_posthoc_pruning(
    run_dir: str | Path,
    output_dir: str | Path | None,
    fractions: Sequence[float],
    round_arg: str | int | None = "final",
    device_arg: str = "auto",
    layer_name: str = "all",
) -> Path:
    """Evaluate BBB-style low-SNR pruning from a saved posterior snapshot.

    This reproduces the logic of BBB Table 2 in the FL setting: sort weights by
    posterior SNR = |mu| / sigma, remove the lowest-SNR fraction, and evaluate
    the pruned posterior-mean model.
    """
    run_dir = Path(run_dir)
    out_dir = Path(output_dir) if output_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _parse_config_csv(run_dir)
    cfg.output_dir = str(run_dir)
    if device_arg != "auto":
        cfg.device = device_arg
    # Keep evaluation deterministic and reasonably fast.
    cfg.num_workers = 0
    cfg.val_ratio = 0.0
    device = model.resolve_device(cfg.device)
    snapshot_path, ckpt = _load_posterior_snapshot(run_dir, round_arg)
    mu, sigma, precision = _payload_from_checkpoint(cfg, ckpt)
    if sigma is None and precision is not None:
        sigma = np.sqrt(1.0 / np.maximum(precision, float(cfg.precision_floor))).astype(np.float32)
    if sigma is None:
        raise ValueError("Post-hoc Bayesian pruning requires posterior sigma/precision; FedAvg checkpoint is not supported.")

    actual_round = int(ckpt.get("round", cfg.num_rounds)) if isinstance(ckpt, dict) else int(cfg.num_rounds)
    snr = np.abs(mu.astype(np.float64)) / (sigma.astype(np.float64) + 1.0e-12)
    num_params = int(mu.size)
    bundle = dataset.load_federated_data(cfg)
    rows: list[dict[str, object]] = []
    for frac in fractions:
        frac_f = float(frac)
        frac_f = min(max(frac_f, 0.0), 0.999999)
        if frac_f <= 0:
            keep_mask = np.ones(num_params, dtype=bool)
            threshold_raw = float(np.min(snr))
        else:
            threshold_raw = float(np.percentile(snr, frac_f * 100.0))
            keep_mask = snr > threshold_raw
            # If ties remove too much, keep exactly ceil((1-frac)*N) largest.
            target_keep = max(1, int(np.ceil((1.0 - frac_f) * num_params)))
            if int(keep_mask.sum()) != target_keep:
                idx = np.argsort(snr)[-target_keep:]
                keep_mask = np.zeros(num_params, dtype=bool)
                keep_mask[idx] = True
        pruned_mu = mu.copy()
        pruned_mu[~keep_mask] = 0.0
        payload = [pruned_mu]
        if cfg.method == "vi":
            payload.append(sigma.astype(np.float32))
        elif cfg.method == "ola":
            if precision is None:
                precision = 1.0 / np.maximum(sigma * sigma, float(cfg.precision_floor))
            payload.append(precision.astype(np.float32))
        metrics, _cal = obs.evaluate_payload(
            cfg=cfg,
            payload=payload,
            input_shape=bundle.input_shape,
            num_classes=bundle.num_classes,
            dataloader=bundle.testloader,
            device=device,
            mc_samples=1,
            posterior_sample_scale=0.0,
            eval_scope="posthoc_prune_global_test",
            run_id=run_dir.name,
            round_idx=actual_round,
        )
        rows.append(
            {
                "schema_version": obs.SCHEMA_VERSION,
                "run_id": run_dir.name,
                "round": actual_round,
                "method": cfg.method,
                "layer_name": layer_name,
                "threshold_type": "snr_percentile",
                "threshold_raw": threshold_raw,
                "threshold_db": float(20.0 * np.log10(threshold_raw + 1.0e-12)),
                "prune_fraction": frac_f,
                "kept_fraction": float(np.mean(keep_mask)),
                "num_params_total": num_params,
                "num_params_kept": int(keep_mask.sum()),
                "num_params_pruned": int(num_params - keep_mask.sum()),
                "accuracy_after_prune": metrics.get("accuracy", ""),
                "loss_after_prune": metrics.get("loss", ""),
                "nll_after_prune": metrics.get("nll", ""),
                "ece_after_prune": metrics.get("ece", ""),
                "brier_after_prune": metrics.get("brier", ""),
                "mean_confidence_after_prune": metrics.get("mean_confidence", ""),
                "mean_entropy_after_prune": metrics.get("mean_entropy", ""),
                "posterior_snapshot_path": str(snapshot_path),
            }
        )
    out_path = out_dir / "pruning_eval.csv"
    obs.write_csv(out_path, rows, obs.PRUNING_EVAL_FIELDS)
    return out_path


def plot_pruning_eval(pruning_csv: str | Path, output_dir: str | Path) -> list[Path]:
    rows = _read_csv_rows(pruning_csv)
    if not rows:
        print(f"[skip] no rows in {pruning_csv}")
        return []
    fractions = [float(r["prune_fraction"]) for r in rows]
    outputs: list[Path] = []
    out_dir = Path(output_dir)
    for metric in ["accuracy_after_prune", "loss_after_prune", "ece_after_prune", "num_params_kept"]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(metric, "")))
            except Exception:
                vals.append(float("nan"))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(fractions, vals, marker="o")
        ax.set_xlabel("Pruned fraction (lowest posterior SNR)")
        ax.set_ylabel(metric)
        ax.set_title(f"BBB-style SNR pruning: {metric}")
        ax.grid(True, alpha=0.3)
        outputs.append(_save_figure(fig, out_dir / f"pruning_{metric}.png"))
    return outputs

def main() -> None:
    parser = argparse.ArgumentParser(description="Offline PNG plotting for Bayesian FL runs")
    sub = parser.add_subparsers(dest="command", required=True)

    metric_parser = sub.add_parser("metric", help="Plot one metric from one metrics.csv")
    metric_parser.add_argument("--history", required=True, help="Path to metrics.csv")
    metric_parser.add_argument("--metric", default="accuracy", help="Column to plot, e.g. accuracy/loss/train_loss")
    metric_parser.add_argument("--output_dir", default="plots")

    mix_parser = sub.add_parser("mix", help="Overlay selected runs on the same metric plot")
    mix_parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help=(
            "Selected runs to include. Each entry can be label=run_dir, "
            "label=metrics.csv, run_dir, or metrics.csv."
        ),
    )
    mix_parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy", "loss"],
        help="Metrics to plot. Default: accuracy loss",
    )
    mix_parser.add_argument("--output_dir", default="plots/mix")
    mix_parser.add_argument("--filename_prefix", default="mix")

    selected_parser = sub.add_parser("selected", help="Plot selected physical-client count per round")
    selected_parser.add_argument("--selection", required=True, help="Path to selected_clients.csv or selection_summary.csv")
    selected_parser.add_argument("--output_dir", default="plots")

    radar_parser = sub.add_parser("radar", help="Plot device distribution radar chart")
    radar_parser.add_argument("--device_summary", required=True, help="Path to device_summary.csv")
    radar_parser.add_argument("--output_dir", default="plots")

    snr_parser = sub.add_parser("snr", help="Plot SNR density/CDF from snr_histograms.csv")
    snr_parser.add_argument("--snr", required=True, help="Path to snr_histograms.csv")
    snr_parser.add_argument("--output_dir", default="plots")
    snr_parser.add_argument("--round", type=int, default=None)
    snr_parser.add_argument("--layer", default="all")
    snr_parser.add_argument("--value_space", choices=["raw", "db"], default="db")

    cal_parser = sub.add_parser("calibration", help="Plot reliability diagram from calibration_bins.csv")
    cal_parser.add_argument("--calibration", required=True, help="Path to calibration_bins.csv")
    cal_parser.add_argument("--output_dir", default="plots")
    cal_parser.add_argument("--round", type=int, default=None)
    cal_parser.add_argument("--eval_scope", default="global_test")

    layer_parser = sub.add_parser("posterior-layer", help="Plot layer-wise metric from posterior_summary.csv")
    layer_parser.add_argument("--posterior", required=True, help="Path to posterior_summary.csv")
    layer_parser.add_argument("--metric", required=True, help="Metric column, e.g. sigma_mean, snr_raw_p50")
    layer_parser.add_argument("--round", type=int, default=None)
    layer_parser.add_argument("--scope", default="global")
    layer_parser.add_argument("--output_dir", default="plots")

    client_parser = sub.add_parser("client-box", help="Plot client-level metric boxplots from client_train_metrics.csv")
    client_parser.add_argument("--client_csv", required=True, help="Path to client_train_metrics.csv or client_eval_metrics.csv")
    client_parser.add_argument("--metrics", nargs="+", required=True)
    client_parser.add_argument("--round", type=int, default=None)
    client_parser.add_argument("--output_dir", default="plots")

    char_parser = sub.add_parser("characteristics", help="Generate method-specific OLA/VI characteristic plots")
    char_parser.add_argument("--run", required=True, help="Run directory containing metrics.csv and related CSVs")
    char_parser.add_argument("--method", required=True, choices=["ola", "vi"])
    char_parser.add_argument("--output_dir", default="plots/characteristics")
    char_parser.add_argument("--final_round", type=int, default=None)
    char_parser.add_argument("--best_round", type=int, default=None, help="Best accuracy round, used for layer/client snapshots")
    char_parser.add_argument("--best_ece_round", type=int, default=None, help="Best ECE round, used for layer snapshots")

    prune_parser = sub.add_parser("prune", help="Post-hoc BBB-style low-SNR pruning from a Bayesian posterior snapshot")
    prune_parser.add_argument("--run", required=True, help="Run directory containing config.csv and posterior_snapshots/final.pt")
    prune_parser.add_argument("--output_dir", default=None, help="Directory for pruning_eval.csv; defaults to the run directory")
    prune_parser.add_argument("--fractions", nargs="+", type=float, default=[0.0, 0.5, 0.75, 0.9, 0.95, 0.98])
    prune_parser.add_argument("--round", default="final", help="Snapshot round or final")
    prune_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    prune_plot_parser = sub.add_parser("prune-plot", help="Plot pruning_eval.csv")
    prune_plot_parser.add_argument("--pruning", required=True, help="Path to pruning_eval.csv")
    prune_plot_parser.add_argument("--output_dir", default="plots/pruning")

    args = parser.parse_args()
    if args.command == "metric":
        print(plot_metric(args.history, args.metric, args.output_dir))
    elif args.command == "mix":
        for path in plot_mixed_metrics(args.runs, args.metrics, args.output_dir, args.filename_prefix):
            print(path)
    elif args.command == "selected":
        print(plot_active_clients(args.selection, args.output_dir))
    elif args.command == "radar":
        print(plot_device_radar(args.device_summary, args.output_dir))
    elif args.command == "snr":
        for path in plot_snr_histogram(args.snr, args.output_dir, args.round, args.layer, args.value_space):
            print(path)
    elif args.command == "calibration":
        print(plot_calibration(args.calibration, args.output_dir, args.round, args.eval_scope))
    elif args.command == "posterior-layer":
        path = plot_posterior_layer_metric(args.posterior, args.output_dir, args.metric, args.round, args.scope)
        if path is not None:
            print(path)
    elif args.command == "client-box":
        for path in plot_client_boxplots(args.client_csv, args.output_dir, args.metrics, args.round):
            print(path)
    elif args.command == "characteristics":
        for path in plot_characteristics(args.run, args.method, args.output_dir, args.final_round, args.best_round, args.best_ece_round):
            print(path)
    elif args.command == "prune":
        print(run_posthoc_pruning(args.run, args.output_dir, args.fractions, args.round, args.device))
    elif args.command == "prune-plot":
        for path in plot_pruning_eval(args.pruning, args.output_dir):
            print(path)


if __name__ == "__main__":
    main()
