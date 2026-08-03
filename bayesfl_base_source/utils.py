"""Offline plotting utilities for Bayesian FL runs.

Training writes CSV/LOG/PT files only. Run this module separately when you
want PNG plots. This version intentionally avoids pandas so it is more robust
against NumPy/Pandas binary-version issues.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

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


def _trusted_torch_load(path: str | Path):
    """Load trusted checkpoints from this project across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
            print(f"[skip] No runs contained usable data for metric {metric!r}")
            continue

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
                return path, _trusted_torch_load(path)
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
    return path, _trusted_torch_load(path)


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


# ---------------------------------------------------------------------------
# Research diagnostics plots added for long-run Bayes-FL analysis
# ---------------------------------------------------------------------------
def _last_and_best(rows: Rows, metric: str = "global_accuracy") -> tuple[dict[str, str] | None, dict[str, str] | None]:
    valid = [r for r in rows if r.get(metric, "") != ""]
    if not valid:
        return (rows[-1] if rows else None), None
    if metric.endswith("ece") or metric.endswith("loss"):
        best = min(valid, key=lambda r: float(r.get(metric, "inf")))
    else:
        best = max(valid, key=lambda r: float(r.get(metric, "-inf")))
    return rows[-1] if rows else None, best


def _series_or_none(rows: Rows, column: str) -> tuple[list[float], list[float]] | None:
    if not rows or column not in rows[0] or "round" not in rows[0]:
        return None
    xs, ys = [], []
    for r in rows:
        if r.get("round", "") == "" or r.get(column, "") == "":
            continue
        try:
            xs.append(float(r["round"]))
            ys.append(float(r[column]))
        except Exception:
            continue
    return (xs, ys) if xs and ys else None


def _cumulative(values: Sequence[float]) -> list[float]:
    out, total = [], 0.0
    for v in values:
        total += float(v)
        out.append(total)
    return out


def plot_run_diagnostics(run_dir: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """Generate extra diagnostic plots for one run.

    These plots are designed to explain behavior like VI late-round degradation:
    best-vs-final gap, posterior uncertainty/SNR, local-global gap, cumulative
    communication, and accuracy over wall-clock time.
    """
    run = Path(run_dir)
    out_dir = Path(output_dir) if output_dir is not None else run / "diagnostic_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv_rows(run / "metrics.csv")
    outputs: list[Path] = []
    if not rows:
        return outputs

    final, best = _last_and_best(rows, "global_accuracy")
    if final is not None and best is not None:
        labels = ["best", "final"]
        values = [float(best.get("global_accuracy", "nan")), float(final.get("global_accuracy", "nan"))]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.bar(labels, values)
        ax.set_ylim(0.0, max(1.0, max(values) * 1.05))
        ax.set_ylabel("global_accuracy")
        ax.set_title(f"Best vs final accuracy (best round={best.get('round','')})")
        ax.grid(True, axis="y", alpha=0.3)
        outputs.append(_save_figure(fig, out_dir / "best_vs_final_accuracy.png"))

    # Accuracy + posterior sigma + SNR p50 in one explanatory figure.
    acc = _series_or_none(rows, "global_accuracy")
    sigma = _series_or_none(rows, "posterior_sigma_mean")
    snr = _series_or_none(rows, "posterior_snr_raw_p50")
    if acc is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(acc[0], acc[1], marker="o", markersize=3, label="global_accuracy")
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        if sigma is not None:
            ax2.plot(sigma[0], sigma[1], linestyle="--", label="posterior_sigma_mean")
        if snr is not None:
            ax2.plot(snr[0], snr[1], linestyle=":", label="posterior_snr_raw_p50")
        ax2.set_ylabel("Posterior diagnostic")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
        ax.set_title("Accuracy with posterior uncertainty/SNR")
        outputs.append(_save_figure(fig, out_dir / "accuracy_uncertainty_snr_overlay.png"))

    # Local-global gap.
    local = _series_or_none(rows, "local_accuracy_weighted")
    if acc is not None and local is not None:
        acc_map = dict(zip(acc[0], acc[1]))
        xs, gap = [], []
        for x, y in zip(local[0], local[1]):
            if x in acc_map:
                xs.append(x)
                gap.append(float(y) - float(acc_map[x]))
        if xs:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(xs, gap, marker="o", markersize=3)
            ax.axhline(0.0, linewidth=1)
            ax.set_xlabel("Round")
            ax.set_ylabel("local_accuracy_weighted - global_accuracy")
            ax.set_title("Local-global accuracy gap")
            ax.grid(True, alpha=0.3)
            outputs.append(_save_figure(fig, out_dir / "local_global_accuracy_gap.png"))

    # Cumulative communication.
    dense_b = _series_or_none(rows, "communication_dense_bytes")
    sparse_b = _series_or_none(rows, "communication_sparse_bytes")
    if dense_b is not None or sparse_b is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if dense_b is not None:
            ax.plot(dense_b[0], _cumulative(dense_b[1]), label="cumulative_dense_bytes")
        if sparse_b is not None:
            ax.plot(sparse_b[0], _cumulative(sparse_b[1]), label="cumulative_sparse_bytes")
        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative bytes")
        ax.set_title("Cumulative communication cost")
        ax.grid(True, alpha=0.3)
        ax.legend()
        outputs.append(_save_figure(fig, out_dir / "cumulative_communication_bytes.png"))

    # Accuracy vs cumulative wall clock time.
    time_s = _series_or_none(rows, "round_time_sec")
    if acc is not None and time_s is not None:
        t_map = dict(zip(time_s[0], time_s[1]))
        xs, ys, total = [], [], 0.0
        for r, a in zip(acc[0], acc[1]):
            total += float(t_map.get(r, 0.0))
            xs.append(total)
            ys.append(a)
        if xs:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(xs, ys, marker="o", markersize=3)
            ax.set_xlabel("Cumulative wall-clock seconds")
            ax.set_ylabel("global_accuracy")
            ax.set_title("Accuracy vs wall-clock time")
            ax.grid(True, alpha=0.3)
            outputs.append(_save_figure(fig, out_dir / "accuracy_vs_wall_time.png"))

    return outputs


def plot_compare_diagnostics(run_specs: Sequence[str], output_dir: str | Path) -> list[Path]:
    """Create best/final comparison summary plots for multiple runs."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for spec in run_specs:
        label, history = _resolve_history_path(spec)
        rows = _read_csv_rows(history)
        if not rows:
            continue
        final, best = _last_and_best(rows, "global_accuracy")
        _, best_ece = _last_and_best(rows, "global_ece")
        if final is None or best is None:
            continue
        final_acc = float(final.get("global_accuracy", "nan"))
        best_acc = float(best.get("global_accuracy", "nan"))
        final_ece = float(final.get("global_ece", "nan")) if final.get("global_ece", "") != "" else float("nan")
        summary_rows.append({
            "label": label,
            "final_round": final.get("round", ""),
            "final_global_accuracy": final_acc,
            "best_global_accuracy": best_acc,
            "best_accuracy_round": best.get("round", ""),
            "accuracy_drop_best_to_final": best_acc - final_acc,
            "final_global_loss": float(final.get("global_loss", "nan")) if final.get("global_loss", "") != "" else float("nan"),
            "final_global_ece": final_ece,
            "best_global_ece": float(best_ece.get("global_ece", "nan")) if best_ece is not None and best_ece.get("global_ece", "") != "" else float("nan"),
            "best_ece_round": best_ece.get("round", "") if best_ece is not None else "",
        })
    summary_path = write_simple_csv(out_dir / "best_final_summary.csv", summary_rows)
    outputs = [summary_path]
    if not summary_rows:
        return outputs

    labels = [r["label"] for r in summary_rows]
    for key, title, ylabel in [
        ("final_global_accuracy", "Final global accuracy", "Accuracy"),
        ("best_global_accuracy", "Best global accuracy", "Accuracy"),
        ("accuracy_drop_best_to_final", "Best-to-final accuracy drop", "Accuracy drop"),
        ("final_global_ece", "Final ECE", "ECE"),
        ("final_global_loss", "Final global loss", "Loss"),
    ]:
        vals = [float(r.get(key, float("nan"))) for r in summary_rows]
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 4.5))
        ax.bar(labels, vals)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=35)
        outputs.append(_save_figure(fig, out_dir / f"{key}.png"))

    # Accuracy-calibration tradeoff.
    xs = [float(r.get("final_global_ece", float("nan"))) for r in summary_rows]
    ys = [float(r.get("final_global_accuracy", float("nan"))) for r in summary_rows]
    if any(np.isfinite(xs)) and any(np.isfinite(ys)):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(xs, ys)
        for label, x, y in zip(labels, xs, ys):
            if np.isfinite(x) and np.isfinite(y):
                ax.annotate(label, (x, y), fontsize=8)
        ax.set_xlabel("Final ECE")
        ax.set_ylabel("Final global accuracy")
        ax.set_title("Accuracy-calibration tradeoff")
        ax.grid(True, alpha=0.3)
        outputs.append(_save_figure(fig, out_dir / "accuracy_calibration_tradeoff.png"))
    return outputs


def write_simple_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path


def plot_client_heterogeneity(run_dir: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """Plot client data heterogeneity against update/sparse behavior."""
    run = Path(run_dir)
    out_dir = Path(output_dir) if output_dir is not None else run / "heterogeneity_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = run / "client_train_metrics.csv"
    if not train_csv.exists():
        print(f"[skip] missing {train_csv}")
        return []
    rows = _read_csv_rows(train_csv)
    outputs: list[Path] = []

    def scatter(xcol: str, ycol: str, filename: str) -> None:
        if not rows or xcol not in rows[0] or ycol not in rows[0]:
            return
        xs, ys = [], []
        for r in rows:
            if r.get(xcol, "") == "" or r.get(ycol, "") == "":
                continue
            try:
                xs.append(float(r[xcol])); ys.append(float(r[ycol]))
            except Exception:
                pass
        if not xs:
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(xs, ys, s=10, alpha=0.45)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(f"{ycol} vs {xcol}")
        ax.grid(True, alpha=0.3)
        outputs.append(_save_figure(fig, out_dir / filename))

    scatter("label_entropy", "update_l2_norm", "label_entropy_vs_update_l2.png")
    scatter("kl_to_global_label_distribution", "update_l2_norm", "label_kl_vs_update_l2.png")
    scatter("label_entropy", "sparse_num_params_sent", "label_entropy_vs_sparse_sent_params.png")
    scatter("kl_to_global_label_distribution", "sparse_threshold", "label_kl_vs_sparse_threshold.png")
    scatter("label_entropy", "local_epochs", "label_entropy_vs_local_epochs.png")
    return outputs


def plot_snr_evolution(snr_csv: str | Path, output_dir: str | Path, rounds: Sequence[int] | None = None, layer: str = "all", value_space: str = "db") -> list[Path]:
    """Overlay SNR density/CDF for multiple rounds."""
    rows = _read_csv_rows(snr_csv)
    if not rows:
        return []
    available = sorted({int(float(r.get("round", "0"))) for r in rows if r.get("round", "") != ""})
    if not available:
        return []
    if not rounds:
        if len(available) <= 5:
            chosen = available
        else:
            idxs = np.linspace(0, len(available) - 1, 5).round().astype(int)
            chosen = [available[int(i)] for i in idxs]
    else:
        chosen = []
        for requested in rounds:
            nearest = min(available, key=lambda r: abs(r - int(requested)))
            if nearest not in chosen:
                chosen.append(nearest)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for kind in ["density", "cdf"]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plotted = 0
        for rnd in chosen:
            sub = [r for r in rows if int(float(r.get("round", "-1"))) == rnd and r.get("layer_name") == layer and r.get("value_space") == value_space]
            if not sub:
                continue
            xs = [float(r["bin_center"]) for r in sub]
            ys = [float(r[kind]) for r in sub]
            ax.plot(xs, ys, label=f"round {rnd}")
            plotted += 1
        if plotted:
            ax.set_xlabel(f"SNR ({value_space})")
            ax.set_ylabel(kind)
            ax.set_title(f"SNR {kind} evolution ({layer})")
            ax.grid(True, alpha=0.3)
            ax.legend()
            outputs.append(_save_figure(fig, out_dir / f"snr_{kind}_evolution_{layer}_{value_space}.png"))
        else:
            plt.close(fig)
    return outputs



# ---------------------------------------------------------------------------
# Sparse-selection ablation plots
# ---------------------------------------------------------------------------
def _read_run_config(run_dir: Path) -> dict[str, str]:
    cfg_path = run_dir / "config.csv"
    if not cfg_path.exists():
        return {}
    rows = _read_csv_rows(cfg_path)
    out: dict[str, str] = {}
    for r in rows:
        if "key" in r and "value" in r:
            out[str(r.get("key", ""))] = str(r.get("value", ""))
        else:
            vals = list(r.values())
            if len(vals) >= 2:
                out[str(vals[0])] = str(vals[1])
    return out


def _float_from_row(row: Mapping[str, str] | None, key: str, default: float = float("nan")) -> float:
    if row is None:
        return default
    try:
        v = row.get(key, "")
        if v == "" or v is None:
            return default
        return float(v)
    except Exception:
        return default


def _infer_keep_from_label(label: str) -> float | None:
    import re
    m = re.search(r"keep(\d+)", label.lower())
    if not m:
        return None
    val = int(m.group(1))
    if val > 100:
        return val / 1000.0
    return val / 100.0


def _infer_label_and_run(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(spec)
    return path.name, path


def _mean_from_rows(rows: Rows, key: str) -> float:
    vals = []
    for r in rows:
        try:
            v = r.get(key, "")
            if v != "":
                vals.append(float(v))
        except Exception:
            continue
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _sum_from_rows(rows: Rows, key: str) -> float:
    vals = []
    for r in rows:
        try:
            v = r.get(key, "")
            if v != "":
                vals.append(float(v))
        except Exception:
            continue
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.sum(vals)) if vals else float("nan")


def plot_sparse_ablation(run_specs: Sequence[str], output_dir: str | Path) -> list[Path]:
    """Create Bayesian-vs-random sparse selection ablation summary plots.

    Each run spec can be ``label=run_dir`` or just ``run_dir``.  The function
    reads ``metrics.csv``, ``config.csv``, and optionally
    ``sparse_comm_metrics.csv`` from every run.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for spec in run_specs:
        label, run_dir = _infer_label_and_run(spec)
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            print(f"[skip] missing metrics.csv for {label}: {metrics_path}")
            continue
        rows = _read_csv_rows(metrics_path)
        if not rows:
            print(f"[skip] empty metrics.csv for {label}")
            continue
        cfg = _read_run_config(run_dir)
        last, best_acc = _last_and_best(rows, "global_accuracy")
        _last_ece, best_ece = _last_and_best(rows, "global_ece")
        method = cfg.get("method") or (last or {}).get("method", "") or label.split("_")[0]
        selection = cfg.get("sparse_selection") or cfg.get("sparse_selection_method") or (last or {}).get("sparse_selection_method", "")
        if not selection:
            selection = "random" if "random" in label.lower() else "bayesian"
        sparse_metric = str(cfg.get("sparse_metric", "") or (last or {}).get("sparse_metric", "") or "")
        seed_value = cfg.get("seed", "") or (last or {}).get("seed", "")
        try:
            seed_numeric = int(float(seed_value)) if str(seed_value) != "" else None
        except Exception:
            seed_numeric = None

        try:
            keep_ratio = float(cfg.get("sparse_ratio", ""))
        except Exception:
            inferred = _infer_keep_from_label(label)
            keep_ratio = inferred if inferred is not None else float("nan")
        if not np.isfinite(keep_ratio):
            inferred = _infer_keep_from_label(label)
            keep_ratio = inferred if inferred is not None else float("nan")

        sparse_rows: Rows = []
        sparse_path = run_dir / "sparse_comm_metrics.csv"
        if sparse_path.exists():
            try:
                sparse_rows = _read_csv_rows(sparse_path)
            except Exception:
                sparse_rows = []

        cumulative_sparse = _float_from_row(last, "communication_cumulative_sparse_bytes")
        cumulative_dense = _float_from_row(last, "communication_cumulative_dense_bytes")
        if not np.isfinite(cumulative_sparse):
            cumulative_sparse = _sum_from_rows(sparse_rows, "sparse_sent_bytes")
        if not np.isfinite(cumulative_dense):
            cumulative_dense = _sum_from_rows(sparse_rows, "sparse_dense_bytes")
        if np.isfinite(cumulative_sparse) and np.isfinite(cumulative_dense) and cumulative_dense > 0:
            cumulative_saving = 1.0 - cumulative_sparse / cumulative_dense
        else:
            cumulative_saving = _float_from_row(last, "communication_cumulative_saving_ratio")

        retention = _float_from_row(last, "sparse_update_energy_retention_mean")
        if not np.isfinite(retention):
            retention = _float_from_row(last, "sparse_sent_update_fraction_l2_mean")
        if not np.isfinite(retention):
            retention = _mean_from_rows(sparse_rows, "sparse_update_energy_retention")
        if not np.isfinite(retention):
            retention = _mean_from_rows(sparse_rows, "sparse_sent_update_fraction_l2")

        selected_score = _float_from_row(last, "sparse_selected_score_mean_mean")
        dropped_score = _float_from_row(last, "sparse_dropped_score_mean_mean")
        if not np.isfinite(selected_score):
            selected_score = _mean_from_rows(sparse_rows, "sparse_selected_score_mean")
        if not np.isfinite(dropped_score):
            dropped_score = _mean_from_rows(sparse_rows, "sparse_dropped_score_mean")
        score_gap = selected_score - dropped_score if np.isfinite(selected_score) and np.isfinite(dropped_score) else float("nan")

        summary.append({
            "label": label,
            "run_dir": str(run_dir),
            "method": method,
            "sparse_selection": selection,
            "sparse_metric": sparse_metric,
            "seed": seed_numeric if seed_numeric is not None else "",
            "curve_label": (f"{selection}:{sparse_metric}" if selection == "bayesian" and sparse_metric else selection),
            "keep_ratio": keep_ratio,
            "keep_percent": keep_ratio * 100.0 if np.isfinite(keep_ratio) else float("nan"),
            "final_round": _float_from_row(last, "round"),
            "final_global_accuracy": _float_from_row(last, "global_accuracy"),
            "final_global_loss": _float_from_row(last, "global_loss"),
            "final_global_ece": _float_from_row(last, "global_ece"),
            "best_global_accuracy": _float_from_row(best_acc, "global_accuracy"),
            "best_global_accuracy_round": _float_from_row(best_acc, "round"),
            "best_global_ece": _float_from_row(best_ece, "global_ece"),
            "best_global_ece_round": _float_from_row(best_ece, "round"),
            "cumulative_dense_bytes": cumulative_dense,
            "cumulative_sparse_bytes": cumulative_sparse,
            "cumulative_sparse_MB": cumulative_sparse / (1024.0 * 1024.0) if np.isfinite(cumulative_sparse) else float("nan"),
            "cumulative_saving_ratio": cumulative_saving,
            "mean_update_energy_retention": retention,
            "selected_score_mean": selected_score,
            "dropped_score_mean": dropped_score,
            "selected_minus_dropped_score_mean": score_gap,
        })

    if not summary:
        print("[skip] no valid runs for sparse ablation")
        return []

    fields = []
    for row in summary:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    summary_path = out_dir / "sparse_ablation_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    outputs: list[Path] = [summary_path]

    def _plot_vs_keep(metric: str, ylabel: str, filename: str, title: str) -> None:
        """Plot mean ± std over seeds for every selection/metric curve."""
        methods = sorted({str(r["method"]) for r in summary})
        for method in methods:
            sub = [r for r in summary if str(r["method"]) == method]
            if not sub:
                continue
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            plotted = False
            for selection in sorted({str(r.get("curve_label", r.get("sparse_selection", ""))) for r in sub}):
                rows_s = [r for r in sub if str(r.get("curve_label", r.get("sparse_selection", ""))) == selection]
                grouped: dict[float, list[float]] = {}
                for r in rows_s:
                    try:
                        x = float(r.get("keep_percent", float("nan")))
                        y = float(r.get(metric, float("nan")))
                    except Exception:
                        continue
                    if np.isfinite(x) and np.isfinite(y):
                        grouped.setdefault(x, []).append(y)
                if grouped:
                    xs = sorted(grouped.keys(), reverse=True)
                    ys = [float(np.mean(grouped[x])) for x in xs]
                    yerr = [float(np.std(grouped[x])) if len(grouped[x]) > 1 else 0.0 for x in xs]
                    ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=selection)
                    plotted = True
            if plotted:
                ax.set_xlabel("Keep ratio (%)")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{method.upper()}: {title}")
                ax.grid(True, alpha=0.3)
                ax.legend()
                outputs.append(_save_figure(fig, out_dir / f"{method}_{filename}"))
            else:
                plt.close(fig)

    _plot_vs_keep("final_global_accuracy", "Final global accuracy", "accuracy_vs_keep_ratio.png", "accuracy vs keep ratio")
    _plot_vs_keep("best_global_accuracy", "Best global accuracy", "best_accuracy_vs_keep_ratio.png", "best accuracy vs keep ratio")
    _plot_vs_keep("final_global_ece", "Final ECE", "ece_vs_keep_ratio.png", "calibration vs keep ratio")
    _plot_vs_keep("cumulative_saving_ratio", "Cumulative communication saving ratio", "communication_saving_vs_keep_ratio.png", "communication saving vs keep ratio")
    _plot_vs_keep("mean_update_energy_retention", "Mean retained update L2 fraction", "update_energy_retention_vs_keep_ratio.png", "information retention vs keep ratio")
    _plot_vs_keep("selected_minus_dropped_score_mean", "Selected - dropped Bayesian score", "selected_dropped_score_gap_vs_keep_ratio.png", "selection-quality gap")

    # Accuracy vs communication cost Pareto-style plot.
    methods = sorted({str(r["method"]) for r in summary})
    for method in methods:
        sub = [r for r in summary if str(r["method"]) == method]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        plotted = False
        for selection in sorted({str(r.get("curve_label", r.get("sparse_selection", ""))) for r in sub}):
            rows_s = [r for r in sub if str(r.get("curve_label", r.get("sparse_selection", ""))) == selection]
            xs, ys = [], []
            for r in rows_s:
                try:
                    x = float(r.get("cumulative_sparse_MB", float("nan")))
                    y = float(r.get("final_global_accuracy", float("nan")))
                except Exception:
                    continue
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(x); ys.append(y)
            if xs:
                order = np.argsort(xs)
                xs = [xs[i] for i in order]
                ys = [ys[i] for i in order]
                ax.plot(xs, ys, marker="o", label=selection)
                plotted = True
        if plotted:
            ax.set_xlabel("Cumulative communication (MB)")
            ax.set_ylabel("Final global accuracy")
            ax.set_title(f"{method.upper()}: accuracy vs communication cost")
            ax.grid(True, alpha=0.3)
            ax.legend()
            outputs.append(_save_figure(fig, out_dir / f"{method}_accuracy_vs_communication_cost.png"))
        else:
            plt.close(fig)


    # Round-by-round overlays. These show every run directly, which is useful
    # for checking whether Bayesian top-k consistently trains better than the
    # random top-k baseline under the same keep ratio.
    round_metrics = [
        ("global_accuracy", "Global accuracy"),
        ("global_loss", "Global loss"),
        ("global_ece", "Global ECE"),
    ]
    for method in sorted({str(r["method"]) for r in summary}):
        method_rows = [r for r in summary if str(r["method"]) == method]
        for metric_name, ylabel in round_metrics:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            plotted = False
            for r in sorted(method_rows, key=lambda x: (str(x.get("curve_label", "")), float(x.get("keep_ratio", 0.0)), str(x.get("seed", "")))):
                run_dir = Path(str(r["run_dir"]))
                rows = _read_csv_rows(run_dir / "metrics.csv")
                xs, ys = [], []
                for row in rows:
                    try:
                        x = float(row.get("round", ""))
                        y = float(row.get(metric_name, ""))
                    except Exception:
                        continue
                    if np.isfinite(x) and np.isfinite(y):
                        xs.append(x); ys.append(y)
                if xs:
                    seed_suffix = f",s{r.get('seed')}" if r.get("seed", "") != "" else ""
                    label = f"{r.get('curve_label')} keep{float(r.get('keep_percent', 0.0)):.0f}%{seed_suffix}"
                    ax.plot(xs, ys, linewidth=1.2, alpha=0.75, label=label)
                    plotted = True
            if plotted:
                ax.set_xlabel("Round")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{method.upper()}: {ylabel} vs round, all sparse settings")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7, ncol=2)
                outputs.append(_save_figure(fig, out_dir / f"{method}_{metric_name}_round_all_settings.png"))
            else:
                plt.close(fig)

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

    diag_parser = sub.add_parser("diagnostics", help="Generate extra explanatory plots for one run")
    diag_parser.add_argument("--run", required=True, help="Run directory containing metrics.csv")
    diag_parser.add_argument("--output_dir", default=None)

    compare_diag_parser = sub.add_parser("compare-diagnostics", help="Best/final summary plots for multiple runs")
    compare_diag_parser.add_argument("--runs", nargs="+", required=True, help="label=run_dir entries")
    compare_diag_parser.add_argument("--output_dir", required=True)

    het_parser = sub.add_parser("heterogeneity", help="Client heterogeneity vs update/sparse behavior plots")
    het_parser.add_argument("--run", required=True)
    het_parser.add_argument("--output_dir", default=None)

    snr_evo_parser = sub.add_parser("snr-evolution", help="Overlay SNR density/CDF across rounds")
    snr_evo_parser.add_argument("--snr", required=True)
    snr_evo_parser.add_argument("--rounds", nargs="*", type=int, default=None)
    snr_evo_parser.add_argument("--layer", default="all")
    snr_evo_parser.add_argument("--value_space", choices=["raw", "db"], default="db")
    snr_evo_parser.add_argument("--output_dir", default="plots/snr_evolution")

    prune_parser = sub.add_parser("prune", help="Post-hoc BBB-style low-SNR pruning from a Bayesian posterior snapshot")
    prune_parser.add_argument("--run", required=True, help="Run directory containing config.csv and posterior_snapshots/final.pt")
    prune_parser.add_argument("--output_dir", default=None, help="Directory for pruning_eval.csv; defaults to the run directory")
    prune_parser.add_argument("--fractions", nargs="+", type=float, default=[0.0, 0.5, 0.75, 0.9, 0.95, 0.98])
    prune_parser.add_argument("--round", default="final", help="Snapshot round or final")
    prune_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    prune_plot_parser = sub.add_parser("prune-plot", help="Plot pruning_eval.csv")
    prune_plot_parser.add_argument("--pruning", required=True, help="Path to pruning_eval.csv")
    prune_plot_parser.add_argument("--output_dir", default="plots/pruning")

    sparse_ablation_parser = sub.add_parser("sparse-ablation", help="Compare Bayesian vs random sparse selection runs")
    sparse_ablation_parser.add_argument("--runs", nargs="+", required=True, help="label=run_dir entries")
    sparse_ablation_parser.add_argument("--output_dir", required=True)

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
    elif args.command == "diagnostics":
        for path in plot_run_diagnostics(args.run, args.output_dir):
            print(path)
    elif args.command == "compare-diagnostics":
        for path in plot_compare_diagnostics(args.runs, args.output_dir):
            print(path)
    elif args.command == "heterogeneity":
        for path in plot_client_heterogeneity(args.run, args.output_dir):
            print(path)
    elif args.command == "snr-evolution":
        for path in plot_snr_evolution(args.snr, args.output_dir, args.rounds, args.layer, args.value_space):
            print(path)
    elif args.command == "prune":
        print(run_posthoc_pruning(args.run, args.output_dir, args.fractions, args.round, args.device))
    elif args.command == "prune-plot":
        for path in plot_pruning_eval(args.pruning, args.output_dir):
            print(path)
    elif args.command == "sparse-ablation":
        for path in plot_sparse_ablation(args.runs, args.output_dir):
            print(path)


if __name__ == "__main__":
    main()
