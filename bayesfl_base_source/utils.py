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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    selected_parser.add_argument("--selection", required=True, help="Path to selected_clients.csv")
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


if __name__ == "__main__":
    main()
