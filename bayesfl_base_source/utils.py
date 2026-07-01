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
    rows = _read_csv_rows(selection_csv)
    rounds = _column_as_float(rows, "round", selection_csv)
    selected = _column_as_float(rows, "selected_count", selection_csv)

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


if __name__ == "__main__":
    main()
