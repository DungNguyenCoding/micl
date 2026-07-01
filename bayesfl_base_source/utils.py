"""Offline plotting utilities.

Training intentionally writes CSV/LOG/PT files only. Run this module separately
when you want PNG plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(history_csv: str | Path, metric: str, output_dir: str | Path) -> Path:
    df = pd.read_csv(history_csv)
    if metric not in df.columns:
        raise ValueError(f"Metric {metric!r} not found in {history_csv}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["round"], df[metric], marker="o")
    ax.set_xlabel("Round")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} / round")
    ax.grid(True, alpha=0.3)
    out_path = out_dir / f"{metric}_round.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_active_clients(selection_csv: str | Path, output_dir: str | Path) -> Path:
    df = pd.read_csv(selection_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["round"], df["selected_count"], marker="o")
    ax.set_xlabel("Round")
    ax.set_ylabel("Selected physical clients")
    ax.set_title("Selected clients / round")
    ax.grid(True, alpha=0.3)
    out_path = out_dir / "selected_clients_round.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_device_radar(device_summary_csv: str | Path, output_dir: str | Path) -> Path:
    df = pd.read_csv(device_summary_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    scatter = ax.scatter(df["angle_rad"], df["radius_m"], c=df["virtual_client_id"], s=18, alpha=0.75)
    ax.set_title("Physical devices around central server")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    fig.colorbar(scatter, ax=ax, pad=0.1, label="Flower virtual client")
    out_path = out_dir / "device_distribution_radar.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline PNG plotting for Bayesian FL runs")
    sub = parser.add_subparsers(dest="command", required=True)

    metric_parser = sub.add_parser("metric", help="Plot a metric from metrics.csv")
    metric_parser.add_argument("--history", required=True, help="Path to metrics.csv")
    metric_parser.add_argument("--metric", default="accuracy", help="Column to plot, e.g. accuracy/loss/train_loss")
    metric_parser.add_argument("--output_dir", default="plots")

    selected_parser = sub.add_parser("selected", help="Plot selected physical-client count per round")
    selected_parser.add_argument("--selection", required=True, help="Path to selected_clients.csv")
    selected_parser.add_argument("--output_dir", default="plots")

    radar_parser = sub.add_parser("radar", help="Plot device distribution radar chart")
    radar_parser.add_argument("--device_summary", required=True, help="Path to device_summary.csv")
    radar_parser.add_argument("--output_dir", default="plots")

    args = parser.parse_args()
    if args.command == "metric":
        print(plot_metric(args.history, args.metric, args.output_dir))
    elif args.command == "selected":
        print(plot_active_clients(args.selection, args.output_dir))
    elif args.command == "radar":
        print(plot_device_radar(args.device_summary, args.output_dir))


if __name__ == "__main__":
    main()
