"""Plot generation only.

Training and numerical metrics intentionally live outside this module so Ray client
workers never import Matplotlib during local optimization.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _numeric(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=np.float64)


def plot_global_metrics(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    rows = _read_csv(run_dir / "metrics" / "global_metrics.csv")
    if not rows:
        return []
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    rounds = _numeric(rows, "round")
    paths: list[Path] = []
    for metric in ("accuracy", "nll", "ece", "brier", "mutual_information"):
        y = _numeric(rows, metric)
        if np.all(np.isnan(y)):
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(rounds, y)
        ax.set_xlabel("Round")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} vs Round")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"global_{metric}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_training_metrics(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    rows = _read_csv(run_dir / "metrics" / "round_train_metrics.csv")
    if not rows:
        rows = _read_csv(run_dir / "metrics" / "client_metrics.csv")
    if not rows:
        return []
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    rounds = _numeric(rows, "round")
    paths: list[Path] = []
    for metric in ("train_loss", "task_loss", "prior_loss", "effective_kl_weight", "variance_floor_fraction", "lr"):
        y = _numeric(rows, metric)
        if np.all(np.isnan(y)):
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(rounds, y)
        ax.set_xlabel("Round")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Client {metric.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"client_{metric}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_reliability(run_dir: str | Path, round_number: int | None = None) -> Path | None:
    run_dir = Path(run_dir)
    files = sorted((run_dir / "reliability").glob("round_*.npz"))
    if not files:
        return None
    if round_number is None:
        path = files[-1]
    else:
        path = run_dir / "reliability" / f"round_{round_number:04d}.npz"
        if not path.exists():
            return None
    with np.load(path) as data:
        edges = data["bin_edges"]
        acc = data["bin_accuracy"]
        conf = data["bin_confidence"]
        count = data["bin_count"]
    mask = count > 0
    centers = (edges[:-1] + edges[1:]) / 2.0
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(centers[mask], acc[mask], marker="o", label="Accuracy")
    ax.plot(centers[mask], conf[mask], marker="x", label="Confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Value")
    ax.set_title(f"Reliability Diagram ({path.stem})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = run_dir / "plots" / f"reliability_{path.stem}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_posterior_summary(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    rows = _read_csv(run_dir / "posterior" / "posterior_summary.csv")
    if not rows:
        return []
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    available = set().union(*(row.keys() for row in rows))
    metrics = [m for m in ("sigma_mean", "precision_mean", "snr_mean", "mean_abs") if m in available]
    paths: list[Path] = []
    parameters = sorted({row.get("parameter", "") for row in rows})
    # Plot the mean across parameter tensors to keep figures readable.
    for metric in metrics:
        round_values: dict[int, list[float]] = {}
        for row in rows:
            try:
                rnd = int(float(row["round"]))
                value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                continue
            round_values.setdefault(rnd, []).append(value)
        if not round_values:
            continue
        rounds = sorted(round_values)
        means = [float(np.mean(round_values[r])) for r in rounds]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(rounds, means)
        ax.set_xlabel("Round")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Posterior {metric.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"posterior_{metric}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def generate_all_plots(run_dir: str | Path) -> list[Path]:
    paths = []
    paths.extend(plot_global_metrics(run_dir))
    paths.extend(plot_training_metrics(run_dir))
    paths.extend(plot_posterior_summary(run_dir))
    reliability = plot_reliability(run_dir)
    if reliability is not None:
        paths.append(reliability)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots for a BayesFL run")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    paths = generate_all_plots(args.run_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
