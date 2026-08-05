"""Standalone plotting utility for metrics.csv and reliability.csv."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "scaffold": "SCAFFOLD",
    "proposed": "Proposed",
}


def load_results(input_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    input_dir = Path(input_dir)
    metrics = pd.read_csv(input_dir / "metrics.csv")
    reliability_path = input_dir / "reliability.csv"
    reliability = (
        pd.read_csv(reliability_path)
        if reliability_path.exists()
        else pd.DataFrame()
    )
    return metrics, reliability


def _mean_curve(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby("channel_uses_cumulative", as_index=False)
        .agg(accuracy=("accuracy", "mean"), std=("accuracy", "std"))
        .sort_values("channel_uses_cumulative")
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def _finish_axis(axis: plt.Axes, title: str | None = None) -> None:
    axis.set_xlabel("Number of channel uses")
    axis.set_ylabel("Accuracy for test dataset")
    axis.set_ylim(0.05, 1.0)
    if title:
        axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()


def plot_fig2(metrics: pd.DataFrame, output_dir: Path) -> Path:
    frame = metrics[metrics["experiment"] == "fig2"]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for method in ("fedavg", "fedprox", "scaffold", "proposed"):
        subset = frame[frame["method"] == method]
        if subset.empty:
            continue
        curve = _mean_curve(subset)
        x = curve["channel_uses_cumulative"].to_numpy()
        y = curve["accuracy"].to_numpy()
        std = curve["std"].to_numpy()
        axis.plot(x, y, label=METHOD_LABELS[method])
        if np.any(std > 0):
            axis.fill_between(x, y - std, y + std, alpha=0.15)
    _finish_axis(axis)
    figure.tight_layout()
    path = output_dir / "figure2_accuracy_methods.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _plot_condition_family(
    metrics: pd.DataFrame,
    experiment: str,
    condition_order: Sequence[str],
    condition_labels: Dict[str, str],
    filename: str,
    output_dir: Path,
) -> Path:
    frame = metrics[metrics["experiment"] == experiment]
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    line_styles = ["-", "--", ":"]
    for method in ("fedavg", "fedprox", "proposed"):
        for condition_index, condition in enumerate(condition_order):
            subset = frame[
                (frame["method"] == method) & (frame["condition"] == condition)
            ]
            if subset.empty:
                continue
            curve = _mean_curve(subset)
            label = f"{METHOD_LABELS[method]} — {condition_labels[condition]}"
            axis.plot(
                curve["channel_uses_cumulative"],
                curve["accuracy"],
                linestyle=line_styles[condition_index % len(line_styles)],
                label=label,
            )
    _finish_axis(axis)
    figure.tight_layout()
    path = output_dir / filename
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def plot_fig3(metrics: pd.DataFrame, output_dir: Path) -> Path:
    order = ["labels_1", "labels_2", "labels_10"]
    labels = {
        "labels_1": "1-class local",
        "labels_2": "2-class local",
        "labels_10": "10-class local",
    }
    return _plot_condition_family(
        metrics, "fig3", order, labels, "figure3_label_skew.png", output_dir
    )


def plot_fig4(metrics: pd.DataFrame, output_dir: Path) -> Path:
    order = ["mean_samples_10", "mean_samples_20", "mean_samples_50"]
    labels = {
        "mean_samples_10": "E[|Dk|]=10",
        "mean_samples_20": "E[|Dk|]=20",
        "mean_samples_50": "E[|Dk|]=50",
    }
    return _plot_condition_family(
        metrics, "fig4", order, labels, "figure4_dataset_sizes.png", output_dir
    )


def plot_fig5(metrics: pd.DataFrame, output_dir: Path) -> Path:
    order = ["power_3dbm", "power_23dbm", "power_33dbm"]
    labels = {
        "power_3dbm": "P=3 dBm",
        "power_23dbm": "P=23 dBm",
        "power_33dbm": "P=33 dBm",
    }
    return _plot_condition_family(
        metrics, "fig5", order, labels, "figure5_power_budgets.png", output_dir
    )


def _final_reliability_rows(reliability: pd.DataFrame, experiment: str) -> pd.DataFrame:
    frame = reliability[reliability["experiment"] == experiment].copy()
    if frame.empty:
        return frame
    final_rounds = frame.groupby("run_id")["round"].max().rename("final_round")
    frame = frame.join(final_rounds, on="run_id")
    return frame[frame["round"] == frame["final_round"]]


def plot_fig6(reliability: pd.DataFrame, output_dir: Path) -> Path:
    frame = _final_reliability_rows(reliability, "fig6")
    if frame.empty:
        # Figure 6 may be derived from a Figure 2 default run as a convenience.
        frame = _final_reliability_rows(reliability, "fig2")
    methods = [method for method in ("fedavg", "fedprox", "proposed") if method in set(frame.get("method", []))]
    if not methods:
        raise ValueError("No final reliability data for fig6 or fig2")

    figure, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 4.3), squeeze=False)
    for axis, method in zip(axes[0], methods):
        subset = frame[frame["method"] == method].copy()
        grouped_rows = []
        for bin_index, group in subset.groupby("bin"):
            weights = group["count"].to_numpy(dtype=float)
            total = max(weights.sum(), 1.0)
            grouped_rows.append(
                {
                    "bin": bin_index,
                    "lower": group["lower"].iloc[0],
                    "upper": group["upper"].iloc[0],
                    "count": weights.sum(),
                    "confidence": np.dot(group["confidence"], weights) / total,
                    "accuracy": np.dot(group["accuracy"], weights) / total,
                }
            )
        grouped = pd.DataFrame(grouped_rows).sort_values("bin")
        centers = 0.5 * (grouped["lower"] + grouped["upper"])
        widths = grouped["upper"] - grouped["lower"]
        axis.bar(centers, grouped["accuracy"], width=widths * 0.9, alpha=0.75, label="Outputs")
        gap_bottom = np.minimum(grouped["accuracy"], grouped["confidence"])
        gap_height = np.abs(grouped["accuracy"] - grouped["confidence"])
        axis.bar(centers, gap_height, bottom=gap_bottom, width=widths * 0.9, alpha=0.3, label="Gap")
        axis.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
        ece = np.sum(grouped["count"] * np.abs(grouped["accuracy"] - grouped["confidence"])) / max(grouped["count"].sum(), 1.0)
        axis.set_title(f"{METHOD_LABELS[method]} — ECE={ece:.3f}")
        axis.set_xlabel("Confidence")
        axis.set_ylabel("Accuracy")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    path = output_dir / "figure6_reliability_diagrams.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def plot_ece(metrics: pd.DataFrame, output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for method in sorted(metrics["method"].dropna().unique()):
        subset = metrics[metrics["method"] == method]
        grouped = (
            subset.groupby("channel_uses_cumulative", as_index=False)["ece"]
            .mean()
            .sort_values("channel_uses_cumulative")
        )
        axis.plot(grouped["channel_uses_cumulative"], grouped["ece"], label=METHOD_LABELS.get(method, method))
    axis.set_xlabel("Number of channel uses")
    axis.set_ylabel("Expected calibration error")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "ece_curves.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-style plots from CSV logs")
    parser.add_argument("--input", default="results", help="Directory containing metrics.csv")
    parser.add_argument("--output", default=None, help="Plot output directory")
    parser.add_argument(
        "--figure",
        default="all",
        choices=["fig2", "fig3", "fig4", "fig5", "fig6", "ece", "all"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, reliability = load_results(input_dir)

    requested = (
        ["fig2", "fig3", "fig4", "fig5", "fig6", "ece"]
        if args.figure == "all"
        else [args.figure]
    )
    functions = {
        "fig2": lambda: plot_fig2(metrics, output_dir),
        "fig3": lambda: plot_fig3(metrics, output_dir),
        "fig4": lambda: plot_fig4(metrics, output_dir),
        "fig5": lambda: plot_fig5(metrics, output_dir),
        "fig6": lambda: plot_fig6(reliability, output_dir),
        "ece": lambda: plot_ece(metrics, output_dir),
    }
    for name in requested:
        try:
            path = functions[name]()
            print(f"Created {path}")
        except (ValueError, KeyError) as exc:
            print(f"Skipped {name}: {exc}")


if __name__ == "__main__":
    main()
