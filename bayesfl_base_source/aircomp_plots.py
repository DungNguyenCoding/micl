"""Plot utilities for AirComp Bayesian FL paper reproduction outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure(out: str | Path) -> Path:
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mean_curve(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    # Interpolate only by exact recorded x positions across realizations. This is
    # enough because all realizations of one method/condition have the same x grid.
    cols = [c for c in ["accuracy", "loss", "ece", "mc_accuracy", "mc_ece"] if c in df.columns]
    agg = df.groupby(list(group_cols) + ["channel_uses"], dropna=False)[cols].mean(numeric_only=True).reset_index()
    return agg


def _plot_accuracy_curves(df: pd.DataFrame, scenario: str, out_dir: Path, filename: str, title: str, condition_style: bool = False) -> Path | None:
    sub = df[df["scenario"] == scenario].copy()
    if sub.empty:
        return None
    agg = _mean_curve(sub, ["condition_name", "method"])
    fig, ax = plt.subplots(figsize=(9, 5))
    if condition_style:
        for (condition, method), g in agg.groupby(["condition_name", "method"]):
            g = g.sort_values("channel_uses")
            label = f"{method} / {condition}"
            ax.plot(g["channel_uses"], g["accuracy"], label=label)
    else:
        for method, g in agg.groupby("method"):
            g = g.sort_values("channel_uses")
            ax.plot(g["channel_uses"], g["accuracy"], label=str(method))
    ax.set_title(title)
    ax.set_xlabel("Channel uses (OFDM sub-channel uses)")
    ax.set_ylabel("Accuracy for test dataset")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_reliability(cal: pd.DataFrame, scenario: str, condition: str, out_dir: Path) -> Path | None:
    sub = cal[(cal["scenario"] == scenario) & (cal["condition_name"] == condition) & (cal["eval_scope"] == "global_test")].copy()
    if sub.empty:
        return None
    # Use final round for each method/realization, then average bins.
    final_rows = []
    for (method, realization), g in sub.groupby(["method", "realization"]):
        final_round = g["round"].max()
        final_rows.append(g[g["round"] == final_round])
    final = pd.concat(final_rows, ignore_index=True)
    agg = final.groupby(["method", "bin_id", "bin_left", "bin_right"], dropna=False)[["bin_accuracy", "bin_confidence", "bin_count"]].mean(numeric_only=True).reset_index()
    methods = [m for m in ["fedavg", "fedprox", "proposed"] if m in set(agg["method"])]
    if not methods:
        methods = sorted(agg["method"].unique())[:3]
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), squeeze=False)
    for ax, method in zip(axes[0], methods):
        g = agg[agg["method"] == method].sort_values("bin_id")
        centers = (g["bin_left"].to_numpy() + g["bin_right"].to_numpy()) / 2.0
        width = (g["bin_right"].to_numpy() - g["bin_left"].to_numpy()) * 0.9
        acc = g["bin_accuracy"].to_numpy()
        conf = g["bin_confidence"].to_numpy()
        ece = float(np.sum(np.abs(acc - conf) * g["bin_count"].to_numpy() / max(g["bin_count"].sum(), 1.0)))
        ax.bar(centers, acc, width=width, alpha=0.75, label="Outputs")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
        ax.plot(centers, conf, marker="o", label="Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{method} (ECE={ece:.3f})")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "fig6_reliability_default.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_summary_bars(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if summary.empty:
        return paths
    agg = summary.groupby(["scenario", "condition_name", "method"], dropna=False)[["final_accuracy", "best_accuracy", "final_ece"]].mean(numeric_only=True).reset_index()
    path = out_dir / "aircomp_summary.csv"
    agg.to_csv(path, index=False)
    paths.append(path)
    for scenario in sorted(agg["scenario"].unique()):
        sub = agg[agg["scenario"] == scenario]
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"{r.condition_name}\n{r.method}" for r in sub.itertuples()]
        x = np.arange(len(labels))
        ax.bar(x, sub["final_accuracy"].to_numpy())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Final accuracy")
        ax.set_title(f"Final accuracy summary: {scenario}")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"summary_final_accuracy_{scenario}.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_aircomp_results(run_dir: str | Path, output_dir: str | Path) -> list[Path]:
    run = Path(run_dir)
    out = _ensure(output_dir)
    metrics_path = run / "metrics.csv"
    cal_path = run / "calibration_bins.csv"
    summary_path = run / "run_summary.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    df = pd.read_csv(metrics_path)
    paths: list[Path] = []
    p = _plot_accuracy_curves(df, "default", out, "fig2_accuracy_default.png", "Fig. 2-style: Default scarce heterogeneous data")
    if p:
        paths.append(p)
    p = _plot_accuracy_curves(df, "label_skew", out, "fig3_accuracy_label_skew.png", "Fig. 3-style: Different local label skewness", condition_style=True)
    if p:
        paths.append(p)
    p = _plot_accuracy_curves(df, "dataset_size", out, "fig4_accuracy_dataset_size.png", "Fig. 4-style: Different local dataset sizes", condition_style=True)
    if p:
        paths.append(p)
    p = _plot_accuracy_curves(df, "power", out, "fig5_accuracy_power_budget.png", "Fig. 5-style: Different transmission power budgets", condition_style=True)
    if p:
        paths.append(p)
    if cal_path.exists():
        cal = pd.read_csv(cal_path)
        p = _plot_reliability(cal, "default", "default", out)
        if p:
            paths.append(p)
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        paths.extend(_plot_summary_bars(summary, out))
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="Plot AirComp Bayesian FL reproduction results")
    p.add_argument("--run", required=True, help="Output folder containing metrics.csv")
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()
    out = args.output_dir or str(Path(args.run).with_name(Path(args.run).name + "_plots"))
    for path in plot_aircomp_results(args.run, out):
        print(path)


if __name__ == "__main__":
    main()
