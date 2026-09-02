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



def plot_proposed_debug(metrics: pd.DataFrame, output_dir: Path) -> Path:
    """Plot posterior-predictive vs posterior-mean accuracy for Proposed.

    This diagnostic is not one of the paper figures. It helps determine whether
    poor Bayesian accuracy comes from the learned mean or from posterior
    sampling/variance.
    """
    frame = metrics[metrics["method"] == "proposed"].copy()
    if frame.empty:
        raise ValueError("No proposed-method rows found")
    required = {"posterior_predictive_accuracy", "posterior_mean_accuracy"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing v1.4 diagnostic columns: {sorted(missing)}")

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for column, label in (
        ("posterior_predictive_accuracy", "Posterior predictive"),
        ("posterior_mean_accuracy", "Posterior mean"),
    ):
        grouped = (
            frame.groupby("channel_uses_cumulative", as_index=False)[column]
            .mean()
            .sort_values("channel_uses_cumulative")
        )
        axis.plot(grouped["channel_uses_cumulative"], grouped[column], label=label)
    axis.set_xlabel("Number of channel uses")
    axis.set_ylabel("Accuracy for test dataset")
    axis.set_ylim(0.05, 1.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "proposed_posterior_mean_vs_predictive.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _sparse_final_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics[metrics["experiment"] == "sparse"].copy()
    if frame.empty:
        raise ValueError("No sparse experiment rows found")
    final_rounds = frame.groupby("run_id")["round"].transform("max")
    final = frame[frame["round"] == final_rounds].copy()
    if "sparse_selection" not in final.columns:
        final["sparse_selection"] = final["condition"].str.split("_keep").str[0]
    if "sparse_keep_ratio" not in final.columns:
        final["sparse_keep_ratio"] = (
            final["condition"].str.split("keep").str[-1].astype(float) / 100.0
        )
    final["keep_percent"] = 100.0 * final["sparse_keep_ratio"].astype(float)
    return final


def _sparse_target_round(metrics: pd.DataFrame) -> int:
    """Return the sparse experiment round used for the shared dense baseline."""
    frame = metrics[metrics["experiment"] == "sparse"].copy()
    if frame.empty:
        raise ValueError("No sparse experiment rows found")
    rounds = pd.to_numeric(frame["round"], errors="coerce").dropna()
    if rounds.empty:
        raise ValueError("Sparse metrics do not contain a valid round column")
    return int(rounds.max())


def _seed_values(frame: pd.DataFrame) -> set[int]:
    if "seed" not in frame.columns:
        return set()
    values = pd.to_numeric(frame["seed"], errors="coerce").dropna()
    return {int(value) for value in values.tolist()}


def _shared_dense_keep100_rows(
    sparse_metrics: pd.DataFrame,
    dense_metrics: pd.DataFrame,
    *,
    target_round: int,
) -> pd.DataFrame:
    """Create Bayesian/random keep100 rows from an existing dense Fig.2 run.

    The dense Proposed trajectory is mathematically identical to both sparse
    selectors at keep_ratio=1.0.  We therefore reuse the requested Figure-2
    round rather than rerunning two redundant sparse conditions.
    """
    baseline = dense_metrics[
        (dense_metrics["experiment"] == "fig2")
        & (dense_metrics["method"] == "proposed")
        & (pd.to_numeric(dense_metrics["round"], errors="coerce") <= int(target_round))
    ].copy()
    if baseline.empty:
        raise ValueError(
            "Dense baseline has no Proposed Fig.2 rows at or before "
            f"round {target_round}"
        )
    if int(pd.to_numeric(baseline["round"], errors="coerce").max()) < int(target_round):
        raise ValueError(
            f"Dense baseline stops before sparse target round {target_round}"
        )

    sparse_seeds = _seed_values(sparse_metrics)
    dense_seeds = _seed_values(baseline)
    if sparse_seeds and dense_seeds:
        missing = sparse_seeds.difference(dense_seeds)
        if missing:
            raise ValueError(
                "Dense baseline is missing sparse realization seed(s): "
                + ", ".join(str(value) for value in sorted(missing))
            )
        baseline = baseline[
            pd.to_numeric(baseline["seed"], errors="coerce").isin(sparse_seeds)
        ].copy()

    copies = []
    for selection in ("bayesian", "random"):
        part = baseline.copy()
        part["experiment"] = "sparse"
        part["condition"] = f"{selection}_keep100"
        part["sparse_selection"] = selection
        part["sparse_keep_ratio"] = 1.0
        part["run_id"] = part["run_id"].astype(str) + f"__{selection}_keep100_reuse"
        copies.append(part)
    return pd.concat(copies, ignore_index=True, sort=False)


def _shared_dense_keep100_reliability(
    sparse_metrics: pd.DataFrame,
    dense_reliability: pd.DataFrame,
    *,
    target_round: int,
) -> pd.DataFrame:
    """Create keep100 reliability rows from Fig.2 at exactly target_round."""
    if dense_reliability.empty:
        return pd.DataFrame()
    baseline = dense_reliability[
        (dense_reliability["experiment"] == "fig2")
        & (dense_reliability["method"] == "proposed")
        & (pd.to_numeric(dense_reliability["round"], errors="coerce") == int(target_round))
    ].copy()
    if baseline.empty:
        raise ValueError(
            "Dense baseline reliability.csv has no Proposed Fig.2 rows at "
            f"round {target_round}"
        )

    sparse_seeds = _seed_values(sparse_metrics)
    dense_seeds = _seed_values(baseline)
    if sparse_seeds and dense_seeds:
        missing = sparse_seeds.difference(dense_seeds)
        if missing:
            raise ValueError(
                "Dense baseline reliability data is missing sparse seed(s): "
                + ", ".join(str(value) for value in sorted(missing))
            )
        baseline = baseline[
            pd.to_numeric(baseline["seed"], errors="coerce").isin(sparse_seeds)
        ].copy()

    copies = []
    for selection in ("bayesian", "random"):
        part = baseline.copy()
        part["experiment"] = "sparse"
        part["condition"] = f"{selection}_keep100"
        part["run_id"] = part["run_id"].astype(str) + f"__{selection}_keep100_reuse"
        copies.append(part)
    return pd.concat(copies, ignore_index=True, sort=False)


def _aggregate_reliability_condition(group: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bin_index, bin_group in group.groupby("bin"):
        counts = bin_group["count"].to_numpy(dtype=float)
        total = max(float(counts.sum()), 1.0)
        rows.append(
            {
                "bin": int(bin_index),
                "lower": float(bin_group["lower"].iloc[0]),
                "upper": float(bin_group["upper"].iloc[0]),
                "count": float(counts.sum()),
                "confidence": float(np.dot(bin_group["confidence"], counts) / total),
                "accuracy": float(np.dot(bin_group["accuracy"], counts) / total),
            }
        )
    return pd.DataFrame(rows).sort_values("bin")


def _draw_reliability(axis: plt.Axes, grouped: pd.DataFrame, title: str) -> None:
    centers = 0.5 * (grouped["lower"] + grouped["upper"])
    widths = grouped["upper"] - grouped["lower"]
    axis.bar(
        centers,
        grouped["accuracy"],
        width=widths * 0.9,
        alpha=0.75,
        label="Outputs",
    )
    gap_bottom = np.minimum(grouped["accuracy"], grouped["confidence"])
    gap_height = np.abs(grouped["accuracy"] - grouped["confidence"])
    axis.bar(
        centers,
        gap_height,
        bottom=gap_bottom,
        width=widths * 0.9,
        alpha=0.3,
        label="Gap",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    ece = (
        np.sum(grouped["count"] * np.abs(grouped["accuracy"] - grouped["confidence"]))
        / max(float(grouped["count"].sum()), 1.0)
    )
    axis.set_title(f"{title} — ECE={ece:.3f}")
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Accuracy")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.2)


def plot_sparse_suite(
    metrics: pd.DataFrame,
    reliability: pd.DataFrame,
    output_dir: Path,
    dense_metrics: pd.DataFrame | None = None,
    dense_reliability: pd.DataFrame | None = None,
    dense_baseline_round: int | None = None,
) -> Path:
    """Create the Bayesian-vs-random posterior-sparsity experiment plots."""
    sparse_only = metrics[metrics["experiment"] == "sparse"].copy()
    if sparse_only.empty:
        raise ValueError("No sparse experiment rows found")
    target_round = (
        int(dense_baseline_round)
        if dense_baseline_round is not None
        else _sparse_target_round(metrics)
    )

    plot_metrics = metrics.copy()
    plot_reliability = reliability.copy()
    if dense_metrics is not None:
        dense_keep100 = _shared_dense_keep100_rows(
            sparse_only,
            dense_metrics,
            target_round=target_round,
        )
        plot_metrics = pd.concat(
            [plot_metrics, dense_keep100], ignore_index=True, sort=False
        )
        if dense_reliability is not None and not dense_reliability.empty:
            dense_keep100_rel = _shared_dense_keep100_reliability(
                sparse_only,
                dense_reliability,
                target_round=target_round,
            )
            if not dense_keep100_rel.empty:
                plot_reliability = pd.concat(
                    [plot_reliability, dense_keep100_rel],
                    ignore_index=True,
                    sort=False,
                )
        print(
            "Sparse plotting: reusing dense Proposed Fig.2 as shared keep100 "
            f"baseline at round {target_round}."
        )
    else:
        print(
            "Sparse plotting: no --dense-baseline supplied; keep100 will be "
            "omitted from sparse plots."
        )

    final = _sparse_final_metrics(plot_metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    # 1) Final accuracy vs keep ratio.
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for selection in ("bayesian", "random"):
        sub = final[final["sparse_selection"] == selection]
        if sub.empty:
            continue
        grouped = (
            sub.groupby("keep_percent", as_index=False)
            .agg(value=("accuracy", "mean"), std=("accuracy", "std"))
            .sort_values("keep_percent")
        )
        grouped["std"] = grouped["std"].fillna(0.0)
        axis.errorbar(
            grouped["keep_percent"],
            grouped["value"],
            yerr=grouped["std"],
            marker="o",
            capsize=3,
            label="Bayesian update-SNR" if selection == "bayesian" else "Random",
        )
    axis.set_xlabel("Keep ratio (%)")
    axis.set_ylabel("Final global accuracy")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "sparse_accuracy_vs_keep_ratio.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    created.append(path)

    # 2) Final ECE vs keep ratio.
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for selection in ("bayesian", "random"):
        sub = final[final["sparse_selection"] == selection]
        if sub.empty:
            continue
        grouped = (
            sub.groupby("keep_percent", as_index=False)["ece"]
            .mean()
            .sort_values("keep_percent")
        )
        axis.plot(
            grouped["keep_percent"],
            grouped["ece"],
            marker="o",
            label="Bayesian update-SNR" if selection == "bayesian" else "Random",
        )
    axis.set_xlabel("Keep ratio (%)")
    axis.set_ylabel("Final expected calibration error")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "sparse_ece_vs_keep_ratio.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    created.append(path)

    # 3) Round-by-round accuracy vs sparse payload channel uses, all settings.
    frame = plot_metrics[plot_metrics["experiment"] == "sparse"].copy()
    figure, axis = plt.subplots(figsize=(9.0, 5.8))
    for condition in sorted(frame["condition"].dropna().unique()):
        # Bayesian/random keep100 are the exact same reused dense trajectory.
        # Draw it only once in the round-by-round overlay to avoid duplicate
        # indistinguishable legend entries.
        if condition == "random_keep100":
            continue
        sub = frame[frame["condition"] == condition]
        curve = _mean_curve(sub)
        if condition == "bayesian_keep100":
            label = "Dense 100% (shared)"
        else:
            label = str(condition).replace("bayesian_keep", "Bayes ").replace(
                "random_keep", "Random "
            ) + "%"
        axis.plot(
            curve["channel_uses_cumulative"],
            curve["accuracy"],
            label=label,
        )
    axis.set_xlabel("Sparse posterior values transmitted (cumulative)")
    axis.set_ylabel("Accuracy for test dataset")
    axis.set_ylim(0.05, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    path = output_dir / "sparse_accuracy_vs_channel_uses_all.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    created.append(path)

    # 4) Final accuracy vs total communication cost (Pareto-style).
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for selection in ("bayesian", "random"):
        sub = final[final["sparse_selection"] == selection]
        if sub.empty:
            continue
        grouped = (
            sub.groupby("keep_percent", as_index=False)
            .agg(
                accuracy=("accuracy", "mean"),
                channel_uses=("channel_uses_cumulative", "mean"),
            )
            .sort_values("channel_uses")
        )
        axis.plot(
            grouped["channel_uses"],
            grouped["accuracy"],
            marker="o",
            label="Bayesian update-SNR" if selection == "bayesian" else "Random",
        )
        for _, row in grouped.iterrows():
            axis.annotate(
                f"{row['keep_percent']:.0f}%",
                (row["channel_uses"], row["accuracy"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.set_xlabel("Total sparse posterior values transmitted")
    axis.set_ylabel("Final global accuracy")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "sparse_final_accuracy_vs_total_channel_uses.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    created.append(path)

    # 5) Figure-6-style reliability: one file per selection/keep ratio + grid.
    rel = _final_reliability_rows(plot_reliability, "sparse")
    if not rel.empty:
        condition_order = [
            f"{selection}_keep{keep}"
            for selection in ("bayesian", "random")
            for keep in (100, 75, 50, 25, 10, 5, 2)
        ]
        grid_fig, grid_axes = plt.subplots(
            2, 7, figsize=(24.0, 7.0), squeeze=False, sharex=True, sharey=True
        )
        for index, condition in enumerate(condition_order):
            sub = rel[rel["condition"] == condition]
            if sub.empty:
                continue
            grouped = _aggregate_reliability_condition(sub)
            selection, keep_text = condition.split("_keep", 1)
            title = (
                ("Bayes" if selection == "bayesian" else "Random")
                + f" — keep {keep_text}%"
            )
            row = 0 if selection == "bayesian" else 1
            col = (100, 75, 50, 25, 10, 5, 2).index(int(keep_text))
            _draw_reliability(grid_axes[row][col], grouped, title)

            single_fig, single_axis = plt.subplots(figsize=(5.2, 4.5))
            _draw_reliability(single_axis, grouped, title)
            single_axis.legend()
            single_fig.tight_layout()
            single_path = output_dir / f"sparse_reliability_{condition}.png"
            single_fig.savefig(single_path, dpi=200)
            plt.close(single_fig)
            created.append(single_path)

        handles, labels = grid_axes[0][0].get_legend_handles_labels()
        if handles:
            grid_fig.legend(handles, labels, loc="upper center", ncol=3)
        grid_fig.tight_layout(rect=(0, 0, 1, 0.95))
        grid_path = output_dir / "sparse_reliability_grid.png"
        grid_fig.savefig(grid_path, dpi=180)
        plt.close(grid_fig)
        created.append(grid_path)

    print("Sparse suite created:")
    for item in created:
        print(f"  {item}")
    return created[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-style plots from CSV logs")
    parser.add_argument("--input", default="results", help="Directory containing metrics.csv")
    parser.add_argument("--output", default=None, help="Plot output directory")
    parser.add_argument(
        "--dense-baseline",
        default=None,
        help=(
            "Optional Figure-2 result directory used as the shared 100%% keep "
            "baseline for --figure sparse. The plotter reuses the Proposed row "
            "at the sparse experiment's final round instead of requiring two "
            "redundant keep100 sparse runs."
        ),
    )
    parser.add_argument(
        "--dense-baseline-round",
        type=int,
        default=None,
        help=(
            "Round to reuse from --dense-baseline. By default, use the highest "
            "round present in the sparse experiment (normally 120)."
        ),
    )
    parser.add_argument(
        "--figure",
        default="all",
        choices=["fig2", "fig3", "fig4", "fig5", "fig6", "ece", "proposed_debug", "sparse", "all"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, reliability = load_results(input_dir)
    dense_metrics = None
    dense_reliability = None
    if args.dense_baseline is not None:
        dense_metrics, dense_reliability = load_results(args.dense_baseline)

    requested = (
        ["fig2", "fig3", "fig4", "fig5", "fig6", "ece", "proposed_debug"]
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
        "proposed_debug": lambda: plot_proposed_debug(metrics, output_dir),
        "sparse": lambda: plot_sparse_suite(
            metrics,
            reliability,
            output_dir,
            dense_metrics=dense_metrics,
            dense_reliability=dense_reliability,
            dense_baseline_round=args.dense_baseline_round,
        ),
    }
    for name in requested:
        try:
            path = functions[name]()
            print(f"Created {path}")
        except (ValueError, KeyError) as exc:
            print(f"Skipped {name}: {exc}")


# ============================================================================
# v1.6.2: 3-seed sparse final-accuracy plot with one shared dense-100% line
# ============================================================================

def plot_sparse_final_accuracy_rep3(
    input_dir,
    dense_baseline_dirs,
    target_round=160,
    output_path=None,
):
    """Plot final sparse accuracy vs keep ratio averaged over seeds.

    Sparse Bayesian and random curves use final accuracy at ``target_round``.
    The 100% dense reference is *not* duplicated as Bayesian/Random points:
    for each dense seed we take the best accuracy observed up to target_round,
    then draw one grey dashed horizontal line at the seed-average best value.

    Returns ``(png_path, csv_path)``.
    """
    import csv as _csv
    import math as _math
    from collections import defaultdict as _defaultdict
    from pathlib import Path as _Path

    import matplotlib.pyplot as _plt
    import numpy as _np

    input_dir = _Path(input_dir)
    dense_baseline_dirs = [_Path(p) for p in dense_baseline_dirs]
    target_round = int(target_round)
    metrics_path = input_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing sparse metrics: {metrics_path}")

    def _read(path):
        with _Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
            return list(_csv.DictReader(handle))

    def _float(row, key, default=float("nan")):
        try:
            return float(row.get(key, ""))
        except (TypeError, ValueError):
            return default

    def _int(row, key, default=-1):
        try:
            return int(float(row.get(key, "")))
        except (TypeError, ValueError):
            return default

    def _first_present(row, names, default=""):
        for name in names:
            value = row.get(name, "")
            if value not in (None, ""):
                return value
        return default

    # ----- sparse final rows -----
    sparse_rows = _read(metrics_path)
    groups = _defaultdict(list)
    for row in sparse_rows:
        if str(row.get("method", "")).lower() != "proposed":
            continue
        if _int(row, "round") != target_round:
            continue
        selection = str(
            _first_present(row, ("sparse_selection", "sparse_selection_method"), "")
        ).lower()
        if selection not in {"bayesian", "random"}:
            continue
        keep = _float(
            {"v": _first_present(row, ("sparse_keep_ratio", "sparse_ratio"), "nan")},
            "v",
        )
        acc = _float(row, "accuracy")
        if not (_math.isfinite(keep) and _math.isfinite(acc)):
            continue
        groups[(selection, round(keep * 100.0, 10))].append(acc)

    if not groups:
        raise RuntimeError(
            f"No sparse Proposed rows found at round {target_round} in {metrics_path}"
        )

    summary = []
    for (selection, keep_percent), values in sorted(groups.items()):
        arr = _np.asarray(values, dtype=float)
        summary.append(
            {
                "selection": selection,
                "keep_percent": float(keep_percent),
                "n_seeds": int(arr.size),
                "accuracy_mean": float(arr.mean()),
                "accuracy_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "metric": "final_accuracy",
            }
        )

    # ----- shared dense 100% line -----
    dense_by_run = _defaultdict(list)
    for folder in dense_baseline_dirs:
        path = folder / "metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing dense metrics: {path}")
        for row in _read(path):
            if str(row.get("method", "")).lower() != "proposed":
                continue
            rnd = _int(row, "round")
            if rnd < 0 or rnd > target_round:
                continue
            dense_by_run[(str(row.get("run_id", "")), _int(row, "seed"))].append(row)

    if not dense_by_run:
        raise RuntimeError("No dense Proposed baseline rows were found.")

    # One best value per seed.  If the same seed appears in multiple folders,
    # keep the best valid run for that seed.
    dense_seed_best = {}
    dense_seed_round = {}
    for (_run_id, seed), rows in dense_by_run.items():
        valid = [r for r in rows if _math.isfinite(_float(r, "accuracy"))]
        if not valid:
            continue
        best = max(valid, key=lambda r: _float(r, "accuracy"))
        best_acc = _float(best, "accuracy")
        if seed not in dense_seed_best or best_acc > dense_seed_best[seed]:
            dense_seed_best[seed] = best_acc
            dense_seed_round[seed] = _int(best, "round")

    dense_values = _np.asarray(
        [dense_seed_best[s] for s in sorted(dense_seed_best)], dtype=float
    )
    if dense_values.size == 0:
        raise RuntimeError("Dense baseline contains no finite accuracy values.")
    dense_mean = float(dense_values.mean())
    dense_std = float(dense_values.std(ddof=1)) if dense_values.size > 1 else 0.0

    # ----- save numerical summary -----
    plots_dir = input_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    png_path = _Path(output_path) if output_path else plots_dir / "final_accuracy_vs_keep_ratio_rep3.png"
    csv_path = png_path.with_suffix(".csv")

    csv_rows = list(summary)
    csv_rows.append(
        {
            "selection": "dense_shared",
            "keep_percent": 100.0,
            "n_seeds": int(dense_values.size),
            "accuracy_mean": dense_mean,
            "accuracy_std": dense_std,
            "metric": "best_accuracy_up_to_target_round",
        }
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(
            handle,
            fieldnames=[
                "selection",
                "keep_percent",
                "n_seeds",
                "accuracy_mean",
                "accuracy_std",
                "metric",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    # ----- plot -----
    fig, ax = _plt.subplots(figsize=(8.4, 5.4))
    for selection in ("bayesian", "random"):
        sub = sorted(
            [r for r in summary if r["selection"] == selection],
            key=lambda r: r["keep_percent"],
        )
        if not sub:
            continue
        x = [r["keep_percent"] for r in sub]
        y = [r["accuracy_mean"] for r in sub]
        e = [r["accuracy_std"] for r in sub]
        ax.errorbar(
            x,
            y,
            yerr=e,
            marker="o",
            linewidth=2,
            capsize=4,
            label="Bayesian sparse" if selection == "bayesian" else "Random sparse",
        )

    ax.axhline(
        dense_mean,
        linestyle="--",
        linewidth=2,
        color="grey",
        label=f"Dense Proposed 100% (best={dense_mean:.4f})",
    )
    if dense_std > 0.0:
        ax.axhspan(
            dense_mean - dense_std,
            dense_mean + dense_std,
            color="grey",
            alpha=0.10,
        )

    ax.set_xlabel("Keep ratio (%)", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title(
        f"Bayesian vs Random Sparse Posterior Communication\n"
        f"seed average, {target_round} logical rounds",
        fontsize=13,
    )
    ax.set_xticks([2, 5, 10, 25, 50, 75, 100])
    ax.set_xlim(0, 102)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    _plt.close(fig)

    print("Sparse final-accuracy summary:")
    for row in summary:
        print(
            f"  {row['selection']:8s} keep={row['keep_percent']:>5.0f}% "
            f"n={row['n_seeds']} final={row['accuracy_mean']:.4f} "
            f"+/- {row['accuracy_std']:.4f}"
        )
    print("Dense 100% per-seed best (up to target round):")
    for seed in sorted(dense_seed_best):
        print(
            f"  seed={seed}: best={dense_seed_best[seed]:.4f} "
            f"at round={dense_seed_round[seed]}"
        )
    print(f"Dense shared line: {dense_mean:.4f} +/- {dense_std:.4f}")
    print(f"Created {png_path}")
    print(f"Created {csv_path}")
    return png_path, csv_path


def _v162_sparse_final_accuracy_cli_intercept():
    """Handle the new figure mode before the legacy utils.py parser runs."""
    import argparse as _argparse
    import sys as _sys

    if "--figure" not in _sys.argv:
        return
    try:
        value = _sys.argv[_sys.argv.index("--figure") + 1]
    except (ValueError, IndexError):
        return
    if value != "sparse-final-accuracy":
        return

    parser = _argparse.ArgumentParser(
        description="Plot 3-seed sparse final accuracy with one shared dense-100% line."
    )
    parser.add_argument("--input", required=True, help="Sparse experiment output directory")
    parser.add_argument("--figure", required=True)
    parser.add_argument(
        "--dense-baseline",
        action="append",
        required=True,
        help="Dense Proposed result directory. Repeat this option for multiple folders.",
    )
    parser.add_argument(
        "--dense-baseline-round",
        type=int,
        default=160,
        help="Target logical round; sparse uses final accuracy at this round, dense uses best <= round.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path. CSV is written beside it.",
    )
    args = parser.parse_args()
    plot_sparse_final_accuracy_rep3(
        input_dir=args.input,
        dense_baseline_dirs=args.dense_baseline,
        target_round=args.dense_baseline_round,
        output_path=args.output,
    )
    raise SystemExit(0)


_v162_sparse_final_accuracy_cli_intercept()
# ============================================================================
# end v1.6.2 insertion
# ============================================================================


if __name__ == "__main__":
    main()
