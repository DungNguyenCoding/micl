"""Plot generation only.

Training and numerical metrics intentionally live outside this module so Ray client
workers never import Matplotlib during local optimization.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
from typing import Iterable, Sequence

import yaml

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



def _load_run_config(run_dir: Path) -> dict:
    """Load resolved/source config metadata for labeling comparison plots."""
    for name in ("resolved_config.yaml", "source_config.yaml"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _display_method(method: str) -> str:
    key = str(method or "").strip().lower()
    return {
        "fedavg": "FedAvg",
        "fola": "FOLA",
        "bbb": "BBB",
    }.get(key, method or "Unknown")


def _display_model(model_name: str) -> str:
    key = str(model_name or "").strip().lower()
    return {
        "paper_basiccnn": "BasicCNN",
        "resnet56_gn8": "ResNet-56",
    }.get(key, model_name or "Unknown")


def _comparison_label(run_dir: Path, cfg: dict) -> tuple[str, str, str]:
    method = str(cfg.get("method", ""))
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "")) if isinstance(model_cfg, dict) else ""

    method_label = _display_method(method)
    model_label = _display_model(model_name)

    if method and model_name:
        label = f"{method_label} — {model_label}"
    elif method:
        label = method_label
    else:
        label = run_dir.name

    return label, method_label, model_label


def _accuracy_for_comparison(
    rows: list[dict[str, str]],
    method: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return finite round/accuracy arrays using each method's primary metric."""
    rounds = _numeric(rows, "round")

    candidates = (
        ("fola_mean_accuracy", "accuracy")
        if str(method).lower() == "fola"
        else ("accuracy",)
    )

    selected = "accuracy"
    values = np.full_like(rounds, np.nan, dtype=np.float64)

    for metric in candidates:
        y = _numeric(rows, metric)
        if not np.all(np.isnan(y)):
            values = y
            selected = metric
            break

    mask = np.isfinite(rounds) & np.isfinite(values)
    return rounds[mask], values[mask], selected


def plot_accuracy_comparison(
    run_dirs: Sequence[str | Path],
    *,
    labels: Sequence[str] | None = None,
    output_dir: str | Path = "outputs/plots",
    output_stem: str = "accuracy_comparison",
    title: str = "Accuracy vs Communication Round",
) -> list[Path]:
    """Plot primary global accuracy for multiple BayesFL runs.

    FOLA uses ``fola_mean_accuracy`` when present (falling back to ``accuracy``).
    FedAvg and BBB use the generic ``accuracy`` field. Values are rendered as
    percentages. The function saves PNG, PDF, and a long-form CSV containing the
    merged curves.
    """
    run_paths = [Path(p) for p in run_dirs]
    if not run_paths:
        raise ValueError("run_dirs must contain at least one run directory")

    if labels is not None and len(labels) != len(run_paths):
        raise ValueError("labels must have the same length as run_dirs")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded: list[dict[str, object]] = []

    for idx, run_dir in enumerate(run_paths):
        rows = _read_csv(run_dir / "metrics" / "global_metrics.csv")
        if not rows:
            print(f"WARNING: missing global_metrics.csv for {run_dir}")
            continue

        cfg = _load_run_config(run_dir)
        auto_label, method_label, model_label = _comparison_label(run_dir, cfg)
        method = str(cfg.get("method", ""))

        rounds, accuracy, metric = _accuracy_for_comparison(rows, method)
        if rounds.size == 0:
            print(f"WARNING: no finite accuracy data for {run_dir}")
            continue

        label = labels[idx] if labels is not None else auto_label
        linestyle = "--" if model_label == "ResNet-56" else "-"

        loaded.append(
            {
                "run_dir": run_dir,
                "label": label,
                "method": method_label,
                "model": model_label,
                "metric": metric,
                "rounds": rounds,
                "accuracy": accuracy * 100.0,
                "linestyle": linestyle,
            }
        )

    if not loaded:
        raise RuntimeError("No usable runs were found for accuracy comparison")

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for item in loaded:
        ax.plot(
            item["rounds"],
            item["accuracy"],
            linestyle=item["linestyle"],
            linewidth=2.0,
            label=item["label"],
        )

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    png_path = out_dir / f"{output_stem}.png"
    pdf_path = out_dir / f"{output_stem}.pdf"
    csv_path = out_dir / f"{output_stem}.csv"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "method",
                "model",
                "metric",
                "round",
                "accuracy_percent",
                "run_dir",
            ]
        )
        for item in loaded:
            for rnd, acc in zip(item["rounds"], item["accuracy"]):
                writer.writerow(
                    [
                        item["label"],
                        item["method"],
                        item["model"],
                        item["metric"],
                        int(rnd),
                        float(acc),
                        str(item["run_dir"]),
                    ]
                )

    return [png_path, pdf_path, csv_path]


def latest_matching_run(pattern: str) -> Path:
    """Return the newest directory matching a shell-style glob pattern."""
    matches = [Path(p) for p in glob.glob(pattern) if Path(p).is_dir()]
    if not matches:
        raise FileNotFoundError(f"No run directory matches pattern: {pattern}")
    return max(matches, key=lambda p: p.stat().st_mtime)

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
    parser = argparse.ArgumentParser(description="Generate plots for BayesFL runs")
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Run directory. Repeat for comparison plots.",
    )
    parser.add_argument(
        "--latest-glob",
        action="append",
        default=[],
        help="Add the newest run matching this glob. Repeat as needed.",
    )
    parser.add_argument(
        "--compare-accuracy",
        action="store_true",
        help="Generate one multi-run accuracy-vs-round comparison.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional custom label; repeat once per comparison run.",
    )
    parser.add_argument("--output-dir", default="outputs/plots")
    parser.add_argument("--output-stem", default="accuracy_comparison")
    parser.add_argument("--title", default="Accuracy vs Communication Round")
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.run_dir]
    for pattern in args.latest_glob:
        run_dirs.append(latest_matching_run(pattern))

    if args.compare_accuracy:
        if not run_dirs:
            parser.error("--compare-accuracy requires --run-dir and/or --latest-glob")
        paths = plot_accuracy_comparison(
            run_dirs,
            labels=args.label,
            output_dir=args.output_dir,
            output_stem=args.output_stem,
            title=args.title,
        )
    else:
        if len(run_dirs) != 1:
            parser.error("normal plotting mode requires exactly one --run-dir")
        paths = generate_all_plots(run_dirs[0])

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
