"""Plotting utilities for grouped Flower OTA-FL CIFAR-10 simulation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Sequence


def plot_histories(history_paths: Sequence[Path], output_dir: Path) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plot creation because matplotlib could not be imported: {exc}")
        return None

    histories = {}
    for path in history_paths:
        key = path.stem.replace("_history", "")
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        histories[key] = {
            "Nt": [float(r["Nt"]) for r in rows if float(r["round"]) > 0],
            "accuracy": [float(r["accuracy"]) for r in rows if float(r["round"]) > 0],
            "distortion": [float(r["distortion"]) for r in rows if float(r["round"]) > 0],
        }

    if not histories:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 8.0), sharex=True)
    for name, data in histories.items():
        label = name.replace("m0_", r"$|M_0|=$")
        ax1.plot(data["Nt"], data["accuracy"], label=f"Alg.2, {label}")
        ax2.plot(data["Nt"], data["distortion"], label=f"Alg.2, {label}")

    ax1.set_ylabel("Accuracy for test dataset")
    ax1.set_title("(a) Test accuracy")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.set_xlabel("Number of symbol transmissions, Nt")
    ax2.set_ylabel(r"2-Norm of Aggregated Distortion, $||\xi_t||^2$")
    ax2.set_title("(b) Aggregated update distortion")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "fig1_cifar10_ota_flower_grouped.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
