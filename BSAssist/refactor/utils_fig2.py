"""Plotting utilities for Fig. 2 coverage-radius simulations."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional, Sequence


def _label_from_path(path: Path) -> str:
    # Expected stem: rcvge_550m_m0_160_history
    match = re.search(r"rcvge_(\d+)m", path.stem)
    if match:
        return rf"Alg.2, $r_{{cvge}}={match.group(1)}$m"
    return path.stem.replace("_history", "")


def plot_fig2_histories(history_paths: Sequence[Path], output_dir: Path) -> Optional[Path]:
    """Create the Fig. 2-style test-accuracy plot without TCI curves."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plot creation because matplotlib could not be imported: {exc}")
        return None

    if not history_paths:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8))
    for path in history_paths:
        with path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        nt = [float(r["Nt"]) for r in rows if float(r["round"]) > 0]
        acc = [float(r["accuracy"]) for r in rows if float(r["round"]) > 0]
        ax.plot(nt, acc, label=_label_from_path(path))

    ax.set_xlabel("Number of symbol transmissions, Nt")
    ax.set_ylabel("Accuracy for test dataset")
    ax.set_title("Effect of BS coverage distance on test accuracy")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "fig2_coverage_accuracy.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
