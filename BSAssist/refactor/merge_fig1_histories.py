"""Merge/plot Fig. 1 histories after rerunning only the missing M0 curve.

Example:
python merge_fig1_histories.py \
  --input outputs/fig1_cov550 outputs/fig1_cov550_m0_20_rerun \
  --output outputs/fig1_cov550_merged
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def find_history(input_dirs: list[Path], m0: int) -> Path:
    name = f"m0_{m0}_history.csv"
    for d in input_dirs:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {name} in: {input_dirs}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="One or more output directories")
    parser.add_argument("--output", required=True, help="Merged output directory")
    parser.add_argument("--m0-values", nargs="+", type=int, default=[1600, 160, 20])
    args = parser.parse_args()

    input_dirs = [Path(x) for x in args.input]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    histories = {}
    for m0 in args.m0_values:
        p = find_history(input_dirs, m0)
        df = pd.read_csv(p)
        histories[m0] = df
        df.to_csv(out / f"m0_{m0}_history.csv", index=False)
        print(f"Using |M0|={m0}: {p} ({len(df)} rows)")

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    for m0 in args.m0_values:
        df = histories[m0].dropna(subset=["Nt"]).sort_values("Nt")
        label = rf"Proposed, $|M_0|={m0}$"
        axes[0].plot(df["Nt"], df["accuracy"], label=label)
        axes[1].plot(df["Nt"], df["distortion"], label=label)

    axes[0].set_title("(a) Test accuracy")
    axes[0].set_ylabel("Accuracy for test dataset")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_title("(b) Aggregated update distortion")
    axes[1].set_xlabel("Number of symbol transmissions, Nt")
    axes[1].set_ylabel(r"2-Norm of Aggregated Distortion, $||\\xi_t||^2$")
    axes[1].set_yscale("log")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    fig_path = out / "fig1_merged.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
