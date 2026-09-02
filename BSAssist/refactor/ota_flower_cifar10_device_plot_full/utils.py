"""Plotting utilities for grouped Flower OTA-FL CIFAR-10 simulation."""

from __future__ import annotations

import csv
import numpy as np
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


def save_device_distance_visualization(
    distances_m: np.ndarray,
    coverage_m: float,
    output_dir: Path,
    *,
    filename_prefix: str = "device_distribution",
    active_radii_m: Optional[Sequence[float]] = None,
    angle_seed: int = 2026,
    label_devices: bool = True,
    ring_step_m: float = 50.0,
) -> Optional[Path]:
    """Visualize 1D device-to-BS distances on a 2D radar-like map.

    The simulation itself only uses radial distance. This helper assigns a
    reproducible random angle to each distance purely for visualization,
    converts polar coordinates to Cartesian coordinates, and saves both a PNG
    figure and CSV metadata into the experiment output directory.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping device distance plot because matplotlib could not be imported: {exc}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    distances_m = np.asarray(distances_m, dtype=np.float64).reshape(-1)
    if distances_m.size == 0:
        print("Skipping device distance plot because no distances were provided.")
        return None

    rng = np.random.default_rng(int(angle_seed))
    theta = 2.0 * np.pi * rng.random(distances_m.size)

    r_km = distances_m / 1000.0
    coverage_km = float(coverage_m) / 1000.0
    x_km = r_km * np.cos(theta)
    y_km = r_km * np.sin(theta)

    if active_radii_m is None:
        active_radii_m = []
    active_radii_m = [float(r) for r in active_radii_m]
    active_radii_m = sorted(set(active_radii_m), reverse=True)

    # Save device coordinates/distances as CSV
    csv_path = output_dir / f"{filename_prefix}_devices.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["device_id", "distance_m", "distance_km", "theta_rad", "x_km", "y_km"]
        for rad in active_radii_m:
            header.append(f"active_le_{int(rad)}m")
        writer.writerow(header)
        for idx in range(distances_m.size):
            row = [
                int(idx),
                float(distances_m[idx]),
                float(r_km[idx]),
                float(theta[idx]),
                float(x_km[idx]),
                float(y_km[idx]),
            ]
            for rad in active_radii_m:
                row.append(int(distances_m[idx] <= rad))
            writer.writerow(row)

    # Save active-radius summary as CSV
    if active_radii_m:
        summary_path = output_dir / f"{filename_prefix}_active_summary.csv"
        with summary_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["active_radius_m", "num_active_devices"])
            for rad in active_radii_m:
                writer.writerow([float(rad), int(np.sum(distances_m <= rad))])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Concentric rings every ring_step_m (default 50 m)
    max_ring = int(np.floor(float(coverage_m) / max(float(ring_step_m), 1.0)))
    for k in range(1, max_ring + 1):
        rad_m = k * float(ring_step_m)
        rad_km = rad_m / 1000.0
        is_outer = abs(rad_m - float(coverage_m)) < 1e-9
        circle = plt.Circle(
            (0.0, 0.0),
            rad_km,
            color="gray",
            fill=False,
            linestyle="--",
            linewidth=1.0 if is_outer else 0.7,
            alpha=0.5 if is_outer else 0.25,
            zorder=0,
        )
        ax.add_patch(circle)
        ax.text(
            rad_km,
            0.0,
            f"{int(rad_m)}m",
            fontsize=7,
            color="gray",
            alpha=0.8,
            ha="left",
            va="bottom",
        )

    # Highlight requested active radii (useful for Fig. 2)
    highlight_styles = [
        ("tab:orange", "-"),
        ("tab:green", "-"),
        ("tab:red", ":"),
        ("tab:purple", "-."),
    ]
    for i, rad_m in enumerate(sorted(set(active_radii_m))):
        color, ls = highlight_styles[i % len(highlight_styles)]
        rad_km = rad_m / 1000.0
        circle = plt.Circle(
            (0.0, 0.0),
            rad_km,
            color=color,
            fill=False,
            linestyle=ls,
            linewidth=1.7,
            alpha=0.9,
            zorder=1,
            label=f"active radius = {int(rad_m)} m",
        )
        ax.add_patch(circle)

    ax.scatter(0.0, 0.0, c="red", marker="^", s=160, label="Edge Server", zorder=5)
    ax.scatter(
        x_km,
        y_km,
        c="skyblue",
        edgecolors="navy",
        s=70,
        alpha=0.85,
        label="Edge Devices",
        zorder=3,
    )

    if label_devices:
        for i in range(distances_m.size):
            ax.annotate(
                f"{i}",
                (x_km[i], y_km[i]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                zorder=6,
            )

    ax.set_title(f"Device Distribution ({int(coverage_m)}m Radius)")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Distance (km)")
    ax.grid(True, alpha=0.2)
    ax.axis("equal")
    pad_km = max(0.05, coverage_km * 0.08)
    ax.set_xlim(-coverage_km - pad_km, coverage_km + pad_km)
    ax.set_ylim(-coverage_km - pad_km, coverage_km + pad_km)
    ax.legend(loc="upper right")

    fig.tight_layout()
    png_path = output_dir / f"{filename_prefix}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved device distance visualization: {png_path}")
    print(f"Saved device coordinate CSV: {csv_path}")
    return png_path
