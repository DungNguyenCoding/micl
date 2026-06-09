"""Run grouped Flower simulation for Fig. 2: effect of BS coverage distance.

This script is additive and does not change the Fig. 1 pipeline in main.py.

Interpretation used for Fig. 2 in this project:
    - Generate/place K=300 physical devices once inside the maximum BS coverage
      disk, cfg.coverage_m, e.g. 550 m.
    - For each requested active radius r in fig2_active_radii, keep only devices
      whose distance to the BS is <= r.
    - Train the proposed BS-assisted OTA-FL method without the TCI benchmark.

This matches the user-requested experiment: total 550 m placement, then evaluate
what happens when the system only cares about devices within 550/300/50 m.
"""

from __future__ import annotations

# Must be set before importing numpy/torch in the main process. Ray workers also
# inherit these values, preventing CPU oversubscription.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import math
from pathlib import Path
from typing import Any, Sequence

import flwr as fl
import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset

import client
import dataset
import model
from config import build_sim_config
from strategy import BsDatasetAssistedOtaStrategy
from utils_fig2 import plot_fig2_histories


def _list_float(values: Any) -> list[float]:
    """Convert Hydra/OmegaConf list-like values into a plain list of floats."""
    if isinstance(values, str):
        stripped = values.strip().strip("[]")
        if not stripped:
            return []
        return [float(x.strip()) for x in stripped.split(",")]
    if isinstance(values, Sequence):
        return [float(v) for v in values]
    return [float(values)]


def _sample_distances_with_seed(num_devices: int, coverage_m: float, seed: int) -> np.ndarray:
    """Sample device distances uniformly in a disk using the same law as Fig. 1.

    For a uniform spatial distribution over a disk, the radius is R*sqrt(U).
    The 10 m lower clip avoids unrealistically tiny path loss, matching the
    existing dataset.sample_client_distances helper.
    """
    rng = np.random.default_rng(int(seed) + 999)
    return np.clip(float(coverage_m) * np.sqrt(rng.random(int(num_devices))), 10.0, float(coverage_m))


def _active_counts_for_radii(distances: np.ndarray, radii: Sequence[float]) -> dict[float, int]:
    return {float(r): int(np.sum(distances <= float(r))) for r in radii}


def _sample_fig2_distances(sim_cfg, cfg, active_radii: Sequence[float]) -> tuple[np.ndarray, int]:
    """Sample Fig. 2 distances and optionally reroll seed if a small radius is empty.

    With only 300 devices uniformly placed in a 550 m disk, the expected number
    inside 50 m is 300*(50/550)^2 ~= 2.48, so some seeds contain zero devices.
    This helper keeps Fig. 1 untouched and only gives Fig. 2 a separate distance
    seed/resampling path.
    """
    base_seed = int(getattr(cfg, "fig2_distance_seed", sim_cfg.split_seed))
    min_active = int(getattr(cfg, "fig2_min_active_devices_per_radius", 1))
    auto_resample = bool(getattr(cfg, "fig2_auto_resample_distance_seed", True))
    max_tries = int(getattr(cfg, "fig2_max_distance_seed_tries", 10_000))

    for offset in range(max_tries if auto_resample else 1):
        seed = base_seed + offset
        distances = _sample_distances_with_seed(sim_cfg.num_devices, sim_cfg.coverage_m, seed)
        counts = _active_counts_for_radii(distances, active_radii)
        if all(counts[float(r)] >= min_active for r in active_radii):
            if offset > 0:
                print(
                    f"Fig. 2 distance seed {base_seed} had too few active devices; "
                    f"using seed {seed} with counts {counts}"
                )
            else:
                print(f"Fig. 2 distance seed {seed} selected with counts {counts}")
            return distances, seed

    raise RuntimeError(
        "Could not sample a Fig. 2 device placement satisfying "
        f"min_active={min_active} for radii={list(active_radii)} after {max_tries} tries. "
        "Reduce fig2_min_active_devices_per_radius or increase num_devices."
    )


def _split_device_ids(device_ids: Sequence[int], num_groups: int) -> list[list[int]]:
    """Split a selected set of physical device IDs across Flower virtual clients."""
    ids = [int(x) for x in device_ids]
    if not ids:
        raise ValueError("Cannot create Flower groups from an empty active-device set")
    num_groups = min(max(1, int(num_groups)), len(ids))
    base = len(ids) // num_groups
    rem = len(ids) % num_groups
    groups: list[list[int]] = []
    cursor = 0
    for gid in range(num_groups):
        size = base + (1 if gid < rem else 0)
        groups.append(ids[cursor : cursor + size])
        cursor += size
    return groups


def _save_active_device_summary(
    path: Path,
    radii: Sequence[float],
    active_ids_by_radius: dict[float, list[int]],
    client_distances: np.ndarray,
    client_sizes: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "active_radius_m",
                "num_active_devices",
                "total_active_client_examples",
                "min_distance_m",
                "max_distance_m",
                "mean_distance_m",
                "active_device_ids",
            ]
        )
        for r in radii:
            ids = active_ids_by_radius[float(r)]
            d = client_distances[np.asarray(ids, dtype=np.int64)] if ids else np.asarray([])
            writer.writerow(
                [
                    float(r),
                    len(ids),
                    int(sum(client_sizes[i] for i in ids)),
                    float(np.min(d)) if len(d) else "",
                    float(np.max(d)) if len(d) else "",
                    float(np.mean(d)) if len(d) else "",
                    ids,
                ]
            )


@hydra.main(config_path="docs/conf", config_name="config_fig2", version_base=None)
def main(cfg: DictConfig) -> None:
    sim_cfg = build_sim_config(cfg)
    fig2_m0 = int(getattr(cfg, "fig2_m0", 160))
    active_radii = _list_float(getattr(cfg, "fig2_active_radii", [550.0, 300.0, 50.0]))
    if not active_radii:
        raise ValueError("fig2_active_radii cannot be empty")
    if any(r <= 0 for r in active_radii):
        raise ValueError("All fig2_active_radii must be positive")
    if any(r > sim_cfg.coverage_m for r in active_radii):
        raise ValueError(
            f"All active radii must be <= coverage_m={sim_cfg.coverage_m}; "
            f"got {active_radii}"
        )

    model.set_seed(sim_cfg.runtime_seed)
    model.configure_torch_threads(sim_cfg.torch_threads)
    device = model.resolve_device(sim_cfg.device)

    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    data_dir = to_absolute_path(str(cfg.data_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = model.Cifar10CNN().to(device)
    d_model = model.count_trainable_params(base_model)
    if d_model != 307_498:
        raise RuntimeError(f"Expected D=307,498 trainable parameters, got {d_model}")
    initial_parameters = model.get_parameters(base_model)

    print("Configuration:\n" + OmegaConf.to_yaml(cfg))
    print(f"Using device: {device}")
    print(f"Flower version: {fl.__version__}")
    print(f"Physical candidate devices K={sim_cfg.num_devices}; placement radius={sim_cfg.coverage_m} m")
    print(f"Fig. 2 active radii: {active_radii}")
    print(f"Fixed BS dataset size |M0|={fig2_m0}")
    print(f"CIFAR-10 CNN trainable parameters D={d_model:,}")
    print(
        f"N=ceil(D/F)={math.ceil(d_model / sim_cfg.num_subchannels)} "
        f"OFDM symbols per update round with F={sim_cfg.num_subchannels}"
    )

    # Build the full 550 m candidate-device population once. Each active-radius
    # experiment will select a subset from these same 300 devices.
    trainset, testset = dataset.load_cifar10(data_dir)
    full_bs_dataset, client_datasets = dataset.partition_cifar10_non_iid(
        trainset=trainset,
        cfg=sim_cfg,
        max_m0=fig2_m0,
    )
    # Fig. 2 uses its own optional distance seed/resampling logic. This leaves
    # the Fig. 1 main.py path untouched, but prevents the 50 m case from failing
    # when a random 300-device placement inside 550 m contains no close device.
    client_distances, used_distance_seed = _sample_fig2_distances(sim_cfg, cfg, active_radii)
    all_client_sizes = [len(ds) for ds in client_datasets]
    print(f"Using Fig. 2 device-distance seed: {used_distance_seed}")

    split_path = output_dir / "data_split_summary.csv"
    dataset.save_split_summary(split_path, full_bs_dataset, client_datasets)
    print(f"Saved full candidate-device split summary: {split_path}")

    # Select active device IDs for each radius. The largest radius should include
    # all 300 devices if it equals cfg.coverage_m.
    active_ids_by_radius: dict[float, list[int]] = {}
    for r in active_radii:
        active_ids = np.where(client_distances <= float(r))[0].astype(int).tolist()
        if not active_ids:
            raise RuntimeError(
                f"No active devices found within radius {r} m. "
                "Set fig2_auto_resample_distance_seed=true, use a different "
                "fig2_distance_seed, or increase num_devices."
            )
        active_ids_by_radius[float(r)] = active_ids
        print(
            f"Active radius {float(r):.1f} m: "
            f"{len(active_ids)} / {sim_cfg.num_devices} devices selected"
        )

    active_summary_path = output_dir / "fig2_active_device_summary.csv"
    _save_active_device_summary(
        active_summary_path,
        active_radii,
        active_ids_by_radius,
        client_distances,
        all_client_sizes,
    )
    print(f"Saved active-device summary: {active_summary_path}")

    bs_subset = Subset(full_bs_dataset.dataset, list(full_bs_dataset.indices[:fig2_m0]))
    client_resources = {
        "num_cpus": float(cfg.client_cpus),
        "num_gpus": float(cfg.client_gpus),
    }
    print(f"Using client_resources: {client_resources}")

    history_paths: list[Path] = []
    for radius in active_radii:
        radius = float(radius)
        active_ids = active_ids_by_radius[radius]
        device_groups = _split_device_ids(active_ids, int(sim_cfg.num_flower_clients))
        active_client_sizes = [all_client_sizes[did] for did in active_ids]
        num_groups = len(device_groups)
        experiment_name = f"rcvge_{int(radius)}m_m0_{fig2_m0}"

        client_fn = client.gen_client_fn(
            device_groups=device_groups,
            client_datasets=client_datasets,
            client_distances_m=client_distances,
            cfg=sim_cfg,
        )

        strategy = BsDatasetAssistedOtaStrategy(
            initial_parameters_ndarrays=initial_parameters,
            bs_dataset=bs_subset,
            testset=testset,
            client_sizes=active_client_sizes,
            cfg=sim_cfg,
            output_dir=output_dir,
            experiment_name=experiment_name,
            total_rounds=int(cfg.num_rounds),
            num_flower_clients=num_groups,
        )

        print(
            f"\n=== Fig. 2 / Proposed OTA-FL / active r_cvge={radius:.1f} m / "
            f"candidate K={sim_cfg.num_devices} / active K={len(active_ids)} / "
            f"C_flower={num_groups} / |M0|={fig2_m0} / "
            f"D={strategy.d_model} / F={sim_cfg.num_subchannels} / N={strategy.n_symbols} ==="
        )
        print(f"Global examples: {strategy.global_examples}, BS weight w0={strategy.w0:.6f}")

        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_groups,
            config=fl.server.ServerConfig(num_rounds=int(cfg.num_rounds)),
            strategy=strategy,
            client_resources=client_resources,
        )

        csv_path = strategy.save_history_csv()
        model_path = strategy.save_model()
        history_paths.append(csv_path)
        print(f"Saved history: {csv_path}")
        print(f"Saved model: {model_path}")

    if bool(cfg.plot):
        plot_path = plot_fig2_histories(history_paths, output_dir)
        if plot_path is not None:
            print(f"Saved Fig. 2 plot: {plot_path}")


if __name__ == "__main__":
    main()
