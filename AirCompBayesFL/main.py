"""Command-line entry point for AirCompBayesFL simulations."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from bayesian_protocol import physical_round_count
from config import SimulationConfig
from dataset import ensure_mnist, prepare_partitions
from experiments import RunSpec, derive_rounds, experiment_conditions
from models import build_model, count_parameters
from runtime_utils import (
    configure_runtime_environment,
    resolve_backend,
    validate_runtime,
)
from server import run_configured_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the simulation section of Distribution-Level AirComp for Wireless FL"
    )
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument(
        "--experiment",
        default="fig2",
        choices=["fig2", "fig3", "fig4", "fig5", "fig6", "all"],
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated subset: fedavg,fedprox,scaffold,proposed",
    )
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-wireless", action="store_true")
    parser.add_argument(
        "--power-control-mode",
        default=None,
        choices=["paper_reference_kkt"],
        help=(
            "v1.5.0 paper-reference mode: one shared KKT magnitude optimizer "
            "with target-2025 scaling for Proposed and Hong-2023 reference-[13] "
            "scaling for deterministic benchmarks."
        ),
    )
    parser.add_argument(
        "--deterministic-reference-power-mode",
        default=None,
        choices=["coordinated_aggregate", "weighted_local"],
        help=(
            "rho_ref adaptation for Hong-2023 deterministic benchmark power "
            "control. Default: coordinated_aggregate."
        ),
    )
    parser.add_argument(
        "--path-loss-reference-m",
        type=float,
        default=None,
        help=(
            "Override wireless.path_loss_reference_m. The default configs use "
            "1000 m, equivalent to expressing distance in km."
        ),
    )
    parser.add_argument("--force-partitions", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def completed_runs(metrics_path: Path) -> Dict[str, int]:
    if not metrics_path.exists():
        return {}
    maximum_round: dict[str, int] = {}
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            run_id = row.get("run_id", "")
            try:
                server_round = int(row.get("round", "0"))
            except ValueError:
                continue
            maximum_round[run_id] = max(maximum_round.get(run_id, -1), server_round)
    return maximum_round


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = SimulationConfig.from_yaml(config_path)

    if args.replications is not None:
        config.runtime.replications = args.replications
    if args.rounds is not None:
        config.training.num_rounds = args.rounds
    if args.seed is not None:
        config.runtime.seed = args.seed
    if args.output is not None:
        config.output.directory = args.output
    if args.no_wireless:
        config.wireless.enabled = False
    if args.power_control_mode is not None:
        config.wireless.power_control_mode = args.power_control_mode
    if args.deterministic_reference_power_mode is not None:
        config.wireless.deterministic_reference_power_mode = (
            args.deterministic_reference_power_mode
        )
    if args.path_loss_reference_m is not None:
        config.wireless.path_loss_reference_m = args.path_loss_reference_m

    # Apply CLI overrides before validating runtime/GPU consistency.
    config.validate()
    configure_runtime_environment(config)
    gpu_status = validate_runtime(config)

    data_root = Path(config.data.root)
    if not data_root.is_absolute():
        config.data.root = str((project_root / data_root).resolve())
    output_root = Path(config.output.directory)
    if not output_root.is_absolute():
        config.output.directory = str((project_root / output_root).resolve())
    Path(config.output.directory).mkdir(parents=True, exist_ok=True)

    methods_override: Optional[List[str]] = None
    if args.methods:
        methods_override = [part.strip().lower() for part in args.methods.split(",") if part.strip()]

    experiments = (
        ["fig2", "fig3", "fig4", "fig5", "fig6"]
        if args.experiment == "all"
        else [args.experiment]
    )
    model_dimension = count_parameters(
        build_model(config.model.name, config.model.num_classes)
    )

    ensure_mnist(config.data.root)
    partition_dir = Path(config.output.directory) / "partitions"
    metrics_path = Path(config.output.directory) / config.output.metrics_filename
    finished = completed_runs(metrics_path) if args.resume else {}

    planned: List[tuple[SimulationConfig, RunSpec, str]] = []
    condition_counter = 0
    for experiment in experiments:
        for condition in experiment_conditions(experiment, config, methods_override):
            condition_counter += 1
            condition_cfg = config.copy()
            condition_cfg.data.labels_per_client = condition.labels_per_client
            condition_cfg.data.mean_samples_per_client = condition.mean_samples_per_client
            condition_cfg.wireless.power_dbm = condition.power_dbm

            for realization in range(condition_cfg.runtime.replications):
                partition_seed = (
                    condition_cfg.runtime.seed
                    + 10_000 * condition_counter
                    + realization
                )
                partition_path = prepare_partitions(
                    condition_cfg.data,
                    partition_seed,
                    partition_dir,
                    force=args.force_partitions,
                )
                for method in condition.methods:
                    rounds = derive_rounds(condition_cfg, method, model_dimension)
                    run_id = (
                        f"{condition.experiment}_{condition.name}_{method}_"
                        f"rep{realization:02d}_seed{partition_seed}"
                    )
                    run_spec = RunSpec(
                        run_id=run_id,
                        experiment=condition.experiment,
                        condition=condition.name,
                        method=method,
                        realization=realization,
                        seed=partition_seed,
                        rounds=rounds,
                    )
                    planned.append((condition_cfg.copy(), run_spec, str(partition_path.resolve())))

    print(f"Model dimension: {model_dimension:,}")
    backend = resolve_backend(config)
    print(
        "Runtime: "
        f"backend={backend}, "
        f"client_device={config.runtime.client_device}, "
        f"client_num_gpus={config.runtime.client_num_gpus}, "
        f"server_device={config.runtime.server_device}, "
        f"pin_memory={config.data.pin_memory}, "
        f"power_control={config.wireless.power_control_mode}, "
        "proposed_power=target2025_eq27_28_31, "
        "deterministic_power=hong2023_eq8_10_20, "
        f"deterministic_reference_power="
        f"{config.wireless.deterministic_reference_power_mode}"
    )
    if backend == "local" and config.runtime.client_device.lower().startswith("cuda"):
        print(
            "Native-Windows safe mode: CUDA clients run sequentially in the "
            "launcher process. Use WSL2/Linux with runtime.backend: ray for "
            "parallel Ray GPU clients."
        )
    if config.runtime.client_device.lower().startswith("cuda"):
        print(
            "CUDA: "
            f"torch={gpu_status.torch_version}, build={gpu_status.cuda_build}, "
            f"available={gpu_status.available}, devices={gpu_status.device_count}, "
            f"GPU={gpu_status.device_name}"
        )
    print(f"Planned simulations: {len(planned)}")
    for index, (_, run_spec, partition_path) in enumerate(planned, start=1):
        status = "SKIP" if finished.get(run_spec.run_id, -1) >= run_spec.rounds else "RUN"
        physical_rounds = physical_round_count(run_spec.method, run_spec.rounds)
        print(
            f"[{index:03d}/{len(planned):03d}] {status} {run_spec.run_id} "
            f"logical_rounds={run_spec.rounds} physical_fit_rounds={physical_rounds} "
            f"partition={Path(partition_path).name}"
        )

    if args.dry_run:
        return

    for index, (run_cfg, run_spec, partition_path) in enumerate(planned, start=1):
        if finished.get(run_spec.run_id, -1) >= run_spec.rounds:
            continue
        print("=" * 88)
        print(f"Starting {index}/{len(planned)}: {run_spec.run_id}")
        run_configured_simulation(run_cfg, run_spec, partition_path)
        print(f"Finished: {run_spec.run_id}")

    print("All requested simulations finished.")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
