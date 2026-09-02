"""Canonical CLI for the unified MNIST/CIFAR-10 baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from bayesian_torch_backend import build_initial_states
from bayesian_training import resolved_base_kl_weight
from config import SimulationConfig
from dataset import ensure_dataset, load_partition, prepare_partitions
from experiments import RunSpec, normalize_method
from models import build_model, count_parameters
from runtime_utils import configure_runtime_environment, validate_runtime
from server import run_configured_simulation
from training_schedule import learning_rate_for_round


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Flower/Ray baseline: FedAvg vs Bayesian-Torch BayesAvg"
    )
    parser.add_argument("--dataset", required=True, choices=["mnist", "cifar10"])
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML. Without it, configs/baseline_<dataset>.yaml is used.",
    )
    parser.add_argument(
        "--methods",
        default="fedavg,bayesavg",
        help="Comma-separated: fedavg,bayesavg. 'proposed' is accepted as bayesavg alias.",
    )
    parser.add_argument("--method", default=None, help="Single-method alias for --methods")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--client-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--backend", choices=["ray", "local", "auto"], default=None)
    parser.add_argument("--force-partitions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_methods(args: argparse.Namespace) -> List[str]:
    text = args.method if args.method is not None else args.methods
    methods = [normalize_method(part) for part in str(text).split(",") if part.strip()]
    result = []
    for method in methods:
        if method not in result:
            result.append(method)
    if not result:
        raise ValueError("At least one method is required")
    return result


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    dataset = str(args.dataset).lower()

    if args.config is None:
        config_path = project_root / "configs" / f"baseline_{dataset}.yaml"
    else:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path
    config = SimulationConfig.from_yaml(config_path)
    if config.data.dataset != dataset:
        raise ValueError(
            f"--dataset={dataset} conflicts with config data.dataset={config.data.dataset}"
        )

    if args.rounds is not None:
        config.training.num_rounds = int(args.rounds)
    if args.client_fraction is not None:
        config.federation.client_fraction = float(args.client_fraction)
    if args.seed is not None:
        config.runtime.seed = int(args.seed)
    if args.replications is not None:
        config.runtime.replications = int(args.replications)
    if args.output is not None:
        config.output.directory = str(args.output)
    if args.backend is not None:
        config.runtime.backend = str(args.backend)
    config.validate()

    data_root = Path(config.data.root)
    if not data_root.is_absolute():
        config.data.root = str((project_root / data_root).resolve())
    output_root = Path(config.output.directory)
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
        config.output.directory = str(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    configure_runtime_environment(config)
    gpu = validate_runtime(config)
    ensure_dataset(config.data)

    deterministic_model = build_model(
        config.data.dataset,
        config.model.name,
        config.model.num_classes,
    )
    model_dimension = count_parameters(deterministic_model)
    _bstate, _mean, bayesian_d, deterministic_d = build_initial_states(
        dataset=config.data.dataset,
        model_cfg=config.model,
        variational_cfg=config.variational,
        seed=config.runtime.seed,
    )
    resolved_kl = resolved_base_kl_weight(config.variational, bayesian_d)

    print(f"Dataset: {config.data.dataset}")
    print(f"Model: {config.model.name}")
    print(f"Model dimension: {model_dimension:,}")
    print(f"Bayesian Conv/Linear dimension: {bayesian_d:,}")
    print(f"Deterministic normalization dimension: {deterministic_d:,}")
    print(f"Resolved KL weight: {resolved_kl:.12g}")
    print(
        "Training: "
        f"E={config.training.local_epochs}, batch={config.training.batch_size}, "
        f"lr={config.training.learning_rate}, momentum={config.training.momentum}, "
        f"rounds={config.training.num_rounds}, scheduler={config.training.lr_scheduler}, "
        f"horizon={config.training.lr_decay_rounds}"
    )
    print(
        f"Clients: total={config.data.num_clients}, fraction={config.federation.client_fraction}, "
        f"per_round={config.participating_clients()}"
    )
    print(
        f"CUDA: torch_device_available={gpu.available}, devices={gpu.device_count}, "
        f"GPU={gpu.device_name}"
    )

    if config.training.lr_scheduler == "cosine":
        points = [1, 50, 100, 150, 200, 250, 300, config.training.lr_decay_rounds]
        points = sorted(set(r for r in points if r <= config.training.lr_decay_rounds))
        print("LR schedule checkpoints:")
        for r in points:
            print(f"  round {r:3d}: {learning_rate_for_round(config.training, r):.8f}")

    methods = _resolve_methods(args)
    partition_dir = output_root / "partitions"
    planned = []
    for rep in range(int(config.runtime.replications)):
        seed = int(config.runtime.seed) + rep
        partition_path = prepare_partitions(
            config.data,
            seed,
            partition_dir,
            force=bool(args.force_partitions),
        )
        partition = load_partition(partition_path)
        print(
            f"Partition seed={seed}: {partition_path.name} | "
            f"total={partition['total_samples_used']} mean={partition['mean_size']:.4f} "
            f"min={partition['min_size']} max={partition['max_size']} "
            f"classes/client={partition['mean_classes_per_client']:.4f}"
        )
        for method in methods:
            run_id = f"{dataset}_{config.model.name}_{method}_rep{rep:02d}_seed{seed}"
            planned.append(
                (
                    config.copy(),
                    RunSpec(
                        run_id=run_id,
                        method=method,
                        seed=seed,
                        rounds=int(config.training.num_rounds),
                        realization=rep,
                    ),
                    str(partition_path),
                )
            )

    print(f"Planned simulations: {len(planned)}")
    for index, (_cfg, run_spec, partition_path) in enumerate(planned, start=1):
        print(
            f"[{index:03d}/{len(planned):03d}] RUN {run_spec.run_id} "
            f"rounds={run_spec.rounds} fit_rounds={run_spec.rounds} "
            f"partition={Path(partition_path).name}"
        )

    if config.output.save_resolved_config:
        config.save_yaml(output_root / "resolved_config.yaml")

    if args.dry_run:
        return

    for cfg, run_spec, partition_path in planned:
        print(f"\n===== START {run_spec.run_id} =====")
        run_configured_simulation(cfg, run_spec, partition_path)
        print(f"===== FINISHED {run_spec.run_id} =====")


if __name__ == "__main__":
    main()
