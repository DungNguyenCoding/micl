"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from bayesfl.config import apply_overrides, load_config
from bayesfl.data.datasets import prepare_partition
from bayesfl.logging_utils import (
    create_run_paths,
    save_environment,
    save_resolved_config,
    setup_logging,
)
from bayesfl.server import run_flower_simulation


def _project_root() -> Path:
    # Works in editable installs from this repository.
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "scripts" / "configs").exists() and (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _default_config(dataset: str, method: str) -> Path:
    return _project_root() / "scripts" / "configs" / f"{method}_{dataset}.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bayesian Federated Learning baseline")
    parser.add_argument("--config", type=str, default=None, help="YAML experiment config")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default=None)
    parser.add_argument("--method", choices=["fedavg", "bbb", "fola"], default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config is None:
        dataset = args.dataset or "mnist"
        method = args.method or "fedavg"
        config_path = _default_config(dataset, method)
    else:
        config_path = Path(args.config)
    cfg = load_config(config_path)
    cfg = apply_overrides(
        cfg,
        dataset=args.dataset,
        method=args.method,
        rounds=args.rounds,
        seed=args.seed,
    )

    run_dir, log_path = create_run_paths(cfg)
    logger = setup_logging(log_path)
    save_resolved_config(cfg, run_dir)
    save_environment(run_dir)
    shutil.copy2(config_path, run_dir / "source_config.yaml")

    logger.info("Run directory: %s", run_dir)
    logger.info("Configuration: dataset=%s method=%s", cfg.data.dataset, cfg.method)
    if cfg.data.dataset == "cifar10" and not cfg.data.augment:
        logger.info(
            "CIFAR augmentation disabled: crop_padding=%s and random_flip=%s are inert; normalization remains active.",
            cfg.data.crop_padding,
            cfg.data.random_flip,
        )

    partition_path, partition_metadata = prepare_partition(cfg)
    logger.info("Partition manifest: %s", partition_path)
    logger.info("Partition stats: %s", json.dumps(partition_metadata, sort_keys=True))
    with (run_dir / "partition_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(partition_metadata, handle, indent=2, sort_keys=True)

    run_flower_simulation(
        cfg,
        partition_path=partition_path,
        partition_metadata=partition_metadata,
        run_dir=run_dir,
        logger=logger,
    )
    logger.info("Simulation finished successfully: %s", run_dir)


if __name__ == "__main__":
    main()
