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
from bayesfl.experiment_state import config_fingerprint, load_resume_bundle


def _project_root() -> Path:
    # Works in editable installs from this repository.
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "scripts" / "configs").exists() and (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _default_config(dataset: str, method: str) -> Path:
    base_method = "fola" if method.startswith("fola") else "fedavg" if method.startswith("fedavg") else method
    return _project_root() / "scripts" / "configs" / f"{base_method}_{dataset}.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bayesian Federated Learning baseline")
    parser.add_argument("--config", type=str, default=None, help="YAML experiment config")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default=None)
    parser.add_argument("--method", choices=["fedavg", "bbb", "fola", "fedavg_dense", "fola_dense", "fola_sparse", "fola_sparse_kl_global_local", "fola_sparse_kl_local_global", "fola_sparse_random"], default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--selection-rule", choices=["dense", "kl_global_local", "kl_local_global", "random"])
    parser.add_argument("--keep-ratio", type=float)
    parser.add_argument("--precision-policy", choices=["strict", "floor_for_score"])
    parser.add_argument("--max-communication-bytes", type=int)
    parser.add_argument("--resume", type=Path, help="Resume an existing run directory at its latest completed evaluated round")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume is not None and args.config is None:
        config_path = args.resume.resolve() / "resolved_config.yaml"
    elif args.config is None:
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
        selection_rule=args.selection_rule,
        keep_ratio=args.keep_ratio,
        precision_policy=args.precision_policy,
        max_communication_bytes=args.max_communication_bytes,
    )

    if args.resume is None:
        run_dir, log_path = create_run_paths(cfg)
        logger = setup_logging(log_path)
        save_resolved_config(cfg, run_dir)
        save_environment(run_dir)
        shutil.copy2(config_path, run_dir / "source_config.yaml")
    else:
        from datetime import datetime
        run_dir = args.resume.resolve()
        saved, _ = load_resume_bundle(run_dir)
        if saved["config_fingerprint"] != config_fingerprint(cfg):
            raise ValueError("Resume may extend only training.rounds and communication.max_communication_bytes")
        if not cfg.communication.deterministic_client_schedule:
            raise ValueError("Resume requires a deterministic client schedule configured from the start")
        if cfg.training.rounds < saved["round_id"]:
            raise ValueError("Round cap is smaller than the completed round")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = Path(cfg.output.logs_dir).resolve() / f"{run_dir.name}_resume_{stamp}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(log_path)
        shutil.copy2(run_dir / "resolved_config.yaml", run_dir / f"resolved_config.before_resume_{stamp}.yaml")

    logger.info("Run directory: %s", run_dir)
    logger.info("Configuration: dataset=%s method=%s", cfg.data.dataset, cfg.method)
    if cfg.data.dataset == "cifar10" and not cfg.data.augment:
        logger.info(
            "CIFAR augmentation disabled: crop_padding=%s and random_flip=%s are inert; normalization remains active.",
            cfg.data.crop_padding,
            cfg.data.random_flip,
        )

    partition_path, partition_metadata = prepare_partition(cfg)
    # A cached JSON hash alone is not evidence of the current NPZ contents.
    # Verification does not rebuild/reorder the baseline data partition.
    from bayesfl.data.partition import _hash_partitions, load_partition
    actual_partition_sha = _hash_partitions(load_partition(partition_path))
    if actual_partition_sha != partition_metadata["sha256"]:
        raise ValueError("Partition NPZ content differs from its metadata SHA-256; preserve and inspect the cache")
    logger.info("Partition manifest: %s", partition_path)
    logger.info("Partition stats: %s", json.dumps(partition_metadata, sort_keys=True))
    if args.resume is not None:
        prior_partition = json.loads((run_dir / "partition_metadata.json").read_text())
        if prior_partition["sha256"] != partition_metadata["sha256"]:
            raise ValueError("Resume partition does not match the original run")
    with (run_dir / "partition_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(partition_metadata, handle, indent=2, sort_keys=True)

    from bayesfl.server import run_flower_simulation
    run_flower_simulation(
        cfg,
        partition_path=partition_path,
        partition_metadata=partition_metadata,
        run_dir=run_dir,
        logger=logger,
        resume=args.resume is not None,
    )
    logger.info("Simulation finished successfully: %s", run_dir)


if __name__ == "__main__":
    main()
