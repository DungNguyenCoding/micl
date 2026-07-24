"""Command-line configuration for Bayesian Federated Learning experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List


def str2bool(value: str | bool) -> bool:
    """Parse a boolean value from CLI-friendly strings."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def int_list(value: str | List[int]) -> List[int]:
    """Parse comma-separated integers, e.g. ``256,128``."""
    if isinstance(value, list):
        return value
    value = value.strip()
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


@dataclass
class RunConfig:
    # Core experiment
    method: str = "fedavg"  # fedavg | vi | ola
    dataset: str = "mnist"  # mnist | cifar10
    model: str = "mlp"  # mlp | cnn ; VI currently uses mlp
    output_dir: str = "outputs/debug_run"
    data_dir: str = "./data"
    num_rounds: int = 5
    eval_every: int = 1
    heavy_eval_every: int = 5
    local_eval_every: int = 0
    local_eval_fraction: float = 1.0
    eval_mc_samples: int = 1
    # Scales posterior sigma during MC predictive evaluation.
    # Use a small value for OLA/FOLA when diagonal precision is not calibrated.
    posterior_sample_scale: float = 1.0
    calibration_bins: int = 15
    snr_hist_bins: int = 80
    save_posterior_every: int = 0
    save_prediction_snapshots: bool = False
    metrics_level: str = "bayes"  # basic | bayes | full
    save_every: int = 0  # deprecated alias; prefer save_posterior_every
    seed: int = 42

    # Federated population
    num_devices: int = 300
    num_virtual_clients: int = 24
    client_fraction: float = 0.1
    selector: str = "random"  # random now; wireless_todo placeholder for future

    # Data partitioning
    iid: bool = True
    balanced: bool = True
    noniid_alpha: float = 0.3
    unbalanced_alpha: float = 0.5
    val_ratio: float = 0.0
    min_client_examples: int = 1
    class_balance: bool = False

    # Synthetic radio/device layout metadata for later wireless scheduling
    area_radius_m: float = 550.0

    # Local optimization
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.05
    momentum: float = 0.0
    weight_decay: float = 0.0
    optimizer: str = "sgd"  # sgd | adam

    # Model dimensions
    mlp_hidden: List[int] | str = "200"

    # Online Laplace Approximation / FOLA
    ola_prior_lambda: float = 1.0
    precision_init: float = 1.0
    precision_floor: float = 1.0e-8
    fisher_clip: float = 1.0e6

    # Variational inference / Bayes-by-Backprop-style local learning with Pyro
    bayes_aggregation: str = "product"  # product | mean, used by vi
    vi_prior_scale: float = 0.05
    vi_init_scale: float = 0.05
    vi_min_scale: float = 1.0e-4
    vi_particles: int = 1
    vi_lr: float = 1.0e-3
    # Optional VI stabilization for long non-IID runs.
    # Example: --vi_lr_decay_milestones 80,120 --vi_lr_decay_gamma 0.5
    vi_lr_decay_milestones: str = ""
    vi_lr_decay_gamma: float = 1.0
    # 0 disables clamping. When >0, local VI posterior scales are upper-clamped
    # before being sent to the server. Useful when posterior uncertainty explodes.
    vi_max_scale: float = 0.0

    # Best-checkpoint tracking
    save_best_checkpoints: bool = True
    best_checkpoint_metric: str = "global_accuracy"  # global_accuracy | global_ece | global_loss

    # Sparse Bayesian communication / BBB-style pruning experiments
    sparse_comm: bool = False
    sparse_metric: str = "update_snr"  # snr | update_snr | precision_update | kl
    sparse_selection: str = "bayesian"  # bayesian | random
    sparse_ratio: float = 1.0
    sparse_warmup_rounds: int = 0
    sparse_min_keep: int = 1
    sparse_apply_to: str = "vi,ola"

    # Flower/Ray runtime control
    client_cpus: float = 1.0
    client_gpus: float = 0.0
    num_workers: int = 0
    torch_threads: int = 1
    device: str = "auto"  # auto | cpu | cuda
    cache_clients: bool = True
    accept_failures: bool = False

    # Debugging
    dry_run: bool = False

    def normalized_hidden(self) -> List[int]:
        """Return MLP hidden sizes as a list of integers."""
        return int_list(self.mlp_hidden)

    def as_rows(self) -> list[tuple[str, str]]:
        """Represent the config as rows suitable for CSV output."""
        rows: list[tuple[str, str]] = []
        for item in fields(self):
            value = getattr(self, item.name)
            rows.append((item.name, ",".join(map(str, value)) if isinstance(value, list) else str(value)))
        return rows


def parse_args() -> RunConfig:
    """Parse command-line arguments into a :class:`RunConfig`."""
    parser = argparse.ArgumentParser(
        description=(
            "Flower simulation base source for FedAvg, mean-field variational "
            "Bayesian FL, and Federated Online Laplace Approximation."
        )
    )

    parser.add_argument("--method", choices=["fedavg", "vi", "ola"], default=RunConfig.method)
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default=RunConfig.dataset)
    parser.add_argument("--model", choices=["mlp", "cnn"], default=RunConfig.model)
    parser.add_argument("--output_dir", default=RunConfig.output_dir)
    parser.add_argument("--data_dir", default=RunConfig.data_dir)
    parser.add_argument("--num_rounds", type=int, default=RunConfig.num_rounds)
    parser.add_argument("--eval_every", type=int, default=RunConfig.eval_every)
    parser.add_argument("--heavy_eval_every", type=int, default=RunConfig.heavy_eval_every)
    parser.add_argument("--local_eval_every", type=int, default=RunConfig.local_eval_every)
    parser.add_argument("--local_eval_fraction", type=float, default=RunConfig.local_eval_fraction)
    parser.add_argument("--eval_mc_samples", type=int, default=RunConfig.eval_mc_samples)
    parser.add_argument("--posterior_sample_scale", type=float, default=RunConfig.posterior_sample_scale)
    parser.add_argument("--calibration_bins", type=int, default=RunConfig.calibration_bins)
    parser.add_argument("--snr_hist_bins", type=int, default=RunConfig.snr_hist_bins)
    parser.add_argument("--save_posterior_every", type=int, default=RunConfig.save_posterior_every)
    parser.add_argument("--save_prediction_snapshots", type=str2bool, default=RunConfig.save_prediction_snapshots)
    parser.add_argument("--metrics_level", choices=["basic", "bayes", "full"], default=RunConfig.metrics_level)
    parser.add_argument("--save_every", type=int, default=RunConfig.save_every, help="Deprecated alias; prefer --save_posterior_every")
    parser.add_argument("--seed", type=int, default=RunConfig.seed)

    parser.add_argument("--num_devices", type=int, default=RunConfig.num_devices)
    parser.add_argument("--num_virtual_clients", type=int, default=RunConfig.num_virtual_clients)
    parser.add_argument("--client_fraction", type=float, default=RunConfig.client_fraction)
    parser.add_argument("--selector", choices=["random", "wireless_todo"], default=RunConfig.selector)

    parser.add_argument("--iid", type=str2bool, default=RunConfig.iid)
    parser.add_argument("--balanced", type=str2bool, default=RunConfig.balanced)
    parser.add_argument("--noniid_alpha", type=float, default=RunConfig.noniid_alpha)
    parser.add_argument("--unbalanced_alpha", type=float, default=RunConfig.unbalanced_alpha)
    parser.add_argument("--val_ratio", type=float, default=RunConfig.val_ratio)
    parser.add_argument("--min_client_examples", type=int, default=RunConfig.min_client_examples)
    parser.add_argument("--class_balance", type=str2bool, default=RunConfig.class_balance)
    parser.add_argument("--area_radius_m", type=float, default=RunConfig.area_radius_m)

    parser.add_argument("--local_epochs", type=int, default=RunConfig.local_epochs)
    parser.add_argument("--batch_size", type=int, default=RunConfig.batch_size)
    parser.add_argument("--lr", type=float, default=RunConfig.lr)
    parser.add_argument("--momentum", type=float, default=RunConfig.momentum)
    parser.add_argument("--weight_decay", type=float, default=RunConfig.weight_decay)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default=RunConfig.optimizer)
    parser.add_argument("--mlp_hidden", type=str, default=RunConfig.mlp_hidden)

    parser.add_argument("--ola_prior_lambda", type=float, default=RunConfig.ola_prior_lambda)
    parser.add_argument("--precision_init", type=float, default=RunConfig.precision_init)
    parser.add_argument("--precision_floor", type=float, default=RunConfig.precision_floor)
    parser.add_argument("--fisher_clip", type=float, default=RunConfig.fisher_clip)

    parser.add_argument("--bayes_aggregation", choices=["product", "mean"], default=RunConfig.bayes_aggregation)
    parser.add_argument("--vi_prior_scale", type=float, default=RunConfig.vi_prior_scale)
    parser.add_argument("--vi_init_scale", type=float, default=RunConfig.vi_init_scale)
    parser.add_argument("--vi_min_scale", type=float, default=RunConfig.vi_min_scale)
    parser.add_argument("--vi_particles", type=int, default=RunConfig.vi_particles)
    parser.add_argument("--vi_lr", type=float, default=RunConfig.vi_lr)
    parser.add_argument("--vi_lr_decay_milestones", default=RunConfig.vi_lr_decay_milestones, help="Comma-separated server rounds where VI lr is multiplied by --vi_lr_decay_gamma")
    parser.add_argument("--vi_lr_decay_gamma", type=float, default=RunConfig.vi_lr_decay_gamma)
    parser.add_argument("--vi_max_scale", type=float, default=RunConfig.vi_max_scale, help="Optional upper clamp for VI posterior scale; 0 disables")

    parser.add_argument("--save_best_checkpoints", type=str2bool, default=RunConfig.save_best_checkpoints)
    parser.add_argument("--best_checkpoint_metric", choices=["global_accuracy", "global_ece", "global_loss"], default=RunConfig.best_checkpoint_metric)

    parser.add_argument("--sparse_comm", type=str2bool, default=RunConfig.sparse_comm)
    parser.add_argument("--sparse_metric", choices=["snr", "update_snr", "precision_update", "kl"], default=RunConfig.sparse_metric)
    parser.add_argument("--sparse_selection", choices=["bayesian", "random"], default=RunConfig.sparse_selection)
    parser.add_argument("--sparse_ratio", type=float, default=RunConfig.sparse_ratio)
    parser.add_argument("--sparse_warmup_rounds", type=int, default=RunConfig.sparse_warmup_rounds)
    parser.add_argument("--sparse_min_keep", type=int, default=RunConfig.sparse_min_keep)
    parser.add_argument("--sparse_apply_to", default=RunConfig.sparse_apply_to, help="Comma-separated methods using sparse comm, e.g. vi,ola")

    parser.add_argument("--client_cpus", type=float, default=RunConfig.client_cpus)
    parser.add_argument("--client_gpus", type=float, default=RunConfig.client_gpus)
    parser.add_argument("--num_workers", type=int, default=RunConfig.num_workers)
    parser.add_argument("--torch_threads", type=int, default=RunConfig.torch_threads)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=RunConfig.device)
    parser.add_argument("--cache_clients", type=str2bool, default=RunConfig.cache_clients)
    parser.add_argument("--accept_failures", type=str2bool, default=RunConfig.accept_failures)
    parser.add_argument("--dry_run", type=str2bool, default=RunConfig.dry_run)

    args = parser.parse_args()
    cfg = RunConfig(**vars(args))

    if not 0 < cfg.client_fraction <= 1:
        raise ValueError("client_fraction must be in (0, 1]")
    if cfg.num_virtual_clients <= 0 or cfg.num_devices <= 0:
        raise ValueError("num_devices and num_virtual_clients must be positive")
    if cfg.num_virtual_clients > cfg.num_devices:
        raise ValueError("num_virtual_clients cannot exceed num_devices")
    if cfg.method == "vi" and cfg.model != "mlp":
        raise ValueError("The Pyro VI scaffold currently supports --model mlp. Use --model mlp for --method vi.")
    if not 0 < cfg.sparse_ratio <= 1:
        raise ValueError("sparse_ratio must be in (0, 1]")
    if cfg.sparse_warmup_rounds < 0:
        raise ValueError("sparse_warmup_rounds must be >= 0")
    if cfg.sparse_min_keep < 1:
        raise ValueError("sparse_min_keep must be >= 1")
    if cfg.sparse_comm and cfg.method not in {x.strip() for x in str(cfg.sparse_apply_to).split(",") if x.strip()}:
        raise ValueError("sparse_comm=True but cfg.method is not listed in --sparse_apply_to")
    if cfg.sparse_comm and cfg.method == "ola" and cfg.sparse_metric not in {"precision_update", "update_snr", "snr", "kl"}:
        raise ValueError("Unsupported OLA sparse_metric")
    if cfg.sparse_comm and cfg.method == "vi" and cfg.sparse_metric not in {"update_snr", "snr", "kl"}:
        raise ValueError("Recommended VI sparse_metric values: update_snr, snr, kl")
    if cfg.sparse_comm and cfg.method == "vi" and cfg.bayes_aggregation != "product":
        raise ValueError("Sparse VI communication currently supports --bayes_aggregation product")
    if cfg.eval_every <= 0:
        raise ValueError("eval_every must be positive")
    if cfg.heavy_eval_every <= 0:
        raise ValueError("heavy_eval_every must be positive")
    if cfg.local_eval_every < 0:
        raise ValueError("local_eval_every must be non-negative; use 0 to disable")
    if not 0 < cfg.local_eval_fraction <= 1:
        raise ValueError("local_eval_fraction must be in (0, 1]")
    if cfg.eval_mc_samples <= 0:
        raise ValueError("eval_mc_samples must be positive")
    if cfg.posterior_sample_scale < 0:
        raise ValueError("posterior_sample_scale must be non-negative")
    if cfg.calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive")
    if cfg.snr_hist_bins <= 1:
        raise ValueError("snr_hist_bins must be at least 2")
    if cfg.save_posterior_every == 0 and cfg.save_every > 0:
        cfg.save_posterior_every = cfg.save_every
    if cfg.local_epochs <= 0:
        raise ValueError("local_epochs must be positive")

    # Normalize to a string path early so CSV logs are simple.
    cfg.output_dir = str(Path(cfg.output_dir))
    cfg.data_dir = str(Path(cfg.data_dir))
    cfg.mlp_hidden = cfg.normalized_hidden()
    return cfg
