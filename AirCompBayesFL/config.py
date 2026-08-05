"""Configuration loading and validation for AirCompBayesFL."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    root: str = "data"
    num_clients: int = 40
    labels_per_client: int = 1
    mean_samples_per_client: float = 10.0
    min_samples_per_client: int = 1
    bs_radius_m: float = 200.0
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class ModelConfig:
    name: str = "paper_cnn"
    num_classes: int = 10
    initial_prior_std: float = 0.05
    min_posterior_std: float = 1.0e-4
    max_posterior_std: float = 10.0
    min_precision: float = 1.0e-6
    max_precision: float = 1.0e8


@dataclass
class TrainingConfig:
    local_epochs: int = 3
    batch_size: int = 10
    learning_rate: float = 0.1
    kl_weight: float = 1.0 / 50_000.0
    mc_train_samples: int = 5
    mc_eval_samples: int = 5
    bayesian_local_mode: str = "two_phase"  # joint | two_phase
    fedprox_mu: float = 0.01
    gradient_clip_norm: float = 10.0
    num_rounds: Optional[int] = 3
    max_channel_uses: int = 30_000_000
    evaluate_every: int = 1


@dataclass
class WirelessConfig:
    enabled: bool = True
    power_dbm: float = 23.0
    noise_dbm: float = -74.0
    num_subchannels: int = 1024
    path_loss_exponent: float = 4.0
    gamma_db: float = 10.0
    min_channel_power: float = 1.0e-14
    bisection_steps: int = 60


@dataclass
class RuntimeConfig:
    seed: int = 2025
    replications: int = 1

    # auto: native Windows + CUDA -> local sequential GPU backend;
    #       otherwise -> Flower/Ray backend.
    # ray:  force Flower/Ray (recommended on Linux/WSL2).
    # local: run virtual clients sequentially in the launcher process.
    backend: str = "auto"  # auto | ray | local

    client_num_cpus: float = 1.0
    client_num_gpus: float = 0.0
    ray_include_dashboard: bool = False
    torch_num_threads: int = 1
    client_device: str = "cpu"  # cpu | cuda | cuda:N | auto
    server_device: str = "cpu"
    verbose_flower: bool = False
    cleanup_cuda_after_fit: bool = True

    # Do not silently continue when all or some client jobs fail. This prevents
    # random/untrained models from being written to metrics.csv as valid runs.
    fail_on_client_failure: bool = True


@dataclass
class OutputConfig:
    directory: str = "results"
    metrics_filename: str = "metrics.csv"
    reliability_filename: str = "reliability.csv"
    clients_filename: str = "client_metrics.csv"
    save_checkpoints: bool = True


@dataclass
class SimulationConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wireless: WirelessConfig = field(default_factory=WirelessConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            raw: Dict[str, Any] = yaml.safe_load(handle) or {}
        cfg = cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            wireless=WirelessConfig(**raw.get("wireless", {})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
            output=OutputConfig(**raw.get("output", {})),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.data.num_clients <= 0:
            raise ValueError("data.num_clients must be positive")
        if self.data.labels_per_client not in range(1, 11):
            raise ValueError("data.labels_per_client must be between 1 and 10")
        if self.training.local_epochs <= 0:
            raise ValueError("training.local_epochs must be positive")
        if self.training.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
        if self.training.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if self.training.mc_train_samples <= 0 or self.training.mc_eval_samples <= 0:
            raise ValueError("Monte Carlo sample counts must be positive")
        if self.training.bayesian_local_mode not in {"joint", "two_phase"}:
            raise ValueError("training.bayesian_local_mode must be 'joint' or 'two_phase'")
        if self.wireless.num_subchannels <= 0:
            raise ValueError("wireless.num_subchannels must be positive")
        if self.runtime.replications <= 0:
            raise ValueError("runtime.replications must be positive")
        if self.runtime.client_num_cpus <= 0:
            raise ValueError("runtime.client_num_cpus must be positive")
        if self.runtime.client_num_gpus < 0:
            raise ValueError("runtime.client_num_gpus cannot be negative")
        if str(self.runtime.backend).strip().lower() not in {"auto", "ray", "local"}:
            raise ValueError("runtime.backend must be auto, ray, or local")

        valid_devices = {"cpu", "cuda", "auto"}
        for field_name, value in {
            "runtime.client_device": self.runtime.client_device,
            "runtime.server_device": self.runtime.server_device,
        }.items():
            normalized = str(value).strip().lower()
            if normalized not in valid_devices and not normalized.startswith("cuda:"):
                raise ValueError(f"{field_name} must be cpu, cuda, cuda:N, or auto")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def copy(self) -> "SimulationConfig":
        """Return a deep-enough dataclass copy for experiment overrides."""
        return SimulationConfig(
            data=replace(self.data),
            model=replace(self.model),
            training=replace(self.training),
            wireless=replace(self.wireless),
            runtime=replace(self.runtime),
            output=replace(self.output),
        )
