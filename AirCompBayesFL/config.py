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
    # v1.3 implements Algorithm 1 exactly as two server-controlled phases.
    # "two_phase" and "paper_two_phase" are accepted aliases.
    bayesian_local_mode: str = "paper_two_phase"
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
    # The power-law r^{-alpha} is dimensionless. Distances stored by the data
    # module are in metres, so divide them by this reference before applying
    # the exponent. Set this to 1.0 to reproduce the legacy raw-metre model.
    # The paper does not disclose the numerical distance normalization used by
    # the authors; 1000 m (distances expressed in km) is the documented default.
    path_loss_reference_m: float = 1000.0
    gamma_db: float = 10.0
    min_channel_power: float = 1.0e-14
    bisection_steps: int = 60
    # v1.5.0 shares the KKT/QCQP magnitude optimizer across methods but uses
    # the source-appropriate power-scale normalization for each algorithm:
    # - Proposed: target-2025 Eqs. (27),(28),(31)
    # - FedAvg/FedProx/SCAFFOLD: Hong-2023 ref. [13] Eqs. (8),(10),(20)
    # This corrects v1.4.x, which applied the Proposed normalization to every
    # method and therefore did not faithfully implement the cited benchmark.
    power_control_mode: str = "paper_reference_kkt"
    deterministic_payload_mode: str = "update"
    # Reference [13] obtains rho_ref from a BS-local update. The target paper
    # borrows its power allocation for conventional FL but has no BS dataset.
    # Its exact adaptation is not disclosed. The default below implements the
    # conventional coordinated-aggregate interpretation described in Remark 6:
    # rho_ref = ||sum_k pi_k Delta_k||^2 / d. ``weighted_local`` is available
    # as a documented sensitivity alternative.
    deterministic_reference_power_mode: str = "coordinated_aggregate"




@dataclass
class SparsePosteriorConfig:
    """Optional research extension for sparse posterior-evidence communication.

    Disabled by default so the paper-reproduction path is unchanged.  When
    enabled for Proposed with keep_ratio < 1, each client selects a fixed-size
    coordinate mask once per logical round and applies that same mask to both
    Delta-rho and Delta-nu transmissions.
    """

    enabled: bool = False
    selection: str = "bayesian"  # bayesian | random
    keep_ratio: float = 1.0
    score_epsilon: float = 1.0e-12
    min_keep: int = 1

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
    # Delete client-specific rho state after the corresponding nu phase.
    cleanup_phase_state: bool = True

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
    sparse: SparsePosteriorConfig = field(default_factory=SparsePosteriorConfig)
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
            sparse=SparsePosteriorConfig(**raw.get("sparse", {})),
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
        if self.training.bayesian_local_mode not in {"two_phase", "paper_two_phase"}:
            raise ValueError(
                "training.bayesian_local_mode must be 'two_phase' or "
                "'paper_two_phase'; joint client-side optimization was removed "
                "because it does not implement Algorithm 1"
            )
        if self.wireless.num_subchannels <= 0:
            raise ValueError("wireless.num_subchannels must be positive")
        if self.wireless.path_loss_reference_m <= 0:
            raise ValueError("wireless.path_loss_reference_m must be positive")
        if (
            str(self.wireless.power_control_mode).strip().lower()
            != "paper_reference_kkt"
        ):
            raise ValueError(
                "wireless.power_control_mode must be paper_reference_kkt; "
                "v1.5.0 uses one shared KKT magnitude optimizer with "
                "source-specific Proposed/Hong-2023 power scaling"
            )
        if str(self.wireless.deterministic_payload_mode).strip().lower() != "update":
            raise ValueError(
                "wireless.deterministic_payload_mode must be 'update'; "
                "reference [13] transmits local update vectors"
            )
        if (
            str(self.wireless.deterministic_reference_power_mode).strip().lower()
            not in {"coordinated_aggregate", "weighted_local"}
        ):
            raise ValueError(
                "wireless.deterministic_reference_power_mode must be "
                "coordinated_aggregate or weighted_local"
            )
        if str(self.sparse.selection).strip().lower() not in {"bayesian", "random"}:
            raise ValueError("sparse.selection must be bayesian or random")
        if not (0.0 < float(self.sparse.keep_ratio) <= 1.0):
            raise ValueError("sparse.keep_ratio must be in (0, 1]")
        if float(self.sparse.score_epsilon) <= 0.0:
            raise ValueError("sparse.score_epsilon must be positive")
        if int(self.sparse.min_keep) <= 0:
            raise ValueError("sparse.min_keep must be positive")
        if self.sparse.enabled and self.training.bayesian_local_mode not in {"two_phase", "paper_two_phase"}:
            raise ValueError("Sparse posterior evidence requires the Proposed two-phase VI mode")

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
            sparse=replace(self.sparse),
            runtime=replace(self.runtime),
            output=replace(self.output),
        )
