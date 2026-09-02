"""Configuration profiles for the unified MNIST/CIFAR-10 Bayesian-FL baseline.

This baseline intentionally contains no wireless/AirComp configuration.  Every
federated round has exactly one Flower fit round for both FedAvg and BayesAvg.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    dataset: str = "mnist"  # mnist | cifar10
    root: str = "./data"
    num_classes: int = 10
    num_clients: int = 40

    # MNIST v1.6.1-compatible scarce single-label partition.
    partition: str = "single_label"  # single_label | sparse_dirichlet
    labels_per_client: int = 1

    # Scarce-data client size.  CIFAR sparse_dirichlet uses the exact seed-0
    # size rule 1 + Poisson(avg_samples_per_client - 1), which gives the
    # requested 10046 total examples for avg=100, K=100, seed=0.
    avg_samples_per_client: float = 10.0
    min_samples_per_client: int = 1

    # CIFAR sparse Dirichlet settings.
    dirichlet_alpha: float = 0.1
    sparse_classes_per_client: int = 4

    augment: bool = False
    crop_padding: int = 4
    random_flip: bool = True
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class ModelConfig:
    name: str = "paper_cnn"  # paper_cnn | resnet56_gn
    num_classes: int = 10


@dataclass
class VariationalConfig:
    # Fixed standard-normal prior, used on every client and every round.
    prior_mu: float = 0.0
    prior_sigma: float = 1.0

    # Native Bayesian-Torch initialization centers.  Bayesian-Torch 0.5.0
    # samples tensor initializations around these values.
    posterior_mu_init: float = 0.0
    posterior_rho_init: float = -3.0

    # null/None resolves to 1 / Bayesian_dimension (Conv+Linear only).
    kl_weight: Optional[float] = None
    kl_weight_schedule: bool = False
    kl_warmup_rounds: int = 20
    lambda_scale_by_size: bool = True

    mc_train: int = 2
    mc_eval: int = 5

    # Enforce sigma_local >= variance_floor_ratio * sigma_global after every
    # optimizer step.  Despite the historical name, the ratio is applied to
    # posterior standard deviation, matching the project specification.
    variance_floor_ratio: float = 0.5


@dataclass
class TrainingConfig:
    optimizer: str = "sgd"
    learning_rate: float = 0.1
    momentum: float = 0.0
    weight_decay: float = 0.0
    batch_size: int = 10
    local_epochs: int = 3
    num_rounds: int = 240

    lr_scheduler: str = "constant"  # constant | cosine
    min_learning_rate: float = 0.0
    # Fixed cosine horizon.  It is intentionally independent of num_rounds.
    lr_decay_rounds: int = 240

    gradient_clip_norm: float = 10.0
    evaluate_every: int = 1


@dataclass
class FederationConfig:
    client_fraction: float = 1.0


@dataclass
class RuntimeConfig:
    seed: int = 0
    replications: int = 1
    backend: str = "ray"  # ray | local | auto

    client_num_cpus: float = 1.0
    client_num_gpus: float = 0.125
    ray_include_dashboard: bool = False
    torch_num_threads: int = 1
    client_device: str = "cuda"
    server_device: str = "cpu"
    verbose_flower: bool = False
    cleanup_cuda_after_fit: bool = True
    fail_on_client_failure: bool = True


@dataclass
class OutputConfig:
    directory: str = "results"
    metrics_filename: str = "metrics.csv"
    clients_filename: str = "client_metrics.csv"
    reliability_filename: str = "reliability.csv"
    participation_filename: str = "participation.csv"
    save_checkpoints: bool = True
    save_resolved_config: bool = True


@dataclass
class SimulationConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    variational: VariationalConfig = field(default_factory=VariationalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def profile(cls, dataset: str) -> "SimulationConfig":
        """Return the project baseline profile for one dataset."""
        dataset = str(dataset).strip().lower()
        if dataset == "mnist":
            cfg = cls(
                data=DataConfig(
                    dataset="mnist",
                    root="./data",
                    num_classes=10,
                    num_clients=40,
                    partition="single_label",
                    labels_per_client=1,
                    avg_samples_per_client=10.0,
                    min_samples_per_client=1,
                    augment=False,
                ),
                model=ModelConfig(name="paper_cnn", num_classes=10),
                variational=VariationalConfig(),
                training=TrainingConfig(
                    optimizer="sgd",
                    learning_rate=0.1,
                    momentum=0.0,
                    weight_decay=0.0,
                    batch_size=10,
                    local_epochs=3,
                    # No wireless channel-use budget exists in this baseline.
                    # 240 preserves the old Proposed logical-round budget.
                    num_rounds=240,
                    lr_scheduler="constant",
                    min_learning_rate=0.0,
                    lr_decay_rounds=240,
                    gradient_clip_norm=10.0,
                    evaluate_every=1,
                ),
                federation=FederationConfig(client_fraction=1.0),
                runtime=RuntimeConfig(seed=0),
                output=OutputConfig(directory="results/mnist_baseline"),
            )
        elif dataset == "cifar10":
            cfg = cls(
                data=DataConfig(
                    dataset="cifar10",
                    root="./data",
                    num_classes=10,
                    num_clients=100,
                    partition="sparse_dirichlet",
                    labels_per_client=4,
                    avg_samples_per_client=100.0,
                    min_samples_per_client=1,
                    dirichlet_alpha=0.1,
                    sparse_classes_per_client=4,
                    augment=False,
                    crop_padding=4,
                    random_flip=True,
                ),
                model=ModelConfig(name="resnet56_gn", num_classes=10),
                variational=VariationalConfig(
                    prior_mu=0.0,
                    prior_sigma=1.0,
                    posterior_mu_init=0.0,
                    posterior_rho_init=-3.0,
                    kl_weight=None,
                    kl_weight_schedule=False,
                    kl_warmup_rounds=20,
                    lambda_scale_by_size=True,
                    mc_train=2,
                    mc_eval=5,
                    variance_floor_ratio=0.5,
                ),
                training=TrainingConfig(
                    optimizer="sgd",
                    learning_rate=0.05,
                    momentum=0.9,
                    weight_decay=0.0,
                    batch_size=128,
                    local_epochs=10,
                    num_rounds=300,
                    lr_scheduler="cosine",
                    min_learning_rate=0.0001,
                    lr_decay_rounds=400,
                    gradient_clip_norm=10.0,
                    evaluate_every=1,
                ),
                federation=FederationConfig(client_fraction=1.0),
                runtime=RuntimeConfig(seed=0),
                output=OutputConfig(directory="results/cifar10_baseline"),
            )
        else:
            raise ValueError("dataset must be 'mnist' or 'cifar10'")
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            raw: Dict[str, Any] = yaml.safe_load(handle) or {}
        dataset = str(raw.get("data", {}).get("dataset", "mnist")).strip().lower()
        cfg = cls.profile(dataset)
        for section_name, section_cls in (
            ("data", DataConfig),
            ("model", ModelConfig),
            ("variational", VariationalConfig),
            ("training", TrainingConfig),
            ("federation", FederationConfig),
            ("runtime", RuntimeConfig),
            ("output", OutputConfig),
        ):
            values = raw.get(section_name, {}) or {}
            current = asdict(getattr(cfg, section_name))
            current.update(values)
            setattr(cfg, section_name, section_cls(**current))
        cfg.validate()
        return cfg

    def copy(self) -> "SimulationConfig":
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    def validate(self) -> None:
        dataset = str(self.data.dataset).strip().lower()
        if dataset not in {"mnist", "cifar10"}:
            raise ValueError("data.dataset must be mnist or cifar10")
        self.data.dataset = dataset

        if int(self.data.num_classes) != 10 or int(self.model.num_classes) != 10:
            raise ValueError("This baseline currently supports exactly 10 classes")
        if int(self.data.num_clients) <= 0:
            raise ValueError("data.num_clients must be positive")
        if float(self.data.avg_samples_per_client) <= 0:
            raise ValueError("data.avg_samples_per_client must be positive")
        if int(self.data.min_samples_per_client) <= 0:
            raise ValueError("data.min_samples_per_client must be positive")

        partition = str(self.data.partition).strip().lower()
        if partition not in {"single_label", "sparse_dirichlet"}:
            raise ValueError("data.partition must be single_label or sparse_dirichlet")
        self.data.partition = partition
        if partition == "single_label" and not (1 <= int(self.data.labels_per_client) <= 10):
            raise ValueError("data.labels_per_client must be in [1,10]")
        if partition == "sparse_dirichlet":
            if float(self.data.dirichlet_alpha) <= 0:
                raise ValueError("data.dirichlet_alpha must be positive")
            if not (1 <= int(self.data.sparse_classes_per_client) <= 10):
                raise ValueError("data.sparse_classes_per_client must be in [1,10]")

        expected_model = "paper_cnn" if dataset == "mnist" else "resnet56_gn"
        if str(self.model.name).strip().lower() != expected_model:
            raise ValueError(
                f"dataset={dataset!r} requires model.name={expected_model!r} in this baseline"
            )
        self.model.name = expected_model

        if str(self.training.optimizer).strip().lower() != "sgd":
            raise ValueError("training.optimizer must be sgd")
        if float(self.training.learning_rate) <= 0:
            raise ValueError("training.learning_rate must be positive")
        if not (0 <= float(self.training.momentum) < 1):
            raise ValueError("training.momentum must be in [0,1)")
        if float(self.training.weight_decay) < 0:
            raise ValueError("training.weight_decay cannot be negative")
        if int(self.training.batch_size) <= 0 or int(self.training.local_epochs) <= 0:
            raise ValueError("batch_size and local_epochs must be positive")
        if int(self.training.num_rounds) <= 0:
            raise ValueError("training.num_rounds must be positive")
        scheduler = str(self.training.lr_scheduler).strip().lower()
        if scheduler not in {"constant", "cosine"}:
            raise ValueError("training.lr_scheduler must be constant or cosine")
        self.training.lr_scheduler = scheduler
        if float(self.training.min_learning_rate) < 0:
            raise ValueError("training.min_learning_rate cannot be negative")
        if int(self.training.lr_decay_rounds) <= 0:
            raise ValueError("training.lr_decay_rounds must be positive")
        if int(self.training.evaluate_every) <= 0:
            raise ValueError("training.evaluate_every must be positive")

        if float(self.variational.prior_sigma) <= 0:
            raise ValueError("variational.prior_sigma must be positive")
        if self.variational.kl_weight is not None and float(self.variational.kl_weight) < 0:
            raise ValueError("variational.kl_weight cannot be negative")
        if int(self.variational.kl_warmup_rounds) <= 0:
            raise ValueError("variational.kl_warmup_rounds must be positive")
        if int(self.variational.mc_train) <= 0 or int(self.variational.mc_eval) <= 0:
            raise ValueError("variational MC sample counts must be positive")
        if not (0 < float(self.variational.variance_floor_ratio) <= 1):
            raise ValueError("variational.variance_floor_ratio must be in (0,1]")

        fraction = float(self.federation.client_fraction)
        if not (0 < fraction <= 1):
            raise ValueError("federation.client_fraction must be in (0,1]")

        if str(self.runtime.backend).strip().lower() not in {"ray", "local", "auto"}:
            raise ValueError("runtime.backend must be ray, local, or auto")

    def participating_clients(self) -> int:
        return max(
            1,
            int(float(self.federation.client_fraction) * int(self.data.num_clients)),
        )
