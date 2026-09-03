"""Configuration loading, validation, and round-level schedules."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    dataset: str = "mnist"
    root: str = "./data"
    num_classes: int = 10
    augment: bool = False
    crop_padding: int = 4
    crop_fill: int = 0
    random_flip: bool = True
    autoaugment_policy: str = "none"
    cutout_holes: int = 0
    cutout_length: int = 16
    normalize_mean: list[float] = field(default_factory=lambda: [0.1307])
    normalize_std: list[float] = field(default_factory=lambda: [0.3081])
    partition: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FederationConfig:
    num_clients: int = 100
    clients_per_round: int = 100


@dataclass
class ModelConfig:
    name: str = "mlp_784_500_300_10"
    group_norm_groups: int = 8


@dataclass
class TrainingConfig:
    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    batch_size: int = 64
    local_epochs: int = 5
    rounds: int = 100
    lr_schedule: str = "cosine"
    lr_min: float = 0.0001
    lr_decay_rounds: int = 100
    grad_clip_norm: Optional[float] = None


@dataclass
class BBBConfig:
    posterior_mu_init: float = 0.0
    posterior_rho_init: float = -3.0
    # BBB prior is configurable so the CIFAR paper-environment extension can
    # keep the requested standard Normal N(0,1) prior while MNIST can retain
    # the existing scale-mixture prior if desired.
    prior_type: str = "scale_mixture"
    prior_mean: float = 0.0
    prior_sigma: float = 1.0
    prior_pi: float = 0.5
    prior_sigma1: float = 1.0
    prior_sigma2: float = math.exp(-6.0)
    kl_weight: Optional[float] = None
    # Optional denominator override used only when kl_weight is null.
    # This is needed for the hybrid CIFAR profile: the paper BasicCNN has
    # 878,538 Bayesian variables, but the user explicitly asked to preserve
    # the historical 1/851,514 KL normalization.
    kl_reference_dimension: Optional[int] = None
    kl_scheme: str = "equal_minibatch"
    kl_weight_schedule: bool = False
    kl_warmup_rounds: int = 20
    lambda_scale_by_size: bool = True
    mc_train: int = 2
    mc_eval: int = 5
    variance_floor_ratio: float = 0.5
    rho_lr_multiplier: float = 0.1
    aggregation: str = "gaussian_product"
    # Required for the paper BasicCNN comparison so BBB starts from the same
    # Xavier point as deterministic FedAvg/FOLA. BBB itself is not in the FOLA paper.
    match_deterministic_init: bool = False


@dataclass
class FOLAConfig:
    prior_lambda: float = 1.0
    lambda_scale_by_size: bool = True
    mc_eval: int = 5
    variance_floor_ratio: float = 0.5
    precision_min: float = 1e-8
    precision_max: float = 1e8
    initial_precision: float = 1.0
    # online_recurrence preserves the existing project implementation.
    # paper_reference reproduces the released CIFAR implementation semantics:
    # omega starts at zero, accumulates task-gradient squares online, and the
    # server performs omega-weighted Gaussian-product aggregation.
    mode: str = "online_recurrence"
    aggregation_epsilon: float = 1e-5
    paper_mean_only_eval: bool = False


@dataclass
class RuntimeConfig:
    client_num_cpus: float = 2.0
    client_num_gpus: float = 0.0
    central_eval_device: str = "auto"
    seed: int = 0
    torch_num_threads: int = 1
    verbose_flower: bool = False


@dataclass
class OutputConfig:
    logs_dir: str = "./logs"
    outputs_dir: str = "./outputs"
    checkpoint_every: int = 10
    save_full_client_posteriors: bool = False


@dataclass
class ExperimentConfig:
    run_name: str
    method: str
    data: DataConfig
    federation: FederationConfig
    model: ModelConfig
    training: TrainingConfig
    bbb: BBBConfig = field(default_factory=BBBConfig)
    fola: FOLAConfig = field(default_factory=FOLAConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        self.method = self.method.lower()
        self.data.dataset = self.data.dataset.lower()
        self.data.autoaugment_policy = self.data.autoaugment_policy.lower()
        self.fola.mode = self.fola.mode.lower()

        if self.method not in {"fedavg", "bbb", "fola"}:
            raise ValueError(f"Unsupported method: {self.method}")
        if self.data.dataset not in {"mnist", "cifar10"}:
            raise ValueError(f"Unsupported dataset: {self.data.dataset}")
        if self.training.optimizer.lower() != "sgd":
            raise ValueError("This baseline intentionally supports SGD only.")
        if self.federation.clients_per_round > self.federation.num_clients:
            raise ValueError("clients_per_round cannot exceed num_clients")
        if self.training.rounds < 1 or self.training.local_epochs < 1:
            raise ValueError("rounds and local_epochs must be positive")
        if self.training.lr_decay_rounds < 1:
            raise ValueError("lr_decay_rounds must be positive")
        if self.training.grad_clip_norm is not None and self.training.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive when provided")
        if self.bbb.mc_train < 1 or self.bbb.mc_eval < 1 or self.fola.mc_eval < 1:
            raise ValueError("Monte Carlo sample counts must be positive")
        self.bbb.prior_type = self.bbb.prior_type.lower()
        if self.bbb.prior_type not in {"scale_mixture", "standard_normal", "normal"}:
            raise ValueError("bbb.prior_type must be scale_mixture, standard_normal, or normal")
        if self.bbb.prior_sigma <= 0:
            raise ValueError("BBB prior_sigma must be positive")
        if not 0.0 < self.bbb.prior_pi < 1.0:
            raise ValueError("BBB prior_pi must be in (0, 1)")
        if self.bbb.prior_sigma1 <= 0 or self.bbb.prior_sigma2 <= 0:
            raise ValueError("BBB prior sigmas must be positive")
        if self.bbb.kl_reference_dimension is not None and self.bbb.kl_reference_dimension <= 0:
            raise ValueError("bbb.kl_reference_dimension must be positive when provided")
        if self.bbb.aggregation not in {"gaussian_product", "fedavg_variational"}:
            raise ValueError("bbb.aggregation must be gaussian_product or fedavg_variational")
        if self.fola.mode not in {"online_recurrence", "paper_reference"}:
            raise ValueError("fola.mode must be online_recurrence or paper_reference")
        if self.fola.initial_precision < 0:
            raise ValueError("fola.initial_precision must be >= 0")
        if self.fola.aggregation_epsilon <= 0:
            raise ValueError("fola.aggregation_epsilon must be > 0")
        for ratio in (self.bbb.variance_floor_ratio, self.fola.variance_floor_ratio):
            if ratio <= 0:
                raise ValueError("variance_floor_ratio must be > 0")
        if self.data.autoaugment_policy not in {"none", "cifar10"}:
            raise ValueError("autoaugment_policy must be none or cifar10")
        if self.data.cutout_holes < 0 or self.data.cutout_length < 1:
            raise ValueError("cutout settings are invalid")

        if self.data.dataset == "cifar10" and self.model.name not in {
            "resnet56_gn8",
            "paper_basiccnn",
        }:
            raise ValueError(
                "CIFAR-10 model.name must be resnet56_gn8 or paper_basiccnn"
            )
        if self.data.dataset == "mnist" and self.model.name != "mlp_784_500_300_10":
            raise ValueError("MNIST baseline expects model.name=mlp_784_500_300_10")

    def resolved_kl_weight(self, bayesian_dimension: int) -> float:
        if self.bbb.kl_weight is not None:
            return float(self.bbb.kl_weight)
        if bayesian_dimension <= 0:
            raise ValueError("bayesian_dimension must be positive")
        denominator = (
            int(self.bbb.kl_reference_dimension)
            if self.bbb.kl_reference_dimension is not None
            else int(bayesian_dimension)
        )
        return 1.0 / float(denominator)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _make_section(cls: type, raw: Dict[str, Any] | None):
    return cls(**(raw or {}))


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = ExperimentConfig(
        run_name=str(raw.get("run_name", path.stem)),
        method=str(raw.get("method", "fedavg")),
        data=_make_section(DataConfig, raw.get("data")),
        federation=_make_section(FederationConfig, raw.get("federation")),
        model=_make_section(ModelConfig, raw.get("model")),
        training=_make_section(TrainingConfig, raw.get("training")),
        bbb=_make_section(BBBConfig, raw.get("bbb")),
        fola=_make_section(FOLAConfig, raw.get("fola")),
        runtime=_make_section(RuntimeConfig, raw.get("runtime")),
        output=_make_section(OutputConfig, raw.get("output")),
    )
    cfg.validate()
    return cfg


def apply_overrides(
    cfg: ExperimentConfig,
    *,
    dataset: str | None = None,
    method: str | None = None,
    rounds: int | None = None,
    seed: int | None = None,
) -> ExperimentConfig:
    out = copy.deepcopy(cfg)
    if dataset is not None:
        out.data.dataset = dataset.lower()
    if method is not None:
        out.method = method.lower()
    if rounds is not None:
        out.training.rounds = int(rounds)
    if seed is not None:
        out.runtime.seed = int(seed)
    out.validate()
    return out


def round_learning_rate(cfg: TrainingConfig, server_round: int) -> float:
    """Return the round-level LR; server_round is one-based."""
    if server_round < 1:
        raise ValueError("server_round must be >= 1")
    schedule = cfg.lr_schedule.lower()
    if schedule in {"none", "constant"}:
        return float(cfg.lr)
    if schedule != "cosine":
        raise ValueError(f"Unsupported LR schedule: {cfg.lr_schedule}")
    horizon = max(1, int(cfg.lr_decay_rounds))
    if horizon == 1:
        return float(cfg.lr_min)
    r0 = min(server_round - 1, horizon - 1)
    phase = r0 / float(horizon - 1)
    return float(
        cfg.lr_min
        + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * phase))
    )
