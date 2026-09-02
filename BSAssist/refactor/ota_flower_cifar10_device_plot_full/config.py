"""Configuration helpers for paper-like grouped Flower OTA-FL CIFAR-10 simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SimConfig:
    """Simulation parameters corresponding to the ICC Fig. 1 notation."""

    # Physical OTA-FL setting
    num_devices: int = 300
    num_flower_clients: int = 24
    coverage_m: float = 550.0
    max_symbol_power_dbm: float = 20.0
    noise_power_dbm: float = -50.0
    power_scaling_db: float = -10.0
    num_subchannels: int = 1024
    path_loss_exp: float = 4.0

    # CIFAR-10 non-IID/imbalanced client data
    mean_client_size: int = 160
    min_client_size: int = 10
    classes_per_client: int = 3
    bs_stratified: bool = True

    # Local training; paper does not disclose these exactly.
    # The defaults below are conservative/paper-like reproduction candidates.
    batch_size: int = 32
    local_epochs: int = 1
    lr: float = 0.02
    optimizer: str = "sgd"
    momentum: float = 0.0
    weight_decay: float = 0.0
    num_workers: int = 0

    # Reproducibility. Split seed controls the fixed data/distance split;
    # runtime seed controls local shuffles/channels/noise. Keeping these separate
    # makes it possible to average runtime randomness without changing datasets.
    split_seed: int = 42
    runtime_seed: int = 42

    # Runtime / metrics
    track_distortion: bool = True
    channel_eps: float = 1e-30
    rho_eps: float = 1e-30
    power_tol: float = 1e-5
    power_max_iters: int = 60
    device: str = "auto"
    eval_every: int = 1
    torch_threads: int = 1
    cache_clients: bool = True

    @property
    def max_symbol_power_mw(self) -> float:
        return 10.0 ** (self.max_symbol_power_dbm / 10.0)

    @property
    def noise_power_mw(self) -> float:
        return 10.0 ** (self.noise_power_dbm / 10.0)

    @property
    def power_scaling_linear(self) -> float:
        return 10.0 ** (self.power_scaling_db / 10.0)

    @property
    def seed(self) -> int:
        """Backward-compatible alias for older code paths."""
        return self.runtime_seed


def list_int(values: Any) -> list[int]:
    """Convert Hydra/OmegaConf list-like values into a plain list of ints."""
    if isinstance(values, str):
        stripped = values.strip().strip("[]")
        if not stripped:
            return []
        return [int(x.strip()) for x in stripped.split(",")]
    if isinstance(values, Sequence):
        return [int(v) for v in values]
    return [int(values)]


def _get_bool(cfg: Any, key: str, default: bool) -> bool:
    return bool(getattr(cfg, key, default))


def _get_int(cfg: Any, key: str, default: int) -> int:
    return int(getattr(cfg, key, default))


def build_sim_config(cfg: Any) -> SimConfig:
    """Build a frozen dataclass from Hydra/OmegaConf config."""
    legacy_seed = int(getattr(cfg, "seed", 42))
    split_seed = int(getattr(cfg, "split_seed", legacy_seed))
    runtime_seed = int(getattr(cfg, "runtime_seed", legacy_seed))
    return SimConfig(
        num_devices=int(cfg.num_devices),
        num_flower_clients=int(cfg.num_flower_clients),
        coverage_m=float(cfg.coverage_m),
        max_symbol_power_dbm=float(cfg.max_symbol_power_dbm),
        noise_power_dbm=float(cfg.noise_power_dbm),
        power_scaling_db=float(cfg.power_scaling_db),
        num_subchannels=int(cfg.num_subchannels),
        path_loss_exp=float(cfg.path_loss_exp),
        mean_client_size=int(cfg.mean_client_size),
        min_client_size=int(cfg.min_client_size),
        classes_per_client=int(cfg.classes_per_client),
        bs_stratified=_get_bool(cfg, "bs_stratified", True),
        batch_size=int(cfg.batch_size),
        local_epochs=int(cfg.local_epochs),
        lr=float(cfg.lr),
        optimizer=str(cfg.optimizer),
        momentum=float(cfg.momentum),
        weight_decay=float(cfg.weight_decay),
        num_workers=int(cfg.num_workers),
        split_seed=split_seed,
        runtime_seed=runtime_seed,
        track_distortion=bool(cfg.track_distortion),
        device=str(cfg.device),
        eval_every=max(1, int(cfg.eval_every)),
        torch_threads=max(1, int(cfg.torch_threads)),
        cache_clients=bool(cfg.cache_clients),
    )
