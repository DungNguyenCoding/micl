"""Configuration helpers for grouped Flower OTA-FL CIFAR-10 simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SimConfig:
    """Simulation parameters corresponding to the paper/screenshot notation."""

    num_devices: int = 300
    num_flower_clients: int = 48
    coverage_m: float = 550.0
    max_symbol_power_dbm: float = 20.0
    noise_power_dbm: float = -50.0
    power_scaling_db: float = -10.0
    num_subchannels: int = 1024
    path_loss_exp: float = 4.0
    mean_client_size: int = 160
    min_client_size: int = 10
    classes_per_client: int = 3
    batch_size: int = 10
    local_epochs: int = 3
    lr: float = 0.05
    optimizer: str = "sgd"
    momentum: float = 0.0
    weight_decay: float = 0.0
    num_workers: int = 0
    seed: int = 42
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


def build_sim_config(cfg: Any) -> SimConfig:
    """Build a frozen dataclass from Hydra/OmegaConf config."""
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
        batch_size=int(cfg.batch_size),
        local_epochs=int(cfg.local_epochs),
        lr=float(cfg.lr),
        optimizer=str(cfg.optimizer),
        momentum=float(cfg.momentum),
        weight_decay=float(cfg.weight_decay),
        num_workers=int(cfg.num_workers),
        seed=int(cfg.seed),
        track_distortion=bool(cfg.track_distortion),
        device=str(cfg.device),
        eval_every=max(1, int(cfg.eval_every)),
        torch_threads=max(1, int(cfg.torch_threads)),
        cache_clients=bool(cfg.cache_clients),
    )
