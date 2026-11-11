"""Explicit opt-in configuration. No numerical wireless settings are paper locks."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields
import math
from typing import Any
from bayesfl.wireless.payload import integer, positive


@dataclass
class SolverConfig:
    initial_price: float = 1.0
    max_bracket_queries: int = 48
    max_bisection_queries: int = 64
    price_atol: float = 0.0
    price_rtol: float = 1e-12
    gap_atol: float = 1e-10
    gap_rtol: float = 1e-6


@dataclass
class ChannelConfig:
    mode: str = "fixed"  # fixed, block_rayleigh, trace
    seed: int = 0
    path_gains: list[float] = field(default_factory=list)
    uplink_psd_w_hz: list[float] = field(default_factory=list)
    downlink_psd_w_hz: float | None = None
    noise_psd_w_hz: float | None = None
    reciprocal_fading: bool = True
    trace_csv: str | None = None
    # Optional conservative cap: PSD * entire system bandwidth <= cap.
    client_power_cap_w: list[float] | None = None


@dataclass
class ComputeConfig:
    # All arrays indexed by logical client ID. Explicit device model, not GPU runtime.
    seconds_per_sample_epoch: list[float] = field(default_factory=list)
    fixed_seconds: list[float] = field(default_factory=list)
    score_seconds: list[float] = field(default_factory=list)


@dataclass
class WirelessConfig:
    enabled: bool = False
    backend: str = "flower"  # flower (Ray), local (serial deterministic emulator)
    policy: str = "joint"  # joint, equal_bandwidth, uniform, dense
    uplink_bandwidth_hz: float | None = None
    downlink_bandwidth_hz: float | None = None
    round_target_s: float | None = None
    total_time_budget_s: float | None = None
    control_reserve_s: float | None = None
    control_rate_bps: float | None = None
    control_exchange_latency_s: float | None = None
    rpc_timeout_s: float = 3600.0
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    def validate(self, cfg) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("wireless.enabled must be boolean")
        if not self.enabled:
            return
        k = cfg.federation.num_clients
        if self.backend not in {"flower", "local"}:
            raise ValueError("wireless.backend must be flower or local")
        if self.policy not in {"joint", "equal_bandwidth", "uniform", "dense"}:
            raise ValueError("Invalid wireless.policy")
        if cfg.method not in {"fola", "fedavg"}:
            raise ValueError("Wireless mode supports FOLA and dense FedAvg, not BBB")
        if cfg.method == "fedavg" and self.policy != "dense":
            raise ValueError("FedAvg wireless policy must be dense")
        if cfg.method == "fola":
            if self.policy != "dense" and cfg.compression.selection_rule not in {"kl_global_local", "random"}:
                raise ValueError("Adaptive wireless FOLA requires kl_global_local or random selection")
            if self.policy == "dense" and cfg.compression.selection_rule != "dense":
                raise ValueError("Dense wireless policy requires dense selection")
            if cfg.compression.keep_ratio != 1.0:
                raise ValueError("Set keep_ratio=1.0 as placeholder; wireless uses integer grants, not this ratio")
        if cfg.federation.clients_per_round != k:
            raise ValueError("This wireless protocol requires all clients; no silent client dropping")
        if cfg.training.local_epochs > 10:
            raise ValueError("Wireless studies preserve the project E<=10 rule")
        if cfg.output.save_full_client_posteriors:
            raise ValueError("Server-side full local posterior snapshots defeat the wireless protocol")
        if cfg.communication.max_communication_bytes is not None:
            raise ValueError("Do not combine legacy fixed-byte stopping with dynamic wireless allocation")
        if cfg.communication.downlink_mode != "multicast":
            raise ValueError("Wireless experiments require modeled multicast")
        if not cfg.communication.deterministic_client_schedule:
            raise ValueError("Wireless mode requires deterministic logical client identities")
        for name in ("uplink_bandwidth_hz", "downlink_bandwidth_hz", "round_target_s",
                     "control_rate_bps", "rpc_timeout_s"):
            positive(getattr(self, name), f"wireless.{name}")
        for name in ("control_reserve_s", "control_exchange_latency_s"):
            value = getattr(self, name)
            if value is None or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"wireless.{name} must be explicitly nonnegative")
        if self.control_reserve_s >= self.round_target_s:
            raise ValueError("Control reservation must be smaller than the round target")
        if self.total_time_budget_s is not None:
            positive(self.total_time_budget_s, "total_time_budget_s")
        c = self.channel
        if c.mode not in {"fixed", "block_rayleigh", "trace"}:
            raise ValueError("channel.mode must be fixed, block_rayleigh or trace")
        integer(c.seed, "channel.seed", 0)
        if not isinstance(c.reciprocal_fading, bool):
            raise ValueError("reciprocal_fading must be boolean")
        positive(c.noise_psd_w_hz, "noise_psd_w_hz")
        positive(c.downlink_psd_w_hz, "downlink_psd_w_hz")
        _vector(c.uplink_psd_w_hz, k, "uplink_psd_w_hz", allow_zero=False)
        if c.mode != "trace":
            _vector(c.path_gains, k, "path_gains", allow_zero=False)
            if c.trace_csv is not None:
                raise ValueError("trace_csv requires channel.mode=trace")
        elif not c.trace_csv:
            raise ValueError("channel.mode=trace requires trace_csv")
        if c.client_power_cap_w is not None:
            _vector(c.client_power_cap_w, k, "client_power_cap_w", allow_zero=False)
            for psd, cap in zip(c.uplink_psd_w_hz, c.client_power_cap_w):
                if psd * self.uplink_bandwidth_hz > cap:
                    raise ValueError("PSD * total bandwidth exceeds the declared conservative client power cap")
        for name in ("seconds_per_sample_epoch", "fixed_seconds", "score_seconds"):
            _vector(getattr(self.compute, name), k, "compute." + name, allow_zero=True)
        positive(self.solver.initial_price, "solver.initial_price")
        integer(self.solver.max_bracket_queries, "max_bracket_queries", 1)
        integer(self.solver.max_bisection_queries, "max_bisection_queries", 0)
        for name in ("price_atol", "price_rtol", "gap_atol", "gap_rtol"):
            value = getattr(self.solver, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"solver.{name} must be finite and nonnegative")


def _vector(values, k: int, name: str, *, allow_zero: bool) -> None:
    if not isinstance(values, (list, tuple)) or len(values) != k:
        raise ValueError(f"{name} requires exactly {k} explicit client values")
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
            raise ValueError(f"Invalid {name}: {v!r}")


def _strict(cls, raw):
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unexpected = set(raw) - {f.name for f in fields(cls)}
    if unexpected:
        raise ValueError(f"Unknown {cls.__name__} field(s): {sorted(unexpected)}")
    return cls(**raw)


def parse_wireless(raw: Any) -> WirelessConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("wireless must be a mapping")
    values = dict(raw)
    for name, cls in (("channel", ChannelConfig), ("compute", ComputeConfig), ("solver", SolverConfig)):
        values[name] = _strict(cls, values.get(name, {}))
    return _strict(WirelessConfig, values)
