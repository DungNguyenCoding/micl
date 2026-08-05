"""Experiment matrix corresponding to Figures 2-6 of the paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from config import SimulationConfig


ALL_METHODS = ("fedavg", "fedprox", "scaffold", "proposed")


@dataclass(frozen=True)
class ExperimentCondition:
    experiment: str
    name: str
    labels_per_client: int
    mean_samples_per_client: float
    power_dbm: float
    methods: tuple[str, ...]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    experiment: str
    condition: str
    method: str
    realization: int
    seed: int
    rounds: int


def payload_multiplier(method: str) -> int:
    return 2 if method in {"proposed", "scaffold"} else 1


def derive_rounds(
    config: SimulationConfig,
    method: str,
    model_dimension: int,
) -> int:
    if config.training.num_rounds is not None:
        return int(config.training.num_rounds)
    per_round = payload_multiplier(method) * int(model_dimension)
    return max(1, int(config.training.max_channel_uses // per_round))


def experiment_conditions(
    experiment: str,
    base: SimulationConfig,
    methods_override: Optional[Sequence[str]] = None,
) -> List[ExperimentCondition]:
    experiment = experiment.lower()
    override = tuple(method.lower() for method in methods_override) if methods_override else None

    def methods(default: Sequence[str]) -> tuple[str, ...]:
        selected = override or tuple(default)
        invalid = set(selected) - set(ALL_METHODS)
        if invalid:
            raise ValueError(f"Unsupported methods: {sorted(invalid)}")
        return tuple(selected)

    if experiment == "fig2":
        return [
            ExperimentCondition(
                experiment="fig2",
                name="default",
                labels_per_client=1,
                mean_samples_per_client=10,
                power_dbm=23,
                methods=methods(ALL_METHODS),
            )
        ]
    if experiment == "fig3":
        return [
            ExperimentCondition(
                experiment="fig3",
                name=f"labels_{labels}",
                labels_per_client=labels,
                mean_samples_per_client=10,
                power_dbm=23,
                methods=methods(("fedavg", "fedprox", "proposed")),
            )
            for labels in (1, 2, 10)
        ]
    if experiment == "fig4":
        return [
            ExperimentCondition(
                experiment="fig4",
                name=f"mean_samples_{mean}",
                labels_per_client=1,
                mean_samples_per_client=float(mean),
                power_dbm=23,
                methods=methods(("fedavg", "fedprox", "proposed")),
            )
            for mean in (10, 20, 50)
        ]
    if experiment == "fig5":
        return [
            ExperimentCondition(
                experiment="fig5",
                name=f"power_{power}dbm",
                labels_per_client=1,
                mean_samples_per_client=10,
                power_dbm=float(power),
                methods=methods(("fedavg", "fedprox", "proposed")),
            )
            for power in (3, 23, 33)
        ]
    if experiment == "fig6":
        return [
            ExperimentCondition(
                experiment="fig6",
                name="default",
                labels_per_client=1,
                mean_samples_per_client=10,
                power_dbm=23,
                methods=methods(("fedavg", "fedprox", "proposed")),
            )
        ]
    raise ValueError("experiment must be one of fig2, fig3, fig4, fig5, fig6")
