"""Minimal run specification for the future-work baseline."""

from __future__ import annotations

from dataclasses import dataclass


METHODS = ("fedavg", "bayesavg")
ALIASES = {"proposed": "bayesavg", "bayesian": "bayesavg", "bt": "bayesavg"}


def normalize_method(method: str) -> str:
    value = str(method).strip().lower().replace("-", "")
    value = ALIASES.get(value, value)
    if value not in METHODS:
        raise ValueError(f"Unsupported method {method!r}; expected fedavg or bayesavg")
    return value


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    method: str
    seed: int
    rounds: int
    realization: int = 0
