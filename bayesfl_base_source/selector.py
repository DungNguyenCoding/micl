"""Physical-client selection policies.

Only random selection is active now. Wireless-aware hooks are kept explicit so
analog OTA, digital-link, or channel-quality policies can be added later without
rewriting the Flower strategy or client code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np


@dataclass
class SelectionResult:
    round_idx: int
    selected_ids: List[int]
    policy_name: str

    @property
    def selected_count(self) -> int:
        return len(self.selected_ids)

    def as_csv_string(self) -> str:
        return ",".join(str(x) for x in self.selected_ids)


class BaseClientSelector:
    policy_name = "base"

    def select(self, round_idx: int, num_devices: int, fraction: float) -> SelectionResult:
        raise NotImplementedError


class RandomClientSelector(BaseClientSelector):
    """Uniformly choose a fixed fraction of physical devices each round."""

    policy_name = "random"

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed + 991)

    def select(self, round_idx: int, num_devices: int, fraction: float) -> SelectionResult:
        count = max(1, int(round(float(fraction) * int(num_devices))))
        count = min(count, int(num_devices))
        selected = self.rng.choice(np.arange(num_devices), size=count, replace=False)
        return SelectionResult(round_idx=round_idx, selected_ids=sorted(int(x) for x in selected), policy_name=self.policy_name)


class WirelessQualitySelector(BaseClientSelector):
    """Placeholder for future communication-aware client selection.

    TODO(wireless): implement a policy interface that accepts per-device channel
    state, SNR, path loss, battery/energy limits, and OTA aggregation constraints.
    Suggested future schema:

        select(round_idx, num_devices, fraction, channel_state_df, device_summary_df)

    Initial policies to add:
        1. analog_ota_top_snr: choose clients with high channel quality subject
           to fairness constraints and OTA power normalization.
        2. digital_link_budget: choose clients whose uplink rate can deliver the
           update before a round deadline.
        3. hybrid_quality_importance: combine dataset importance and wireless
           reliability, then randomize with epsilon-greedy exploration.
    """

    policy_name = "wireless_todo"

    def select(self, round_idx: int, num_devices: int, fraction: float) -> SelectionResult:
        raise NotImplementedError(
            "WirelessQualitySelector is a TODO hook. Use --selector random now, "
            "then implement channel-aware selection in selector.py."
        )


def build_selector(policy: str, seed: int) -> BaseClientSelector:
    if policy == "random":
        return RandomClientSelector(seed=seed)
    if policy == "wireless_todo":
        return WirelessQualitySelector()
    raise ValueError(f"Unknown selector policy: {policy}")


def parse_selected_ids(value: str | bytes | Sequence[int]) -> set[int]:
    """Parse selected physical IDs from a Flower Scalar-compatible string."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        if not value.strip():
            return set()
        return {int(part) for part in value.split(",") if part.strip()}
    return {int(x) for x in value}
