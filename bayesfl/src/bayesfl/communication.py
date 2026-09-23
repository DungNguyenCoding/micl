"""Logical message-boundary array accounting, not physical network traffic.

Final ndarray.nbytes determines the payload. Serialized tensor bytes are
reported separately (e.g. Flower Parameters.tensors including NumPy headers).
Control/metrics envelopes, retries/partial messages not observed by the API,
transport framing and local copies are not guessed or counted as wire traffic.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np


COUNTERS = ("train_uplink_array_bytes", "train_downlink_array_bytes", "train_total_array_bytes",
            "initialization_array_bytes", "evaluation_array_bytes", "all_array_bytes")


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def array_bytes(arrays: Sequence[np.ndarray]) -> int:
    return sum(int(a.nbytes) for a in arrays)


def core_round_bytes(d: int, k: int, *, sparse: bool = False, m: int | None = None) -> dict:
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (d, k)) or d <= 0 or k < 0:
        raise ValueError("Require integer d>0 and k>=0")
    down = int(k) * 8 * int(d)
    if sparse:
        if isinstance(m, bool) or not isinstance(m, Integral) or not 0 <= m <= d:
            raise ValueError("Sparse cost requires an integer 0<=m<=d")
        up = int(k) * (8 * int(m) + (int(d) + 7) // 8)
    else:
        up = down
    return dict(down_bytes=down, up_bytes=up, total_bytes=down + up)


def payload_components(arrays, *, method: str, sparse_upload: bool = False, tensor_count: int = 0) -> dict:
    total = array_bytes(arrays)
    if sparse_upload and len(arrays) == 3:
        bitmap, mean, cov = (int(a.nbytes) for a in arrays)
    elif method == "fola" and len(arrays) == 2 * tensor_count:
        bitmap, mean, cov = 0, array_bytes(arrays[:tensor_count]), array_bytes(arrays[tensor_count:])
    else:
        bitmap, mean, cov = 0, total, 0
    return dict(mean_bytes=mean, covariance_bytes=cov, bitmap_bytes=bitmap,
                auxiliary_array_bytes=0, array_payload_bytes=total)


class CommunicationLedger:
    """One append-only event log with duplicate prevention and replay validation.

    A disposition is a status update, not another transfer. Failed rounds still
    retain every observed download/upload; they cannot be silently rolled back.
    """
    def __init__(self, path: Path, *, run_id: str, method: str, seed: int, restore: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id, self.method, self.seed = run_id, method, int(seed)
        self.totals = {name: 0 for name in COUNTERS}
        self.rounds: dict[int, dict] = {}
        self.ids: set[str] = set()
        self.event_count = 0
        self._digest = hashlib.sha256()
        self.serialized_tensor_bytes = 0
        self.serialized_tensor_observed = False
        self.unmeasured_array_messages = 0
        if self.path.exists():
            if not restore:
                raise FileExistsError(f"Refusing to overwrite communication ledger: {self.path}")
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    event = json.loads(line)
                    self._apply(event)
                    self._digest.update(line.encode())
                    self.event_count += 1
        else:
            self.path.touch()

    def _apply(self, e: dict) -> None:
        if (e["run_id"], e["method"], e["seed"]) != (self.run_id, self.method, self.seed):
            raise ValueError("Ledger belongs to a different run")
        if e["event_type"] == "disposition":
            if e["message_id"] not in self.ids:
                raise ValueError("Disposition refers to an unknown transfer")
            return
        mid = e["message_id"]
        if mid in self.ids:
            raise ValueError(f"Duplicate transfer: {mid}")
        if e["direction"] not in {"uplink", "downlink"} or e["phase"] not in {"initialize", "fit", "evaluate"}:
            raise ValueError("Invalid communication event")
        size = e["array_payload_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("Payload bytes must be a nonnegative integer")
        self.ids.add(mid)
        if e.get("array_payload_observed", True) is False:
            self.unmeasured_array_messages += 1
        r = self.rounds.setdefault(int(e["round_id"]), {name: 0 for name in COUNTERS})
        keys = ["all_array_bytes"]
        if e["phase"] == "fit":
            keys += ["train_total_array_bytes", f"train_{e['direction']}_array_bytes"]
        elif e["phase"] == "initialize":
            keys.append("initialization_array_bytes")
        else:
            keys.append("evaluation_array_bytes")
        for key in keys:
            self.totals[key] += size
            r[key] += size
        value = e.get("serialized_tensor_bytes")
        if value is not None:
            if not isinstance(value, int) or value < 0:
                raise ValueError("Invalid serialized tensor byte count")
            self.serialized_tensor_bytes += value
            self.serialized_tensor_observed = True

    def _append(self, event: dict) -> None:
        event = dict(run_id=self.run_id, method=self.method, seed=self.seed, **event)
        self._apply(event)
        line = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
        self._digest.update(line.encode())
        self.event_count += 1

    def record(self, *, round_id: int, client_id, direction: str, phase: str,
               components: dict, serialized_tensor_bytes: int | None = None,
               attempt_id: int = 0, status: str | None = None, **metadata) -> str:
        mid = f"{phase}:{int(round_id)}:{client_id}:{direction}:{int(attempt_id)}"
        self._append(dict(event_type="transfer", round_id=int(round_id), model_version=int(round_id),
                          client_id=str(client_id), message_id=mid, attempt_id=int(attempt_id),
                          direction=direction, phase=phase,
                          status=status or ("dispatched" if direction == "downlink" else "received"),
                          accounting_kind="logical_payload", serialized_tensor_bytes=serialized_tensor_bytes,
                          serialized_message_bytes=None, **components, **metadata))
        return mid

    def record_undecodable(self, *, round_id: int, client_id, serialized_tensor_bytes: int,
                           reason: str) -> str:
        """Store what was observed; do not invent ndarray sizes for corrupt bytes.

        Integer array counters become a known lower bound for this failed run.
        Valid no-failure runs always have unmeasured_array_messages == 0.
        """
        return self.record(round_id=round_id, client_id=client_id, direction="uplink", phase="fit",
                           status="rejected", components=payload_components([], method="unknown"),
                           serialized_tensor_bytes=serialized_tensor_bytes,
                           array_payload_observed=False, array_payload_count_is_lower_bound=True,
                           reason=str(reason))

    def reject(self, message_id: str, reason: str) -> None:
        self._append(dict(event_type="disposition", message_id=message_id,
                          status="rejected", reason=str(reason)))

    def cumulative(self) -> dict:
        return {"cumulative_" + key: value for key, value in self.totals.items()}

    def round_fields(self, round_id: int) -> dict:
        r = self.rounds.get(round_id, {name: 0 for name in COUNTERS})
        return {"round_" + key: value for key, value in r.items()}

    def state(self) -> dict:
        return dict(event_count=self.event_count, event_sha256=self._digest.hexdigest(),
                    totals=dict(self.totals), serialized_tensor_bytes=self.serialized_tensor_bytes,
                    serialized_tensor_observed=self.serialized_tensor_observed,
                    unmeasured_array_messages=self.unmeasured_array_messages)

    def assert_checkpoint(self, state: dict) -> None:
        if state != self.state():
            raise ValueError(
                "Ledger differs from the last completed checkpoint. This may be an interrupted "
                "in-flight round. Resume refuses to discard already-consumed communication."
            )
