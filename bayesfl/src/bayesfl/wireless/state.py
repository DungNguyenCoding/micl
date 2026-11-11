"""Separate wireless ledger/checkpoints; never reinterpret legacy resume files."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
from bayesfl.communication import atomic_json
from bayesfl.posterior.sparse import layout_manifest, snapshot_id, posterior_vectors
from bayesfl.wireless.protocol import canonical
from bayesfl.wireless import PROTOCOL


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(cfg) -> str:
    data = cfg.to_dict()
    data["training"].pop("rounds")
    data["wireless"].pop("total_time_budget_s")
    # Copy of the trace plus its hash is committed independently. The original
    # external trace path is still immutable provenance in resolved_config.
    return hashlib.sha256(canonical(data)).hexdigest()


@contextmanager
def run_lock(run_dir: Path):
    """Linux advisory lock; held for the complete simulation, not only commit."""
    import fcntl
    root = run_dir / "wireless"
    root.mkdir(exist_ok=True)
    with (root / "writer.lock").open("a+") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another wireless writer owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class WirelessState:
    TRACKED = ("metrics/global_metrics.csv", "metrics/wireless_rounds.csv", "metrics/wireless_clients.csv")

    def __init__(self, cfg, layout, initial_arrays, run_dir: Path, *, partition_sha256: str,
                 trace_sha256: str | None = None, resume=False):
        self.cfg, self.layout, self.run_dir = cfg, layout, Path(run_dir)
        self.partition_sha256, self.trace_sha256 = partition_sha256, trace_sha256
        self.current = [np.asarray(a).copy() for a in initial_arrays]
        self.round_id, self.pending_round = 0, None
        self.total_bytes, self.downlink_bytes, self.uplink_bytes, self.control_bytes = 0, 0, 0, 0
        self.elapsed_s, self.energy_j, self.local_steps, self.examples_seen = 0.0, 0.0, 0, 0
        self.event_count = 0
        self.last_round_stats, self.last_evaluation = {}, {}
        self.failure = None
        self.incomplete_accounting = False
        self.rep = "mean_and_raw_precision" if cfg.method == "fola" else "mean_only"
        self.layout_id = layout_manifest(layout, representation=self.rep)["layout_id"]
        self.events = self.run_dir / "wireless" / "events.jsonl"
        self.events.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            self._restore()
        elif self.events.exists() or (self.run_dir / "wireless_resume" / "latest.json").exists():
            raise ValueError("New wireless run refuses an existing ledger/checkpoint")
        else:
            self.events.touch(exist_ok=False)
            atomic_json(self.run_dir / "wireless" / "protocol.json", {
                "protocol": PROTOCOL, "method": cfg.method, "policy": cfg.wireless.policy,
                "selection_rule": cfg.compression.selection_rule, "num_clients": cfg.federation.num_clients,
                "round_target_s": cfg.wireless.round_target_s, "array_accounting": "one multicast model + individual actual packet arrays",
                "control_accounting": "canonical UTF-8 JSON of instructions/replies; excludes Flower/network framing",
                "clock": "fixed slots, control reservation, modeled preparation max; no simulated sleep",
                "power_model": "fixed PSD; ideal successful digital delivery; not AirComp",
                "layout_id": self.layout_id, "trace_sha256": self.trace_sha256})
            atomic_json(self.run_dir / "wireless" / "layout_manifest.json", layout_manifest(layout, representation=self.rep))

    @property
    def run_token(self):
        return hashlib.sha256((self.run_dir.name + ":" + fingerprint(self.cfg)).encode()).hexdigest()

    def event(self, phase: str, *, downlink_bytes=0, uplink_bytes=0, control_bytes=0,
              modeled_seconds=0.0, energy_j=0.0, **details):
        for v in (downlink_bytes, uplink_bytes, control_bytes):
            if type(v) is not int or v < 0:
                raise ValueError("Event byte counts must be nonnegative integers")
        for v in (modeled_seconds, energy_j):
            if not math.isfinite(v) or v < 0:
                raise ValueError("Invalid event time/energy")
        row = dict(event_id=self.event_count + 1, round=self.pending_round if self.pending_round is not None else self.round_id,
                   phase=phase, downlink_array_bytes=downlink_bytes, uplink_array_bytes=uplink_bytes,
                   control_bytes=control_bytes, modeled_seconds=modeled_seconds, energy_j=energy_j, **details)
        data = canonical(row) + b"\n"
        with self.events.open("ab") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        self.event_count += 1
        self.total_bytes += downlink_bytes + uplink_bytes
        self.downlink_bytes += downlink_bytes
        self.uplink_bytes += uplink_bytes
        self.control_bytes += control_bytes
        self.elapsed_s += modeled_seconds
        self.energy_j += energy_j

    def stop_reason(self):
        if self.failure:
            return "failed"
        if self.round_id >= self.cfg.training.rounds:
            return "max_rounds"
        cap = self.cfg.wireless.total_time_budget_s
        if cap is not None and (self.round_id + 1) * self.cfg.wireless.round_target_s > cap:
            return "modeled_time_budget"
        return "running"

    def evaluation_fields(self):
        return dict(round_id=self.round_id, method=self.cfg.method_id,
            wireless_policy=self.cfg.wireless.policy, wireless_backend=self.cfg.wireless.backend,
            protocol=PROTOCOL, selection_rule=self.cfg.compression.selection_rule,
            cumulative_all_array_bytes=self.total_bytes,
            cumulative_train_total_array_bytes=self.total_bytes,
            cumulative_train_downlink_array_bytes=self.downlink_bytes,
            cumulative_train_uplink_array_bytes=self.uplink_bytes,
            cumulative_control_bytes=self.control_bytes,
            cumulative_modeled_time_seconds=self.elapsed_s,
            cumulative_modeled_tx_energy_j=self.energy_j,
            cumulative_local_training_steps=self.local_steps,
            cumulative_local_examples_seen=self.examples_seen,
            array_accounting_complete=not self.incomplete_accounting,
            stop_reason=self.stop_reason(), **self.last_round_stats)

    def save_resume(self, evaluation=None):
        if self.pending_round is not None:
            raise RuntimeError("Cannot commit a partial wireless round")
        if evaluation is not None:
            self.last_evaluation = {k: float(v) for k, v in evaluation.items()
                                    if isinstance(v, (int, float, np.number)) and math.isfinite(v)}
        root = self.run_dir / "wireless_resume"
        root.mkdir(exist_ok=True)
        stem = f"round_{self.round_id:06d}"
        arrays_path = root / (stem + ".npz")
        tmp = arrays_path.with_suffix(".npz.tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f, **{f"array_{i:04d}": a for i, a in enumerate(self.current)})
        tmp.replace(arrays_path)
        state = dict(version=2, protocol=PROTOCOL, round_id=self.round_id, fingerprint=fingerprint(self.cfg),
            partition_sha256=self.partition_sha256, trace_sha256=self.trace_sha256, layout_id=self.layout_id,
            snapshot_id=snapshot_id(self.current, self.layout, self.round_id, representation=self.rep),
            arrays_file=arrays_path.name, arrays_sha256=file_sha(arrays_path), array_count=len(self.current),
            events_sha256=file_sha(self.events), event_count=self.event_count,
            tracked_sha256={name: file_sha(self.run_dir/name) for name in self.TRACKED if (self.run_dir/name).is_file()},
            counters={k:getattr(self, k) for k in ("total_bytes", "downlink_bytes", "uplink_bytes", "control_bytes",
                      "elapsed_s", "energy_j", "local_steps", "examples_seen")},
            last_round_stats=self.last_round_stats, last_evaluation=self.last_evaluation,
            rng_policy="client/round reseed; separate counter-based channel/mask RNG; fresh local optimizers")
        path = root / (stem + ".json")
        atomic_json(path, state)
        atomic_json(root / "latest.json", {"state_file":path.name, "state_sha256":file_sha(path)})
        # Retain latest and previous state; scheduled evaluator checkpoints remain.
        for old in sorted(root.glob("round_*.npz"))[:-2]:
            old.unlink()
            old.with_suffix(".json").unlink(missing_ok=True)

    def _restore(self):
        root = self.run_dir / "wireless_resume"
        latest = json.loads((root / "latest.json").read_text())
        name = latest["state_file"]
        if Path(name).name != name:
            raise ValueError("Invalid checkpoint path")
        path = root / name
        if file_sha(path) != latest["state_sha256"]:
            raise ValueError("Checkpoint JSON digest mismatch")
        state = json.loads(path.read_text())
        for key, expected in (("protocol", PROTOCOL), ("fingerprint", fingerprint(self.cfg)),
                              ("partition_sha256", self.partition_sha256), ("trace_sha256", self.trace_sha256),
                              ("layout_id", self.layout_id)):
            if state.get(key) != expected:
                raise ValueError(f"Wireless resume mismatch: {key}; only total rounds/time cap may change")
        if not self.events.is_file() or file_sha(self.events) != state["events_sha256"]:
            raise ValueError("Uncommitted/corrupt wireless events; partial-round resume is forbidden")
        for name in self.TRACKED:
            p = self.run_dir / name
            if (p.exists()) != (name in state["tracked_sha256"]):
                raise ValueError(f"Missing/uncommitted metrics: {name}")
            if p.exists() and file_sha(p) != state["tracked_sha256"][name]:
                raise ValueError(f"Metrics differ from committed checkpoint: {name}")
        name = state["arrays_file"]
        if Path(name).name != name or file_sha(root / name) != state["arrays_sha256"]:
            raise ValueError("Wireless state arrays digest mismatch")
        with np.load(root / name, allow_pickle=False) as data:
            self.current = [data[f"array_{i:04d}"].copy() for i in range(state["array_count"])]
        if self.cfg.method == "fola":
            posterior_vectors(self.current, self.layout)
        else:
            self.layout.validate(self.current)
        self.round_id = int(state["round_id"])
        if snapshot_id(self.current, self.layout, self.round_id, representation=self.rep) != state["snapshot_id"]:
            raise ValueError("Wireless model snapshot mismatch")
        for key, value in state["counters"].items():
            setattr(self, key, value)
        self.event_count = state["event_count"]
        self.last_round_stats, self.last_evaluation = state["last_round_stats"], state["last_evaluation"]
        if self.cfg.training.rounds < self.round_id:
            raise ValueError("Round cap below completed round")
        if self.cfg.wireless.total_time_budget_s is not None and self.cfg.wireless.total_time_budget_s < self.elapsed_s:
            raise ValueError("Time cap below consumed modeled time")

    def finalize(self, failed=None):
        if failed is not None:
            self.failure = str(failed)
        atomic_json(self.run_dir / "run_summary.json", dict(self.evaluation_fields(),
            completed_rounds=self.round_id, pending_round=self.pending_round,
            failure=self.failure, last_evaluation=self.last_evaluation))
