"""Shared run state for the Flower strategy and offline integration checks.

Owns only communicated global state, packet reconstruction and accounting.
Local training is not implemented here and full local sparse updates never
enter this object. A resume checkpoint is committed after central evaluation.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from bayesfl.communication import CommunicationLedger, array_bytes, atomic_json, payload_components
from bayesfl.config import ExperimentConfig
from bayesfl.posterior.packing import ParameterLayout
from bayesfl.posterior.sparse import (
    decode_packet, dimension, flatten, keep_count, layout_manifest,
    posterior_vectors, snapshot_id, unflatten,
)


def config_fingerprint(cfg: ExperimentConfig) -> str:
    data = cfg.to_dict()
    # These two limits may be extended on a clean completed-round resume.
    data["training"].pop("rounds")
    data["communication"].pop("max_communication_bytes")
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_resume_bundle(run_dir: Path) -> tuple[dict, list[np.ndarray]]:
    root = Path(run_dir) / "resume"
    pointer = json.loads((root / "latest.json").read_text())
    for name in (pointer["state_file"], pointer["arrays_file"]):
        if Path(name).name != name:
            raise ValueError("Invalid checkpoint path")
    state = json.loads((root / pointer["state_file"]).read_text())
    with np.load(root / pointer["arrays_file"], allow_pickle=False) as data:
        arrays = [data[f"array_{i:04d}"].copy() for i in range(state["array_count"])]
    return state, arrays


class RunState:
    def __init__(self, cfg: ExperimentConfig, layout: ParameterLayout, initial_arrays: Sequence[np.ndarray],
                 run_dir: Path, *, partition_sha256: str = "unspecified", resume: bool = False):
        self.cfg, self.layout, self.run_dir = cfg, layout, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._layout_representation = "mean_and_raw_precision" if cfg.method == "fola" else "parameter"
        self.manifest = layout_manifest(layout, representation=self._layout_representation)
        self.layout_id = self.manifest["layout_id"]
        self.d = dimension(layout)
        self.partition_sha256 = str(partition_sha256)
        self.current = [np.asarray(a).copy() for a in initial_arrays]
        self.round_id = 0
        self.cumulative_local_steps = 0
        self.cumulative_examples_seen = 0
        self.last_round_stats = {}
        self.last_evaluation = {}
        saved = None
        if resume:
            if not cfg.communication.deterministic_client_schedule:
                raise ValueError("Resume requires communication.deterministic_client_schedule: true")
            saved, self.current = load_resume_bundle(self.run_dir)
            if saved["config_fingerprint"] != config_fingerprint(cfg):
                raise ValueError("Resume changed training/protocol settings; only rounds and byte limit may change")
            if saved["layout_id"] != self.layout_id or saved["partition_sha256"] != self.partition_sha256:
                raise ValueError("Resume layout or partition hash does not match")
            self.round_id = int(saved["round_id"])
            if cfg.training.rounds < self.round_id:
                raise ValueError("Round safety cap precedes the saved model")
            if snapshot_id(self.current, layout, self.round_id, representation=self._layout_representation) != saved["state_snapshot_id"]:
                raise ValueError("Checkpoint arrays do not match the committed state")
            self.cumulative_local_steps = int(saved["cumulative_local_steps"])
            self.cumulative_examples_seen = int(saved["cumulative_examples_seen"])
            self.last_round_stats = saved["last_round_stats"]
            self.last_evaluation = saved["last_evaluation"]
        self.ledger = CommunicationLedger(
            self.run_dir / "communication" / "events.jsonl", run_id=self.run_dir.name,
            method=cfg.method_id, seed=cfg.runtime.seed, restore=resume,
        )
        if saved is not None:
            self.ledger.assert_checkpoint(saved["ledger"])
            limit = cfg.communication.max_communication_bytes
            if limit is not None and self.budget_used > limit:
                raise ValueError("New byte budget is below communication already consumed")
            metrics = self.run_dir / "metrics" / "global_metrics.csv"
            if metrics.exists():
                import csv
                with metrics.open(newline="") as f:
                    rounds = [int(float(row["round"])) for row in csv.DictReader(f)]
                if not rounds or rounds[-1] != self.round_id or len(rounds) != len(set(rounds)):
                    raise ValueError("Metrics do not end at the committed evaluated round")
        if cfg.method == "fola":
            posterior_vectors(self.current, layout)
        else:
            layout.validate(self.current)
        self.broadcast = None
        self.base_snapshot_id = None
        self.pending_round = None
        self._received = set()
        self._recipients = set()
        atomic_json(self.run_dir / "communication" / "layout_manifest.json", self.manifest)
        atomic_json(self.run_dir / "communication" / "protocol.json", dict(
            method=cfg.method_id, covariance_representation="precision" if cfg.method == "fola" else "not_applicable",
            precision_policy=cfg.compression.precision_policy,
            score_precision_floor=cfg.score_precision_floor, score_dtype="float64",
            payload_dtype="float32" if cfg.method == "fola" else "actual_array_dtypes",
            sparse_enabled=cfg.sparse_enabled, bitmap_policy="always", bitorder="little",
            missing_coordinate_policy="broadcast_posterior", sparse_downlink=False,
            aggregation="uploaded_baseline_with_all_omitted_coordinate_restoration" if cfg.sparse_enabled else "uploaded_baseline",
            accounting_kind="logical_payload", budget_metric=cfg.communication.budget_metric,
            control_metadata_and_transport_headers_included=False,
            initialization="server_local; identity queries have no model arrays",
            evaluation="central_only; no additional model communication",
            auxiliary_policy="same named_parameters-only layout as baseline; no buffers transmitted",
            precision_zero_note="Raw paper_reference omega may be zero. floor_for_score is a score-only surrogate, not KL of raw zero precision.",
            next_round_prediction=self.next_round_cost(),
        ))

    @property
    def budget_used(self) -> int:
        return self.ledger.cumulative()[self.cfg.communication.budget_metric]

    def next_round_cost(self) -> int:
        k = self.cfg.federation.clients_per_round
        down = array_bytes(self.current)
        up = (8 * keep_count(self.d, self.cfg.compression.keep_ratio) + (self.d + 7) // 8
              if self.cfg.sparse_enabled else down)
        return int(k) * (down + up)

    def stop_reason(self) -> str:
        if self.round_id >= self.cfg.training.rounds:
            return "max_rounds"
        limit = self.cfg.communication.max_communication_bytes
        if limit is not None and self.budget_used + self.next_round_cost() > limit:
            return "communication_budget"
        return "running"

    def prepare_round(self, round_id: int, recipients, arrays, *, serialized_tensor_bytes: int | None = None) -> dict:
        if self.pending_round is not None or round_id != self.round_id + 1:
            raise ValueError("Rounds must be sequential and the preceding round completed")
        if self.stop_reason() != "running":
            raise RuntimeError(f"Cannot dispatch a round after {self.stop_reason()}")
        recipients = [str(cid) for cid in recipients]
        if len(recipients) != self.cfg.federation.clients_per_round or len(set(recipients)) != len(recipients):
            raise ValueError("Expected the configured number of distinct recipients")
        if len(arrays) != len(self.current) or any(
            a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b)
            for a, b in zip(arrays, self.current)
        ):
            raise ValueError("Flower broadcast differs from the current committed global state")
        self.broadcast = [np.asarray(a).copy() for a in arrays]
        for a in self.broadcast:
            a.setflags(write=False)
        self.base_snapshot_id = snapshot_id(self.broadcast, self.layout, round_id, representation=self._layout_representation)
        self.pending_round, self._recipients, self._received = round_id, set(recipients), set()
        self._round_start = time.perf_counter()
        self._cost_before = self.budget_used
        self._predicted_cost = self.next_round_cost()
        components = payload_components(arrays, method=self.cfg.method, tensor_count=self.layout.size)
        for cid in recipients:
            self.ledger.record(round_id=round_id, client_id=cid, direction="downlink", phase="fit",
                               components=components, serialized_tensor_bytes=serialized_tensor_bytes,
                               layout_id=self.layout_id, d=self.d, m=self.d,
                               base_snapshot_id=self.base_snapshot_id)
        return dict(server_round=round_id, round_id=round_id, base_snapshot_id=self.base_snapshot_id,
                    layout_id=self.layout_id, covariance_representation="precision" if self.cfg.method == "fola" else "not_applicable", d=self.d)

    def receive_packet(self, *, round_id: int, recipient_id, arrays, metrics: dict,
                       num_examples: int, expected_client_id: int | None = None,
                       serialized_tensor_bytes: int | None = None):
        # Charge actual received arrays BEFORE protocol validation. Rejected data
        # has consumed logical communication even though it cannot be aggregated.
        components = payload_components(arrays, method=self.cfg.method, sparse_upload=self.cfg.sparse_enabled,
                                        tensor_count=self.layout.size)
        mid = self.ledger.record(round_id=round_id, client_id=recipient_id, direction="uplink", phase="fit",
                                 components=components, serialized_tensor_bytes=serialized_tensor_bytes,
                                 layout_id=self.layout_id, d=self.d,
                                 m=keep_count(self.d, self.cfg.compression.keep_ratio) if self.cfg.sparse_enabled else self.d)
        try:
            if self.pending_round != round_id or str(recipient_id) not in self._recipients:
                raise ValueError("Reply is not from this round's dispatched recipient set")
            if str(recipient_id) in self._received:
                raise ValueError("Duplicate client reply")
            cid_raw = metrics.get("client_id")
            if isinstance(cid_raw, bool) or not isinstance(cid_raw, (int, float, np.number)) or not np.isfinite(cid_raw) or int(cid_raw) != cid_raw:
                raise ValueError("Invalid client identity")
            cid = int(cid_raw)
            if not 0 <= cid < self.cfg.federation.num_clients:
                raise ValueError("Client ID out of range")
            if expected_client_id is not None and cid != expected_client_id:
                raise ValueError("Client identity differs from its registered partition ID")
            if isinstance(num_examples, bool) or not isinstance(num_examples, (int, np.integer)) or num_examples <= 0:
                raise ValueError("num_examples must be a positive local dataset size")
            if self.cfg.sparse_enabled:
                reconstructed, mask = decode_packet(
                    arrays, metrics, global_arrays=self.broadcast, layout=self.layout, cfg=self.cfg,
                    round_id=round_id, client_id=cid, base_snapshot_id=self.base_snapshot_id,
                )
            else:
                if self.cfg.method == "fola":
                    posterior_vectors(arrays, self.layout)
                else:
                    self.layout.validate(arrays)
                    if any(not np.isfinite(a).all() for a in arrays):
                        raise ValueError("Nonfinite dense update")
                reconstructed, mask = arrays, None
            self._received.add(str(recipient_id))
            return reconstructed, mask
        except Exception as exc:
            self.ledger.reject(mid, str(exc))
            raise

    def finish_round(self, aggregated, *, masks, counts, client_metrics):
        if self.pending_round is None or self._received != self._recipients:
            raise RuntimeError("Cannot finish a partial round")
        output = [np.asarray(a).copy() for a in aggregated]
        if self.cfg.sparse_enabled:
            union = np.logical_or.reduce(masks)
            omitted = ~union
            # The uploaded paper_reference reducer divides by omega + epsilon.
            # Even an all-prior coordinate would otherwise shrink. Restore those
            # entries exactly; touched coordinates retain the original reducer.
            if omitted.any():
                from bayesfl.posterior.packing import unpack_fola, pack_fola
                means, precs = unpack_fola(output, self.layout)
                mu, precision = flatten(means, self.layout), flatten(precs, self.layout)
                old_mu, old_precision = posterior_vectors(self.broadcast, self.layout)
                mu[omitted], precision[omitted] = old_mu[omitted], old_precision[omitted]
                output = pack_fola(unflatten(mu, self.layout), unflatten(precision, self.layout))
            posterior_vectors(output, self.layout)
        actual_cost = self.budget_used - self._cost_before
        if actual_cost != self._predicted_cost:
            raise RuntimeError(f"Observed round payload {actual_cost} differs from reserved {self._predicted_cost}")
        if self.cfg.communication.max_communication_bytes is not None and self.budget_used > self.cfg.communication.max_communication_bytes:
            raise RuntimeError("Array-payload budget exceeded")
        self.round_id = int(self.pending_round)
        self.current = output
        steps = sum(int(m.get("local_steps", 0)) for m in client_metrics)
        seen = sum(int(n) * self.cfg.training.local_epochs for n in counts)
        self.cumulative_local_steps += steps
        self.cumulative_examples_seen += seen
        self.last_round_stats = dict(num_download_recipients=len(self._recipients),
                                     num_upload_replies=len(self._received), num_aggregated_clients=len(counts),
                                     local_training_steps=steps, local_examples_seen=seen,
                                     selection_time_seconds=sum(float(m.get("selection_time_seconds", 0)) for m in client_metrics),
                                     round_wall_time_seconds=time.perf_counter() - self._round_start)
        self.pending_round = None
        self.broadcast = None
        return self.current

    def evaluation_fields(self) -> dict:
        sparse = self.cfg.sparse_enabled
        m = keep_count(self.d, self.cfg.compression.keep_ratio) if sparse else self.d
        defaults = dict(num_download_recipients=0, num_upload_replies=0, num_aggregated_clients=0,
                        local_training_steps=0, local_examples_seen=0, selection_time_seconds=0.0,
                        round_wall_time_seconds=0.0)
        return dict(run_id=self.run_dir.name, method=self.cfg.method_id,
                    selection_rule=self.cfg.compression.selection_rule, seed=self.cfg.runtime.seed,
                    round_id=self.round_id, model_version=self.round_id, d=self.d, m=m,
                    keep_ratio_requested=self.cfg.compression.keep_ratio if sparse else 1.0,
                    keep_ratio_actual=m / self.d, layout_id=self.layout_id,
                    covariance_representation="precision" if self.cfg.method == "fola" else "not_applicable",
                    precision_policy=self.cfg.compression.precision_policy,
                    score_precision_floor=self.cfg.score_precision_floor,
                    **{**defaults, **self.last_round_stats},
                    **self.ledger.round_fields(self.round_id), **self.ledger.cumulative(),
                    cumulative_local_training_steps=self.cumulative_local_steps,
                    cumulative_local_examples_seen=self.cumulative_examples_seen,
                    cumulative_serialized_tensor_bytes=(self.ledger.serialized_tensor_bytes
                                                        if self.ledger.serialized_tensor_observed else None),
                    cumulative_serialized_message_bytes=None,
                    unmeasured_array_messages=self.ledger.unmeasured_array_messages,
                    array_accounting_complete=self.ledger.unmeasured_array_messages == 0,
                    budget_metric=self.cfg.communication.budget_metric,
                    budget_bytes=self.cfg.communication.max_communication_bytes,
                    stop_reason=self.stop_reason(), accounting_kind="logical_payload")

    def save_resume(self, evaluation: dict | None = None) -> None:
        if self.pending_round is not None:
            raise RuntimeError("Only completed, evaluated rounds can be committed")
        if evaluation is not None:
            self.last_evaluation = {k: float(v) for k, v in evaluation.items()
                                    if isinstance(v, (int, float, np.number)) and np.isfinite(v)}
        root = self.run_dir / "resume"
        root.mkdir(parents=True, exist_ok=True)
        stem = f"round_{self.round_id:06d}"
        arrays_path = root / (stem + ".npz")
        tmp = root / (stem + ".npz.tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f, **{f"array_{i:04d}": a for i, a in enumerate(self.current)})
        tmp.replace(arrays_path)
        state = dict(version=1, round_id=self.round_id, array_count=len(self.current),
                     config_fingerprint=config_fingerprint(self.cfg), partition_sha256=self.partition_sha256,
                     layout_id=self.layout_id, state_snapshot_id=snapshot_id(self.current, self.layout, self.round_id, representation=self._layout_representation),
                     ledger=self.ledger.state(), cumulative_local_steps=self.cumulative_local_steps,
                     cumulative_examples_seen=self.cumulative_examples_seen, last_round_stats=self.last_round_stats,
                     last_evaluation=self.last_evaluation,
                     rng_policy="baseline per-client/per-round reseed; separate SHA256 mask/schedule streams",
                     experiment_seed=self.cfg.runtime.seed,
                     optimizer_state_policy="fresh local optimizer each fit, as in supplied baseline",
                     deterministic_client_schedule=self.cfg.communication.deterministic_client_schedule)
        atomic_json(root / (stem + ".json"), state)
        atomic_json(root / "latest.json", dict(state_file=stem + ".json", arrays_file=stem + ".npz"))
        # Retain latest and previous committed generations, not a second full
        # archive of every global model. Scheduled baseline checkpoints remain.
        generations = sorted(root.glob("round_*.npz"))
        for old in generations[:-2]:
            old.unlink()
            old.with_suffix(".json").unlink(missing_ok=True)

    def finalize(self, *, failed: str | None = None) -> None:
        summary = {**self.evaluation_fields(), **self.last_evaluation}
        summary["round_id"] = self.round_id
        summary["stop_reason"] = "failed" if failed else self.stop_reason()
        summary["failure"] = failed
        summary["pending_round_id"] = self.pending_round
        summary["next_round_array_bytes"] = self.next_round_cost()
        atomic_json(self.run_dir / "run_summary.json", summary)
