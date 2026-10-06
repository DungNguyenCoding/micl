#!/usr/bin/env python3
"""Safely migrate an existing CIFAR or legacy-MNIST BayesFL output from repeated-unicast
logical accounting to a modeled single shared multicast downlink.

The optimization trajectory is NOT changed. Model/posterior arrays, accuracy,
loss, partitions, and reliability files are untouched. The script rewrites only
communication/accounting metadata and the ledger/checkpoint ledger hashes that
must agree for a later --resume.

Default mode is DRY RUN. Pass --apply to modify the run directory. A timestamped
backup is created before any in-place change.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

COUNTERS = (
    "train_uplink_array_bytes",
    "train_downlink_array_bytes",
    "train_total_array_bytes",
    "initialization_array_bytes",
    "evaluation_array_bytes",
    "all_array_bytes",
)


def _canonical_line(event: dict[str, Any]) -> str:
    return json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSON in {path}:{lineno}: {exc}") from exc
    if not events:
        raise ValueError(f"Communication ledger is empty: {path}")
    return events


def _effective_rounds(events: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in events:
        if e.get("event_type") == "transfer":
            out[str(e["message_id"])] = int(e["round_id"])
    return out


def _transform_events(events: list[dict[str, Any]], clients_per_round: int) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Collapse K fit-downlink transfers in each training round to one event."""
    by_round: dict[int, list[int]] = defaultdict(list)
    for idx, event in enumerate(events):
        if (
            event.get("event_type") == "transfer"
            and event.get("phase") == "fit"
            and event.get("direction") == "downlink"
            and int(event.get("round_id", 0)) > 0
        ):
            by_round[int(event["round_id"])].append(idx)

    if not by_round:
        raise ValueError("No training downlink events were found")

    # Refuse ledgers with dispositions pointing at a downlink that would be removed.
    disposition_targets = {
        str(e.get("message_id"))
        for e in events
        if e.get("event_type") == "disposition"
    }

    first_idx: dict[int, int] = {}
    skipped: set[int] = set()
    original_counts: dict[int, int] = {}

    for round_id, indices in sorted(by_round.items()):
        original_counts[round_id] = len(indices)
        if len(indices) == 1 and str(events[indices[0]].get("client_id")) == "__multicast__":
            raise ValueError(f"Round {round_id} already appears to use multicast accounting")
        if len(indices) != clients_per_round:
            raise ValueError(
                f"Round {round_id}: expected {clients_per_round} repeated-unicast downlinks, "
                f"found {len(indices)}; refusing an ambiguous migration"
            )
        payloads = {int(events[i]["array_payload_bytes"]) for i in indices}
        if len(payloads) != 1:
            raise ValueError(f"Round {round_id}: downlink payloads are not identical")
        serialized = {events[i].get("serialized_tensor_bytes") for i in indices}
        if len(serialized) != 1:
            raise ValueError(f"Round {round_id}: serialized downlink sizes are not identical")
        for i in indices:
            if str(events[i]["message_id"]) in disposition_targets:
                raise ValueError(
                    f"Round {round_id}: a disposition references a downlink message; "
                    "manual inspection is required"
                )
        first_idx[round_id] = indices[0]
        skipped.update(indices[1:])

    transformed: list[dict[str, Any]] = []
    for idx, original in enumerate(events):
        if idx in skipped:
            continue
        event = dict(original)
        if event.get("event_type") == "transfer":
            event["accounting_kind"] = "modeled_multicast_payload"
            if (
                event.get("phase") == "fit"
                and event.get("direction") == "downlink"
                and int(event.get("round_id", 0)) > 0
            ):
                r = int(event["round_id"])
                if idx != first_idx[r]:
                    continue
                attempt = int(event.get("attempt_id", 0))
                event["client_id"] = "__multicast__"
                event["message_id"] = f"fit:{r}:__multicast__:downlink:{attempt}"
                event["status"] = "modeled_multicast"
                event["recipient_count"] = clients_per_round
                event["downlink_mode"] = "multicast"
        transformed.append(event)

    ids: set[str] = set()
    for event in transformed:
        if event.get("event_type") == "transfer":
            mid = str(event["message_id"])
            if mid in ids:
                raise ValueError(f"Migration produced duplicate message_id: {mid}")
            ids.add(mid)
        elif event.get("event_type") == "disposition":
            if str(event["message_id"]) not in ids:
                raise ValueError("Migration produced a disposition with no preceding transfer")
    return transformed, original_counts


def _event_round(event: dict[str, Any], message_round: dict[str, int]) -> int:
    if event.get("event_type") == "transfer":
        return int(event["round_id"])
    return int(message_round[str(event["message_id"])])


def _ledger_state(events: list[dict[str, Any]], max_round: int | None = None) -> dict[str, Any]:
    message_round = _effective_rounds(events)
    totals = {name: 0 for name in COUNTERS}
    digest = hashlib.sha256()
    event_count = 0
    serialized_total = 0
    serialized_observed = False
    unmeasured = 0
    ids: set[str] = set()

    for event in events:
        eround = _event_round(event, message_round)
        if max_round is not None and eround > max_round:
            continue
        line = _canonical_line(event)
        digest.update(line.encode())
        event_count += 1
        if event.get("event_type") == "disposition":
            if str(event["message_id"]) not in ids:
                raise ValueError("Disposition prefix state is inconsistent")
            continue
        mid = str(event["message_id"])
        if mid in ids:
            raise ValueError(f"Duplicate message_id in migrated ledger: {mid}")
        ids.add(mid)
        size = int(event["array_payload_bytes"])
        if event.get("array_payload_observed", True) is False:
            unmeasured += 1
        totals["all_array_bytes"] += size
        phase = event["phase"]
        if phase == "fit":
            totals["train_total_array_bytes"] += size
            totals[f"train_{event['direction']}_array_bytes"] += size
        elif phase == "initialize":
            totals["initialization_array_bytes"] += size
        elif phase == "evaluate":
            totals["evaluation_array_bytes"] += size
        else:
            raise ValueError(f"Unknown phase {phase!r}")
        serialized = event.get("serialized_tensor_bytes")
        if serialized is not None:
            serialized_total += int(serialized)
            serialized_observed = True

    return {
        "event_count": event_count,
        "event_sha256": digest.hexdigest(),
        "totals": totals,
        "serialized_tensor_bytes": serialized_total,
        "serialized_tensor_observed": serialized_observed,
        "unmeasured_array_messages": unmeasured,
    }


def _round_accounting(events: list[dict[str, Any]]) -> tuple[dict[int, dict[str, int]], dict[int, dict[str, int]]]:
    rounds: dict[int, dict[str, int]] = defaultdict(lambda: {name: 0 for name in COUNTERS})
    serialized_by_round: dict[int, int] = defaultdict(int)
    unmeasured_by_round: dict[int, int] = defaultdict(int)
    message_round = _effective_rounds(events)

    for event in events:
        if event.get("event_type") != "transfer":
            continue
        r = int(event["round_id"])
        size = int(event["array_payload_bytes"])
        rounds[r]["all_array_bytes"] += size
        if event["phase"] == "fit":
            rounds[r]["train_total_array_bytes"] += size
            rounds[r][f"train_{event['direction']}_array_bytes"] += size
        elif event["phase"] == "initialize":
            rounds[r]["initialization_array_bytes"] += size
        elif event["phase"] == "evaluate":
            rounds[r]["evaluation_array_bytes"] += size
        serialized = event.get("serialized_tensor_bytes")
        if serialized is not None:
            serialized_by_round[r] += int(serialized)
        if event.get("array_payload_observed", True) is False:
            unmeasured_by_round[r] += 1

    cumulative: dict[int, dict[str, int]] = {}
    running = {name: 0 for name in COUNTERS}
    running_serialized = 0
    running_unmeasured = 0
    for r in range(0, max(rounds) + 1):
        rr = rounds.get(r, {name: 0 for name in COUNTERS})
        for name in COUNTERS:
            running[name] += int(rr[name])
        running_serialized += serialized_by_round.get(r, 0)
        running_unmeasured += unmeasured_by_round.get(r, 0)
        cumulative[r] = {
            **running,
            "serialized_tensor_bytes": running_serialized,
            "unmeasured_array_messages": running_unmeasured,
        }
    return dict(rounds), cumulative


def _rewrite_metrics_csv(path: Path, round_data: dict[int, dict[str, int]], cumulative: dict[int, dict[str, int]], apply: bool) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or "round" not in fields:
        return 0

    communication_fields = {
        *("round_" + name for name in COUNTERS),
        *("cumulative_" + name for name in COUNTERS),
    }
    if not any(field in communication_fields for field in fields):
        return 0

    for extra in ("downlink_mode", "accounting_kind", "num_downlink_transmissions"):
        if extra not in fields:
            fields.append(extra)

    changed = 0
    for row in rows:
        r = int(float(row["round"]))
        rr = round_data.get(r, {name: 0 for name in COUNTERS})
        cc = cumulative.get(r)
        if cc is None:
            raise ValueError(f"{path}: no migrated ledger accounting available for round {r}")
        for name in COUNTERS:
            row["round_" + name] = str(int(rr[name]))
            row["cumulative_" + name] = str(int(cc[name]))
        if "cumulative_serialized_tensor_bytes" in fields:
            row["cumulative_serialized_tensor_bytes"] = str(int(cc["serialized_tensor_bytes"]))
        if "unmeasured_array_messages" in fields:
            row["unmeasured_array_messages"] = str(int(cc["unmeasured_array_messages"]))
        if "array_accounting_complete" in fields:
            row["array_accounting_complete"] = str(cc["unmeasured_array_messages"] == 0)
        row["downlink_mode"] = "multicast"
        row["accounting_kind"] = "modeled_multicast_payload"
        row["num_downlink_transmissions"] = "1" if r > 0 and rr["train_downlink_array_bytes"] > 0 else "0"
        changed += 1

    if apply:
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    return changed


def _next_round_cost_from_config(cfg: dict[str, Any], d: int) -> int:
    method = str(cfg["method"]).lower()
    k = int(cfg["federation"]["clients_per_round"])
    compression = cfg.get("compression") or {}
    rule = str(compression.get("selection_rule", "dense"))
    sparse = method == "fola" and rule != "dense"
    dense_message = (8 if method == "fola" else 4) * int(d)
    if sparse:
        ratio = float(compression["keep_ratio"])
        m = min(int(d), math.ceil(ratio * int(d)))
        per_upload = 8 * m + (int(d) + 7) // 8
    else:
        per_upload = dense_message
    return int(dense_message + k * per_upload)


def _backup_files(run_dir: Path, files: list[Path]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = run_dir / f"multicast_migration_backup_{stamp}"
    if backup.exists():
        raise FileExistsError(backup)
    for src in files:
        if not src.exists():
            continue
        rel = src.relative_to(run_dir)
        dst = backup / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return backup


def migrate(run_dir: Path, *, apply: bool, require_resumable: bool) -> None:
    run_dir = run_dir.resolve()
    config_path = run_dir / "resolved_config.yaml"
    ledger_path = run_dir / "communication" / "events.jsonl"
    protocol_path = run_dir / "communication" / "protocol.json"
    summary_path = run_dir / "run_summary.json"

    if not config_path.exists() or not ledger_path.exists():
        raise FileNotFoundError("run_dir must contain resolved_config.yaml and communication/events.jsonl")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset = str((cfg.get("data") or {}).get("dataset", "")).lower()
    model_name = str((cfg.get("model") or {}).get("name", ""))

    # CIFAR trajectories remain reusable because only communication accounting
    # changed. MNIST is allowed ONLY for the legacy 784-500-300-10 model.
    # Outputs from the newer 256x5 MLP must never be mixed with legacy-MNIST
    # trajectories.
    if dataset == "mnist":
        if model_name != "mlp_784_500_300_10":
            raise ValueError(
                "Refusing MNIST migration for model "
                f"{model_name!r}; only legacy mlp_784_500_300_10 "
                "trajectories are reusable."
            )
    elif dataset != "cifar10":
        raise ValueError(
            f"Refusing unsupported dataset={dataset!r}"
        )
    comm_cfg = cfg.setdefault("communication", {})
    if str(comm_cfg.get("downlink_mode", "unicast")).lower() == "multicast":
        raise ValueError("resolved_config.yaml already says downlink_mode=multicast; refusing a second migration")

    k = int(cfg["federation"]["clients_per_round"])
    if require_resumable and not bool(comm_cfg.get("deterministic_client_schedule", False)):
        raise ValueError("This run is not resumable: deterministic_client_schedule was not enabled from the start")

    original = _read_events(ledger_path)
    transformed, original_counts = _transform_events(original, k)
    full_state = _ledger_state(transformed)
    round_data, cumulative = _round_accounting(transformed)
    training_rounds = sorted(r for r in round_data if r > 0 and round_data[r]["train_total_array_bytes"] > 0)
    final_round = max(training_rounds)

    old_total = _ledger_state(original)["totals"]["all_array_bytes"]
    new_total = full_state["totals"]["all_array_bytes"]
    print("=" * 78)
    print(f"RUN             : {run_dir}")
    print(f"DATASET         : {dataset}")
    print(f"METHOD          : {cfg.get('method')}")
    print(f"CLIENTS/ROUND   : {k}")
    print(f"TRAINING ROUNDS : {training_rounds[0]}..{final_round}")
    print(f"OLD ARRAY BYTES : {old_total} ({old_total/1e9:.6f} GB)")
    print(f"NEW ARRAY BYTES : {new_total} ({new_total/1e9:.6f} GB)")
    print(f"SAVING          : {100.0*(1.0-new_total/old_total):.3f}%")
    print(f"MODE            : {'APPLY' if apply else 'DRY RUN'}")

    files_to_backup: list[Path] = [config_path, ledger_path, protocol_path, summary_path]
    for csv_name in ("global_metrics.csv", "round_train_metrics.csv"):
        files_to_backup.append(run_dir / "metrics" / csv_name)
    resume_root = run_dir / "resume"
    resume_jsons = sorted(resume_root.glob("round_*.json")) if resume_root.exists() else []
    files_to_backup.extend(resume_jsons)

    if apply:
        backup = _backup_files(run_dir, files_to_backup)
        print(f"BACKUP          : {backup}")

    # Ledger
    ledger_text = "".join(_canonical_line(e) for e in transformed)
    if apply:
        _atomic_text(ledger_path, ledger_text)

    # Config: accounting mode only. Keep the historical byte limit unchanged;
    # continuation commands may override it safely because max bytes are excluded
    # from the optimization fingerprint.
    comm_cfg["downlink_mode"] = "multicast"
    if apply:
        tmp = config_path.with_name(config_path.name + ".tmp")
        tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        tmp.replace(config_path)

    # CSV metrics
    for csv_name in ("global_metrics.csv", "round_train_metrics.csv"):
        path = run_dir / "metrics" / csv_name
        changed = _rewrite_metrics_csv(path, round_data, cumulative, apply)
        if changed:
            print(f"METRICS         : {csv_name}: {changed} rows recomputed")

    # Protocol
    protocol: dict[str, Any] = _load_json(protocol_path) if protocol_path.exists() else {}
    protocol["downlink_mode"] = "multicast"
    protocol["downlink_transmissions_per_round"] = 1
    protocol["accounting_kind"] = "modeled_multicast_payload"
    protocol["control_metadata_and_transport_headers_included"] = False
    protocol["multicast_note"] = (
        "Modeled single shared downlink payload; Flower transport itself is not claimed to implement wireless multicast."
    )
    d = int(protocol.get("d") or (_load_json(summary_path).get("d") if summary_path.exists() else 0) or 0)
    if d:
        protocol["next_round_prediction"] = _next_round_cost_from_config(cfg, d)
    if apply:
        _atomic_json(protocol_path, protocol)

    # Summary
    if summary_path.exists():
        summary = _load_json(summary_path)
        d = int(summary.get("d", d or 0))
        rr = round_data.get(final_round, {name: 0 for name in COUNTERS})
        cc = cumulative[final_round]
        for name in COUNTERS:
            summary["round_" + name] = int(rr[name])
            summary["cumulative_" + name] = int(cc[name])
        if summary.get("cumulative_serialized_tensor_bytes") is not None:
            summary["cumulative_serialized_tensor_bytes"] = int(cc["serialized_tensor_bytes"])
        summary["unmeasured_array_messages"] = int(cc["unmeasured_array_messages"])
        summary["array_accounting_complete"] = cc["unmeasured_array_messages"] == 0
        summary["downlink_mode"] = "multicast"
        summary["accounting_kind"] = "modeled_multicast_payload"
        summary["num_downlink_transmissions"] = 1
        if d:
            summary["next_round_array_bytes"] = _next_round_cost_from_config(cfg, d)
        summary["accounting_migration"] = (
            "Repeated-unicast downlink ledger collapsed to one modeled multicast downlink per training round; "
            "optimization trajectory and accuracy were not changed."
        )
        if apply:
            _atomic_json(summary_path, summary)

    # Resume states: only ledger state/hash and accounting-only last-round stat change.
    for state_path in resume_jsons:
        state = _load_json(state_path)
        r = int(state["round_id"])
        state["ledger"] = _ledger_state(transformed, max_round=r)
        if r > 0:
            stats = dict(state.get("last_round_stats") or {})
            stats["num_downlink_transmissions"] = 1
            state["last_round_stats"] = stats
        state["downlink_mode"] = "multicast"
        state["accounting_kind"] = "modeled_multicast_payload"
        state["accounting_migration"] = "modeled_multicast_payload"
        if apply:
            _atomic_json(state_path, state)
    if resume_jsons:
        print(f"RESUME STATES   : {len(resume_jsons)} checkpoint JSON file(s) recomputed")

    if apply:
        # Validate the just-written ledger against its expected final state.
        reloaded = _read_events(ledger_path)
        if _ledger_state(reloaded) != full_state:
            raise RuntimeError("Post-write ledger validation failed")
        print("RESULT           : PASS - migration written and ledger hash/totals revalidated")
    else:
        print("RESULT           : PASS - dry-run validation only; no files changed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Apply in place after creating a timestamped backup")
    parser.add_argument(
        "--require-resumable",
        action="store_true",
        help="Fail if deterministic_client_schedule was not enabled from the original run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    migrate(args.run_dir, apply=args.apply, require_resumable=args.require_resumable)
