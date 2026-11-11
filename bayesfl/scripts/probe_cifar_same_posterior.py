#!/usr/bin/env python3
"""CIFAR-10 same-prepared-posterior allocator probe (no global training round).

Uses one existing server posterior checkpoint and one local FOLA PREPARE per
client. The two policies only query the same persistent client-owned oracles;
no second local train, no upload, no aggregation, no evaluation, and no edits
to original output runs/configuration/partition/checkpoint files.

Run from the bayesfl repository root with PYTHONPATH=src.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from bayesfl.config import load_config
from bayesfl.data.datasets import partition_stem
from bayesfl.data.partition import _hash_partitions, load_partition
from bayesfl.models.factory import initialize_model
from bayesfl.posterior.packing import ParameterLayout, pack_fola
from bayesfl.posterior.sparse import dimension, posterior_vectors, snapshot_id, layout_manifest
from bayesfl.runtime_utils import seed_everything
from bayesfl.wireless.allocation import allocate, InfeasibleAllocation
from bayesfl.wireless.channel import Channel, round_timing
from bayesfl.wireless.client import train_local
from bayesfl.wireless.oracle import Response
from bayesfl.wireless.payload import payload_bytes, payload_bits
from bayesfl.wireless.protocol import binding, canonical
from bayesfl.wireless.transport import LocalTransport


SOURCE_TAG = "cifar_eqb_pilot20_v1"
EXPECTED_PARTITION_SHA = "914fe035f7b504b0059ca85b5a6a194bd41027fb2cfe2765ee777142d68280f2"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_source(root: Path, source_run: str | None) -> Path:
    if source_run:
        run = Path(source_run).expanduser().resolve()
        if not run.is_dir():
            raise ValueError(f"Source run does not exist: {run}")
        return run
    study = root / "scripts/configs/wireless" / SOURCE_TAG / "study_manifest.json"
    if not study.is_file():
        raise ValueError(f"Expected original study manifest: {study}")
    jobs = json.loads(study.read_text())["jobs"]
    if len(jobs) != 1:
        raise ValueError(f"Expected exactly one job for source study, got {len(jobs)}")
    name = jobs[0]["run_name"]
    matches = sorted(p.resolve() for p in (root / "outputs").glob(name + "_*") if p.is_dir())
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source output for {name}, found {len(matches)}; pass --source-run explicitly")
    return matches[0]


def load_checkpoint_arrays(checkpoint: Path, layout: ParameterLayout) -> list[np.ndarray]:
    expected_keys = [f"mean__{s}" for s in layout.names] + [f"precision__{s}" for s in layout.names]
    with np.load(checkpoint, allow_pickle=False) as z:
        if set(z.files) != set(expected_keys) or len(z.files) != len(expected_keys):
            raise ValueError("Server checkpoint keys differ from the current FOLA parameter layout")
        arrays = [np.asarray(z[k]) for k in expected_keys]
        for key, arr, shape in zip(expected_keys, arrays, list(layout.shapes) * 2):
            if arr.shape != shape or arr.dtype != np.float32 or not np.isfinite(arr).all():
                raise ValueError(f"Invalid checkpoint array {key}: {arr.shape}, {arr.dtype}")
    output = pack_fola(arrays[:layout.size], arrays[layout.size:])
    posterior_vectors(output, layout)  # raw precision must be finite/nonnegative
    return output


def load_existing_inputs(root: Path, source_run: str | None, checkpoint_round: int):
    source = resolve_source(root, source_run)
    config_path = source / "resolved_config.yaml"
    if not config_path.is_file():
        raise ValueError(f"Missing source resolved config: {config_path}")
    cfg = load_config(config_path)
    cfg.validate()
    if not cfg.wireless_enabled or cfg.data.dataset != "cifar10" or cfg.model.name != "paper_basiccnn":
        raise ValueError("Probe requires the real wireless CIFAR-10 BasicCNN FOLA configuration")
    if cfg.method != "fola" or cfg.compression.selection_rule != "kl_global_local":
        raise ValueError("Probe requires FOLA and kl_global_local; no silent policy changes")
    if cfg.wireless.policy not in {"equal_bandwidth", "joint"} or cfg.federation.num_clients != 20:
        raise ValueError("Source must be an established 20-client equal/joint CIFAR run")
    if cfg.wireless.channel.mode != "fixed":
        raise ValueError("The stage-1 probe expects the pinned fixed-gain scenario")
    if checkpoint_round < 0 or checkpoint_round >= cfg.training.rounds:
        raise ValueError("Checkpoint round must be within the original experiment (before its last round)")
    summary = json.loads((source / "run_summary.json").read_text())
    if (summary.get("failure") is not None or summary.get("pending_round") is not None or
            int(summary["completed_rounds"]) < checkpoint_round + 1):
        raise ValueError("Source is not a completed, healthy run covering the requested checkpoint")
    part_root = (root / cfg.output.outputs_dir / "partitions").resolve()
    stem = partition_stem(cfg)
    partition = part_root / (stem + ".npz")
    partition_json = part_root / (stem + ".json")
    if not partition.is_file() or not partition_json.is_file():
        raise ValueError("Pinned partition missing; refusing to generate/download/replace any partition")
    part_meta = json.loads(partition_json.read_text())
    original_meta = json.loads((source / "partition_metadata.json").read_text())
    if part_meta["sha256"] != original_meta["sha256"] or part_meta["sha256"] != EXPECTED_PARTITION_SHA:
        raise ValueError("Partition manifest hash differs from the original CIFAR-10 pilot")
    parts = load_partition(partition)
    if _hash_partitions(parts) != EXPECTED_PARTITION_SHA:
        raise ValueError("Actual partition NPZ contents differ from the pinned content digest")
    samples = [int(len(x)) for x in parts]
    if len(samples) != 20 or any(n < 1 for n in samples) or sum(samples) != 49988:
        raise ValueError("Unexpected partition client sizes; no silent population change")
    checkpoint = source / "checkpoints" / f"global_round_{checkpoint_round:04d}.npz"
    if not checkpoint.is_file():
        raise ValueError(f"Missing original server checkpoint: {checkpoint}")
    model = initialize_model(cfg)
    layout = ParameterLayout.from_model(model)
    if dimension(layout) != 878538 or list(model.named_buffers()):
        raise ValueError("CIFAR model/layout differs from the locked BasicCNN")
    del model
    global_arrays = load_checkpoint_arrays(checkpoint, layout)
    ch = Channel(cfg.wireless, cfg.federation.num_clients).round(checkpoint_round + 1)
    timing = round_timing(cfg, samples=samples,
                          downlink_bytes=sum(a.nbytes for a in global_arrays), channel=ch)
    return dict(root=root, source=source, cfg=cfg, config_path=config_path,
                partition=partition, meta=part_meta, samples=samples,
                checkpoint=checkpoint, layout=layout, global_arrays=global_arrays,
                channel=ch, timing=timing, round_id=checkpoint_round + 1)


class QueryRecorder:
    """Server receives only scalar Tail/Price replies; clients retain scores."""
    def __init__(self, transport, bindings, samples, cfg, round_id):
        self.transport = transport
        self.bindings = bindings
        self.samples = samples
        self.cfg = cfg
        self.round_id = round_id
        self.events: list[dict] = []
        self.count_requests = 0
        self.price_requests = 0
        self.control_bytes = 0
        self.control_seconds = 0.0

    def exchange(self, phase, additions=None, parameters=None):
        k = len(self.samples)
        additions = additions or [{} for _ in range(k)]
        if len(additions) != k:
            raise ValueError("Incorrect number of client query instructions")
        instructions = [dict(base, phase=phase, **extra)
                        for base, extra in zip(self.bindings, additions)]
        request_bytes = len(canonical(instructions))
        reply_map = self.transport.exchange(parameters or [], instructions, self.round_id)
        if set(reply_map) != set(range(k)):
            raise ValueError("Missing/duplicate scalar reply")
        replies = [reply_map[i] for i in range(k)]
        reply_bytes = 0
        for cid, reply in enumerate(replies):
            if reply.arrays:
                raise ValueError("A client leaked an array instead of a scalar control reply")
            if reply.num_examples != self.samples[cid]:
                raise ValueError("Client sample count changed during frozen-state query")
            for key, value in self.bindings[cid].items():
                if reply.metrics.get(key) != value:
                    raise ValueError("Client response binding differs from pinned source")
            reply_bytes += len(canonical(dict(client_id=cid, num_examples=reply.num_examples,
                                              metrics=reply.metrics)))
        seconds = (8*(request_bytes+reply_bytes) / self.cfg.wireless.control_rate_bps +
                   self.cfg.wireless.control_exchange_latency_s)
        self.control_bytes += request_bytes+reply_bytes
        self.control_seconds += seconds
        self.events.append(dict(phase=phase, clients=k, request_bytes=request_bytes,
                                reply_bytes=reply_bytes, control_bytes=request_bytes+reply_bytes,
                                modeled_control_seconds=seconds))
        if phase == "tail":
            self.count_requests += 1
        elif phase == "price":
            self.price_requests += 1
        return replies

    def counts(self, counts):
        replies = self.exchange("tail", [{"count": int(m)} for m in counts])
        return [Response(int(r.metrics["count"]), float(r.metrics["discarded_kl"])) for r in replies]

    def price(self, price, weights, window_s, efficiencies):
        replies = self.exchange("price", [dict(price=float(price), weight=float(w),
                                                 window_s=float(window_s), efficiency=float(c))
                                          for w,c in zip(weights,efficiencies)])
        return [Response(int(r.metrics["count"]), float(r.metrics["discarded_kl"])) for r in replies]


def compare_frozen(transport, bindings, samples, cfg, layout, channel, timing, round_id):
    """Invoke production allocators twice; only scalar queries, no PREPARE or training."""
    weights = [n/sum(samples) for n in samples]
    args = dict(d=dimension(layout), weights=weights,
                window_s=timing["upload_window_seconds"],
                efficiencies=channel["efficiencies"],
                bandwidth_hz=cfg.wireless.uplink_bandwidth_hz,
                settings=cfg.wireless.solver, posterior=True)
    recorder_equal = QueryRecorder(transport, bindings, samples, cfg, round_id)
    equal = allocate(recorder_equal, policy="equal_bandwidth", **args)
    recorder_joint = QueryRecorder(transport, bindings, samples, cfg, round_id)
    joint = allocate(recorder_joint, policy="joint", **args)
    tol = 1e-8 * max(1.0, abs(equal.objective))
    if joint.objective > equal.objective + tol:
        raise AssertionError("Joint allocation failed to preserve the equal-bandwidth feasible incumbent")
    for label, alloc in (("equal", equal), ("joint", joint)):
        if any(not 1 <= m <= args["d"] for m in alloc.counts):
            raise AssertionError(f"Invalid integer counts: {label}")
        if math.fsum(alloc.bandwidth_hz) > args["bandwidth_hz"]*(1+1e-12):
            raise AssertionError(f"Infeasible bandwidth budget: {label}")
        for cid, (m, b, c) in enumerate(zip(alloc.counts, alloc.bandwidth_hz, args["efficiencies"])):
            duration = payload_bits(m, args["d"]) / b / c
            if duration > args["window_s"] * (1 + 1e-11):
                raise AssertionError(f"Infeasible upload duration in {label} client={cid}")
        if not math.isclose(alloc.objective,math.fsum(w*t for w,t in zip(weights,alloc.discarded_kl)), rel_tol=1e-10,abs_tol=1e-10):
            raise AssertionError(f"Invalid weighted KL sum: {label}")
    return equal, joint, recorder_equal, recorder_joint


def info_for_allocation(alloc, recorder, source, cfg, channel, timing):
    d = dimension(source["layout"])
    packet_bytes = [payload_bytes(m,d) for m in alloc.counts]
    uplink_j = math.fsum(p * (8*size) / c for p,size,c in zip(
        cfg.wireless.channel.uplink_psd_w_hz,packet_bytes,channel["efficiencies"]))
    dl_bytes = sum(a.nbytes for a in source["global_arrays"])
    downlink_j = cfg.wireless.channel.downlink_psd_w_hz * cfg.wireless.downlink_bandwidth_hz * timing["downlink_seconds"]
    return dict(policy=alloc.status if alloc.status == 'equal_bandwidth' else 'joint',
                objective_weighted_discarded_kl=alloc.objective,
                mean_keep_ratio=float(np.mean([m/d for m in alloc.counts])),
                data_weighted_keep_ratio=math.fsum(n/sum(source["samples"])*m/d
                                               for n,m in zip(source["samples"],alloc.counts)),
                total_bandwidth_hz=math.fsum(alloc.bandwidth_hz),
                downlink_array_bytes=dl_bytes,
                uplink_array_bytes=sum(packet_bytes),
                model_array_bytes=dl_bytes+sum(packet_bytes),
                allocation_control_bytes=recorder.control_bytes,
                allocation_control_modeled_seconds=recorder.control_seconds,
                allocation_tail_requests=recorder.count_requests,
                allocation_price_requests=recorder.price_requests,
                modeled_data_plane_tx_energy_j=downlink_j+uplink_j,
                lower_bound_diagnostic=alloc.lower_bound_diagnostic,
                numerical_gap_diagnostic=alloc.gap_diagnostic,
                solver_status=alloc.status,
                certified_optimal=bool(alloc.certified_optimal))


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows for CSV")
    with path.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preflight_print(source):
    cfg = source['cfg']
    print('PASS: READ-ONLY PREFLIGHT')
    print('Source run:',source['source'])
    print('Source round:',source['round_id']-1,'-> probe PREPARE round',source['round_id'])
    print('Checkpoint:',source['checkpoint'])
    print('Partition:',source['partition'])
    print('Checkpoint SHA256:',file_sha(source['checkpoint']))
    print('Partition content SHA256:',source['meta']['sha256'])
    print('Dataset/model:',cfg.data.dataset,cfg.model.name,'d =',dimension(source['layout']))
    print('Client count / samples:',len(source['samples']),sum(source['samples']))
    print('Upload window [s]:',source['timing']['upload_window_seconds'])
    print('Total UL bandwidth [Hz]:',cfg.wireless.uplink_bandwidth_hz)
    print('Original source files and historical result files have not been modified.')


def execute(source, output_dir: Path):
    cfg = source['cfg']
    if output_dir.exists() or Path(str(output_dir)+'.zip').exists():
        raise ValueError('Output directory or result ZIP already exists; refusing overwrite')
    layout = source['layout']
    round_id = source['round_id']
    run_token = 'frozen-counterfactual-r'+str(round_id)+'-'+file_sha(source['checkpoint'])[:12]
    bindings = [binding(source['global_arrays'],layout,round_id=round_id,client_id=cid,
                        run_token=run_token,posterior=True) for cid in range(len(source['samples']))]
    calls = [0]*len(source['samples'])
    def counted_trainer(config, cid, partition_path, mean_size, parameters, instruction):
        calls[cid] += 1
        if calls[cid] != 1:
            raise RuntimeError(f'Duplicate client PREPARE/training detected for client {cid}')
        return train_local(config,cid,partition_path,mean_size,parameters,instruction)
    transport = LocalTransport(cfg,source['partition'], source['meta']['mean_size'],trainer=counted_trainer)
    seed_everything(cfg.runtime.seed)
    print(f'START: One local FOLA preparation for each of {len(source["samples"])} clients; no global update',flush=True)
    started = time.perf_counter()
    prepare_recorder = QueryRecorder(transport,bindings,source['samples'],cfg,round_id)
    prep = prepare_recorder.exchange('prepare',parameters=source['global_arrays'])
    if calls != [1]*len(calls):
        raise AssertionError('Each client must train exactly once during PREPARE')
    print('PREPARE complete | clients=%d | wall_s=%.3f'%(len(calls), time.perf_counter()-started), flush=True)
    equal,joint,re, rj = compare_frozen(transport,bindings,source['samples'],cfg,layout,
                                       source['channel'],source['timing'],round_id)
    if calls != [1]*len(calls):
        raise AssertionError('Allocator queries unexpectedly retrained local models')
    d = dimension(layout)
    rows=[]
    for cid in range(len(source['samples'])):
        count_e,count_j=equal.counts[cid],joint.counts[cid]
        rows.append(dict(client_id=cid, num_examples=source['samples'][cid],
                         weight=source['samples'][cid]/sum(source['samples']),
                         efficiency=source['channel']['efficiencies'][cid],
                         equal_bandwidth_hz=equal.bandwidth_hz[cid],
                         joint_bandwidth_hz=joint.bandwidth_hz[cid],
                         equal_count=count_e,joint_count=count_j,
                         equal_keep_ratio=count_e/d,joint_keep_ratio=count_j/d,
                         equal_discarded_kl=equal.discarded_kl[cid],
                         joint_discarded_kl=joint.discarded_kl[cid],
                         equal_weighted_discarded_kl=source['samples'][cid]/sum(source['samples'])*equal.discarded_kl[cid],
                         joint_weighted_discarded_kl=source['samples'][cid]/sum(source['samples'])*joint.discarded_kl[cid],
                         equal_upload_bytes=payload_bytes(count_e,d),
                         joint_upload_bytes=payload_bytes(count_j,d),
                         equal_upload_seconds=payload_bits(count_e,d)/equal.bandwidth_hz[cid]/source['channel']['efficiencies'][cid],
                         joint_upload_seconds=payload_bits(count_j,d)/joint.bandwidth_hz[cid]/source['channel']['efficiencies'][cid]))
    ie=info_for_allocation(equal,re,source,cfg,source['channel'],source['timing'])
    ij=info_for_allocation(joint,rj,source,cfg,source['channel'],source['timing'])
    summary=dict(experiment='CIFAR10_same_prepared_posterior_allocation_replay_v1',
                 preparation_passes_per_client=1, global_training_rounds_executed=0,
                 selected_checkpoint_round=round_id-1,
                 probe_local_preparation_round=round_id,
                 source_run=str(source['source']),
                 source_resolved_config_sha256=file_sha(source['config_path']),
                 source_checkpoint_sha256=file_sha(source['checkpoint']),
                 source_partition_file_sha256=file_sha(source['partition']),
                 partition_content_sha256=source['meta']['sha256'],
                 client_states_frozen=True,
                 compared_on_same_prepared_scores=True,
                 model_arrays_uploaded=False,
                 notes=['One new local preparation on the original R10 server checkpoint is necessary because the old R20 archives retained no full KL score vectors.',
                        'Both allocators issue only scalar tail/price queries against the same client-owned memory states.',
                        'No policy-specific retraining, upload, server aggregation, or test evaluation occurs.',
                        'Local sequential emulation, not Flower/Ray airtime or RF measurement.',
                        'Canonical JSON control bytes cover allocation-phase queries only. PREPARE is separate; upload and release not simulated.',
                        'Float64 numerical gap is diagnostic; not global integer-optimality certificate.',
                        'No accuracy claim is made; compares one fixed batch of posteriors.'],
                 d=d, num_clients=len(source['samples']), sum_examples=sum(source['samples']),
                 total_uplink_bandwidth_hz=cfg.wireless.uplink_bandwidth_hz,
                 modeled_round_target_s=cfg.wireless.round_target_s,
                 modeled_downlink_s=source['timing']['downlink_seconds'],
                 modeled_upload_window_s=source['timing']['upload_window_seconds'],
                 shared_preparation_control_bytes=prepare_recorder.control_bytes,
                 shared_preparation_control_modeled_seconds=prepare_recorder.control_seconds,
                 preparation_calls_each_client=calls,
                 local_preparation_wall_seconds=[r.metrics.get('preparation_wall_seconds') for r in prep],
                 equal=ie, joint=ij,
                 joint_kl_improvement=ie['objective_weighted_discarded_kl']-ij['objective_weighted_discarded_kl'],
                 joint_kl_improvement_percent=(100*(1-ij['objective_weighted_discarded_kl']/ie['objective_weighted_discarded_kl'])
                                               if ie['objective_weighted_discarded_kl']>0 else 0.0))
    for label, rec in [('equal',re),('joint',rj)]:
        if rec.control_seconds + prepare_recorder.control_seconds > cfg.wireless.control_reserve_s+1e-10:
            raise InfeasibleAllocation(f'PREPARE+{label} allocation queries exceed modeled control reserve')
    output_dir.mkdir(parents=True,exist_ok=False)
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    write_csv(output_dir/'per_client.csv',rows)
    all_events=[dict(policy='shared_prepare',index=1,**e) for e in prepare_recorder.events]
    for label,rec in [('equal',re),('joint',rj)]:
        all_events.extend(dict(policy=label,index=j,**e) for j,e in enumerate(rec.events,1))
    write_csv(output_dir/'allocation_query_events.csv',all_events)
    (output_dir/'README.txt').write_text(
        'Same-posterior allocation-only counterfactual. One local CIFAR preparation per client; no extra training between policies.\n'
        'Input source run and checkpoint provenance are in summary.json.\n'
        'Per-client CSV holds only counts and scalar KL tails, not raw scores, data or model arrays.\n'
        'Control bytes represent canonical JSON query/reply messages only, not RF measurements.\n')
    print('\nPASS: EXACT SAME-PREPARED-POSTERIOR ALLOCATION PROBE')
    print('Equal KL objective: %.10g'%equal.objective)
    print('Joint KL objective: %.10g'%joint.objective)
    print('Joint KL reduction: %.4f%%'%summary['joint_kl_improvement_percent'])
    print('Equal model-array bytes: %d'%ie['model_array_bytes'])
    print('Joint model-array bytes: %d'%ij['model_array_bytes'])
    print('Equal allocation control bytes: %d'%re.control_bytes)
    print('Joint allocation control bytes: %d'%rj.control_bytes)
    print('Joint price queries:',rj.price_requests)
    print('Training calls by client:',calls)
    print('Result:',output_dir)
    zip_path = Path(str(output_dir) + '.zip')
    with ZipFile(zip_path, 'x', compression=ZIP_DEFLATED) as archive:
        for item in sorted(output_dir.iterdir()):
            if item.is_file():
                archive.write(item, arcname=output_dir.name+'/'+item.name)
    print('Upload this ZIP:',zip_path)
    print('No historical experiment files, checkpoints, partitions or configs changed.')
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    phase = p.add_mutually_exclusive_group(required=True)
    phase.add_argument('--preflight',action='store_true',help='Read-only compatibility, checkpoint and partition check')
    phase.add_argument('--execute',action='store_true',help='One new local FOLA preparation, both allocations, no global training')
    p.add_argument('--source-run',default=None, help='Existing successfully completed CIFAR equal-bandwidth output directory')
    p.add_argument('--checkpoint-round',type=int,default=10)
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--require-cuda',action='store_true',help='Reject CPU fallback before local preparation')
    p.add_argument('--root',type=Path,default=Path.cwd())
    args=p.parse_args(argv)
    root=args.root.expanduser().resolve()
    if not (root/'src/bayesfl/wireless/allocation.py').is_file():
        p.error('Run from the installed BayesFL repository root or pass --root')
    if args.execute and args.output_dir is None:
        p.error('--execute requires a new --output-dir outside existing run folders')
    if args.output_dir is not None:
        out=args.output_dir.expanduser().resolve()
        if out.exists() or Path(str(out)+'.zip').exists():
            p.error(f'Output path or result ZIP already exists: {out}')
        if out == root or root in out.parents:
            p.error('Place diagnostic output in a sibling folder, not under the locked BayesFL repository')
    source=load_existing_inputs(root,args.source_run,args.checkpoint_round)
    preflight_print(source)
    if args.preflight:
        return 0
    if args.require_cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('--require-cuda set, but no visible CUDA GPU is available')
    execute(source,args.output_dir.expanduser().resolve())
    return 0


if __name__=='__main__':
    raise SystemExit(main())
