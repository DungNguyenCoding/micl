#!/usr/bin/env python3
"""One-preparation CIFAR-10 R11 paired posterior-aggregation/evaluation gate.

Run from bayesfl root, with PYTHONPATH including src and scripts.
No historical run, checkpoint, partition, configuration, or queue is modified.
The two *independent counterfactual branches* restore a frozen local-state
snapshot; neither branch performs another local training call. Packets and
aggregates use the installed wireless protocol, not a hand-built approximation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time
import zipfile

import numpy as np

import probe_cifar_same_posterior as prior_probe
from bayesfl.data.datasets import load_test_loader
from bayesfl.evaluation import CentralEvaluator
from bayesfl.metrics import predictive_metric_bundle
from bayesfl.posterior.aggregation import aggregate_fola
from bayesfl.posterior.packing import pack_fola
from bayesfl.posterior.sparse import dimension, posterior_vectors, unflatten
from bayesfl.runtime_utils import seed_everything, release_cuda_memory
from bayesfl.wireless.protocol import aggregate, make_grant, decode, canonical
from bayesfl.wireless.payload import payload_bytes, payload_bits
from bayesfl.wireless.transport import LocalTransport


PROBE_ID = 'cifar10_r11_paired_aggregate_evaluate_v1'
SOURCE_ROUND = 10
TARGET_ROUND = 11


def sha256(path: Path) -> str:
    return prior_probe.file_sha(path)


def evaluate_on_test(cfg, label: str, arrays: list[np.ndarray], round_id: int, output: Path) -> dict:
    """Use production CentralEvaluator predictions and its identical metric function.

    Deliberately skip its checkpoint/CSV side effects. The prediction branch
    matches paper_mean_only_eval on the original CIFAR experimental config.
    """
    if cfg.method != 'fola' or not cfg.fola.paper_mean_only_eval:
        raise ValueError('Expected locked mean-only FOLA evaluation; no silent evaluation change')
    import torch
    seed_everything(cfg.runtime.seed + 99_991 * round_id)
    ev = CentralEvaluator(cfg, load_test_loader(cfg), output / ('eval_' + label),
                          logger=logging.getLogger('cifar-paired'), state=None)
    with torch.no_grad():
        primary, labels, sample_probs, point = ev._collect_probabilities(arrays)
    if point is None or sample_probs is not None or not np.array_equal(primary, point):
        raise AssertionError('Evaluation disagrees with paper mean-only prediction mode')
    metrics, _ = predictive_metric_bundle(point, labels, sample_probabilities=None, n_bins=15)
    result = {str(k): float(v) for k, v in metrics.items()}
    result.update(round=int(round_id), n_test=int(len(labels)),
                  paper_mean_only_eval=True, evaluator='CentralEvaluator._collect_probabilities+predictive_metric_bundle')
    del ev
    release_cuda_memory()
    return result


def _capture_frozen(transport, sample_sizes, bindings):
    snapshots = []
    for cid, (client, n, base) in enumerate(zip(transport.clients, sample_sizes, bindings)):
        cached = client.store.load()
        if cached is None:
            raise AssertionError('Missing prepared state')
        meta, arrays = cached
        if meta['n'] != n or meta['uploaded'] or meta['binding'] != base:
            raise AssertionError(f'Mismatched snapshot for client {cid}')
        for field in ('mu', 'precision', 'scores', 'order', 'tails', 'negative_sorted'):
            if field not in arrays:
                raise AssertionError(f'Missing cached field {field} client {cid}')
        # Meta must be independent for each policy branch. Arrays remain shared
        # until a branch stores/releases; packets only READ cached arrays.
        for a in arrays.values():
            a.setflags(write=False)
        snapshots.append((copy.deepcopy(meta), arrays))
    return snapshots


def _restore_branch(transport, snapshots):
    for client, (meta, arrays) in zip(transport.clients, snapshots):
        if client.store.load() is not None:
            raise AssertionError('A branch was not released before frozen-state restoration')
        client.store.save(copy.deepcopy(meta), arrays)
        # Original frozen arrays remain in snapshots; store.save makes copies.


def _record_exchange(transport, instructions, r: int, phase: str, policy: str, cfg):
    req_bytes = len(canonical(instructions))
    reply_map = transport.exchange([], instructions, r)
    if set(reply_map) != set(range(cfg.federation.num_clients)):
        raise AssertionError(f'{policy}: missing or duplicated {phase} replies')
    replies = [reply_map[cid] for cid in range(cfg.federation.num_clients)]
    reply_bytes = math.fsum(len(canonical(dict(client_id=cid, num_examples=rep.num_examples,
                                               metrics=rep.metrics))) for cid, rep in enumerate(replies))
    # Canonical JSON bytes count *control* metadata; model ndarrays separate.
    reply_bytes = int(reply_bytes)
    array_bytes = sum(sum(a.nbytes for a in rep.arrays) for rep in replies)
    event = dict(policy=policy, phase=phase, clients=len(replies), request_bytes=req_bytes,
                 reply_bytes=reply_bytes, control_bytes=req_bytes + reply_bytes,
                 array_bytes=array_bytes, modeled_control_s=(8*(req_bytes+reply_bytes)/cfg.wireless.control_rate_bps
                                             + cfg.wireless.control_exchange_latency_s))
    return replies, event


def _run_upload_branch(transport, snapshots, alloc, policy, src, bindings, prepared, *, restore):
    cfg, layout = src['cfg'], src['layout']
    r, d, U = src['round_id'], dimension(layout), src['timing']['upload_window_seconds']
    cs, samples, global_arrays = src['channel']['efficiencies'], src['samples'], src['global_arrays']
    if restore:
        _restore_branch(transport, snapshots)
    grants = [make_grant(base, count=int(m), d=d, bandwidth_hz=float(b), window_s=U,
                         efficiency=float(c), posterior=True)
              for base, m, b, c in zip(bindings, alloc.counts, alloc.bandwidth_hz, cs)]
    requests = [dict(base, phase='upload', grant_json=canonical(g).decode())
                for base, g in zip(bindings, grants)]
    uploads, upload_event = _record_exchange(transport, requests, r, 'upload', policy, cfg)
    decoded, masks, client_rows, actual_tails = [], [], [], []
    for cid, (reply, grant, snapshot) in enumerate(zip(uploads, grants, snapshots)):
        if reply.num_examples != samples[cid]:
            raise AssertionError('Client example count differs from the pinned partition')
        for key, expected in bindings[cid].items():
            if reply.metrics.get(key) != expected:
                raise AssertionError(f'{policy} client {cid} unbound upload metadata: {key}')
        got_bytes = sum(a.nbytes for a in reply.arrays)
        if got_bytes != payload_bytes(grant['count'], d) or got_bytes != grant['payload_bytes']:
            raise AssertionError(f'{policy}: wire-array bytes do not match granted integers')
        reconstructed, mask = decode(reply.arrays, reply.metrics, grant=grant,
                                     global_arrays=global_arrays, layout=layout)
        if int(mask.sum()) != grant['count'] or mask.shape != (d,):
            raise AssertionError('Incorrect actual selection cardinality')
        mu, prec = posterior_vectors(reconstructed, layout)
        g_mu, g_prec = posterior_vectors(global_arrays, layout)
        local = snapshot[1]
        if not (np.array_equal(mu[mask], local['mu'][mask]) and
                np.array_equal(prec[mask], local['precision'][mask]) and
                np.array_equal(mu[~mask], g_mu[~mask]) and
                np.array_equal(prec[~mask], g_prec[~mask])):
            raise AssertionError(f'Packet reconstruction/prior-fill differs from the specified values for {cid}')
        tail = float(reply.metrics['actual_discarded_kl'])
        if not math.isclose(tail, float(alloc.discarded_kl[cid]), rel_tol=1e-8, abs_tol=1e-7):
            raise AssertionError(f'{policy} client {cid}: oracle tail != uploaded mask tail')
        if int(reply.metrics['local_steps']) != int(prepared[cid].metrics['local_steps']):
            raise AssertionError('Local training steps differ across branches')
        actual_tails.append(tail)
        decoded.append(reconstructed)
        masks.append(mask)
        client_rows.append(dict(policy=policy, client_id=cid, examples=samples[cid],
            weight=samples[cid]/sum(samples), efficiency=cs[cid],
            retention_count=int(grant['count']), retention_ratio=int(grant['count'])/d,
            granted_bandwidth_hz=float(grant['bandwidth_hz']),
            observed_upload_bytes=got_bytes,
            upload_duration_s=8*got_bytes/float(grant['bandwidth_hz'])/float(cs[cid]),
            tail_oracle=float(alloc.discarded_kl[cid]), tail_from_uploaded_mask=tail,
            weighted_tail=samples[cid]/sum(samples)*tail,
            packet_format=reply.metrics['packet_format'], local_steps=reply.metrics['local_steps']))
    if upload_event['array_bytes'] != sum(g['payload_bytes'] for g in grants):
        raise AssertionError('Actual returned payload bytes differ from allocated bytes')
    aggregated = aggregate(decoded, masks, samples, global_arrays=global_arrays, layout=layout, cfg=cfg)
    release_requests = [dict(base, phase='release') for base in bindings]
    releases, release_event = _record_exchange(transport, release_requests, r, 'release', policy, cfg)
    if any(rep.arrays or rep.num_examples != samples[i] or not rep.metrics.get('released', False)
           for i, rep in enumerate(releases)):
        raise AssertionError('Release protocol mismatch')
    if any(client.store.load() is not None for client in transport.clients):
        raise AssertionError('Client prepared arrays persisted after release')
    actual_objective = math.fsum(n/sum(samples)*t for n, t in zip(samples, actual_tails))
    if not math.isclose(actual_objective, alloc.objective, rel_tol=1e-8, abs_tol=1e-7):
        raise AssertionError('Weighted actual packet KL does not match solver objective')
    branch_control_s = upload_event['modeled_control_s'] + release_event['modeled_control_s']
    return aggregated, client_rows, actual_objective, [upload_event, release_event], branch_control_s


def _dense_reference(snapshots, src):
    layout, cfg, samples = src['layout'], src['cfg'], src['samples']
    d = dimension(layout)
    complete = [pack_fola(unflatten(a['mu'], layout), unflatten(a['precision'],layout))
                for _,a in snapshots]
    weights = [n/sum(samples) for n in samples]
    dense = aggregate_fola(complete, weights, layout=layout, cfg=cfg)
    through_wireless = aggregate(complete, [np.ones(d, dtype=bool) for _ in samples], samples,
                                 global_arrays=src['global_arrays'], layout=layout, cfg=cfg)
    if not all(np.array_equal(x, y) for x,y in zip(dense, through_wireless)):
        raise AssertionError('Wireless full-mask aggregation differs from original dense FOLA reducer')
    return dense


def _posterior_deviation(output, reference, layout):
    mu, precision = posterior_vectors(output, layout)
    target_mu, target_precision = posterior_vectors(reference, layout)
    # Norm in float64, avoid changing any array.
    md = float(np.linalg.norm(mu.astype(np.float64)-target_mu.astype(np.float64)))
    pn = float(np.linalg.norm(precision.astype(np.float64)-target_precision.astype(np.float64)))
    return dict(mean_l2_difference=md,
                mean_relative_l2=md/max(1e-30, float(np.linalg.norm(target_mu.astype(np.float64)))),
                raw_precision_l2_difference=pn,
                raw_precision_relative_l2=pn/max(1e-30, float(np.linalg.norm(target_precision.astype(np.float64)))))


def save_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for CSV {path}')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_manifest(out: Path):
    return {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p))
            for p in sorted(out.iterdir()) if p.is_file() and p.name != 'RESULT_MANIFEST.json'}


def do_pair(src, output: Path, *, trainer=None, evaluator=None) -> dict:
    """Runs 1 PREPARE per client and 2 independent upload branches; output is new.

    Exposed for synthetic tests; production passes original trainer/evaluator.
    """
    if output.exists() or Path(str(output)+'.zip').exists():
        raise FileExistsError('Output directory/ZIP already exists; refusing overwrite')
    if src['round_id'] != TARGET_ROUND and evaluator is None:
        raise ValueError('Real experiment must use the pinned R10 -> R11 pair')
    output.mkdir(parents=True,exist_ok=False)
    cfg, layout, samples, r = src['cfg'], src['layout'], src['samples'], src['round_id']
    checkpoint_before = sha256(src['checkpoint'])
    partition_before = sha256(src['partition'])
    config_before = sha256(src['config_path'])
    source_summary = src['source']/'run_summary.json'
    summary_before = sha256(source_summary) if source_summary.exists() else None
    n = len(samples)
    calls = [0] * n
    def counted_trainer(*args):
        cid = int(args[1])
        calls[cid] += 1
        if calls[cid] != 1:
            raise AssertionError(f'Client {cid} trained more than once')
        return (trainer or prior_probe.train_local)(*args)
    # LocalTransport makes real ClientEngines, not a fabricated selection routine.
    transport = LocalTransport(cfg, src['partition'], src['meta']['mean_size'], trainer=counted_trainer)
    run_token = 'paired-r'+str(r)+'-'+checkpoint_before[:16]
    from bayesfl.wireless.protocol import binding
    bindings = [binding(src['global_arrays'], layout, round_id=r, client_id=i,
                        run_token=run_token, posterior=True) for i in range(n)]
    seed_everything(cfg.runtime.seed)
    prepare = prior_probe.QueryRecorder(transport, bindings, samples, cfg, r)
    print(f'PREPARE: one local FOLA training pass for each of {n} clients', flush=True)
    started = time.perf_counter()
    prep_replies = prepare.exchange('prepare', parameters=src['global_arrays'])
    elapsed = time.perf_counter()-started
    if calls != [1]*n:
        raise AssertionError('Local preparation was not exactly once per client')
    snapshots = _capture_frozen(transport, samples, bindings)
    print(f'PREPARE done | clients={n} | wall_seconds={elapsed:.3f}',flush=True)
    equal,joint,query_eq,query_joint = prior_probe.compare_frozen(
        transport, bindings, samples, cfg, layout, src['channel'], src['timing'], r)
    dense = _dense_reference(snapshots, src)
    d = dimension(layout)
    dense_required_bandwidth = math.fsum(payload_bits(d, d)/src['timing']['upload_window_seconds']/c
                                        for c in src['channel']['efficiencies'])
    dense_feasible = dense_required_bandwidth <= cfg.wireless.uplink_bandwidth_hz
    print('ALLOCATE: both policies used the same frozen scores; now encoding both branches',flush=True)
    eq_out,eq_rows,eq_actual,eq_events,eq_upload_control_s = _run_upload_branch(
        transport,snapshots,equal,'equal',src,bindings,prep_replies,restore=False)
    joint_out,joint_rows,joint_actual,joint_events,joint_upload_control_s = _run_upload_branch(
        transport,snapshots,joint,'joint',src,bindings,prep_replies,restore=True)
    if calls != [1]*n:
        raise AssertionError('Upload/query phase retrained a client')
    for name,rec,bctrl in [('equal',query_eq,eq_upload_control_s),('joint',query_joint,joint_upload_control_s)]:
        control_time = prepare.control_seconds + rec.control_seconds + bctrl
        if control_time > cfg.wireless.control_reserve_s + 1e-9:
            raise AssertionError(f'{name}: full protocol modeled control time exceeds reserve: {control_time}')
    source_files_after = [sha256(src['checkpoint']),sha256(src['partition']),sha256(src['config_path']),
                          sha256(source_summary) if source_summary.exists() else None]
    if source_files_after != [checkpoint_before,partition_before,config_before,summary_before]:
        raise AssertionError('Protected original experiment input changed during probe')
    real_evaluation = evaluator is None
    if evaluator is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available for original CIFAR paired evaluation')
        evaluator = lambda label,arrays,rr: evaluate_on_test(cfg,label,arrays,rr,output)
    print('EVALUATE: R10 checkpoint, paired R11 outputs, and full-posterior oracle on same test set',flush=True)
    evals = {
        'source_r10': evaluator('source_r10',src['global_arrays'],r-1),
        'equal_r11': evaluator('equal_r11',eq_out,r),
        'joint_r11': evaluator('joint_r11',joint_out,r),
        'dense_oracle_r11': evaluator('dense_oracle_r11',dense,r),
    }
    if calls != [1]*n:
        raise AssertionError('Evaluation retrained a client')
    historical_r10 = None
    if real_evaluation:
        # Check the inherited checkpoint against its own saved R10 test metrics.
        # Newly re-fitted R11 local posteriors need NOT bitwise reproduce old R11.
        original_metrics = src['source'] / 'metrics/global_metrics.csv'
        with original_metrics.open(newline='', encoding='utf-8') as handle:
            matches = [row for row in csv.DictReader(handle)
                       if int(float(row['round'])) == r-1]
        if len(matches) != 1:
            raise AssertionError('Saved source R10 centralized evaluation is missing or duplicated')
        historical_r10 = {key: float(matches[0][key]) for key in ('accuracy','nll')}
        if (abs(evals['source_r10']['accuracy']-historical_r10['accuracy']) > 0.001 or
            abs(evals['source_r10']['nll']-historical_r10['nll']) > 0.02):
            raise AssertionError('R10 checkpoint test metrics disagree with source R10 record')
    if ([sha256(src['checkpoint']),sha256(src['partition']),sha256(src['config_path']),
         sha256(source_summary) if source_summary.exists() else None] !=
        [checkpoint_before,partition_before,config_before,summary_before]):
        raise AssertionError('Protected source files modified during evaluation')
    dl_bytes = sum(a.nbytes for a in src['global_arrays'])
    def details(alloc,query,actual,rows,events,extra_control_s,output_arrays):
        u_bytes = sum(row['observed_upload_bytes'] for row in rows)
        bc = sum(ev['control_bytes'] for ev in events)
        ctrl = prepare.control_bytes + query.control_bytes + bc
        return dict(weighted_discarded_kl=alloc.objective,actual_discarded_kl=actual,
                    mean_keep_ratio=float(np.mean([m/dimension(layout) for m in alloc.counts])),
                    data_weighted_keep_ratio=math.fsum(x['weight']*x['retention_ratio'] for x in rows),
                    mean_bandwidth_hz=float(np.mean(alloc.bandwidth_hz)),
                    min_keep_ratio=min(x['retention_ratio'] for x in rows),
                    max_keep_ratio=max(x['retention_ratio'] for x in rows),
                    uplink_bytes=u_bytes, downlink_bytes=dl_bytes, model_array_bytes=dl_bytes+u_bytes,
                    control_bytes_full_modeled_protocol=ctrl,
                    modeled_control_seconds=prepare.control_seconds+query.control_seconds+extra_control_s,
                    control_allocation_bytes=query.control_bytes,
                    control_prepare_bytes_shared=prepare.control_bytes,
                    control_upload_release_bytes=bc,
                    price_queries=query.price_requests,
                    tail_queries=query.count_requests,
                    solver_status=alloc.status, solver_numerical_gap=alloc.gap_diagnostic,
                    certified_optimal=alloc.certified_optimal,
                    max_upload_seconds=max(x['upload_duration_s'] for x in rows),
                    aggregate_distance_to_full=difference_to_dense(output_arrays))
    def difference_to_dense(a):
        return _posterior_deviation(a,dense,layout)
    eq_detail = details(equal,query_eq,eq_actual,eq_rows,eq_events,eq_upload_control_s,eq_out)
    joint_detail = details(joint,query_joint,joint_actual,joint_rows,joint_events,joint_upload_control_s,joint_out)
    all_bytes_e = eq_detail['model_array_bytes']+eq_detail['control_bytes_full_modeled_protocol']
    all_bytes_j = joint_detail['model_array_bytes']+joint_detail['control_bytes_full_modeled_protocol']
    summary=dict(experiment=PROBE_ID, source_checkpoint_round=r-1, compared_round=r,
        source_run=str(src['source']), checkpoint_path=str(src['checkpoint']),
        checkpoint_sha256=checkpoint_before, partition_path=str(src['partition']),
        partition_file_sha256=partition_before, partition_content_sha256=src['meta']['sha256'],
        resolved_config_sha256=config_before,
        prepared_once_per_client=True, per_client_training_calls=calls,
        client_snapshots_cloned_before_upload=True,
        actual_wire_packets_encoded_decoded=True, FOLA_aggregated_using_installed_protocol=True,
        dense_oracle_is_not_feasible_wireless_baseline=not dense_feasible,
        dense_required_bandwidth_hz=dense_required_bandwidth,
        same_checkpoint_and_local_posteriors=True,
        actual_test_evaluation=real_evaluation,
        model=cfg.model.name, d=dimension(layout), num_clients=n, examples=sum(samples),
        upload_window_s=src['timing']['upload_window_seconds'],
        modeled_round_seconds=cfg.wireless.round_target_s,
        control_reserve_s=cfg.wireless.control_reserve_s,
        input_wireless_policy=cfg.wireless.policy,
        evaluation=evals, historical_source_r10_evaluation=historical_r10,
        source_r10_accuracy_replay_difference=(None if historical_r10 is None else
            evals['source_r10']['accuracy']-historical_r10['accuracy']),
        equal=eq_detail, joint=joint_detail,
        kl_reduction_percent=(100*(1-joint_actual/eq_actual) if eq_actual>0 else 0.0),
        model_bytes_reduction_percent=100*(1-joint_detail['model_array_bytes']/eq_detail['model_array_bytes']),
        full_recorded_bytes_reduction_percent=100*(1-all_bytes_j/all_bytes_e),
        r11_joint_minus_equal_accuracy_pp=100*(evals['joint_r11']['accuracy']-evals['equal_r11']['accuracy']),
        scenario_is_example_not_locked=True,
        caveats=['One checkpoint and one set of client updates; no significance test or multi-round convergence claim.',
                 'Dense oracle ignores wireless deadline/bandwidth feasibility.',
                 'Modeled time and canonical JSON control counters are not RF or GPU wall-time measurements.',
                 'Fixed PSD, assumed compute model, and fixed 180s slot are pilot values, not physical-device calibration.',
                 'Joint method optimizes discarded KL, not test accuracy or minimum transmitted bytes.',
                 'No long training has been launched; readiness is conditional on the assumptions.',
        'The new local R11 fit may not bitwise reproduce the earlier Flower/Ray R11 trajectory; only the common starting R10 is fixed.'])
    save_csv(output/'per_client.csv',eq_rows+joint_rows)
    events = []
    for ev in prepare.events:
        events.append(dict(policy='shared',phase=ev['phase'],clients=ev['clients'],
                  request_bytes=ev['request_bytes'],reply_bytes=ev['reply_bytes'],
                  control_bytes=ev['control_bytes'],array_bytes=0,
                  modeled_control_s=ev['modeled_control_seconds']))
    for label,query,branch in [('equal',query_eq,eq_events),('joint',query_joint,joint_events)]:
        events.extend(dict(policy=label,phase=e['phase'],clients=e['clients'],
               request_bytes=e['request_bytes'],reply_bytes=e['reply_bytes'],
               control_bytes=e['control_bytes'],array_bytes=0,
               modeled_control_s=e['modeled_control_seconds']) for e in query.events)
        events.extend(branch)
    save_csv(output/'protocol_events.csv', events)
    eval_rows=[dict(policy=name, **v) for name,v in evals.items()]
    save_csv(output/'evaluation.csv',eval_rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (output/'README.txt').write_text(
        'Paired R10->R11 CIFAR-10 one-step *counterfactual*: real 20-client FOLA local preparation once;\n'
        'independent equal/joint sparse-encoded uploads and FOLA aggregation/evaluation.\n'
        'Dense R11 oracle is not feasible under the modeled deadline.\n'
        'Only scalar summaries and accounting exported; no scores, CIFAR data, or model arrays.\n')
    (output/'RESULT_MANIFEST.json').write_text(json.dumps(_summary_manifest(output),indent=2,sort_keys=True)+'\n')
    with zipfile.ZipFile(str(output)+'.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.iterdir()):
            if p.is_file():
                z.write(p,arcname=output.name+'/'+p.name)
    print('\nPASS: PAIRED REAL-PACKET R11 AGGREGATION AND EVALUATION')
    print('Equal objective: %.9g | Joint objective: %.9g | Reduction %.4f%%'%(
        eq_actual,joint_actual,summary['kl_reduction_percent']))
    print('R10 accuracy: %.4f%%'%(100*evals['source_r10']['accuracy']))
    print('R11 equal accuracy: %.4f%%'%(100*evals['equal_r11']['accuracy']))
    print('R11 joint accuracy: %.4f%%'%(100*evals['joint_r11']['accuracy']))
    print('R11 dense oracle accuracy: %.4f%%'%(100*evals['dense_oracle_r11']['accuracy']))
    print('Joint - equal: %+.4f percentage points'%(summary['r11_joint_minus_equal_accuracy_pp']))
    print('Equal model bytes: %d | Joint model bytes: %d'%(
        eq_detail['model_array_bytes'],joint_detail['model_array_bytes']))
    print('Equal all-protocol control: %d | Joint all-protocol control: %d'%(
        eq_detail['control_bytes_full_modeled_protocol'],joint_detail['control_bytes_full_modeled_protocol']))
    print('Local training calls:',calls)
    print('Upload ZIP:',str(output)+'.zip')
    print('Protected historical checkpoint/partition/config hashes verified unchanged.')
    return summary


def preflight(src):
    prior_probe.preflight_print(src)
    cfg=src['cfg']
    if src['round_id'] != TARGET_ROUND or src['source'].name.find('equal_bandwidth') < 0:
        raise ValueError('Pinned probe requires original equal-bandwidth R10 source checkpoint')
    if cfg.data.dataset != 'cifar10' or cfg.federation.num_clients!=20 or cfg.method!='fola':
        raise ValueError('Unexpected CIFAR/20-client/FOLA source config')
    if not cfg.fola.paper_mean_only_eval:
        raise ValueError('Locked paper mean-only accuracy must be used')
    if cfg.wireless.policy!='equal_bandwidth' or cfg.wireless.channel.mode!='fixed':
        raise ValueError('Expected equal-bandwidth fixed-gain source')
    import torch
    print('EVALUATION:', 'FOLA mean-only', 'device:',cfg.runtime.central_eval_device,
          'CUDA detected:',torch.cuda.is_available())
    print('PASS: paired source/channel/mean-only evaluation gate')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight',action='store_true')
    group.add_argument('--execute',action='store_true')
    parser.add_argument('--checkpoint-round',type=int,default=SOURCE_ROUND)
    parser.add_argument('--source-run',default=None)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--require-cuda',action='store_true')
    args=parser.parse_args(argv)
    if args.checkpoint_round != SOURCE_ROUND:
        parser.error('This locked experiment accepts only R10 -> R11, no silent checkpoint changes')
    root=Path.cwd().resolve()
    if not (root/'src/bayesfl/wireless/controller.py').exists():
        parser.error('Run from the installed bayesfl repository root')
    src=prior_probe.load_existing_inputs(root,args.source_run,args.checkpoint_round)
    preflight(src)
    if args.preflight:
        return
    if args.output_dir is None:
        parser.error('--execute requires --output-dir')
    import torch
    if args.require_cuda and not torch.cuda.is_available():
        parser.error('CUDA required but unavailable; no silent CPU fallback')
    do_pair(src,args.output_dir.expanduser().resolve())


if __name__=='__main__':
    main()
