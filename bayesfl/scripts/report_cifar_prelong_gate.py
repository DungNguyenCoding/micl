#!/usr/bin/env python3
"""Fail-closed summary and portable ZIP for the one-shot CIFAR pre-long gate.

Only consumes new paired-probe scalar results; never accesses the model arrays.
Run from bayesfl root. Usage: python scripts/report_cifar_prelong_gate.py --gate-dir /abs/path
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import zipfile


def check_summary(summary):
    checks = {}
    def require(name, condition):
        checks[name] = bool(condition)
    ev = summary.get('evaluation',{})
    eq = summary.get('equal',{})
    joint = summary.get('joint',{})
    require('source_R10_to_R11', summary.get('source_checkpoint_round') == 10 and summary.get('compared_round') == 11)
    require('locked_CIFAR10_model_and_clients', summary.get('model') == 'paper_basiccnn' and
            summary.get('d') == 878538 and summary.get('num_clients') == 20 and summary.get('examples') == 49988)
    require('same_source_and_frozen_posteriors', summary.get('same_checkpoint_and_local_posteriors') is True and
            summary.get('client_snapshots_cloned_before_upload') is True)
    require('exactly_one_FOLA_training_per_client', summary.get('per_client_training_calls') == [1] * 20)
    require('production_encode_decode_and_aggregation', summary.get('actual_wire_packets_encoded_decoded') is True and
            summary.get('FOLA_aggregated_using_installed_protocol') is True)
    require('real_mean_only_test_evaluation', summary.get('actual_test_evaluation') is True and
            all(k in ev for k in ('source_r10','equal_r11','joint_r11','dense_oracle_r11')) and
            all(v.get('n_test') == 10000 and v.get('paper_mean_only_eval') is True
                and math.isfinite(v.get('accuracy',float('nan'))) and 0 <= v['accuracy'] <= 1
                and math.isfinite(v.get('nll',float('nan')))
                for v in ev.values()))
    hist=summary.get('historical_source_r10_evaluation')
    require('source_R10_checkpoint_reproduces_saved_test_metrics',
            isinstance(hist,dict) and
            abs(summary['evaluation']['source_r10']['accuracy']-hist.get('accuracy',-1))<=0.001 and
            abs(summary['evaluation']['source_r10']['nll']-hist.get('nll',float('inf')))<=0.02)
    require('same_modeled_round_and_upload_interval', summary.get('modeled_round_seconds') == 180.0 and
            58.0 < summary.get('upload_window_s',float('nan')) < 60.0)
    require('KL_objectives_recomputed_from_actual_packets',
            all(math.isfinite(x.get('actual_discarded_kl',float('nan'))) and
                math.isclose(x['actual_discarded_kl'],x.get('weighted_discarded_kl',float('nan')),rel_tol=1e-8,abs_tol=1e-7)
                for x in (eq,joint)))
    require('joint_feasible_incumbent_KL_not_worse',joint.get('weighted_discarded_kl',float('inf')) <=
            eq.get('weighted_discarded_kl',-float('inf')) + 1e-7)
    require('modeled_control_budget_both_policies',
            all(math.isfinite(x.get('modeled_control_seconds',float('nan'))) and
                0 < x['modeled_control_seconds'] <= summary.get('control_reserve_s',0)+1e-9
                for x in (eq,joint)))
    require('both_wire_payloads_nonempty_and_feasible',
            all(x.get('uplink_bytes',0) > 0 and
                x.get('model_array_bytes',0) == x.get('uplink_bytes',0)+x.get('downlink_bytes',-1) and
                0 < x.get('max_upload_seconds',float('inf')) <= summary.get('upload_window_s',0)*(1+1e-10)
                for x in (eq,joint)))
    require('price_coordination_exercised',joint.get('price_queries',0)>0 and eq.get('price_queries',-1)==0)
    require('dense_oracle_labelled_infeasible_reference',summary.get('dense_oracle_is_not_feasible_wireless_baseline') is True)
    require('scenario_not_falsely_claimed_as_physically_locked',summary.get('scenario_is_example_not_locked') is True)
    return checks


def produce(gate_dir):
    path=gate_dir/'paired_r11'/'summary.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing paired summary: {path}')
    summary=json.loads(path.read_text())
    checks=check_summary(summary)
    failures=[name for name,passed in checks.items() if not passed]
    gate=dict(schema='cifar_prelong_gate_v1', produced_at_utc=datetime.now(timezone.utc).isoformat(),
              all_engineering_checks_passed=(not failures),check_results=checks,failures=failures,
              long_run_launched=False,
              ready_for_controlled_fixed_scenario_pilots=(not failures),
              ready_for_publication_grade_wireless_claims=False,
              substantive_limits=[
                  'Fixed path-loss channels only; no fades/outages or imperfect CSI evaluated.',
                  '180-second slot and client compute/score time are engineering assumptions, not measured edge-device timings.',
                  'PSD is conservatively adapted from SchedulingFL fixed power, not a fixed 24-dBm power per client.',
                  'Control bytes/time are canonical-JSON modeled traffic, not measured wireless packets.',
                  'One seed, one source checkpoint, one next-step evaluation cannot establish long-run accuracy improvement.',
                  'Joint solver is an incumbent/diagnostic, not a proof of global integer optimality.',
                  'Dense oracle is a non-feasible reference under the selected bandwidth/deadline.'
              ],
              next_steps=[
                  'Inspect paired R11 equal/joint/dense oracle accuracy and NLL before longer experiments.',
                  'Choose and freeze the study scenario explicitly; record timing/calibration limitations.',
                  'Use matched seeds/partitions and the identical 150-round learning schedule for both methods; compare R100 and R150.',
                  'Consider multi-seed repetition and sensitivity to bandwidth, deadline, fading and control overhead.'
              ],
              paired_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              pair_result_zip= 'paired_r11.zip')
    (gate_dir/'READINESS_REPORT.json').write_text(json.dumps(gate,indent=2,sort_keys=True,allow_nan=False)+'\n')
    readme=[
        '# CIFAR-10 pre-long gate evidence (pilot scenario)', '',
        f"Engineering validation: **{'PASS' if not failures else 'FAIL'}**",
        'Publication-grade scenario readiness: **NOT CERTIFIED**.',
        'No 100/150-round jobs were started.', '',
        '## Paired R11 actual test accuracy', '',
        '| Model | Accuracy | NLL |', '|---|---:|---:|',
    ]
    for label in ('source_r10','equal_r11','joint_r11','dense_oracle_r11'):
        v=summary.get('evaluation',{}).get(label,{})
        if v:
            readme.append(f"| {label} | {100*v['accuracy']:.3f}% | {v['nll']:.6f} |")
    readme += ['',
        'The R11 outputs were produced from a single common R10 global checkpoint and the same local client preparations.',
        'Each policy encoded/decoded its own sparse packets and used the production weighted FOLA aggregation.',
        'The dense oracle ignores the modeled wireless feasibility constraint and is only a diagnostic.', '',
        '## Checks','']
    readme.extend(f"- {'PASS' if v else 'FAIL'}: {k}" for k,v in checks.items())
    readme += ['', '## Scientific caveats', '']
    readme.extend('- '+x for x in gate['substantive_limits'])
    readme += ['', '## Source material', '',
        '- `paired_r11/summary.json`: measured paired accuracy, KL, model/control array accounting and provenance',
        '- `paired_r11/per_client.csv`: exact integer counts, uploaded bytes and per-client KL reconstruction',
        '- `paired_r11/protocol_events.csv`: price/tail/prepare/upload/release metadata sizes',
        '- `logs/`: actual server test and read-only validation output',
        '- `READINESS_REPORT.json`: machine-readable gate results',
        '', 'No CIFAR images, checkpoint weights, raw KL scores or trained posterior arrays are included in this ZIP.',
        '']
    (gate_dir/'READINESS_REPORT.md').write_text('\n'.join(readme))
    if failures:
        print('FAIL: pre-long gate checks:',', '.join(failures))
        raise SystemExit(2)
    # Zip carefully whitelisted small evidence files only. Refuse symlinks.
    allowed=[]
    for p in sorted(gate_dir.rglob('*')):
        if not p.is_file() or p.is_symlink():
            continue
        relative=p.relative_to(gate_dir)
        name=relative.name
        if relative.parts[0]=='logs' and name.endswith('.txt'):
            allowed.append(p)
        elif len(relative.parts)==1 and name in {'READINESS_REPORT.json','READINESS_REPORT.md'}:
            allowed.append(p)
        elif len(relative.parts)==2 and relative.parts[0]=='paired_r11' and name in {
            'summary.json','per_client.csv','protocol_events.csv','evaluation.csv',
            'README.txt','RESULT_MANIFEST.json'}:
            allowed.append(p)
    result=Path(str(gate_dir)+'.zip')
    if result.exists():
        raise FileExistsError(f'Refusing to overwrite: {result}')
    with zipfile.ZipFile(result,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in allowed:
            archive.write(p,arcname=gate_dir.name+'/'+str(p.relative_to(gate_dir)))
    with zipfile.ZipFile(result) as z:
        assert z.testzip() is None
    print('PASS: pre-long CIFAR-10 fixed-scenario engineering gate')
    print('Paired R11: accuracy equal/joint = %.3f%% / %.3f%%'%(
        100*summary['evaluation']['equal_r11']['accuracy'],
        100*summary['evaluation']['joint_r11']['accuracy']))
    print('Long-run training has NOT started.')
    print('Audit ZIP:',result)
    print('Science note: pilot compute/control assumptions and seed limitations remain.')
    return gate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gate-dir',type=Path,required=True)
    args=p.parse_args()
    produce(args.gate_dir.expanduser().resolve())

if __name__=='__main__':
    main()
