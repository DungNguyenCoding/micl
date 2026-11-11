#!/usr/bin/env python3
"""Read-only R150 joint-FedKLS audit; canonical CSV/JSON are not modified.

Python 3.10+. Requires PyYAML (already used by BayesFL), otherwise stdlib.
No model/dataset files are ever read or exported by this tool.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import zipfile


def fail(message):
    raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def num(x, where):
    try: y=float(x)
    except (TypeError, ValueError) as e: raise ValueError(f'Invalid {where}={x!r}') from e
    if not math.isfinite(y): fail(f'Non-finite {where}={x!r}')
    return y


def integer(x, where):
    y=num(x,where)
    if not y.is_integer(): fail(f'Not integer {where}={x!r}')
    return int(y)


def near(a,b,where,relative=1e-8,absolute=1e-6):
    a=num(a,where);b=num(b,where)
    if not math.isclose(a,b,rel_tol=relative,abs_tol=absolute):
        fail(f'Mismatch {where}: {a!r} != {b!r}')


def payload(m,d):
    return 8*d if m==d else 8*m+(d+7)//8


def select_run(root,tag):
    manifest_path=root/'scripts/configs/wireless'/tag/'study_manifest.json'
    manifest=read_json(manifest_path)
    joint=[j for j in manifest['jobs'] if '_joint_' in j['run_name']]
    if len(joint)!=1: fail(f'Expected exactly one joint job in {manifest_path}; found {len(joint)}')
    name=joint[0]['run_name']
    matches=sorted(p for p in (root/'outputs').glob(name+'_*') if p.is_dir())
    if len(matches)!=1: fail(f'Expected exactly one run for {name}; found {len(matches)}. Supply --run-dir.')
    return matches[0],manifest_path,manifest


def audit(root,run,manifest_path,manifest,expected_rounds):
    import yaml
    summary=read_json(run/'run_summary.json')
    protocol=read_json(run/'wireless/protocol.json')
    layout=read_json(run/'wireless/layout_manifest.json')
    cfg=yaml.safe_load((run/'resolved_config.yaml').read_text(encoding='utf-8'))
    joint=cfg['wireless']; train=cfg['training']
    d=integer(layout['d'],'d'); K=integer(protocol['num_clients'],'K')
    target=num(joint['round_target_s'],'target'); B=num(joint['uplink_bandwidth_hz'],'B')
    reserve=num(joint['control_reserve_s'],'reserve'); E=integer(train['local_epochs'],'E')
    batch=integer(train['batch_size'],'batch'); P=joint['channel']['uplink_psd_w_hz']; cap=joint['channel']['client_power_cap_w']
    hard=[]; cautions=[]
    def check(when,desc):
        if not when: fail(desc)
        hard.append(desc)

    check(cfg['method']=='fola' and cfg['data']['dataset']=='cifar10' and cfg['model']['name']=='paper_basiccnn',
          'CIFAR10/FOLA/paper_basiccnn configuration')
    check(cfg['wireless']['policy']=='joint' and protocol['policy']=='joint' and summary['wireless_policy']=='joint',
          'Joint allocation policy')
    check(cfg['compression']['selection_rule']=='kl_global_local' and protocol['selection_rule']=='kl_global_local',
          'Global-to-local KL selection')
    check(protocol['protocol']=='fedkls_wireless_integer_precision_v2' and d==878538 and K==20,
          'Wireless-v2 layout and 20 clients')
    check(cfg['federation']['clients_per_round']==K and cfg['federation']['num_clients']==K and
          train['local_epochs']==10 and train['batch_size']==32 and cfg['runtime']['seed']==0,
          'Full-participation seed0 E10 B32')
    check(integer(train['rounds'],'configured rounds')==expected_rounds and
          integer(summary['completed_rounds'],'completed')==expected_rounds and
          not summary.get('failure') and summary.get('pending_round') is None and
          summary.get('stop_reason')=='max_rounds', 'All expected rounds committed')
    check(len(P)==K and len(cap)==K and B>0 and target>reserve>0, 'Power/bandwidth/deadline configuration')

    study_manifest_source=manifest.get('scenario_source')
    if study_manifest_source:
        scenario=Path(study_manifest_source)
        check(scenario.is_file() and checksum(scenario)==manifest.get('scenario_sha256'),
              'Pinned scenario source unchanged')
    if manifest.get('scenario_status')!='LOCKED':
        cautions.append('Wireless scenario is provisional/example, not physically calibrated.')
    if joint['channel']['mode']!='fixed':
        cautions.append('Channel mode is not fixed; audit does not certify fading/outage policy.')

    rr=read_rows(run/'metrics/wireless_rounds.csv')
    gr=read_rows(run/'metrics/global_metrics.csv')
    cr=read_rows(run/'metrics/wireless_clients.csv')
    check(len(rr)==expected_rounds and len(gr)==expected_rounds+1 and len(cr)==expected_rounds*K,
          'All round/global/client metric rows present')
    check([integer(x['round'],'round') for x in rr]==list(range(1,expected_rounds+1)) and
          [integer(x['round'],'global round') for x in gr]==list(range(expected_rounds+1)),
          'Round IDs contiguous and nonduplicate')
    by_round=defaultdict(list)
    for row in cr: by_round[integer(row['round'],'client round')].append(row)
    check(set(by_round)==set(range(1,expected_rounds+1)), 'All client rounds present')
    check(all(sorted(integer(c['client_id'],'client id') for c in by_round[r])==list(range(K))
              for r in range(1,expected_rounds+1)), '20 distinct client IDs each round')
    check(gr[0]['array_accounting_complete']=='True', 'Initialization accounting flag')
    check(all(row['array_accounting_complete']=='True' for row in rr), 'All-round array accounting flags')

    events_path=run/'wireless/events.jsonl'
    ev_round=defaultdict(lambda:dict(bytes=0,array=0,energy=0.,price=0,reply_price=0))
    event_phases=Counter()
    ev_count=0
    with events_path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            ev=json.loads(line);ev_count+=1
            r=integer(ev['round'],'event round')
            if not 1<=r<=expected_rounds: fail('Orphan control event')
            o=ev_round[r];o['bytes']+=integer(ev.get('control_bytes',0),'event control')
            o['array']+=integer(ev.get('downlink_array_bytes',0),'event downlink')+integer(ev.get('uplink_array_bytes',0),'event uplink')
            o['energy']+=num(ev.get('energy_j',0),'event energy')
            phase=ev['phase'];event_phases[phase]+=1
            if phase=='control_request' and ev.get('operation')=='price':o['price']+=1
            if phase=='reply' and ev.get('operation')=='price':o['reply_price']+=1
    check(len(ev_round)==expected_rounds, 'Every round has an event ledger')

    perround=[]; perclient=[]
    prev_array=prev_control=prev_steps=prev_examples=0
    prev_energy=0.
    sample_by_id=None
    for rid,(r,g) in enumerate(zip(rr,gr[1:]),1):
        clients=sorted(by_round[rid],key=lambda x:integer(x['client_id'],'client_id'))
        near(r['cumulative_modeled_time_seconds'],rid*target,f'R{rid} clock')
        near(g['cumulative_modeled_time_seconds'],rid*target,f'R{rid} global clock')
        for metric in ('accuracy','nll','brier','ece','mean_confidence'):
            a=num(g[metric],f'R{rid} {metric}')
            if metric in ('accuracy','ece','mean_confidence') and not 0<=a<=1:fail(f'R{rid} {metric} out of [0,1]')
            if metric=='nll' and a<0: fail('negative NLL')
        near(g['accuracy'],g['fola_mean_accuracy'],f'R{rid} FOLA mean metric')
        near(g['nll'],g['fola_mean_nll'],f'R{rid} FOLA mean NLL')
        counts=[];bands=[];actual_tail=[];selected_tail=[];example_count=[];keeps=[]
        uplink=0;training_steps=0
        for c in clients:
            cid=integer(c['client_id'],'client id')
            n=integer(c['num_examples'],f'client {cid} n');m=integer(c['keep_count'],f'client {cid} m')
            if not 1<=m<=d or not n>0 or integer(c['d'],'c.d')!=d:fail(f'R{rid} client{cid} invalid count')
            bytes_expected=payload(m,d)
            if integer(c['upload_array_bytes'],'client bytes')!=bytes_expected:fail(f'R{rid} c{cid} byte/count mismatch')
            b=num(c['bandwidth_hz'],'grant');eff=num(c['spectral_efficiency'],'spectral efficiency')
            if b<=0 or eff<=0: fail('Non-positive bandwidth or link efficiency')
            airtime=8*bytes_expected/(b*eff)
            near(airtime,c['modeled_upload_s'],f'R{rid} c{cid} radio upload',relative=2e-8,absolute=1e-5)
            if airtime>num(r['modeled_upload_window_seconds'],'window')+1e-5:fail(f'R{rid} c{cid} misses modeled upload deadline')
            power=b*num(P[cid],'PSD')
            if power>num(cap[cid],'cap')+1e-9:fail(f'R{rid} c{cid} power cap exceeded')
            near(power,c['implied_tx_power_w'],f'R{rid} c{cid} power',relative=1e-7)
            near(power*airtime,c['modeled_tx_energy_j'],f'R{rid} c{cid} energy',relative=1e-7)
            steps=integer(c['local_steps'],'local steps')
            expected_steps=E*math.ceil(n/batch)
            if steps!=expected_steps: fail(f'R{rid} c{cid} local train steps: {steps}!={expected_steps}')
            keep=num(c['keep_ratio'],'keep ratio')
            near(keep,m/d,f'R{rid} c{cid} keep ratio',relative=1e-8)
            a=num(c['actual_discarded_kl'],'actual discarded KL')
            s=num(c['topm_tail_kl'],'top-m KL')
            if a<0 or s<0:fail('Negative KL tail')
            near(a,s,f'R{rid} c{cid} tail consistency',relative=1e-8,absolute=1e-4)
            counts.append(m);bands.append(b);actual_tail.append(a);selected_tail.append(s);keeps.append(keep)
            uplink+=bytes_expected; training_steps+=steps; example_count.append(n)
            perclient.append(dict(round=rid,client_id=cid,num_examples=n,retained=m,retained_ratio=keep,bandwidth_hz=b,
                discarded_kl=a,weighted_discarded_kl=n*a,upload_bytes=bytes_expected,airtime_s=airtime,
                floor_global=num(c['global_floor_fraction'],'global floor'),floor_local=num(c['local_floor_fraction'],'local floor')))
        if sample_by_id is None: sample_by_id=example_count
        elif example_count!=sample_by_id:fail(f'R{rid} unexpected partition/client sample change')
        total_n=sum(example_count)
        weight=[n/total_n for n in example_count]
        obj=math.fsum(w*s for w,s in zip(weight,actual_tail))
        if math.fsum(bands)>B+1e-4:fail(f'R{rid} sum bandwidth exceeds budget')
        near(math.fsum(bands),r['allocated_bandwidth_hz'],f'R{rid} total bandwidth',relative=1e-8)
        need_bytes=8*d+uplink
        if integer(r['round_array_bytes'],'round bytes')!=need_bytes:fail(f'R{rid} data-plane packet accounting mismatch')
        if integer(r['cumulative_all_array_bytes'],'cumulative bytes')!=prev_array+need_bytes:fail(f'R{rid} cumulative array mismatch')
        if integer(g['cumulative_all_array_bytes'],'global array bytes')!=prev_array+need_bytes:fail('Global array mismatch')
        if integer(r['cumulative_local_training_steps'],'cum steps')!=prev_steps+training_steps:fail(f'R{rid} step total mismatch')
        if integer(r['cumulative_local_examples_seen'],'cum examples')!=prev_examples+total_n*E:fail('Examples total mismatch')
        if num(r['modeled_control_seconds'],'control seconds')>reserve+1e-6:fail(f'R{rid} control reserve exceeded')
        if num(r['modeled_idle_seconds'],'idle seconds') < -1e-6:fail('Negative idle')
        near(num(r['modeled_idle_seconds'],'idle')+num(r['modeled_control_seconds'],'control'),reserve,
             f'R{rid} reserve balance',relative=1e-7,absolute=1e-5)
        near(r['topm_reconstruction_objective'],obj,f'R{rid} recorded topm objective',relative=1e-8,absolute=1e-5)
        near(r['actual_reconstruction_objective'],obj,f'R{rid} actual objective',relative=1e-8,absolute=1e-5)
        near(r['cumulative_modeled_tx_energy_j'],prev_energy+num(ev_round[rid]['energy'],'event energy'),
             f'R{rid} cum energy',relative=1e-8,absolute=1e-4)
        alloc_path=run/'wireless/allocations'/f'round_{rid:06d}.json'
        allocation=read_json(alloc_path)
        check(integer(allocation['round'],'alloc round')==rid,f'R{rid} allocation JSON round')
        al=allocation['allocation'];timing=allocation['timing'];cc=allocation['channel']
        if [integer(x,'alloc.count') for x in al['counts']]!=counts:fail(f'R{rid} exact grants differ from clients')
        for cid in range(K):
            near(al['bandwidth_hz'][cid],bands[cid],f'R{rid} c{cid} JSON grant')
            near(al['discarded_kl'][cid],selected_tail[cid],f'R{rid} c{cid} JSON tail',relative=1e-8,absolute=1e-4)
            near(allocation['actual_discarded_kl'][cid],actual_tail[cid],f'R{rid} c{cid} JSON actual tail',relative=1e-8,absolute=1e-4)
            near(cc['efficiencies'][cid],clients[cid]['spectral_efficiency'],f'R{rid} c{cid} channel efficiency')
        near(al['objective'],obj,f'R{rid} allocation objective',relative=1e-8,absolute=1e-5)
        near(timing['upload_window_seconds'],r['modeled_upload_window_seconds'],f'R{rid} JSON upload window')
        if integer(al['price_queries'],'alloc.price_queries')!=integer(r['price_queries'],'round queries'):
            fail(f'R{rid} query count mismatch')
        if al['status']!=r['allocation_status']:fail('Allocator statuses mismatch')
        if (al['status'] not in ('numerical_gap_target','zero_distortion')):
            cautions.append(f'R{rid}: allocator status {al["status"]}; investigate convergence/tolerance')
        if al.get('certified_optimal') is not True and rid==1:
            cautions.append('Allocator finds a feasible integer incumbent; global integer optimality is NOT certified.')
        if integer(r['round_control_bytes'],'round control')!=ev_round[rid]['bytes'] or \
           integer(r['round_array_bytes'],'round array')!=ev_round[rid]['array']:
            fail(f'R{rid} event ledger bytes mismatch')
        if integer(r['price_queries'],'price queries')!=ev_round[rid]['price'] or ev_round[rid]['reply_price']!=K*ev_round[rid]['price']:
            fail(f'R{rid} price query/reply count mismatch')
        if integer(r['cumulative_control_bytes'],'cumulative control')!=prev_control+ev_round[rid]['bytes']:
            fail(f'R{rid} cumulative control mismatch')
        acc=num(g['accuracy'],'accuracy');nll=num(g['nll'],'NLL')
        perround.append(dict(round=rid,accuracy_pct=100*acc,nll=nll,keep_min=min(keeps),keep_mean=sum(keeps)/K,
                keep_max=max(keeps),keep_weighted=sum(w*v for w,v in zip(weight,keeps)),
                bandwidth_min_hz=min(bands),bandwidth_max_hz=max(bands),round_array_bytes=need_bytes,
                round_control_bytes=ev_round[rid]['bytes'],price_queries=ev_round[rid]['price'],
                weighted_discarded_kl=obj,modeled_control_s=num(r['modeled_control_seconds'],'control'),
                upload_window_s=num(r['modeled_upload_window_seconds'],'upload'),
                max_upload_s=max(num(c['modeled_upload_s'],'client airtime') for c in clients),
                max_power_w=max(num(c['implied_tx_power_w'],'power') for c in clients),
                cumulative_array_bytes=prev_array+need_bytes,cumulative_control_bytes=prev_control+ev_round[rid]['bytes'],
                cumulative_energy_j=num(r['cumulative_modeled_tx_energy_j'],'energy'),
                cumulative_steps=prev_steps+training_steps,cumulative_examples=prev_examples+total_n*E,
                solver_status=r['allocation_status'],solver_gap=num(r['objective_gap_diagnostic'],'gap'),
                solver_certified_optimal=r['certified_optimal']))
        prev_array+=need_bytes;prev_control+=ev_round[rid]['bytes'];prev_steps+=training_steps
        prev_examples+=total_n*E;prev_energy=num(r['cumulative_modeled_tx_energy_j'],'energy')

    check(len(list((run/'wireless/allocations').glob('round_*.json')))==expected_rounds,
          'Exact allocation-file count')
    check(integer(summary['cumulative_all_array_bytes'],'summary arrays')==prev_array and
          integer(summary['cumulative_control_bytes'],'summary control')==prev_control,
          'Summary cumulative bytes match all 150 rounds')
    near(summary['last_evaluation']['accuracy'],gr[-1]['accuracy'],'Summary final accuracy')
    near(summary['last_evaluation']['nll'],gr[-1]['nll'],'Summary final NLL')
    check(integer(summary['price_queries'],'summary price queries')==perround[-1]['price_queries'],
          'Summary latest-round query count')
    latest=run/'wireless_resume/latest.json'
    if latest.exists():
        pointer=read_json(latest);filename=pointer['state_file']
        if Path(filename).name!=filename:fail('Unexpected state pointer path')
        state_file=latest.parent/filename
        if checksum(state_file)!=pointer['state_sha256']:fail('Resume state hash mismatch')
        state=read_json(state_file)
        if integer(state['round_id'],'state round')!=expected_rounds:fail('Resume commit round mismatch')
        for rel,sha in state['tracked_sha256'].items():
            p=Path(rel)
            if p.is_absolute() or '..' in p.parts:fail('Unsafe tracked metrics path')
            if checksum(run/p)!=sha:fail(f'Committed file changed: {rel}')
        hard.append('Resume state and tracked metric hashes valid')
    else:
        cautions.append('No wireless_resume/latest.json found; checkpoint commit not independently checked.')
    scenario_status=manifest.get('scenario_status')
    if perround[0]['weighted_discarded_kl']>1e6:
        cautions.append('Round-1 KL is very large (initial precision flooring); inspect later rounds separately.')
    if all(x['solver_certified_optimal']!='True' for x in perround):
        cautions.append('No round is certified globally integer-optimal; feasible numerical solver is expected.')

    def selected(r):
        x=perround[r-1];return dict(round=r, accuracy_pct=x['accuracy_pct'],nll=x['nll'],
            mean_retention=x['keep_mean'],weighted_kl=x['weighted_discarded_kl'],price_queries=x['price_queries'],
            array_gb=x['cumulative_array_bytes']/1e9,control_mb=x['cumulative_control_bytes']/1e6)
    head=[1,20,50,100,150]
    landmarks={str(i):selected(i) for i in head if i<=expected_rounds}
    best=max(perround,key=lambda x:x['accuracy_pct'])
    accuracy_sequence=[100*num(x['accuracy'],'accuracy') for x in gr]
    worst_drop=min(b-a for a,b in zip(accuracy_sequence,accuracy_sequence[1:]))
    summary_out={
        'status':'PASS_ENGINEERING_RECORD_INTEGRITY',
        'policy':'joint_only','run_name':run.name,'expected_rounds':expected_rounds,
        'configuration':{'dataset':cfg['data']['dataset'],'model':cfg['model']['name'],'d':d,
            'clients':K,'local_epochs':E,'batch_size':batch,'lr':train['lr'],'lr_schedule':train['lr_schedule'],
            'seed':cfg['runtime']['seed'],'bandwidth_hz':B,'round_target_s':target,
            'control_reserve_s':reserve,'wireless_channel_mode':joint['channel']['mode'],
            'scenario_status':scenario_status},
        'landmarks':landmarks,
        'accuracy':{'initial_pct':accuracy_sequence[0],'final_pct':accuracy_sequence[-1],
            'best_pct':best['accuracy_pct'],'best_round':best['round'],
            'last_20_mean_pct':statistics.mean(x['accuracy_pct'] for x in perround[-20:]),
            'worst_single_round_change_pp':worst_drop},
        'resource':{'cumulative_array_bytes':prev_array,'cumulative_control_bytes':prev_control,
            'cumulative_energy_j':prev_energy,'cumulative_steps':prev_steps,
            'cumulative_examples_seen':prev_examples,
            'total_price_queries':sum(x['price_queries'] for x in perround),
            'total_price_replies':sum(ev_round[r]['reply_price'] for r in ev_round),
            'max_control_seconds':max(x['modeled_control_s'] for x in perround),
            'max_airtime_overrun_seconds':max(x['max_upload_s']-x['upload_window_s'] for x in perround),
            'min_client_retention':min(x['keep_min'] for x in perround),
            'max_client_retention':max(x['keep_max'] for x in perround),
            'min_bandwidth_hz':min(x['bandwidth_min_hz'] for x in perround),
            'max_bandwidth_hz':max(x['bandwidth_max_hz'] for x in perround),
            'event_count':ev_count,'event_phase_counts':dict(event_phases)},
        'hard_checks':hard,'cautions':sorted(set(cautions)),
        'interpretation':'Internally consistent recorded execution, not proof of ideal RF, global integer optimality, or learning superiority.',
    }
    return summary_out,perround,perclient,files_for_export(run,manifest_path)


def files_for_export(run,manifest_path):
    paths=[manifest_path,run/'run_summary.json',run/'resolved_config.yaml',run/'source_config.yaml',
            run/'partition_metadata.json',run/'wireless/protocol.json',run/'wireless/layout_manifest.json',
            run/'metrics/global_metrics.csv',run/'metrics/wireless_rounds.csv',
            run/'metrics/wireless_clients.csv',run/'wireless/events.jsonl']
    paths.extend(sorted((run/'wireless/allocations').glob('round_*.json')))
    return [p for p in paths if p.is_file()]


def write_csv(path, records):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.cwd())
    ap.add_argument('--tag',default='cifar_eqb_joint150_fixed_v1')
    ap.add_argument('--run-dir',type=Path)
    ap.add_argument('--rounds',type=int,default=150)
    ap.add_argument('--output-dir',type=Path)
    args=ap.parse_args()
    root=args.root.resolve()
    if args.run_dir:
        run=args.run_dir.resolve();manifest_path=root/'scripts/configs/wireless'/args.tag/'study_manifest.json'
        manifest=read_json(manifest_path)
    else:
        run,manifest_path,manifest=select_run(root,args.tag)
    if not run.is_dir() or not run.is_relative_to(root/'outputs'):
        fail('Run must be a directory inside the selected repository outputs/')
    result,perround,perclient,source_files=audit(root,run,manifest_path,manifest,args.rounds)
    folder=args.output_dir or root.parent / f'bayesfl_joint_R{args.rounds}_audit_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    folder=folder.resolve()
    if folder.exists():fail(f'Audit output directory already exists: {folder}')
    folder.mkdir(parents=True)
    (folder/'joint_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    write_csv(folder/'joint_rounds_audit.csv',perround)
    write_csv(folder/'joint_clients_audit.csv',perclient)
    report=[f'# Joint FedKLS R{args.rounds} engineering audit','',
        f"Status: **{result['status']}**",f"Source: `{run}`",'',
        '## Learning checkpoints','',
        '| Round | Accuracy | NLL | Mean retention | Weighted discarded KL | Price queries |','|---:|---:|---:|---:|---:|---:|']
    for p in result['landmarks'].values():
        report.append(f"| {p['round']} | {p['accuracy_pct']:.3f}% | {p['nll']:.4f} | {p['mean_retention']:.3%} | {p['weighted_kl']:.6g} | {p['price_queries']} |")
    report+=['','## Key observations','',
        f"- Best recorded accuracy: {result['accuracy']['best_pct']:.3f}% at R{result['accuracy']['best_round']}.",
        f"- Final accuracy: {result['accuracy']['final_pct']:.3f}%.",
        f"- Mean accuracy across last 20 rounds: {result['accuracy']['last_20_mean_pct']:.3f}%.",
        f"- Worst one-round accuracy change: {result['accuracy']['worst_single_round_change_pp']:+.3f} pp; interpret with whole trajectory.",
        f"- Total price queries: {result['resource']['total_price_queries']}.",
        f"- Maximum modeled control use: {result['resource']['max_control_seconds']:.3f} seconds (reservation {result['configuration']['control_reserve_s']:.3f}).",
        f"- Control/model-array accounting is internally consistent across {args.rounds} rounds.",
        '', '## Cautions','']
    report.extend('- '+warning for warning in result['cautions'])
    report+=['','Source evidence is copied in the ZIP. Original repository files were never changed.',
             'This is a digital-link simulator audit, not physical radio measurement.']
    (folder/'JOINT_AUDIT_REPORT.md').write_text('\n'.join(report)+'\n')
    base=root
    hashes={}
    with zipfile.ZipFile(str(folder)+'.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in source_files:
            name=p.relative_to(base).as_posix()
            if name.endswith('.npz') or '/checkpoints/' in name or '/data/' in name:fail('Disallowed binary input')
            z.write(p,name)
            hashes[name]=checksum(p)
        for p in folder.iterdir():
            z.write(p,'audit/'+p.name)
        z.writestr('audit/SOURCE_HASHES.json',json.dumps(hashes,indent=2))
    print('PASS: JOINT R%d RECORD INTEGRITY AUDIT'%args.rounds)
    print('Run:',run)
    for point in result['landmarks'].values():
        print(f"R{point['round']}: accuracy={point['accuracy_pct']:.3f}%, NLL={point['nll']:.5f}, mean_keep={point['mean_retention']:.2%}, KL={point['weighted_kl']:.6g}")
    print('Price queries:',result['resource']['total_price_queries'])
    print('Max modeled control seconds:',result['resource']['max_control_seconds'])
    print('Cautions:',len(result['cautions']))
    print('Report:',folder/'JOINT_AUDIT_REPORT.md')
    print('Upload:',str(folder)+'.zip')
    print('No original results, checkpoints, datasets, configs, or docs were modified.')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('FAIL: '+str(exc),file=sys.stderr)
        sys.exit(1)
