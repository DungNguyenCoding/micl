#!/usr/bin/env python3
"""Clone locked learning profiles into explicitly configured wireless studies.

Writes NEW configs and two serial GPU queues, never launches training.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
import yaml
from bayesfl.config import load_config
from bayesfl.wireless.config import parse_wireless
from bayesfl.wireless.state import fingerprint
from bayesfl.wireless.reporting import inspect_run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['mnist_s10','cifar10_a0p01_f1'],required=True)
    p.add_argument('--scenario',type=Path,required=True)
    p.add_argument('--tag',required=True)
    p.add_argument('--rounds',type=int,required=True)
    p.add_argument('--backend',choices=['flower','local'],default='flower')
    p.add_argument('--allow-example',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Explicit <=3 round runtime check, smoke_ output prefix')
    p.add_argument('--prepare-partition',action='store_true')
    p.add_argument('--policies',nargs='+',choices=['joint','joint_random','equal_bandwidth','uniform','dense_fola','fedavg'],
                   default=['joint','joint_random','equal_bandwidth','uniform','dense_fola','fedavg'])
    a=p.parse_args();root=Path.cwd().resolve()
    if not (root/'src/bayesfl/main.py').is_file():p.error('Run from the bayesfl root')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',a.tag):p.error('tag must contain letters, digits, _ or -')
    if a.rounds<1 or (a.smoke and a.rounds>3):p.error('rounds must be positive; smoke is limited to 3 rounds')
    if len(set(a.policies))!=len(a.policies):p.error('Duplicate policies')
    scenario=yaml.safe_load(a.scenario.read_text())
    if not isinstance(scenario, dict):p.error('Scenario must be a YAML mapping')
    if scenario.get('status')=='EXAMPLE_NOT_LOCKED' and not a.allow_example:
        p.error('This scenario is an engineering example, not a locked study. Explicit --allow-example is required.')
    if not isinstance(scenario, dict) or 'wireless' not in scenario:p.error('Scenario requires a wireless mapping')
    reg=json.loads((root/'docs/EXPERIMENT_REGISTRY.json').read_text())
    selected=[r for r in reg['runs'] if r['primary_figure'] and r['profile']==a.profile]
    def source(role):
        candidates=[r for r in selected if r['role']==role]
        if role=='kl_sparse':candidates=[r for r in candidates if r['keep_ratio']==.5]
        if len(candidates)!=1:raise ValueError(f'Cannot uniquely resolve {role}')
        path=root/candidates[0]['relative_output']/'resolved_config.yaml'
        return path,load_config(path)
    fpath,fola=source('dense_fola');apath,fedavg=source('fedavg_baseline');spath,sparse=source('kl_sparse')
    for section in ('data','federation','model'):
        if fola.to_dict()[section]!=fedavg.to_dict()[section]:
            raise ValueError(f'Locked baselines differ in {section}; inspect instead of silently harmonizing')
    parent=root/'scripts/configs/wireless'/a.tag
    if parent.exists():raise FileExistsError(f'Study directory exists: {parent}; use its manifest, not a regenerated queue')
    qdir=root/'scripts/queues'
    for gpu in (0,1):
        q=qdir/f'wireless_{a.tag}_gpu{gpu}.txt'
        if q.exists():raise FileExistsError(f'Refusing to overwrite queue: {q}')
    jobs=[];configs=[];queues=[[],[]]
    for idx,policy in enumerate(a.policies):
        cfg=copy.deepcopy(fedavg if policy=='fedavg' else fola)
        source_path=apath if policy=='fedavg' else fpath
        cfg.run_name=f"{'smoke' if a.smoke else 'run'}_wireless_{a.tag}_{a.profile}_{policy}_seed{cfg.runtime.seed}"
        cfg.training.rounds=a.rounds
        cfg.communication.max_communication_bytes=None
        cfg.communication.downlink_mode='multicast';cfg.communication.deterministic_client_schedule=True
        cfg.output.save_full_client_posteriors=False
        cfg.compression.keep_ratio=1.0
        if policy in ('dense_fola','fedavg'):
            cfg.compression.selection_rule='dense'
        else:
            cfg.compression=copy.deepcopy(sparse.compression)
            cfg.compression.keep_ratio=1.0
            cfg.compression.selection_rule='random' if policy=='joint_random' else 'kl_global_local'
        cfg.wireless=parse_wireless(copy.deepcopy(scenario['wireless']))
        cfg.wireless.enabled=True;cfg.wireless.backend=a.backend
        cfg.wireless.policy='joint' if policy=='joint_random' else 'dense' if policy in ('dense_fola','fedavg') else policy
        if cfg.wireless.channel.mode=='trace':
            trace=Path(cfg.wireless.channel.trace_csv).expanduser()
            if not trace.is_absolute():trace=a.scenario.resolve().parent/trace
            cfg.wireless.channel.trace_csv=str(trace.resolve())
        cfg.validate()
        name=cfg.run_name+'.yaml';relative=f'scripts/configs/wireless/{a.tag}/{name}'
        matches=sorted(Path(cfg.output.outputs_dir).glob(cfg.run_name+'_*'))
        if len(matches)>1:raise ValueError(f'Ambiguous existing outputs for {cfg.run_name}; resolve exact run explicitly')
        command=relative;status='NEW'
        if matches:
            previous=load_config(matches[0]/'resolved_config.yaml')
            if fingerprint(previous)!=fingerprint(cfg):raise ValueError('Existing run has different learning/wireless settings')
            record=inspect_run(matches[0])
            t=cfg.wireless.total_time_budget_s
            if record['completed_rounds']>=cfg.training.rounds or (t is not None and (record['completed_rounds']+1)*cfg.wireless.round_target_s>t):
                status='SKIP_COMPLETE';command=None
            else:
                status='RESUME'
                command=f"WRESUME|{matches[0].resolve()}|{cfg.training.rounds}|{t if t is not None else 'none'}"
        if command:queues[idx%2].append(command)
        configs.append((name,cfg))
        jobs.append(dict(run_name=cfg.run_name, config=relative, gpu=idx%2,status=status,
                         source_resolved_config=str(source_path.relative_to(root)),
                         source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                         fingerprint=fingerprint(cfg), expected_round_cap=cfg.training.rounds))
        print(f'{status:14} GPU{idx%2} {cfg.run_name}')
    # All configs validated before any files are written.
    if a.prepare_partition:
        from bayesfl.data.datasets import prepare_partition
        from bayesfl.data.partition import load_partition,_hash_partitions
        partition,meta=prepare_partition(configs[0][1])
        if _hash_partitions(load_partition(partition))!=meta['sha256']:raise ValueError('Partition content hash mismatch')
        print(f'PARTITION: {partition} | sha256={meta["sha256"]} | min={min(map(len,load_partition(partition)))} max={max(map(len,load_partition(partition)))}')
    parent.mkdir(parents=True,exist_ok=False)
    for name,cfg in configs:
        path=parent/name;path.write_text(yaml.safe_dump(cfg.to_dict(),sort_keys=False))
        load_config(path)
    qdir=root/'scripts/queues';qdir.mkdir(exist_ok=True)
    for gpu,lines in enumerate(queues):
        q=qdir/f'wireless_{a.tag}_gpu{gpu}.txt'
        if q.exists():raise FileExistsError(f'Refusing to overwrite queue: {q}')
        q.write_text('\n'.join(lines)+ ('\n' if lines else ''))
        print(f'QUEUE GPU{gpu}: {len(lines)} jobs | {q.relative_to(root)}')
    (parent/'study_manifest.json').write_text(json.dumps(dict(schema='wireless_study_v2',profile=a.profile,
        scenario_source=str(a.scenario.resolve()),scenario_sha256=hashlib.sha256(a.scenario.read_bytes()).hexdigest(),
        scenario_status=scenario.get('status','USER_SPECIFIED_NOT_INDEPENDENTLY_VALIDATED'),jobs=jobs),indent=2)+'\n')
    print('PASS: configs and queues created; no training started. Dense baselines may be infeasible under this same deadline; no policy is silently relaxed.')
    print(f'Terminal summary: python scripts/summarize_wireless_study.py --study-dir scripts/configs/wireless/{a.tag}')

if __name__=='__main__':main()
