#!/usr/bin/env python
"""Clone locked configurations into a separate, paired sparsification study.

No retuning; original YAMLs are never edited. The only intentional differences
are method/communication controls, explicit run names, seed (when requested),
and the common total-round safety cap. LR decay horizons are unchanged.
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
from bayesfl.models.factory import initialize_model
from bayesfl.posterior.packing import ParameterLayout, model_to_ndarrays
from bayesfl.posterior.sparse import dimension, keep_count
from bayesfl.communication import core_round_bytes


def generate(base_fola: Path, base_fedavg: Path | None, out_dir: Path, *, tag: str,
             keep_ratios=(.5,), rounds: int | None=None, dense_budget_rounds: int | None=None,
             max_communication_bytes: int | None=None, precision_policy='strict', seed: int | None=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', tag):
        raise ValueError('tag must use only letters, digits, underscores and hyphens')
    if len(set(keep_ratios)) != len(keep_ratios):
        raise ValueError('Duplicate keep ratios would overwrite a study condition')
    fola=load_config(base_fola)
    if fola.method!='fola' or fola.sparse_enabled:
        raise ValueError('--base-fola must be an existing dense FOLA configuration')
    fedavg=load_config(base_fedavg) if base_fedavg else None
    if fedavg is not None:
        if fedavg.method!='fedavg': raise ValueError('--base-fedavg must select FedAvg')
        for field in ('data','model','federation'):
            if getattr(fedavg,field)!=getattr(fola,field):
                raise ValueError(f'Paired baseline {field} settings differ')
        if seed is None and fedavg.runtime.seed!=fola.runtime.seed:
            raise ValueError('Paired baseline seeds differ')
        if (fedavg.training.local_epochs,fedavg.training.batch_size)!=(fola.training.local_epochs,fola.training.batch_size):
            raise ValueError('Paired local epochs/batch sizes differ')
    if dense_budget_rounds is not None and max_communication_bytes is not None:
        raise ValueError('Choose one byte-budget specification')
    for cfg in [fola]+([fedavg] if fedavg else []):
        if cfg.training.local_epochs>10:
            raise ValueError('The current study preserves the user E<=10 compute limit')
    model=initialize_model(fola)
    d=dimension(ParameterLayout.from_model(model)); k=fola.federation.clients_per_round
    budget=max_communication_bytes
    if dense_budget_rounds is not None:
        if dense_budget_rounds<1: raise ValueError('dense_budget_rounds must be positive')
        budget=core_round_bytes(d,k)['total_bytes']*dense_budget_rounds
    selected_seed=fola.runtime.seed if seed is None else seed
    out_dir=Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    for target in (out_dir/'queue.txt', out_dir/'study_manifest.json'):
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
    candidates=[('fola_dense',1.0,fola)]
    if fedavg: candidates.insert(0,('fedavg_dense',1.0,fedavg))
    for ratio in keep_ratios:
        keep_count(d,ratio)
        for rule in ('kl_global_local','kl_local_global','random'):
            candidates.append(('fola_sparse_'+rule,ratio,fola))
    records=[]
    for method,ratio,base in candidates:
        cfg=copy.deepcopy(base)
        cfg.method=method
        cfg.compression.selection_rule='dense'
        cfg.compression.keep_ratio=ratio
        cfg.compression.precision_policy=precision_policy
        cfg.compression.covariance_representation='precision'
        cfg.runtime.seed=selected_seed
        cfg.communication.deterministic_client_schedule=True
        cfg.communication.max_communication_bytes=budget
        cfg.communication.budget_metric='cumulative_all_array_bytes'
        if rounds is not None: cfg.training.rounds=rounds
        cfg.output.save_full_client_posteriors=False
        ratio_tag=('_keep'+format(ratio,'.8g').replace('.','p')) if 'sparse_' in method else ''
        cfg.run_name=f'sparse_{tag}_{method}{ratio_tag}_seed{selected_seed}'
        cfg.validate()
        path=out_dir/(cfg.run_name+'.yaml')
        if path.exists(): raise FileExistsError(f'Refusing to overwrite {path}')
        # Validate all candidates before writing any to avoid a partial study.
        m=keep_count(d,ratio)
        round_cost=(core_round_bytes(d,k,sparse=True,m=m)['total_bytes'] if cfg.sparse_enabled else
                    core_round_bytes(d,k)['total_bytes'] if cfg.method=='fola' else
                    2*k*sum(a.nbytes for a in model_to_ndarrays(initialize_model(cfg))))
        records.append((cfg,path,dict(method=cfg.method_id,keep_ratio=ratio,config=str(path),
                                      round_array_bytes=round_cost,
                                      affordable_rounds=None if budget is None else min(cfg.training.rounds,budget//round_cost))))
    if len({path for _,path,_ in records}) != len(records):
        raise ValueError('Keep ratios collide in formatted filenames; choose distinct tags/ratios')
    for cfg,path,_ in records:
        path.write_text(yaml.safe_dump(cfg.to_dict(),sort_keys=False),encoding='utf-8')
    queue=out_dir/'queue.txt'
    if queue.exists(): raise FileExistsError(f'Refusing to overwrite {queue}')
    queue.write_text(''.join(str(path)+'\n' for _,path,_ in records))
    manifest=dict(tag=tag,d=d,clients_per_round=k,seed=selected_seed,max_communication_bytes=budget,
                  covariance_representation='raw_precision',precision_policy=precision_policy,
                  bases=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                         for p in [base_fola,base_fedavg] if p],
                  inherited_lr_schedule_caveat='FOLA versus FedAvg keeps original locked LR/schedule differences; the three sparse rules share the FOLA trainer.',
                  runs=[record for _,_,record in records])
    (out_dir/'study_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-fola',type=Path,required=True)
    parser.add_argument('--base-fedavg',type=Path)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--keep-ratios',type=float,nargs='+',default=[.5])
    parser.add_argument('--rounds',type=int,help='Total-round safety cap; not an LR schedule horizon')
    parser.add_argument('--dense-budget-rounds',type=int,help='Common byte budget equal to this many original dense FOLA rounds')
    parser.add_argument('--max-communication-bytes',type=int)
    parser.add_argument('--precision-policy',choices=['strict','floor_for_score'],default='strict')
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    manifest=generate(args.base_fola,args.base_fedavg,args.out_dir,tag=args.tag,
                      keep_ratios=args.keep_ratios,rounds=args.rounds,dense_budget_rounds=args.dense_budget_rounds,
                      max_communication_bytes=args.max_communication_bytes,
                      precision_policy=args.precision_policy,seed=args.seed)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
