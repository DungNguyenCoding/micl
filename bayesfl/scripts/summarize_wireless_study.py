#!/usr/bin/env python3
"""Resolve exact study jobs and print completed wireless results; never choose best/latest silently."""
import argparse,json
from pathlib import Path
from bayesfl.config import load_config
from bayesfl.wireless.state import fingerprint
from bayesfl.wireless.reporting import inspect_run,print_results

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study-dir',type=Path,required=True)
    p.add_argument('--time-budget-seconds',type=float);a=p.parse_args()
    manifest=json.loads((a.study_dir/'study_manifest.json').read_text());results=[]
    for job in manifest['jobs']:
        cfg=load_config(job['config'])
        matches=sorted(Path(cfg.output.outputs_dir).glob(job['run_name']+'_*'))
        if len(matches)!=1:raise ValueError(f"Expected exactly one output for {job['run_name']}; found {len(matches)}")
        actual=load_config(matches[0]/'resolved_config.yaml')
        if fingerprint(actual)!=job['fingerprint']:raise ValueError('Study/run fingerprint mismatch')
        r=inspect_run(matches[0],time_budget_s=a.time_budget_seconds)
        if a.time_budget_seconds is None:
            cap=cfg.wireless.total_time_budget_s
            done=r['completed_rounds']>=cfg.training.rounds or (cap is not None and (r['completed_rounds']+1)*cfg.wireless.round_target_s>cap)
            if not done:raise ValueError(f"Incomplete job: {job['run_name']}")
        results.append(r)
    print_results(results)
if __name__=='__main__':main()
