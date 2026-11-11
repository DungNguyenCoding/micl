#!/usr/bin/env python3
"""Explicit new wireless figures; never overwrite the legacy Acc_Com figures."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bayesfl.wireless.reporting import inspect_run, rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,action='append',required=True)
    p.add_argument('--label',action='append')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.label and len(a.label)!=len(a.run_dir):p.error('One label per run is required')
    records=[inspect_run(d) for d in a.run_dir]
    trajectories=[rows(d/'metrics/global_metrics.csv') for d in a.run_dir]
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for col,xlabel,filename,scale in [
        ('cumulative_modeled_time_seconds','Modeled elapsed time (s)','accuracy_vs_modeled_time',1.),
        ('cumulative_all_array_bytes','Model-array communication (GB)','accuracy_vs_array_bytes',1e-9)]:
        fig,ax=plt.subplots(figsize=(6.5,4.2))
        for i,(r,tr) in enumerate(zip(records,trajectories)):
            lab=a.label[i] if a.label else f"{r['method']} {r['policy']} {r['selection_rule']}"
            key='fola_mean_accuracy' if r['method']=='fola' else 'accuracy'
            ax.plot([float(v[col])*scale for v in tr],[100*float(v[key]) for v in tr],label=lab)
        ax.set_xlabel(xlabel);ax.set_ylabel('Global test accuracy (%)')
        ax.grid(True,alpha=.3);ax.legend(fontsize=8);fig.tight_layout()
        for extension in ('png','pdf'):
            path=a.output_dir/(filename+'.'+extension);fig.savefig(path,dpi=300);print(path)
        plt.close(fig)
    print('Raw observed evaluation points; no smoothing or extrapolation. Modeled time is not measured runtime.')
if __name__=='__main__':main()
