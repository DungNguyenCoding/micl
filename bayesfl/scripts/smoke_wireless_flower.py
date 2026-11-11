#!/usr/bin/env python3
"""Synthetic CPU real Flower/Ray lifecycle gate; no dataset or paper results."""
import argparse,json,logging
from pathlib import Path
import torch
from bayesfl.wireless.smoke import smoke_config,synthetic_trainer
from bayesfl.offline_smoke import tiny_model,synthetic_dataset
from bayesfl.posterior.packing import ParameterLayout,initial_fola_state,unpack_fola,ndarrays_to_model
from bayesfl.logging_utils import CsvRecorder,save_resolved_config,save_environment
from bayesfl.runtime_utils import seed_everything
from bayesfl.wireless.state import WirelessState,run_lock
from bayesfl.wireless.channel import Channel
from bayesfl.wireless.runner import run_flower
from bayesfl.wireless.reporting import inspect_run,print_results

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=2);a=p.parse_args()
    if not 1<=a.rounds<=3:p.error('This smoke is limited to 1-3 rounds')
    import flwr,ray
    if flwr.__version__!='1.29.0':raise RuntimeError('Use the pinned Flower 1.29.0 environment for this gate')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    cfg=smoke_config(rounds=a.rounds);cfg.wireless.backend='flower'
    cfg.runtime.client_num_cpus=1;cfg.runtime.client_num_gpus=0
    torch.set_num_threads(1);seed_everything(cfg.runtime.seed)
    model=tiny_model();layout=ParameterLayout.from_model(model)
    save_resolved_config(cfg,a.output_dir);save_environment(a.output_dir)
    samples=[len(synthetic_dataset(i)) for i in range(3)]
    with run_lock(a.output_dir):
        state=WirelessState(cfg,layout,initial_fola_state(model,1),a.output_dir,partition_sha256='synthetic-wireless-v2')
        recorder=CsvRecorder(a.output_dir/'metrics/global_metrics.csv')
        class Evaluator:
            def evaluate(self,r,arrays):
                ndarrays_to_model(model,unpack_fola(arrays,layout)[0]);model.eval()
                with torch.no_grad():
                    x,y=synthetic_dataset(20).tensors;logits=model(x)
                    loss=float(torch.nn.functional.cross_entropy(logits,y))
                    acc=float((logits.argmax(1)==y).float().mean())
                metrics=dict(round=r,accuracy=acc,nll=loss,fola_mean_accuracy=acc,global_accuracy=acc)
                recorder.append(dict(metrics,**state.evaluation_fields()));state.save_resume(metrics)
                return loss,{'accuracy':acc}
        run_flower(cfg,state,Channel(cfg.wireless,3),Evaluator(),samples,'synthetic',{'mean_size':7},
                   logging.getLogger('wireless-flower-smoke'),trainer=synthetic_trainer)
    result=inspect_run(a.output_dir)
    if result['completed_rounds']!=a.rounds:raise RuntimeError('Flower smoke did not reach its required endpoint')
    print_results([result])
    print('PASS: synthetic real Flower/Ray multi-phase smoke; not MNIST/CIFAR performance or RF measurement')
if __name__=='__main__':main()
