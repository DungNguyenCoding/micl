"""Real current MNIST CNN and CentralEvaluator on synthetic image tensors."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset
import yaml
from bayesfl.wireless.smoke import smoke_config
from bayesfl.wireless.runner import run_experiment
from bayesfl.wireless.reporting import inspect_run
from bayesfl.data.partition import _hash_partitions


def test_real_cnn_local_runner_and_evaluated_resume(tmp_path,monkeypatch):
    import bayesfl.wireless.runner as runner
    import bayesfl.data.datasets as datasets
    torch.set_num_threads(1)
    cfg=smoke_config(rounds=1)
    cfg.model.name='mnist_cnn_16_32_128';cfg.training.local_epochs=1
    cfg.wireless.uplink_bandwidth_hz=1e6;cfg.wireless.downlink_bandwidth_hz=1e9
    cfg.output.outputs_dir=str(tmp_path/'outputs');cfg.output.logs_dir=str(tmp_path/'logs')
    cfg.wireless.control_reserve_s=.1
    cfg.validate()
    parts=[np.arange(0,5),np.arange(5,12),np.arange(12,21)]
    partition=tmp_path/'partition.npz';np.savez(partition,**{f'client_{i:04d}':a for i,a in enumerate(parts)})
    metadata=dict(sha256=_hash_partitions(parts),mean_size=7.,num_clients=3)
    def data(cid):
        generator=torch.Generator().manual_seed(123+cid)
        return TensorDataset(torch.randn(5+2*cid,1,28,28,generator=generator),torch.arange(5+2*cid)%10)
    def loader(cfg,path,cid,shuffle_seed):
        ds=data(cid)
        return DataLoader(ds,batch_size=cfg.training.batch_size,shuffle=True,
                          generator=torch.Generator().manual_seed(shuffle_seed)),len(ds)
    monkeypatch.setattr(runner,'prepare_partition',lambda cfg:(partition,metadata))
    monkeypatch.setattr(datasets,'load_client_loader',loader)
    monkeypatch.setattr(runner,'load_test_loader',lambda cfg:DataLoader(data(5),batch_size=8))
    # Enforce CPU for this author's test without changing production device policy.
    import bayesfl.runtime_utils as runtime
    monkeypatch.setattr(runtime,'resolve_device',lambda _:torch.device('cpu'))
    source=tmp_path/'source.yaml';source.write_text(yaml.safe_dump(cfg.to_dict()))
    directory=run_experiment(cfg,config_path=source)
    assert inspect_run(directory)['completed_rounds']==1
    cfg.training.rounds=2
    directory=run_experiment(cfg,config_path=source,resume=directory)
    assert inspect_run(directory)['completed_rounds']==2
    assert (directory/'checkpoints/global_round_0002.npz').is_file()
