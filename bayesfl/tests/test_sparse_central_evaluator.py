import csv
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from bayesfl.evaluation import CentralEvaluator
from bayesfl.experiment_state import RunState, load_resume_bundle
from bayesfl.models.factory import initialize_model
from bayesfl.posterior.packing import ParameterLayout, initial_fola_state
from bayesfl.offline_smoke import smoke_config


def test_actual_mlp_central_evaluator_records_bytes_and_commit(tmp_path):
    torch.set_num_threads(1)
    cfg=smoke_config('fola_sparse_kl_global_local',rounds=3,budget=0)
    cfg.model.name='mlp_784_500_300_10'
    cfg.data.num_classes=10
    model=initialize_model(cfg)
    layout=ParameterLayout.from_model(model)
    arrays=initial_fola_state(model,1)
    state=RunState(cfg,layout,arrays,tmp_path,partition_sha256='synthetic-eval')
    for name in ('metrics','reliability','checkpoints','posterior'):(tmp_path/name).mkdir(exist_ok=True)
    x=torch.zeros(5,1,28,28)
    loader=DataLoader(TensorDataset(x,torch.arange(5)),batch_size=2)
    evaluator=CentralEvaluator(cfg,loader,tmp_path,logger=logging.getLogger('test'),state=state)
    loss,metrics=evaluator.evaluate(0,arrays)
    assert np.isfinite(loss) and metrics['accuracy']==metrics['fola_mean_accuracy']
    with (tmp_path/'metrics/global_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==1 and rows[0]['cumulative_all_array_bytes']=='0'
    assert rows[0]['stop_reason']=='communication_budget'
    assert (tmp_path/'checkpoints/global_round_0000.npz').exists()
    saved,restored=load_resume_bundle(tmp_path)
    assert saved['round_id']==0
    for a,b in zip(arrays,restored):np.testing.assert_array_equal(a,b)
