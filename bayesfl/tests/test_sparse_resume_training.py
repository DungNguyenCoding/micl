import copy
import csv
import numpy as np
import pytest
from bayesfl.communication import payload_components
from bayesfl.experiment_state import RunState
from bayesfl.offline_smoke import run_offline, smoke_config


@pytest.mark.parametrize('rule', ['kl_global_local','kl_local_global','random'])
def test_r1_full_local_training_matches_dense(tmp_path,rule):
    dense=run_offline(smoke_config('fola_dense',rounds=2),tmp_path/'dense')
    sparse=run_offline(smoke_config('fola_sparse_'+rule,rounds=2,keep_ratio=1),tmp_path/'sparse')
    for a,b in zip(dense.current,sparse.current): np.testing.assert_array_equal(a,b)
    # d=37 -> 5 bitmap bytes, K=2, T=2.
    assert sparse.budget_used-dense.budget_used==20


@pytest.mark.parametrize('method', ['fedavg_dense','fola_dense','fola_sparse_kl_global_local',
                                    'fola_sparse_kl_local_global','fola_sparse_random'])
def test_clean_resume_matches_uninterrupted_training(tmp_path,method):
    cfg=smoke_config(method,rounds=3)
    continuous=run_offline(cfg,tmp_path/'continuous')
    one=copy.deepcopy(cfg); one.training.rounds=1
    first=run_offline(one,tmp_path/'resumed')
    prior_cost=first.budget_used
    resumed=run_offline(cfg,tmp_path/'resumed',resume=True)
    assert resumed.round_id==3 and resumed.budget_used==continuous.budget_used==3*prior_cost
    for a,b in zip(continuous.current,resumed.current): np.testing.assert_array_equal(a,b)
    assert resumed.cumulative_examples_seen==continuous.cumulative_examples_seen
    assert resumed.cumulative_local_steps==continuous.cumulative_local_steps
    with (tmp_path/'resumed/metrics/global_metrics.csv').open() as f:
        assert [int(row['round']) for row in csv.DictReader(f)]==[0,1,2,3]


def test_resume_refuses_to_discard_inflight_communication(tmp_path):
    cfg=smoke_config('fola_sparse_random',rounds=1)
    state=run_offline(cfg,tmp_path)
    state.ledger.record(round_id=2,client_id=0,phase='fit',direction='downlink',
                         components=payload_components(state.current,method='fola',tensor_count=state.layout.size))
    cfg.training.rounds=3
    with pytest.raises(ValueError,match='interrupted'):
        run_offline(cfg,tmp_path,resume=True)


def test_zero_omega_full_retention_keeps_baseline(tmp_path):
    a=run_offline(smoke_config('fola_dense',rounds=2,initial_precision=0),tmp_path/'a')
    b=run_offline(smoke_config('fola_sparse_kl_global_local',rounds=2,initial_precision=0,keep_ratio=1),tmp_path/'b')
    for x,y in zip(a.current,b.current):np.testing.assert_array_equal(x,y)


def test_changed_training_setting_rejected_on_resume(tmp_path):
    cfg=smoke_config('fola_sparse_random',rounds=1)
    run_offline(cfg,tmp_path)
    cfg.training.rounds=3; cfg.training.lr=.1
    with pytest.raises(ValueError,match='settings'):
        run_offline(cfg,tmp_path,resume=True)
