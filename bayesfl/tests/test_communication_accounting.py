import copy
import numpy as np
import pytest
from bayesfl.communication import CommunicationLedger, core_round_bytes, payload_components
from bayesfl.offline_smoke import run_offline, smoke_config


def test_reference_costs():
    assert core_round_bytes(4,1,sparse=True,m=2)==dict(down_bytes=32,up_bytes=17,total_bytes=49)
    assert core_round_bytes(4,1)['total_bytes']==64
    assert core_round_bytes(1_000_000,10)['total_bytes']==160_000_000
    assert core_round_bytes(1_000_000,10,sparse=True,m=500_000)['total_bytes']==121_250_000
    assert core_round_bytes(1_000_000,10,sparse=True,m=100_000)['total_bytes']==89_250_000


def test_event_counts_not_weighted_rejection_and_extras(tmp_path):
    log=CommunicationLedger(tmp_path/'events.jsonl',run_id='r',method='fola_dense',seed=0)
    comp=payload_components([np.ones(4,np.float32)],method='fedavg')
    for cid in [0,1]:
        log.record(round_id=1,client_id=cid,phase='fit',direction='downlink',components=comp)
        mid=log.record(round_id=1,client_id=cid,phase='fit',direction='uplink',components=comp)
    log.reject(mid,'malformed posterior')
    assert log.totals['train_total_array_bytes']==64
    for phase in ['evaluate','initialize']:
        log.record(round_id=1,client_id=0,phase=phase,direction='downlink',components=comp)
    assert log.totals['all_array_bytes']==96
    assert log.totals['evaluation_array_bytes']==log.totals['initialization_array_bytes']==16
    with pytest.raises(ValueError):
        log.record(round_id=1,client_id=0,phase='fit',direction='downlink',components=comp)
    assert log.totals['all_array_bytes']==96
    loaded=CommunicationLedger(log.path,run_id='r',method='fola_dense',seed=0,restore=True)
    loaded.assert_checkpoint(log.state())


def test_full_retention_cost_has_bitmap_overhead():
    assert core_round_bytes(9,3,sparse=True,m=9)['total_bytes']==core_round_bytes(9,3)['total_bytes']+6


@pytest.mark.parametrize('budget,rounds', [(0,0),(905,0),(906,1),(1811,1),(1812,2)])
def test_strict_budget_never_dispatches_unaffordable_round(tmp_path,budget,rounds):
    cfg=smoke_config('fola_sparse_random',rounds=10,budget=budget)
    state=run_offline(cfg,tmp_path)
    assert state.round_id==rounds
    assert state.ledger.totals['all_array_bytes']==rounds*906<=budget
    assert state.stop_reason()=='communication_budget'


def test_round_cap_is_separate_from_byte_budget(tmp_path):
    cfg=smoke_config('fola_sparse_random',rounds=1,budget=10_000)
    state=run_offline(cfg,tmp_path)
    assert state.stop_reason()=='max_rounds' and state.round_id==1


def test_undecodable_payload_marks_lower_bound_and_keeps_serialized_bytes(tmp_path):
    from bayesfl.communication import CommunicationLedger
    ledger=CommunicationLedger(tmp_path/'events.jsonl',run_id='test',method='fola_dense',seed=0)
    ledger.record_undecodable(round_id=1,client_id=0,serialized_tensor_bytes=123,reason='malformed npy')
    assert ledger.unmeasured_array_messages==1
    assert ledger.serialized_tensor_bytes==123
    replay=CommunicationLedger(tmp_path/'events.jsonl',run_id='test',method='fola_dense',seed=0,restore=True)
    assert replay.state()==ledger.state()
