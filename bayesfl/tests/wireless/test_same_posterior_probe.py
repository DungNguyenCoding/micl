"""Fast isolated verification of same-state queries; no dataset or checkpoint needed."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
from bayesfl.posterior.packing import ParameterLayout, initial_fola_state
from bayesfl.runtime_utils import seed_everything
from bayesfl.wireless.smoke import smoke_config, synthetic_trainer
from bayesfl.offline_smoke import tiny_model, synthetic_dataset
from bayesfl.wireless.channel import Channel, round_timing
from bayesfl.wireless.transport import LocalTransport
from bayesfl.wireless.protocol import binding


def _module():
    path = Path(__file__).resolve().parents[2] / 'scripts' / 'probe_cifar_same_posterior.py'
    spec = importlib.util.spec_from_file_location('same_posterior_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_allocations_share_one_preparation_and_no_state_arrays_uplink():
    probe = _module()
    cfg = smoke_config(policy='joint', rounds=2)
    cfg.wireless.uplink_bandwidth_hz = 3000
    seed_everything(cfg.runtime.seed)
    model = tiny_model()
    layout = ParameterLayout.from_model(model)
    initial = initial_fola_state(model,cfg.fola.initial_precision)
    samples=[len(synthetic_dataset(i)) for i in range(3)]
    calls=[0]*3
    def counted(*args):
        calls[args[1]]+=1
        return synthetic_trainer(*args)
    transport=LocalTransport(cfg,'synthetic',sum(samples)/3,trainer=counted)
    bindings=[binding(initial,layout,round_id=1,client_id=i,run_token='test-frozen',posterior=True) for i in range(3)]
    common=probe.QueryRecorder(transport,bindings,samples,cfg,1)
    result=common.exchange('prepare',parameters=initial)
    assert all(r.metrics['local_steps']>0 for r in result)
    assert calls==[1,1,1]
    ch=Channel(cfg.wireless,3).round(1)
    timing=round_timing(cfg,samples=samples,downlink_bytes=sum(a.nbytes for a in initial),channel=ch)
    eq,joint,req,rjoint=probe.compare_frozen(transport,bindings,samples,cfg,layout,ch,timing,1)
    assert calls==[1,1,1]
    assert len(eq.counts)==len(joint.counts)==3
    assert joint.objective<=eq.objective+1e-9
    assert req.count_requests==1 and req.price_requests==0
    assert rjoint.count_requests>=1
    assert rjoint.price_requests==joint.price_queries
    assert common.control_bytes>0
    assert rjoint.control_bytes>0
    info1=probe.info_for_allocation(eq,req,dict(layout=layout,global_arrays=initial,samples=samples),cfg,ch,timing)
    info2=probe.info_for_allocation(joint,rjoint,dict(layout=layout,global_arrays=initial,samples=samples),cfg,ch,timing)
    assert info1['model_array_bytes']>0 and info2['model_array_bytes']>0


def test_query_recorder_rejects_bad_binding():
    probe=_module()
    cfg=smoke_config(policy='joint')
    class BadReply:
        arrays=[]
        num_examples=1
        metrics={'count':1,'discarded_kl':0.0}
    class BadTransport:
        def exchange(self,arrays, instructions, round_id):
            return {i:BadReply() for i in range(3)}
    b=[dict(protocol='x',client_id=i) for i in range(3)]
    rec=probe.QueryRecorder(BadTransport(),b,[1,1,1],cfg,1)
    try:
        rec.counts([1,1,1])
    except ValueError as e:
        assert 'binding' in str(e)
    else:
        raise AssertionError('Unbound oracle reply was accepted')


def test_full_result_writer_mock_preparation(tmp_path):
    probe=_module()
    cfg=smoke_config(policy='joint',rounds=2)
    seed_everything(cfg.runtime.seed)
    model=tiny_model()
    layout=ParameterLayout.from_model(model)
    initial=initial_fola_state(model,cfg.fola.initial_precision)
    samples=[len(synthetic_dataset(i)) for i in range(3)]
    ckpt=tmp_path/'dummy.npz';ckpt.write_bytes(b'diagnostic source hash only')
    cf=tmp_path/'cfg.yaml';cf.write_text('synthetic-test')
    part=tmp_path/'part.npz';part.write_bytes(b'synthetic-test')
    ch=Channel(cfg.wireless,3).round(1)
    timing=round_timing(cfg,samples=samples,downlink_bytes=sum(a.nbytes for a in initial),channel=ch)
    source=dict(cfg=cfg,source=tmp_path,round_id=1,checkpoint=ckpt,config_path=cf,
                partition=part,meta=dict(mean_size=sum(samples)/3,sha256='toy'),
                layout=layout,global_arrays=initial,samples=samples,channel=ch,timing=timing)
    old=probe.train_local
    try:
        probe.train_local=synthetic_trainer
        out=tmp_path/'new_output'
        result=probe.execute(source,out)
    finally:
        probe.train_local=old
    assert result['preparation_calls_each_client']==[1,1,1]
    assert result['global_training_rounds_executed']==0
    assert result['compared_on_same_prepared_scores']
    assert result['joint']['objective_weighted_discarded_kl']<=result['equal']['objective_weighted_discarded_kl']+1e-9
    assert (out/'summary.json').is_file()
    assert (out/'per_client.csv').is_file()
    assert (out/'allocation_query_events.csv').is_file()
    assert not (out/'scores.npz').exists()
