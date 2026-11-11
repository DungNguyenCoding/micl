from __future__ import annotations
import copy
import json
import numpy as np
import pytest
from bayesfl.wireless.smoke import smoke_config, run_smoke
from bayesfl.wireless.state import fingerprint
from bayesfl.config import load_config
from bayesfl.wireless.channel import Channel


@pytest.mark.parametrize('policy,selection,method',[
 ('joint','kl_global_local','fola'),('joint','random','fola'),
 ('equal_bandwidth','kl_global_local','fola'),('uniform','kl_global_local','fola'),
 ('dense','dense','fola'),('dense','dense','fedavg')])
def test_rounds_and_resume(tmp_path,policy,selection,method):
    cfg=smoke_config(policy,selection,method,rounds=3)
    full=run_smoke(cfg,tmp_path/'full')
    short=copy.deepcopy(cfg);short.training.rounds=1
    first=run_smoke(short,tmp_path/'split')
    resumed=run_smoke(cfg,tmp_path/'split',resume=True)
    assert full.round_id==resumed.round_id==3
    assert all(np.array_equal(a,b) for a,b in zip(full.current,resumed.current))
    assert full.total_bytes==resumed.total_bytes
    assert full.elapsed_s==pytest.approx(3)
    assert full.failure is None


def test_partial_ledger_refuses_resume(tmp_path):
    cfg=smoke_config(rounds=1)
    s=run_smoke(cfg,tmp_path/'run')
    with s.events.open('a') as f:f.write('{}\n')
    cfg.training.rounds=2
    with pytest.raises(ValueError,match='Uncommitted/corrupt'):
        run_smoke(cfg,tmp_path/'run',resume=True)


def test_modified_metrics_refuses_resume(tmp_path):
    cfg=smoke_config(rounds=1);run_smoke(cfg,tmp_path/'run')
    with (tmp_path/'run/metrics/global_metrics.csv').open('a') as f:f.write('\n')
    cfg.training.rounds=2
    with pytest.raises(ValueError,match='Metrics differ'):
        run_smoke(cfg,tmp_path/'run',resume=True)


def test_resume_fingerprint():
    cfg=smoke_config();h=fingerprint(cfg)
    cfg.training.rounds=200;cfg.wireless.total_time_budget_s=1000
    assert fingerprint(cfg)==h
    cfg.wireless.round_target_s=3
    assert fingerprint(cfg)!=h


def test_channel_repeat_and_rng_independence():
    cfg=smoke_config();cfg.wireless.channel.mode='block_rayleigh'
    ch=Channel(cfg.wireless,3)
    np.random.seed(45);expected=np.random.random()
    np.random.seed(45);first=ch.round(4);assert np.random.random()==expected
    assert ch.round(4)==first and ch.round(5)!=first


def test_legacy_serialization_and_typo(tmp_path):
    import yaml
    cfg=smoke_config();raw=cfg.to_dict();raw.pop('wireless')
    p=tmp_path/'legacy.yaml';p.write_text(yaml.safe_dump(raw))
    old=load_config(p)
    assert 'wireless' not in old.to_dict() and not old.wireless_enabled
    raw['wireless']={'enabled':True,'bandwidth_Hz':123}
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError,match='Unknown WirelessConfig'):
        load_config(p)


def test_time_budget_stops_whole_round(tmp_path):
    cfg=smoke_config(rounds=10);cfg.wireless.total_time_budget_s=2.5
    s=run_smoke(cfg,tmp_path/'time')
    assert s.round_id==2 and s.stop_reason()=='modeled_time_budget'


def test_control_failure_preserves_spent_resources(tmp_path):
    cfg=smoke_config();cfg.wireless.control_reserve_s=1e-9
    with pytest.raises(ValueError):run_smoke(cfg,tmp_path/'fail')
    summary=json.loads((tmp_path/'fail/run_summary.json').read_text())
    assert summary['stop_reason']=='failed' and summary['cumulative_all_array_bytes']>0
    assert summary['pending_round']==1
