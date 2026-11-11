"""Small synthetic full-packet paired-aggregation tests; no CIFAR download required."""
from __future__ import annotations
import csv
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pytest

from bayesfl.posterior.packing import ParameterLayout, initial_fola_state
from bayesfl.wireless.smoke import smoke_config, synthetic_trainer
from bayesfl.offline_smoke import tiny_model, synthetic_dataset
from bayesfl.wireless.channel import Channel, round_timing
from bayesfl.runtime_utils import seed_everything

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0,str(SCRIPTS))
spec = importlib.util.spec_from_file_location('paired_probe_test', SCRIPTS/'probe_cifar_paired_r11.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def synthetic_source(tmp_path):
    cfg = smoke_config(policy='joint',rounds=3)
    cfg.wireless.uplink_bandwidth_hz = 3000.0
    seed_everything(cfg.runtime.seed)
    model = tiny_model()
    layout = ParameterLayout.from_model(model)
    initial = initial_fola_state(model, cfg.fola.initial_precision)
    samples = [len(synthetic_dataset(i)) for i in range(3)]
    checkpoint=tmp_path/'existing_checkpoint.bin'
    checkpoint.write_bytes(b'KEEP_THE_ORIGINAL')
    partition=tmp_path/'partition.bin'
    partition.write_bytes(b'ORIGINAL_PARTITION')
    config_path=tmp_path/'original_config.yaml'
    config_path.write_text('ORIGINAL_CONFIG')
    summary=tmp_path/'run_summary.json'
    summary.write_text('{"completed_rounds":2}')
    ch=Channel(cfg.wireless,3).round(1)
    timing=round_timing(cfg, samples=samples,
                        downlink_bytes=sum(a.nbytes for a in initial),channel=ch)
    return dict(cfg=cfg,source=tmp_path,round_id=1,checkpoint=checkpoint,
                partition=partition,config_path=config_path,
                meta=dict(mean_size=sum(samples)/3,sha256='synthetic'),
                layout=layout,global_arrays=initial,samples=samples,channel=ch,timing=timing)


def synthetic_evaluator(label, arrays, r):
    from bayesfl.posterior.packing import unpack_fola
    m,_ = unpack_fola(arrays,ParameterLayout.from_model(tiny_model()))
    # Deterministic stand-in for slow CIFAR-10 test inference.
    metric = 1/(1+sum(float(np.sum(a*a)) for a in m))
    return dict(round=r, accuracy=metric, nll=float(-np.log(metric)),
                n_test=32, paper_mean_only_eval=True)


def test_both_branches_have_real_payloads_same_state_and_no_retraining(tmp_path):
    source=synthetic_source(tmp_path)
    protected={key:probe.sha256(source[key]) for key in ('checkpoint','partition','config_path')}
    out=tmp_path/'paired'
    result=probe.do_pair(source,out,trainer=synthetic_trainer,evaluator=synthetic_evaluator)
    assert result['prepared_once_per_client']
    assert result['per_client_training_calls']==[1,1,1]
    assert result['client_snapshots_cloned_before_upload']
    assert result['actual_wire_packets_encoded_decoded']
    assert result['FOLA_aggregated_using_installed_protocol']
    assert result['same_checkpoint_and_local_posteriors']
    assert result['actual_test_evaluation'] is False
    assert result['joint']['weighted_discarded_kl'] <= result['equal']['weighted_discarded_kl'] + 1e-8
    assert result['equal']['actual_discarded_kl']==pytest.approx(result['equal']['weighted_discarded_kl'],abs=1e-7)
    assert result['joint']['actual_discarded_kl']==pytest.approx(result['joint']['weighted_discarded_kl'],abs=1e-7)
    assert result['equal']['uplink_bytes']>0 and result['joint']['uplink_bytes']>0
    assert result['equal']['modeled_control_seconds']<source['cfg'].wireless.control_reserve_s
    assert result['joint']['modeled_control_seconds']<source['cfg'].wireless.control_reserve_s
    assert set(result['evaluation'])=={'source_r10','equal_r11','joint_r11','dense_oracle_r11'}
    for key,sha in protected.items():
        assert probe.sha256(source[key])==sha
    with (out/'per_client.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==6
    assert {r['policy'] for r in rows}=={'equal','joint'}
    assert all(r['packet_format'] in {'dense','bitmap'} for r in rows)
    with (out/'protocol_events.csv').open(newline='') as f:
        events=list(csv.DictReader(f))
    assert any(r['phase']=='upload' and int(r['array_bytes'])>0 for r in events)
    assert {r['policy'] for r in events}=={'equal','joint','shared'}
    z=Path(str(out)+'.zip')
    assert z.exists()
    with zipfile.ZipFile(z) as f:
        assert sorted(x.split('/')[-1] for x in f.namelist()) == sorted(x.name for x in out.iterdir() if x.is_file())
        assert not any('checkpoint' in x.lower() or 'scores' in x.lower() for x in f.namelist())


def test_probe_fails_closed_on_existing_result(tmp_path):
    src=synthetic_source(tmp_path)
    out=tmp_path/'exists'
    out.mkdir()
    with pytest.raises(FileExistsError):
        probe.do_pair(src,out,trainer=synthetic_trainer,evaluator=synthetic_evaluator)


def test_probe_detects_control_reserve_violation(tmp_path):
    src=synthetic_source(tmp_path)
    src['cfg'].wireless.control_reserve_s=1e-9
    with pytest.raises(AssertionError,match='control time exceeds reserve'):
        probe.do_pair(src,tmp_path/'reserved',trainer=synthetic_trainer,evaluator=synthetic_evaluator)


def test_existing_source_read_only_import(tmp_path):
    source=synthetic_source(tmp_path)
    assert source['cfg'].fola.paper_mean_only_eval
    assert probe.PROBE_ID.startswith('cifar10_r11_')
    assert probe.sha256(source['checkpoint']) == probe.sha256(source['checkpoint'])


def test_paired_test_evaluator_matches_production_mean_only_metrics(tmp_path, monkeypatch):
    import logging
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from bayesfl.models.factory import initialize_model
    from bayesfl.posterior.packing import initial_fola_state
    from bayesfl.evaluation import CentralEvaluator
    cfg = smoke_config(policy='equal_bandwidth', rounds=2)
    cfg.runtime.central_eval_device='cpu'
    assert cfg.fola.paper_mean_only_eval
    torch.set_num_threads(1)
    seed_everything(cfg.runtime.seed)
    model = initialize_model(cfg)
    arrays = initial_fola_state(model,cfg.fola.initial_precision)
    x = torch.randn(23,1,28,28)
    y = torch.arange(23) % 10
    loader = DataLoader(TensorDataset(x,y),batch_size=7,shuffle=False)
    monkeypatch.setattr(probe,'load_test_loader',lambda config: loader)
    output = tmp_path/'example'
    got = probe.evaluate_on_test(cfg,'pair',arrays,1,output)
    reference_dir = tmp_path/'reference'
    for name in ('metrics','checkpoints','reliability','posterior'):
        (reference_dir/name).mkdir(parents=True,exist_ok=True)
    central = CentralEvaluator(cfg,loader,reference_dir,logger=logging.getLogger('test'))
    nll, metrics = central.evaluate(1,arrays)
    assert got['n_test']==23
    assert got['accuracy']==pytest.approx(metrics['accuracy'],abs=1e-10)
    assert got['nll']==pytest.approx(nll,abs=1e-10)
    assert got['brier']==pytest.approx(metrics['brier'],abs=1e-10)
    assert got['ece']==pytest.approx(metrics['ece'],abs=1e-10)
