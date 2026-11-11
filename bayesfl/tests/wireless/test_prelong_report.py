"""Only verifies gate arithmetic and safe export; no training or network access."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts'
spec=importlib.util.spec_from_file_location('prelong_report',SCRIPTS/'report_cifar_prelong_gate.py')
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture_summary():
    ev=dict(accuracy=.38, nll=1.9, n_test=10000, paper_mean_only_eval=True)
    equal=dict(weighted_discarded_kl=51.4,actual_discarded_kl=51.4,
               modeled_control_seconds=.4,uplink_bytes=123,downlink_bytes=50,
               model_array_bytes=173,max_upload_seconds=58.0,price_queries=0)
    joint=dict(weighted_discarded_kl=1.42,actual_discarded_kl=1.42,
               modeled_control_seconds=5.0,uplink_bytes=115,downlink_bytes=50,
               model_array_bytes=165,max_upload_seconds=58.0,price_queries=35)
    return dict(source_checkpoint_round=10,compared_round=11,model='paper_basiccnn',d=878538,
        num_clients=20,examples=49988,same_checkpoint_and_local_posteriors=True,
        client_snapshots_cloned_before_upload=True,per_client_training_calls=[1]*20,
        actual_wire_packets_encoded_decoded=True,FOLA_aggregated_using_installed_protocol=True,
        actual_test_evaluation=True,
        evaluation={k:dict(ev) for k in ('source_r10','equal_r11','joint_r11','dense_oracle_r11')},
        historical_source_r10_evaluation=dict(ev),
        modeled_round_seconds=180.,upload_window_s=58.97274041844295,
        control_reserve_s=30.0,equal=equal,joint=joint,
        dense_oracle_is_not_feasible_wireless_baseline=True,
        scenario_is_example_not_locked=True)


def test_gate_passes_and_zips_only_whitelisted_files(tmp_path):
    gate=tmp_path/'gate'
    pair=gate/'paired_r11'
    pair.mkdir(parents=True)
    (gate/'logs').mkdir()
    (pair/'summary.json').write_text(json.dumps(fixture_summary()))
    (pair/'per_client.csv').write_text('policy,client\nequal,0\n')
    (gate/'logs'/'test.txt').write_text('test pass')
    # Keep secrets & checkpoints OUT OF THE portable ZIP even if in working folder.
    (gate/'PRIVATE_DATA.bin').write_bytes(b'SECRET')
    (pair/'scores.npz').write_bytes(b'SECRET SCORES')
    (pair/'global_checkpoint.npz').write_bytes(b'SECRET MODEL')
    checks=report.check_summary(fixture_summary())
    assert checks and all(checks.values())
    produced=report.produce(gate)
    assert produced['all_engineering_checks_passed']
    z=Path(str(gate)+'.zip')
    assert z.exists()
    with zipfile.ZipFile(z) as archive:
        names=archive.namelist()
        assert not any('PRIVATE' in x or 'scores.npz' in x or 'checkpoint.npz' in x for x in names)
        assert any(x.endswith('READINESS_REPORT.md') for x in names)
        assert any(x.endswith('per_client.csv') for x in names)
        assert archive.testzip() is None


def test_gate_rejects_unverified_accuracy_and_unfeasible_control():
    payload=fixture_summary()
    payload['actual_test_evaluation']=False
    payload['joint']['modeled_control_seconds']=32.
    checks=report.check_summary(payload)
    assert not checks['real_mean_only_test_evaluation']
    assert not checks['modeled_control_budget_both_policies']


def test_gate_rejects_controlled_run_without_20_training_calls():
    payload=fixture_summary()
    payload['per_client_training_calls']=[2]+[1]*19
    assert not report.check_summary(payload)['exactly_one_FOLA_training_per_client']
