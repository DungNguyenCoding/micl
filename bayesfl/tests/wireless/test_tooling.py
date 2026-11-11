"""Study generation, trace handling and portable exports; no GPU/data download."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import pytest
import yaml
from bayesfl.config import load_config
from bayesfl.wireless.smoke import run_smoke, smoke_config
from bayesfl.wireless.reporting import inspect_run
from bayesfl.wireless.channel import Channel

ROOT=Path(__file__).resolve().parents[2]


def test_fixed_trace_and_missing_entries(tmp_path):
    cfg=smoke_config(rounds=2);cfg.wireless.channel.mode='trace'
    path=tmp_path/'trace.csv';cfg.wireless.channel.trace_csv=str(path)
    lines=['round,client_id,uplink_gain,downlink_gain']
    for r in (1,2):
        for cid in range(3):lines.append(f'{r},{cid},1e-9,2e-9')
    path.write_text('\n'.join(lines)+'\n');cfg.validate()
    ch=Channel(cfg.wireless,3)
    assert ch.trace_sha256==hashlib.sha256(path.read_bytes()).hexdigest()
    assert ch.round(1)==ch.round(2)
    with pytest.raises(ValueError,match='Missing channel'):ch.round(3)
    path.write_text(path.read_text()+'1,0,1e-9,2e-9\n')
    with pytest.raises(ValueError,match='duplicate'):Channel(cfg.wireless,3)


def test_wireless_report_rejects_changed_cost(tmp_path):
    run_smoke(smoke_config(rounds=2),tmp_path/'run')
    rec=inspect_run(tmp_path/'run',time_budget_s=1.)
    assert rec['selected_round']<=2
    with pytest.raises(ValueError,match='outside'):inspect_run(tmp_path/'run',time_budget_s=10000.)
    file=tmp_path/'run/metrics/wireless_clients.csv'
    with file.open(newline='') as f:rows=list(csv.DictReader(f));names=list(rows[0])
    rows[0]['upload_array_bytes']=str(int(rows[0]['upload_array_bytes'])+1)
    with file.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='byte mismatch'):inspect_run(tmp_path/'run')


def test_wireless_only_handoff_export(tmp_path):
    repo=tmp_path/'bayesfl';repo.mkdir()
    for name in ('src','scripts','tests','docs'): (repo/name).mkdir()
    shutil.copytree(ROOT/'src/bayesfl',repo/'src/bayesfl',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('_run_gpu_queue.sh','plot_ieee_multicast_results_mnist_s10_stage22.py'):
        shutil.copy2(ROOT/'scripts'/name,repo/'scripts'/name)
    (repo/'docs/CHAT_HANDOFF.md').write_text('Test snapshot\n')
    (repo/'outputs').mkdir()
    run=repo/'outputs/smoke_wireless_export';run_smoke(smoke_config(rounds=2),run)
    command=[sys.executable,str(ROOT/'docs/tools/export_handoff.py'),'--root',str(repo),
             '--wireless-only','--wireless-run-dir',str(run),'--export']
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    archives=list(tmp_path.glob('bayesfl_LIVE_HANDOFF_*.zip'));assert len(archives)==1
    with zipfile.ZipFile(archives[0]) as z:
        index=json.loads(z.read('HANDOFF_FILE_HASHES.json'))
        for entry in index:assert hashlib.sha256(z.read(entry['path'])).hexdigest()==entry['sha256']
        capture=json.loads(z.read('HANDOFF_CAPTURE.json'))
        assert len(capture['wireless_endpoints'])==1 and not capture['primary_endpoints']
        assert not any('/wireless_resume/' in n or n.endswith('events.jsonl') for n in z.namelist())
        assert any(n.endswith('metrics/wireless_clients.csv') for n in z.namelist())


@pytest.mark.parametrize('profile',['mnist_s10','cifar10_a0p01_f1'])
def test_generator_preserves_locked_learning(tmp_path,monkeypatch,profile):
    # A source package may omit primary metrics; only the three actual saved configs are needed here.
    reg=json.loads((ROOT/'docs/EXPERIMENT_REGISTRY.json').read_text())
    selected=[r for r in reg['runs'] if r['primary_figure'] and r['profile']==profile]
    if any(not (ROOT/r['relative_output']/'resolved_config.yaml').is_file() for r in selected):
        pytest.skip('The minimal source copy has no saved primary configurations')
    (tmp_path/'src/bayesfl').mkdir(parents=True);(tmp_path/'src/bayesfl/main.py').write_text('')
    (tmp_path/'docs').mkdir();(tmp_path/'docs/EXPERIMENT_REGISTRY.json').write_text(json.dumps(reg))
    for r in selected:
        target=tmp_path/r['relative_output'];target.mkdir(parents=True)
        shutil.copy2(ROOT/r['relative_output']/'resolved_config.yaml',target/'resolved_config.yaml')
    spec=importlib.util.spec_from_file_location('wireless_study_generator_test',ROOT/'scripts/prepare_wireless_study.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys,'argv',['prepare_wireless_study.py','--profile',profile,'--scenario',
        str(ROOT/'scripts/configs/wireless/scenario_example.yaml'),'--tag','test_run','--rounds','2',
        '--backend','local','--smoke','--allow-example'])
    module.main()
    manifest=json.loads((tmp_path/'scripts/configs/wireless/test_run/study_manifest.json').read_text())
    assert len(manifest['jobs'])==6
    for job in manifest['jobs']:
        old=load_config(tmp_path/job['source_resolved_config']).to_dict()
        new=load_config(tmp_path/job['config']).to_dict()
        for section in ('data','federation','model','fola','runtime'):
            assert old[section]==new[section]
        old['training']['rounds']=2
        assert old['training']==new['training']
    assert sum(len(p.read_text().splitlines()) for p in (tmp_path/'scripts/queues').glob('*.txt'))==6
    with pytest.raises(FileExistsError):module.main()
