#!/usr/bin/env python3
"""Read-only CIFAR10 joint sensitivity six-case check; never trains or writes."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml

sys.dont_write_bytecode = True
ROOT = Path.cwd().resolve()
BASE = ROOT / 'scripts/configs/wireless/cifar10_scheduling_pilot_v1/scenario_fixed.yaml'
FOLDER = ROOT / 'scripts/configs/wireless/cifar10_joint_sensitivity6_v1'
ORIGINAL_HASH = '61fa0744708a75a5a5218f095a48a27bcbc07dba94b5bc264dec574a96a759cb'
EXPECTED = [('u170',170.,1000000.),('u160',160.,1000000.),('u150',150.,1000000.),
            ('b0550',180.,550000.),('b0750',180.,750000.),('b0950',180.,950000.)]


def main():
    if not BASE.is_file(): raise SystemExit('ERROR: Missing original pinned scenario')
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == ORIGINAL_HASH, 'Original scenario hash changed; stop'
    base = yaml.safe_load(BASE.read_text())
    topology = ROOT / 'scripts/configs/wireless/cifar10_scheduling_pilot_v1/topology.json'
    assert topology.is_file(), 'Missing pinned topology'
    original_topology_hash = hashlib.sha256(topology.read_bytes()).hexdigest()
    assert hashlib.sha256((FOLDER/'topology.json').read_bytes()).hexdigest() == original_topology_hash
    manifest = json.loads((FOLDER/'CASE_MANIFEST.json').read_text())
    assert manifest['original_scenario_sha256'] == ORIGINAL_HASH
    assert manifest['original_topology_sha256'] == original_topology_hash
    checker = ROOT / 'scripts/check_cifar_wireless_scenario_v1.py'
    if not checker.is_file(): raise SystemExit('ERROR: Install the original CIFAR scenario checker first')
    print('JOINT CIFAR SENSITIVITY PREFLIGHT — no training or file writes', flush=True)
    for name, target, b in EXPECTED:
        file = FOLDER / (name+'.yaml')
        conf = yaml.safe_load(file.read_text())
        assert conf['status'] == 'EXAMPLE_NOT_LOCKED'
        assert conf['scenario_id'] == 'cifar10_joint_sensitivity6_v1_'+name
        assert hashlib.sha256(file.read_bytes()).hexdigest()==next(row['yaml_sha256'] for row in manifest['cases'] if row['case']==name)
        # The *only* actual wireless changes from the source must be the named axis.
        normalized = copy.deepcopy(conf['wireless'])
        assert normalized['round_target_s'] == target
        assert normalized['uplink_bandwidth_hz'] == b
        normalized['round_target_s'] = base['wireless']['round_target_s']
        normalized['uplink_bandwidth_hz'] = base['wireless']['uplink_bandwidth_hz']
        assert normalized == base['wireless'], f'Unexpected wireless change in {name}'
        assert b <= base['wireless']['uplink_bandwidth_hz'], 'Above-cap PSD handling not included'
        result=subprocess.run([sys.executable,'-B',str(checker),'--scenario',str(file)],cwd=ROOT,
                              text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if result.returncode:
            print(result.stdout)
            raise SystemExit(f'ERROR: Source-backed feasibility check failed for {name}')
        window=[x for x in result.stdout.splitlines() if x.startswith('Available upload window:')]
        status='Joint minimum-payload feasibility: PASS' in result.stdout
        if len(window)!=1 or not status:raise SystemExit(f'ERROR: Missing feasibility details for {name}')
        print(f'PASS {name:5s}  B={b/1e6:.2f}MHz  T_round={target:.0f}s  {window[0]}',flush=True)
    print('PASS: SIX SCENARIOS VERIFIED, fixed PSD/caps and all other configuration fields unchanged.',flush=True)
    print('No training was started. Proceed to short smoke checks before long runs.',flush=True)


if __name__=='__main__':main()
