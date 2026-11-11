#!/usr/bin/env python3
"""Read-only CIFAR-10 resource preflight. No fitting, datasets, queues or file writes.

Uses the installed production config/channel/payload/allocation functions and the
pinned legacy partition. An optional Rayleigh scan reports failures; it never
redraws, clips, repairs, or exports a modified trace.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml
from bayesfl.config import load_config
from bayesfl.data.partition import load_partition, _hash_partitions
from bayesfl.wireless.allocation import InfeasibleAllocation, allocate
from bayesfl.wireless.channel import Channel, round_timing
from bayesfl.wireless.config import parse_wireless
from bayesfl.wireless.oracle import Response
from bayesfl.wireless.payload import payload_bits

PROFILE = 'cifar10_a0p01_f1'
D = 878_538
PARTITION = 'outputs/partitions/cifar10_paper_dirichlet_a0.01_n20_seed0.npz'
PARTITION_HASH = '914fe035f7b504b0059ca85b5a6a194bd41027fb2cfe2765ee777142d68280f2'
DEFAULT = 'scripts/configs/wireless/cifar10_scheduling_pilot_v1/scenario_fixed.yaml'


class CountsOnly:
    """No local posterior is needed to check equal-share capacity.

    Zero tails are placeholders, not estimated CIFAR KL scores. Joint price
    queries are deliberately unsupported in this read-only resource preflight.
    """
    def counts(self, counts):
        return [Response(int(m), 0.0) for m in counts]

    def price(self, *args):
        raise RuntimeError('This preflight must not claim a current-round KL allocation')


def primary(root: Path, registry: dict, role: str):
    rows = [r for r in registry['runs'] if r.get('primary_figure')
            and r['profile'] == PROFILE and r['role'] == role]
    if role == 'kl_sparse':
        rows = [r for r in rows if r['keep_ratio'] == 0.5]
    if len(rows) != 1:
        raise ValueError(f'Cannot uniquely resolve {role}')
    directory = root / rows[0]['relative_output']
    return directory, load_config(directory / 'resolved_config.yaml')


def load_inputs(root: Path, scenario_path: Path):
    registry = json.loads((root / 'docs/EXPERIMENT_REGISTRY.json').read_text())
    directory, cfg = primary(root, registry, 'dense_fola')
    _, sparse = primary(root, registry, 'kl_sparse')
    if cfg.data.dataset != 'cifar10' or cfg.model.name != 'paper_basiccnn':
        raise ValueError('Wrong locked dataset/model')
    if cfg.runtime.seed != 0 or cfg.federation.num_clients != 20 or cfg.federation.clients_per_round != 20:
        raise ValueError('This pilot requires the selected seed-0, 20-client learning profile')
    if cfg.training.local_epochs != 10 or cfg.training.batch_size != 32:
        raise ValueError('Locked E/B changed; review before reusing this pilot')
    if cfg.data.partition.get('dirichlet_alpha') != 0.01 or cfg.data.partition.get('local_data_fraction') != 1.0:
        raise ValueError('Locked data partition settings changed')
    layout = json.loads((directory / 'communication/layout_manifest.json').read_text())
    if layout.get('d') != D:
        raise ValueError('Saved model dimension differs from this pilot')
    parts = load_partition(root / PARTITION)
    meta = json.loads((directory / 'partition_metadata.json').read_text())
    digest = _hash_partitions(parts)
    if digest != meta['sha256'] or digest != PARTITION_HASH:
        raise ValueError('Partition differs from the selected legacy experiment; do not regenerate/overwrite it')
    if len(parts) != 20 or any(len(p) <= 0 for p in parts):
        raise ValueError('All 20 clients need positive realized sample counts')
    raw = yaml.safe_load(scenario_path.read_text())
    cfg.compression = copy.deepcopy(sparse.compression)
    cfg.compression.selection_rule = 'kl_global_local'
    cfg.compression.keep_ratio = 1.0  # unused placeholder; integer grants determine payloads
    cfg.communication.downlink_mode = 'multicast'
    cfg.communication.deterministic_client_schedule = True
    cfg.communication.max_communication_bytes = None
    cfg.output.save_full_client_posteriors = False
    cfg.wireless = parse_wireless(raw['wireless'])
    cfg.wireless.policy = 'equal_bandwidth'
    cfg.validate()
    topo = json.loads((scenario_path.parent / 'topology.json').read_text())
    if len(topo['clients']) != 20:
        raise ValueError('Topology length mismatch')
    for i, item in enumerate(topo['clients']):
        expected = 10**(-(128.1 + 37.6*math.log10(item['distance_m']/1000))/10)
        if item['client_id'] != i or not 0 < item['distance_m'] <= topo['radius_m']:
            raise ValueError('Invalid geometry')
        if not math.isclose(item['power_gain'], expected, rel_tol=1e-12):
            raise ValueError('Topology does not implement the documented LTE path-loss formula')
        if not math.isclose(cfg.wireless.channel.path_gains[i], expected, rel_tol=1e-12):
            raise ValueError('Scenario gains differ from the pinned topology')
    return cfg, [len(p) for p in parts], topo, digest, raw


def evaluate(cfg, samples, round_id):
    w = cfg.wireless
    ch = Channel(w, len(samples)).round(round_id)
    timing = round_timing(cfg, samples=samples, downlink_bytes=8*D, channel=ch)
    weights = [n/sum(samples) for n in samples]
    allocation = allocate(CountsOnly(), d=D, weights=weights,
        window_s=timing['upload_window_seconds'], efficiencies=ch['efficiencies'],
        bandwidth_hz=w.uplink_bandwidth_hz, policy='equal_bandwidth')
    control_capacity = min(min(ch['efficiencies'])*w.uplink_bandwidth_hz,
                           ch['downlink_efficiency']*w.downlink_bandwidth_hz)
    if w.control_rate_bps > control_capacity:
        raise InfeasibleAllocation('Assumed serialized full-band control rate exceeds the weakest ideal link')
    return ch, timing, allocation, control_capacity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--scenario', type=Path, default=Path(DEFAULT))
    p.add_argument('--rayleigh-scan-rounds', type=int, default=0,
        help='Optional diagnostic ONLY; does not change the fixed scenario or choose a favorable seed')
    a = p.parse_args()
    root = a.root.expanduser().resolve()
    if not (root/'src/bayesfl/main.py').is_file():
        p.error('Run from the BayesFL root or pass --root')
    if a.rayleigh_scan_rounds < 0:
        p.error('--rayleigh-scan-rounds must be nonnegative')
    scenario = a.scenario.expanduser()
    if not scenario.is_absolute(): scenario = root/scenario
    cfg, ns, topo, digest, raw = load_inputs(root, scenario)
    if cfg.wireless.channel.mode != 'fixed':
        raise ValueError('This first pilot preflight requires fixed gains; audit fading separately')
    ch, timing, allocation, ctrl_cap = evaluate(cfg, ns, 1)
    w = cfg.wireless; cs = ch['efficiencies']; U = timing['upload_window_seconds']
    ratios = [m/D for m in allocation.counts]
    print('='*100)
    print('CIFAR-10 WIRELESS RESOURCE PREFLIGHT — no training or file writes')
    print('='*100)
    print('Scenario status:', raw.get('status'))
    print('Scenario SHA256:', hashlib.sha256(scenario.read_bytes()).hexdigest())
    print('Partition SHA256:', digest)
    print(f'Clients=20 | d={D:,} | realized samples min/mean/max={min(ns)}/{sum(ns)/20:.1f}/{max(ns)}')
    print(f'Distances min/max: {min(c["distance_m"] for c in topo["clients"]):.6f}/{max(c["distance_m"] for c in topo["clients"]):.6f} m')
    print(f'Noise PSD: {w.channel.noise_psd_w_hz:.12g} W/Hz')
    print(f'UL PSD/client: {w.channel.uplink_psd_w_hz[0]:.12g} W/Hz')
    print(f'Equal bandwidth: {allocation.bandwidth_hz[0]:.3f} Hz/client')
    print(f'Power at equal bandwidth: {w.channel.uplink_psd_w_hz[0]*allocation.bandwidth_hz[0]:.12g} W/client')
    print(f'UL spectral efficiency min/max: {min(cs):.6f}/{max(cs):.6f} bit/s/Hz')
    print(f'Multicast downlink: {timing["downlink_seconds"]:.9f} s')
    print(f'Modeled preparation barrier: {max(timing["preparation_seconds"]):.9f} s (ASSUMED device model)')
    print(f'Control reservation: {w.control_reserve_s:.6f} s; rate={w.control_rate_bps:.0f} bit/s (ASSUMED)')
    print(f'Weakest ideal full-band control capacity: {ctrl_cap:.3f} bit/s (control cost itself not certified)')
    print(f'Available upload window: {U:.9f} s | slot={w.round_target_s:.6f} s')
    print(f'Bitmap: {(D+7)//8:,} bytes | minimum sparse upload={payload_bits(1,D)//8:,} bytes')
    print(f'Dense posterior upload: {payload_bits(D,D)//8:,} bytes')
    print('\nCID  Samples  Distance(m)   UL SNR(dB)       Count      Keep   UL power(mW)')
    for i,(n,c,m,b) in enumerate(zip(ns,topo['clients'],allocation.counts,allocation.bandwidth_hz)):
        required = payload_bits(m,D)/U/cs[i]
        if required>b or m<1 or m>D: raise AssertionError('Invalid production equal-share allocation')
        if w.channel.uplink_psd_w_hz[i]*b>w.channel.client_power_cap_w[i]:
            raise AssertionError('Client cap exceeded')
        print(f'{i:3d} {n:8d} {c["distance_m"]:12.4f} {10*math.log10(ch["uplink_snr"][i]):12.4f} {m:11d} {m/D:9.5f} {1e3*w.channel.uplink_psd_w_hz[i]*b:14.6f}')
    print(f'\nEqual-share keep min/mean/max: {min(ratios):.6f}/{sum(ratios)/20:.6f}/{max(ratios):.6f}')
    print(f'Dense-format clients: {sum(m==D for m in allocation.counts)}/20')
    round_bytes = 8*D+sum(payload_bits(m,D)//8 for m in allocation.counts)
    print(f'Equal-share modeled array bytes/round: {round_bytes:,}; control counted separately')
    threshold = timing['downlink_seconds']+max(timing['preparation_seconds'])+w.control_reserve_s+sum(payload_bits(D,D)/w.uplink_bandwidth_hz/c for c in cs)
    print(f'All-dense FOLA minimum fixed-slot duration with optimally divided bandwidth: {threshold:.6f} s')
    print('All-dense FOLA feasible at this slot:', w.round_target_s>=threshold)
    print('Joint minimum-payload feasibility: PASS (the returned equal-share counts are already feasible).')
    print('No current-round CIFAR KL objective or learning accuracy was estimated.')
    if a.rayleigh_scan_rounds:
        trial=copy.deepcopy(cfg);trial.wireless.channel.mode='block_rayleigh'
        failures=[]
        for r in range(1,a.rayleigh_scan_rounds+1):
            try: evaluate(trial,ns,r)
            except InfeasibleAllocation as exc: failures.append((r,str(exc)))
        print(f'\nRAYLEIGH DIAGNOSTIC ONLY: {len(failures)}/{a.rayleigh_scan_rounds} tested rounds fail at least one equal-share/control assumption.')
        for r,err in failures[:10]: print(f'  R{r}: {err}')
        print('No fading configuration was written; no channel was redrawn, clipped or repaired.')
        print('Do not treat a zero-failure finite scan as a guarantee for future fading realizations.')
    print('\nPASS: fixed-gain resource preflight. Real CIFAR/Flower/CUDA smoke and observed control usage remain required.')


if __name__=='__main__':
    try: main()
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        raise SystemExit(f'ERROR: {exc}') from exc
