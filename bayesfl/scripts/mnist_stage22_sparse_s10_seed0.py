#!/usr/bin/env python3
"""Queue exactly 10 sparse FOLA runs matched to dense R150."""
from pathlib import Path
import copy, csv, json
import yaml
from bayesfl.config import load_config
from bayesfl.models.factory import initialize_model
from bayesfl.data.datasets import prepare_partition, partition_stem
from bayesfl.data.partition import load_partition
from bayesfl.communication import core_round_bytes
from bayesfl.posterior.sparse import keep_count

ROOT = Path.cwd().resolve()
SRC = ROOT / (
    'scripts/configs/mnist_stage18_fola_noclip_cnn/'
    'run_mnist_stage18_cnn16_32_fola_noclip_'
    'fixed1c_s10_n20_seed0_e5_b10_lr0p10_lam0p01_r200.yaml'
)
DEST = ROOT / 'scripts/configs/mnist_stage22_sparse_cnn_s10_seed0'
QUEUES = ROOT / 'scripts/queues'
RATIOS = [('0p75', .75), ('0p50', .50), ('0p25', .25),
          ('0p10', .10), ('0p05', .05)]
RULES = [(0, 'kl_global_local'), (1, 'random')]

assert SRC.is_file(), f'Missing dense FOLA template: {SRC}'
base = load_config(SRC)
assert (base.method, base.model.name, base.runtime.seed) == (
    'fola', 'mnist_cnn_16_32_128', 0)
assert base.data.dataset == 'mnist'
assert base.data.partition == dict(type='fixed_labels',
    samples_per_client=10, labels_per_client=1)
assert (base.federation.num_clients, base.federation.clients_per_round) == (20,20)
assert (base.training.local_epochs, base.training.batch_size, base.training.lr) == (5,10,.1)
assert (base.training.optimizer, base.training.lr_schedule) == ('sgd','constant')
assert base.training.grad_clip_norm is None
assert base.training.momentum == 0 and base.training.weight_decay == 0
assert base.fola.mode == 'paper_reference' and base.fola.prior_lambda == .01
assert base.fola.initial_precision > 0
assert base.communication.downlink_mode == 'multicast'
assert base.compression.selection_rule == 'dense'

p, metadata = prepare_partition(base)  # cache the partition before both GPUs launch
indices = load_partition(p)
assert len(indices) == 20 and all(len(a) == 10 for a in indices)
assert len({int(i) for a in indices for i in a}) == 200

d = sum(p.numel() for p in initialize_model(base).parameters())
k = base.federation.clients_per_round
assert (d,k) == (80202,20)
dense_cost = core_round_bytes(d,k,method='fola',downlink_mode='multicast')['total_bytes']
BUDGET = 150*dense_cost
assert BUDGET == 2021090400

DEST.mkdir(parents=True,exist_ok=True)
QUEUES.mkdir(parents=True,exist_ok=True)
jobs = {0: [], 1: []}
manifest = []

for label, ratio in RATIOS:
    m = keep_count(d,ratio)
    cost = core_round_bytes(d,k,sparse=True,m=m,method='fola',
                            downlink_mode='multicast')['total_bytes']
    rounds = BUDGET // cost
    assert rounds*cost <= BUDGET < (rounds+1)*cost
    for gpu, rule in RULES:
        name = (f'run_mnist_stage22_cnn_s10_n20_seed0_fola_'
                f'{rule}_keep{label}_budget150_r{rounds}')
        conf = copy.deepcopy(base.to_dict())
        conf['run_name'] = name
        conf['training']['rounds'] = rounds
        conf['compression'].update(selection_rule=rule,keep_ratio=ratio,
            covariance_representation='precision', precision_policy='strict',
            score_precision_floor=None)
        conf['communication'].update(downlink_mode='multicast',
            deterministic_client_schedule=True, max_communication_bytes=BUDGET,
            budget_metric='cumulative_all_array_bytes')
        conf['output']['save_full_client_posteriors'] = False
        f = DEST / f'{name}.yaml'
        if f.exists():
            assert yaml.safe_load(f.read_text()) == conf, f'Config conflict: {f}'
        else:
            f.write_text(yaml.safe_dump(conf,sort_keys=False))
        valid = load_config(f)
        assert valid.to_dict() == conf and valid.sparse_enabled
        assert valid.method_id == f'fola_sparse_{rule}'
        assert partition_stem(valid) == partition_stem(base)
        # Don't restart an existing partially completed experiment.
        dirs = [ROOT/'outputs'/name, *(ROOT/'outputs').glob(name+'_*')]
        dirs = [x for x in dirs if x.is_dir()]
        complete = False
        if dirs:
            for run in dirs:
                resolved = run/'resolved_config.yaml'
                assert resolved.is_file(), f'Existing output needs inspection: {run}'
                assert load_config(resolved).to_dict() == valid.to_dict(), run
                metrics = run/'metrics/global_metrics.csv'
                assert metrics.is_file(), f'Existing run is partial: {run}'
                with metrics.open(newline='') as handle:
                    rows = {int(float(row['round'])): row
                            for row in csv.DictReader(handle) if row.get('round')}
                if all(r in rows for r in range(1,rounds+1)) and (
                    int(float(rows[rounds]['cumulative_all_array_bytes'])) == rounds*cost):
                    complete = True
                    break
            assert complete, f'Existing run partial; resume instead: {dirs}'
        if not complete:
            jobs[gpu].append(f.relative_to(ROOT).as_posix())
        manifest.append(dict(method=rule,keep_ratio=ratio,coordinates=m,
            rounds=rounds,round_bytes=cost,total_bytes=rounds*cost,
            leftover_bytes=BUDGET-rounds*cost,gpu=gpu,skipped_completed=complete))
        print(f'GPU{gpu}  {rule:15s}  keep={ratio:.2f}  R={rounds:4d}'
              f'  {"SKIP COMPLETE" if complete else "QUEUE"}')

for gpu, rule in RULES:
    q = QUEUES/f'mnist_stage22_cnn_s10_n20_seed0_gpu{gpu}_{rule}.txt'
    content = ''.join(x+'\n' for x in jobs[gpu])
    if q.exists():
        assert q.read_text() == content, f'Old queue differs: {q}'
    else:
        q.write_text(content)
    print(f'GPU{gpu} QUEUE: {q} ({len(jobs[gpu])} jobs)')

manifest_path = DEST/'study_manifest.json'
content = json.dumps(dict(source=str(SRC.relative_to(ROOT)),dense_rounds=150,
    budget_bytes=BUDGET,seed=0,clients=k,dimension=d,jobs=manifest),indent=2)+'\n'
if manifest_path.exists():
    # The manifest only reflects intended case metadata; status can evolve.
    old = json.loads(manifest_path.read_text())
    new = json.loads(content)
    for row in old.get('jobs',[]): row.pop('skipped_completed',None)
    for row in new.get('jobs',[]): row.pop('skipped_completed',None)
    assert old == new, f'Old manifest differs: {manifest_path}'
else:
    manifest_path.write_text(content)
print('PASS:', sum(map(len,jobs.values())), 'jobs queued; budget', BUDGET, 'bytes')
