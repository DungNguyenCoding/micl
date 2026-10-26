#!/usr/bin/env python3
"""Generate, validate, and summarize Paper 00 strict FedAvg runs.

GPU queues: 15 jobs on each GPU, total 30 runs = 2 sample counts x 5 seeds
            x {old-MLP LR .01, CNN LR .03, CNN LR .10}, all FedAvg R400.

Usage:
  PYTHONPATH="$PWD/src" python scripts/mnist_stage16_paper00.py prepare
  PYTHONPATH="$PWD/src" python scripts/mnist_stage16_paper00.py partition
  PYTHONPATH="$PWD/src" python scripts/mnist_stage16_paper00.py results
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, stdev, pstdev
import csv
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "scripts/configs/mnist_stage16_paper00_fedavg"
QUEUES = tuple(ROOT / f"scripts/queues/mnist_stage16_paper00_gpu{g}.txt" for g in (0, 1))
MODELS = {
    "oldmlp": {"name":"mlp_784_500_300_10", "lrs":(0.01,), "params":545810},
    "cnn16_32": {"name":"mnist_cnn_16_32_128", "lrs":(0.03,0.10), "params":80202},
}
SAMPLES = (5, 10)
SEEDS = range(5)
ROUNDS = 400
MILESTONES = (120, 150, 200, 240, 300, 400)


def tag_lr(lr):
    return f"{lr:.2f}".replace(".", "p")


def case_name(model, samples, lr, seed):
    return (f"run_stage16_paper00_fedavg_{model}_s{samples}_n20_"
            f"seed{seed}_e5_b{128 if model == 'oldmlp' else samples}_"
            f"lr{tag_lr(lr)}_r400")


def template_for(model, samples, lr):
    if model == "oldmlp":
        candidates = [
            ROOT / "scripts/configs/mnist_stage11_r200400_seed0/"
                   "debug_mnist_stage11_oldfixed_fedavg_n20_seed0_e5_b128_lr0p01_r400.yaml",
            ROOT / "scripts/configs/mnist_fig5_n20_large_b_sweep_seed0/"
                   "run_fig5probe_mnist_oldmlp_fixed1c_s2710_n20_"
                   "fedavg_b128_e5_lr0p01_r300_seed0.yaml",
        ]
    else:
        stage = 12 if samples == 10 else 14
        directory = (
            "mnist_stage12_arch_fixed_s10_n20_seed0"
            if stage == 12 else "mnist_stage14_arch_fixed_s5_n20_seed0"
        )
        candidates = [
            ROOT / "scripts/configs" / directory /
                   (f"debug_mnist_stage{stage}_cnn16_32_fixed1c_s{samples}_"
                    f"n20_fedavg_seed0_e5_b{samples}_lr{tag_lr(lr)}_r400.yaml")
        ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError("No compatible historical config template: " + ", ".join(map(str, candidates)))


def paths_for_case():
    return {
        (model, samples, lr, seed): CONF / (case_name(model, samples, lr, seed) + ".yaml")
        for model, spec in MODELS.items()
        for samples in SAMPLES
        for lr in spec["lrs"]
        for seed in SEEDS
    }


def verify_paper00(cfg, *, model, samples, lr, seed):
    assert cfg.method == "fedavg"
    assert cfg.data.dataset == "mnist"
    assert cfg.data.partition["type"] == "fixed_labels"
    assert cfg.data.partition["labels_per_client"] == 1
    assert cfg.data.partition["samples_per_client"] == samples
    assert cfg.model.name == MODELS[model]["name"]
    assert cfg.runtime.seed == seed
    assert cfg.federation.num_clients == cfg.federation.clients_per_round == 20
    t = cfg.training
    assert t.paper00_strict is True
    assert t.optimizer.lower() == "sgd"
    assert t.momentum == t.weight_decay == 0.0
    assert t.grad_clip_norm is None
    assert t.lr_schedule in ("constant", "none")
    assert t.lr == lr and t.local_epochs == 5 and t.rounds == ROUNDS
    assert t.batch_size == (128 if model == "oldmlp" else samples)
    assert cfg.compression.selection_rule == "dense"
    assert cfg.compression.keep_ratio == 1.0
    assert cfg.communication.max_communication_bytes is None
    assert cfg.communication.downlink_mode == "multicast"
    assert cfg.communication.deterministic_client_schedule is True
    assert cfg.data.augment is False


def prepare():
    from bayesfl.config import load_config
    from bayesfl.models.factory import initialize_model
    from bayesfl.data.datasets import partition_stem
    from bayesfl.communication import core_round_bytes

    CONF.mkdir(parents=True, exist_ok=True)
    QUEUES[0].parent.mkdir(parents=True, exist_ok=True)
    paths = paths_for_case()
    partitions = {}
    for (model, samples, lr, seed), path in paths.items():
        raw = yaml.safe_load(template_for(model, samples, lr).read_text())
        assert raw["method"] == "fedavg"
        raw["run_name"] = case_name(model, samples, lr, seed)
        raw["model"]["name"] = MODELS[model]["name"]
        raw["data"]["dataset"] = "mnist"
        raw["data"]["augment"] = False
        raw["data"]["partition"] = dict(type="fixed_labels", samples_per_client=samples, labels_per_client=1)
        raw["federation"]["num_clients"] = 20
        raw["federation"]["clients_per_round"] = 20
        raw["runtime"]["seed"] = seed
        raw["training"].update(
            optimizer="sgd", lr=float(lr), momentum=0.0, weight_decay=0.0,
            batch_size=128 if model == "oldmlp" else samples,
            local_epochs=5, rounds=ROUNDS, lr_schedule="constant",
            lr_min=float(lr), lr_decay_rounds=ROUNDS,
            grad_clip_norm=None, paper00_strict=True,
        )
        raw["training"].pop("max_local_steps_per_round", None)
        raw["compression"].update(selection_rule="dense", keep_ratio=1.0)
        raw["communication"].update(max_communication_bytes=None,
                                     budget_metric="cumulative_all_array_bytes",
                                     deterministic_client_schedule=True,
                                     downlink_mode="multicast")
        raw["output"]["checkpoint_every"] = 10
        raw["output"]["save_full_client_posteriors"] = False
        if path.exists():
            assert yaml.safe_load(path.read_text()) == raw, f"Existing config conflict: {path}"
        else:
            assert not list((ROOT / "outputs").glob(raw["run_name"] + "_*")), (
                "Output already exists; do not overwrite completed runs: " + raw["run_name"])
            path.write_text(yaml.safe_dump(raw, sort_keys=False))
        cfg = load_config(path)
        verify_paper00(cfg, model=model, samples=samples, lr=lr, seed=seed)
        key = (samples, seed)
        stem = partition_stem(cfg)
        assert key not in partitions or partitions[key] == stem
        partitions[key] = stem
        # Check model matches the historical architecture and parameter count.
        pcount = sum(p.numel() for p in initialize_model(cfg).parameters())
        assert pcount == MODELS[model]["params"], (model, pcount)
        bytes_per_round = core_round_bytes(pcount, 20, method="fedavg", downlink_mode="multicast")["total_bytes"]
        print(f"OK {model:8s} s{samples:<2d} seed{seed} LR={lr:.2f} "
              f"plainSGD R400 budget={bytes_per_round*400/1e9:.3f} GB")

    # Balance equal workloads, not just total entry counts: each GPU gets
    # five old-MLP and ten CNN runs, which also share cached partitions.
    groups = [(s, seed) for s in SAMPLES for seed in SEEDS]
    gpu_jobs = {0: [], 1: []}
    for i, (samples, seed) in enumerate(groups):
        gpu = i % 2
        for model, spec in MODELS.items():
            for lr in spec["lrs"]:
                gpu_jobs[gpu].append(paths[(model, samples, lr, seed)].relative_to(ROOT).as_posix())
    assert len(paths) == 30 and len(partitions) == 10
    for gpu, jobs in gpu_jobs.items():
        assert len(jobs) == len(set(jobs)) == 15
        assert sum("oldmlp" in name for name in jobs) == 5
        assert sum("cnn16_32" in name for name in jobs) == 10
        contents = "\n".join(jobs) + "\n"
        q = QUEUES[gpu]
        if q.exists():
            assert q.read_text() == contents, f"Queue file conflict: {q}"
        else:
            q.write_text(contents)
        print(f"GPU{gpu} QUEUE: {len(jobs)} jobs at {q.relative_to(ROOT)}")
    assert set(gpu_jobs[0]).isdisjoint(gpu_jobs[1])
    print("PASS: 30 strict Paper-00 FedAvg jobs, 15 per GPU, no FOLA changes")


def partition():
    from bayesfl.config import load_config
    from bayesfl.data.datasets import prepare_partition
    from bayesfl.data.partition import load_partition
    from torchvision.datasets import MNIST
    import numpy as np

    paths = paths_for_case()
    hashes = {}
    for samples in SAMPLES:
        for seed in SEEDS:
            cfg = load_config(paths[("oldmlp", samples, 0.01, seed)])
            path, meta = prepare_partition(cfg)
            parts = load_partition(path)
            labels = MNIST(root=cfg.data.root, train=True, download=False).targets.numpy()
            assert len(parts) == 20 and all(len(ids) == samples for ids in parts)
            assert len(set(int(i) for ids in parts for i in ids)) == 20*samples
            class_ids = [set(map(int, labels[ids].tolist())) for ids in parts]
            assert all(len(ids) == 1 for ids in class_ids)
            counts = Counter(next(iter(ids)) for ids in class_ids)
            assert all(counts[d] == 2 for d in range(10)), counts
            hashes[(samples, seed)] = meta["sha256"]
            print(f"PARTITION PASS s{samples} seed{seed}: {20*samples} unique samples, "
                  f"exactly two clients/digit, sha={meta['sha256'][:16]}")
    assert len(hashes) == 10
    print("PASS: all ten seeded partitions cached before parallel GPU runs")


def get_accuracy(row):
    for key in ("accuracy", "global_accuracy", "test_accuracy"):
        if row.get(key) not in (None, ""):
            v = float(row[key])
            if math.isfinite(v):
                return v*100 if v <= 1.000001 else v
    raise ValueError(f"No accuracy column in {list(row)}")


def get_run(run_name, min_rounds=400):
    dirs = []
    out = ROOT / "outputs"
    for p in [out / run_name, *out.glob(run_name + "_*")]:
        if p.is_dir() and (p / "metrics/global_metrics.csv").is_file():
            dirs.append(p)
    for p in sorted(set(dirs), key=lambda v: (v/"metrics/global_metrics.csv").stat().st_mtime, reverse=True):
        with (p / "metrics/global_metrics.csv").open(newline="", encoding="utf-8") as f:
            rows = {int(float(r["round"])):r for r in csv.DictReader(f) if r.get("round")}
        if all(r in rows for r in range(1, min_rounds+1)):
            return {r:get_accuracy(rows[r]) for r in range(1, min_rounds+1)}, p
    raise RuntimeError(f"Missing/incomplete: {run_name}, expected R{min_rounds}")


def results():
    from bayesfl.config import load_config
    paths = paths_for_case()
    data = {}
    failures = []
    for key, path in paths.items():
        cfg = load_config(path)
        try:
            data[key], _ = get_run(cfg.run_name, 400)
        except Exception as exc:
            failures.append(str(exc))
    if failures:
        print("\n".join(failures))
        raise RuntimeError(f"Only {len(data)}/30 FedAvg Paper00 runs completed")

    print("\n" + "="*121)
    print("1. PER-SEED FEDAVG ACCURACY AT R120/150/200/240/300/400")
    print("="*121)
    print("Model       S   LR  seed  " + " ".join(f"R{r:3d}" for r in MILESTONES) +
          "  lateSD  delta400  best@round")
    for (model, samples, lr, seed), a in data.items():
        tail = [a[r] for r in range(361,401)]
        late_delta = mean(a[r] for r in range(381,401)) - mean(a[r] for r in range(341,361))
        best_round = max(a, key=a.get)
        print(f"{model:9s} {samples:2d} {lr:4.2f} {seed:4d} " +
              " ".join(f"{a[r]:6.2f}" for r in MILESTONES) +
              f"  {pstdev(tail):6.2f} {late_delta:+9.2f} {a[best_round]:5.2f}@{best_round}")
    print("\n" + "="*112)
    print("2. FIVE-SEED FEDAVG SUMMARY (mean +/- sample SD across seeds)")
    print("="*112)
    print("Model       S   LR    R150 mean+/-SD   R240 mean+/-SD   R300 mean+/-SD   R400 mean+/-SD  delta400")
    for model, spec in MODELS.items():
        for samples in SAMPLES:
            for lr in spec["lrs"]:
                vals = [data[(model, samples, lr, seed)] for seed in SEEDS]
                metrics = []
                for r in (150,240,300,400):
                    v = [a[r] for a in vals]
                    metrics.append(f"{mean(v):6.2f}+/-{stdev(v):5.2f}")
                deltas = [mean(a[r] for r in range(381,401)) - mean(a[r] for r in range(341,361)) for a in vals]
                print(f"{model:9s} {samples:2d} {lr:4.2f} " + " ".join(metrics) + f" {mean(deltas):+7.2f}")
    # Optional diagnostic: compare this new unclipped FedAvg to the previously
    # saved clipped FedAvg and FOLA curves on the exact same seed/partition.
    # Stage-15 had 5 seeds for (s10,LR.10), (s5,LR.10), (s5,LR.03).
    # For (s10,LR.03), only the older seed-0 Stage-12 run exists.
    print("\n" + "="*110)
    print("3. OPTIONAL CHECK: NEW UNCLIPPED VS HISTORICAL CLIPPED FEDAVG @R300")
    print("   Positive = new strict Paper00 FedAvg has better accuracy")
    print("="*110)
    for samples in SAMPLES:
        for lr in (0.03, 0.10):
            diffs = []
            for seed in SEEDS:
                if seed == 0:
                    stage = 12 if samples == 10 else 14
                    old_name = (f"debug_mnist_stage{stage}_cnn16_32_fixed1c_s{samples}_"
                                f"n20_fedavg_seed0_e5_b{samples}_lr{tag_lr(lr)}_r400")
                elif samples == 10 and lr == 0.03:
                    continue  # This pair was not run in Stage-15.
                else:
                    old_name = (f"run_mnist_stage15_cnn16_32_fixed1c_s{samples}_"
                                f"n20_fedavg_seed{seed}_e5_b{samples}_lr{tag_lr(lr)}_r300")
                try:
                    previous, _ = get_run(old_name, 300)
                except RuntimeError:
                    continue
                new = data[("cnn16_32",samples,lr,seed)]
                diff = new[300]-previous[300]
                diffs.append(diff)
                print(f"s{samples:2d} LR={lr:.2f} seed{seed}: "
                      f"unclipped={new[300]:6.2f}% historical={previous[300]:6.2f}% "
                      f"delta={diff:+6.2f} pp")
            if diffs:
                print(f"  Group s{samples} LR={lr:.2f}: "
                      f"mean delta={mean(diffs):+.2f} pp across {len(diffs)} comparable seed(s)")

    print("\n" + "="*115)
    print("4. OPTIONAL CHECK: HISTORICAL FOLA VS NEW PAPER00 FEDAVG AT MATCHED BYTES")
    print("   FOLA curves retain their original clipped configuration: interpret explicitly.")
    print("="*115)
    for samples, lr in ((10,0.10),(5,0.10),(5,0.03)):
        gaps120=[]; gaps150=[]
        for seed in SEEDS:
            if seed == 0:
                stage = 12 if samples == 10 else 14
                fola_name = (f"debug_mnist_stage{stage}_cnn16_32_fixed1c_s{samples}_"
                             f"n20_fola_seed0_e5_b{samples}_lr{tag_lr(lr)}_lam0p01_r200")
            else:
                fola_name = (f"run_mnist_stage15_cnn16_32_fixed1c_s{samples}_"
                             f"n20_fola_seed{seed}_e5_b{samples}_lr{tag_lr(lr)}_lam0p01_r150")
            try:
                fola, _ = get_run(fola_name, 150)
            except RuntimeError:
                continue
            paper00 = data[("cnn16_32",samples,lr,seed)]
            gaps120.append(fola[120]-paper00[240])
            gaps150.append(fola[150]-paper00[300])
            print(f"s{samples:2d} LR={lr:.2f} seed{seed}: "
                  f"F120-A240={gaps120[-1]:+6.2f} pp, F150-A300={gaps150[-1]:+6.2f} pp")
        if gaps120:
            print(f" GROUP s{samples} LR={lr:.2f}: "
                  f"mean120/240={mean(gaps120):+.2f} pp, "
                  f"mean150/300={mean(gaps150):+.2f} pp, n={len(gaps120)}")

    print("\nPASS: all 30 FedAvg Paper00 simulations analyzed")
    print("NOTE: This is strict Paper00 Algorithm 1 under your chosen 20-client fixed-label MNIST environment,")
    print("      NOT a numerical reproduction of Paper00's original 100-client MNIST experiment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "partition", "results"))
    act = parser.parse_args().action
    {"prepare": prepare, "partition": partition, "results": results}[act]()
