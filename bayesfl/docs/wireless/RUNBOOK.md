# Wireless execution runbook

Run from the real project root. Do not upgrade the working CUDA dependencies or install
this over an actively running queue. The overlay installer supplies preview, backup and
verification; it never installs historical experiment output files.

## 1. Validate before dataset queues

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -m pytest -q tests/wireless
"$PY" -m pytest -q
"$PY" docs/tools/export_handoff.py --check
```

Missing Flower/Bayesian-Torch skips must not be described as passes. The last command
rechecks existing primary byte-budget endpoints and does not train.

Synthetic protocol/trainer smoke (six cases; no downloads):

```bash
"$PY" scripts/smoke_wireless_offline.py \
  --output-dir "outputs/smoke_wireless_cpu_$(date +%Y%m%d_%H%M%S)" --rounds 3
```

Dense smoke cases use a larger bandwidth only to exercise dense code, so these toy
results are not a comparative performance study.

On the pinned Flower/Ray environment, run the actual adapter gate and short Ray smoke:

```bash
"$PY" -m pytest -q tests/wireless/test_flower.py
CUDA_VISIBLE_DEVICES="" "$PY" scripts/smoke_wireless_flower.py \
  --output-dir "outputs/smoke_wireless_ray_$(date +%Y%m%d_%H%M%S)" --rounds 2
```

The Ray smoke uses CPU synthetic data, actual Flower ClientApp/ServerApp and persisted
client Context.state. It does not validate CUDA. A separately named 2-round real dataset
smoke is the next target-machine gate, not an automatic long queue.

## 2. Generate a new study without changing locked learning

Prepare a reviewed scenario YAML using CONFIGURATION.md. For an **explicit illustrative
MNIST runtime check**, the included example may be used as follows:

```bash
TAG="mnist_wireless_smoke_$(date +%Y%m%d_%H%M%S)"
"$PY" scripts/prepare_wireless_study.py \
  --profile mnist_s10 \
  --scenario scripts/configs/wireless/scenario_example.yaml \
  --tag "$TAG" --rounds 2 --smoke --allow-example \
  --backend flower --prepare-partition
```

Default: six configs, three jobs/GPU when none already exist. Expected artifacts:
`scripts/configs/wireless/$TAG/study_manifest.json` and
`scripts/queues/wireless_${TAG}_gpu0.txt`, `...gpu1.txt`.
The generator prints NEW/RESUME/SKIP_COMPLETE and actual counts; a valid empty queue
can mean no remaining work. `--prepare-partition` uses the established dataset/cache
logic and can obtain data if absent, so prepare it once before parallel workers.

For a real study, use your reviewed scenario, omit `--smoke`/`--allow-example`, and set
an approved round cap. Use profile `cifar10_a0p01_f1` for CIFAR. Dense feasibility is not
guaranteed under an arbitrary common deadline; no policy gets hidden extra bandwidth.

Validate saved configs and shared partition before launch:

```bash
"$PY" - "$TAG" <<'PYCODE'
from pathlib import Path
import sys
from bayesfl.config import load_config
from bayesfl.data.datasets import prepare_partition
from bayesfl.data.partition import load_partition, _hash_partitions
folder=Path('scripts/configs/wireless')/sys.argv[1]
paths=sorted(folder.glob('*.yaml'))
if not paths: raise SystemExit('No study YAMLs')
for path in paths:
    cfg=load_config(path)
    assert cfg.wireless_enabled
    print('VALID',path.name,cfg.method_id)
    partition,meta=prepare_partition(cfg)
    parts=load_partition(partition)
    assert _hash_partitions(parts)==meta['sha256']
    print('PARTITION',partition,'N=',len(parts),'min/max=',min(map(len,parts)),max(map(len,parts)))
print('PASS:',len(paths),'configs; no training started')
PYCODE
```

## 3. Launch two serial queues only when GPUs are available

```bash
pgrep -af 'bayesfl.main|_run_gpu_queue.sh|_run_wireless_queue.sh' || true
nvidia-smi
mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
for gpu in 0 1; do
    Q="scripts/queues/wireless_${TAG}_gpu${gpu}.txt"
    if [ -s "$Q" ]; then
        nohup bash scripts/_run_wireless_queue.sh "$gpu" "$Q" \
          > "logs/wireless_${TAG}_gpu${gpu}_${TS}.log" 2>&1 &
        echo "GPU${gpu} queue PID=$!"
    else
        echo "GPU${gpu}: no queued jobs"
    fi
done
```

Each new runner uses the explicit dungndh interpreter, masks its assigned GPU and stops
on the first failed job. It prevents duplicate launches of the same queue, not arbitrary
GPU sharing by other applications. Do not run both a historical queue and this queue
on the same GPU concurrently. Inspect START/SUCCESS/FAILED and actual checkpoints.

## 4. Terminal summaries and explicit plots

```bash
"$PY" scripts/summarize_wireless_study.py \
  --study-dir "scripts/configs/wireless/$TAG"
```

For explicit completed outputs:

```bash
"$PY" scripts/summarize_wireless_results.py \
  --run-dir outputs/ACTUAL_JOINT_RUN \
  --run-dir outputs/ACTUAL_EQUAL_BANDWIDTH_RUN
```

Replace uppercase output placeholders with actual directory names printed by the run;
do not guess timestamps. Add `--time-budget-seconds <covered seconds>` for last-observed
common-time checkpoints; no best-test selection or extrapolation. Summary writes no
result report file. Different endpoint clocks trigger a warning, not an equal-time claim.

Only when requesting figures:

```bash
"$PY" scripts/plot_wireless_results.py \
  --run-dir outputs/ACTUAL_JOINT_RUN \
  --run-dir outputs/ACTUAL_EQUAL_BANDWIDTH_RUN \
  --output-dir plots/wireless_COMPARISON_TAG
```

The destination must be new. Separate raw accuracy-versus-modeled-time and accuracy-
versus-array-GB figures are created; legacy Acc_Com/Acc_KeepRatio are never modified.

## 5. Resume a clean completed wireless run

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m bayesfl.main \
  --resume outputs/ACTUAL_WIRELESS_RUN --rounds 200 \
  --max-modeled-time-seconds 24000
```

Both numbers above are **syntax examples**, not locked budgets. Round cap is total,
not additional rounds. A queue resume record uses the separate grammar:

```text
WRESUME|/absolute/existing/wireless/run|TOTAL_ROUND_CAP|TOTAL_TIME_CAP_OR_none
```

`none` leaves the existing optional cap unchanged; it does not remove a prior finite
cap. Legacy `RESUME|...|BYTE_BUDGET|...` belongs only to `_run_gpu_queue.sh`. Never
interchange the runners, delete partial ledgers, or force a failed run to look complete.

## 6. Carry new work into another chat

```bash
"$PY" docs/tools/export_handoff.py --check \
  --wireless-run-dir outputs/ACTUAL_WIRELESS_RUN
"$PY" docs/tools/export_handoff.py --export \
  --wireless-run-dir outputs/ACTUAL_WIRELESS_RUN
```

Add `--wireless-only` to skip the 24 legacy-primary requirement when only wireless runs
are available. Repeat the run-dir flag for every desired new result. Update the handoff,
scenario decisions and run identities before exporting. No automatic newest-run selection.
