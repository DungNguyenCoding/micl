# Operational runbook

Current terminal commands below use the user's Linux machine. They are not commands already executed by the documentation author.

## 1. Read-only environment and process check

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -V
"$PY" -m bayesfl.main --help
pgrep -af 'bayesfl.main|_run_gpu_queue.sh' || true
nvidia-smi
```

The supplied dependency files declare Python >=3.10,<3.11; Flower simulation 1.29.0; Bayesian-Torch 0.5.0; PyYAML >=6,<7; NumPy >=1.26,<3; Matplotlib >=3.8,<4; pytest >=8,<9. requirements.txt intentionally does not install Torch; its comment records Torch 2.7.0+cu128 and corresponding torchvision 0.22.x. These are **declarations/recorded environment**, not a fresh live version inventory. Do not upgrade or reinstall packages simply to update these docs.

For a deliberately prepared new environment, install the suitable working Torch/torchvision pair separately and inspect the declared dependencies. `"$PY" -m pip install -e . --no-deps` is an editable-install command, not a replacement for missing dependencies. The repository's shell runner hard-codes the original interpreter in the inspected snapshot; another PC must inspect/adapt that local interpreter path before launching training.

## 2. Regenerate current paper figures (no training)

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
/home/micl/anaconda3/envs/dungndh/bin/python \
  scripts/plot_ieee_multicast_results_mnist_s10_stage22.py \
  --mnist-fedavg clipped
```

Selected outputs: `plots/Acc_Com.pdf`, `plots/Acc_Com.png`, `plots/Acc_KeepRatio.pdf`, `plots/Acc_KeepRatio.png`. This may overwrite those generated figures; it must not retrain or edit experiment CSVs. The user reported normal plotting after cleanup.

Do not replace the Stage-18 unclipped dense path with Stage-24 clipped FOLA. `--mnist-fedavg unclipped` is an explicit alternative comparison, not the current selected primary legend/curve.

## 3. Print the current primary endpoints without writing a result file

The documentation bundle adds a small read-only helper, separate from the simulation implementation:

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
/home/micl/anaconda3/envs/dungndh/bin/python \
  docs/tools/export_handoff.py --check
```

It uses exact registry outputs, reads each required raw metric round, validates multicast byte totals, and prints primary MNIST/CIFAR results. It does not train, load weights, update plots, or save a summary in check mode. By default accuracy fields are interpreted as fractions, matching the inspected evaluator. The explicit `--accuracy-unit percent` option is for verified percent-valued CSVs, not automatic guessing.

## 4. Existing generators and queues: inspect before rerunning

| Purpose | Generator / queue family |
| --- | --- |
| MNIST unclipped dense | `scripts/mnist_stage18_unclipped_fola.py`; `mnist_stage18_fola_noclip_gpu0.txt`, `...gpu1.txt` |
| MNIST s10 sparse | `scripts/mnist_stage22_sparse_s10_seed0.py`; Stage-22 KL GPU0 / random GPU1 queues |
| MNIST s5 alternative sparse | `scripts/mnist_stage21_sparse_s5_seed3.py`; Stage-21 KL / random queues |
| MNIST clipped FedAvg recovery | `scripts/queues/mnist_stage23_recovery_gpu0.txt`; config in `scripts/configs/mnist_stage23_recovery/` |
| CIFAR R150 continuation | `scripts/prepare_cifar_r150_sparse_resume.py`; current two resume queues |
| CIFAR dense recovery | `scripts/recover_cifar_plot.py`; configs in `scripts/configs/cifar_plot_recovery/` |

These are provenance/maintenance references, **not a new instruction to rerun completed work**. The Stage-23 standalone generator was provided in conversation but was not listed among the latest 73 top-level scripts; do not assume it is installed.

CIFAR continuation status can be inspected with:

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
/home/micl/anaconda3/envs/dungndh/bin/python \
  scripts/prepare_cifar_r150_sparse_resume.py --status
```

The previous status showed all ten runs complete; no launch was needed. Inspect current output rather than treating a saved empty queue as an error.

## 5. Generic launch pattern — only after approved new work and validation

This pattern intentionally requires explicit queue variables, so it cannot accidentally rerun historical jobs by being pasted without review:

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
(
    set -eu
    : "${QUEUE0:?Set QUEUE0 to the newly reviewed GPU0 queue}"
    : "${QUEUE1:?Set QUEUE1 to the newly reviewed GPU1 queue}"
    test -f "$QUEUE0"
    test -f "$QUEUE1"
    mkdir -p logs
    TS="$(date +%Y%m%d_%H%M%S)"
    for gpu in 0 1; do
        if [ "$gpu" = 0 ]; then Q="$QUEUE0"; else Q="$QUEUE1"; fi
        if grep -Eq '^[[:space:]]*[^#[:space:]]' "$Q"; then
            nohup bash scripts/_run_gpu_queue.sh "$gpu" "$Q" \
              > "logs/approved_queue_gpu${gpu}_${TS}.log" 2>&1 &
            echo "GPU$gpu PID=$! | queue=$Q"
        else
            echo "GPU$gpu: no executable records"
        fi
    done
)
```

Before use, perform all five stages in [COMMAND_RULES.md](COMMAND_RULES.md). A `GPU QUEUE FINISHED` banner with no `START` can mean no runnable records; successful child process exit still does not validate the required endpoint.

## 6. Tests after a real source change

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
/home/micl/anaconda3/envs/dungndh/bin/python -m pytest -q
```

The supplied documentation records older tests; no current full test result is asserted. Check current test module names before giving selective test commands. A skipped dependency test is not a passed integration check. Real dataset smoke tests are separate simulations and need explicit unique names and authorization; do not start them as an incidental documentation check.

## 7. Preserve/review cleanup state

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
/home/micl/anaconda3/envs/dungndh/bin/python scripts/bayesfl_same_ssd_organizer.py --verify
/home/micl/anaconda3/envs/dungndh/bin/python scripts/bayesfl_stage3b_organizer.py --verify
```

These tools have exact manifests and expected protected paths from October 26. Do not edit their allowlists or restore/archive everything automatically merely because a future experiment changes a file. See [CLEANUP_STATE.md](CLEANUP_STATE.md).

## 8. New-chat export

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
/home/micl/anaconda3/envs/dungndh/bin/python \
  docs/tools/export_handoff.py --export
```

This writes a timestamped ZIP **beside** bayesfl containing selected current code, scripts/configs/queues, tests, docs, primary metrics/config metadata, and an exact hash list. It excludes model weights, resume arrays, datasets, logs, .git and environment secrets. It also captures primary endpoint JSON and the installed package versions available to that interpreter. It is a handoff/plotting snapshot, **not a resumable training backup**. Inspect the contents before sharing. No original project file is changed.
