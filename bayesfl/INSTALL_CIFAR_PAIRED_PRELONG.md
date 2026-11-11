# FedKLS CIFAR-10 — Paired R11 Evaluation + One-Shot Pre-Long Gate (v1)

**Status:** additive, fail-closed diagnostic package. Existing wireless source and all protected MNIST/CIFAR results are unchanged. **No 100/150-round simulation starts automatically.**

## Purpose

Before starting longer CIFAR-10 wireless runs, verify the complete chain under **the same** prepared client posteriors:

1. Load the verified equal-bandwidth 20-round pilot's **R10 global posterior** and original fixed seed-0 CIFAR-10 partition.
2. Perform one **R11 FOLA local preparation** per each of the original 20 clients.
3. Freeze the client-local KL score/posterior caches.
4. Query the installed `equal_bandwidth` and `joint` allocators on that **identical prepared set**.
5. In two isolated counterfactual branches, run the **production upload encoder, decoder, grant binding, raw-precision reconstruction, prior-fill, and original FOLA reducer**. Each branch may upload once, but no client trains twice.
6. Also compute the **uncompressed dense FOLA aggregate** from the identical prepared local posteriors, strictly as an offline diagnostic oracle; the dense upload is infeasible under the selected 180-second slot.
7. Evaluate the original R10 checkpoint and all three R11 aggregations on the **same CIFAR-10 test set** using the production mean-only FOLA prediction and metric code.
8. Collect control request/reply bytes, actual model-array packet bytes, KL objectives and per-client allocations; verify protected SHA-256 hashes unchanged.
9. Check wireless tests, full regression tests, the original 24 experiment endpoints, and the fixed-channel radio scenario. Produce one portable, model/data-free evidence ZIP.

## Dependencies already installed on the server

- The **wireless source v1** and earlier **CIFAR pilot scenario**.
- `scripts/probe_cifar_same_posterior.py` from the earlier allocation-only probe.
- A successful `cifar_eqb_pilot20_v1` run, including `checkpoints/global_round_0010.npz`.
- Original CIFAR-10 data files and partition NPZ already present on the server.
- The original `/home/micl/anaconda3/envs/dungndh/bin/python` environment with CUDA.

## Install (does not overwrite)

Copy `FedKLS_CIFAR10_Paired_R11_PreLong_Gate_v1.zip` to `~/DungNDH/micl/`, then from the server:

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
for f in \
  CIFAR_PAIRED_PRELONG_MANIFEST.json \
  INSTALL_CIFAR_PAIRED_PRELONG.md \
  scripts/probe_cifar_paired_r11.py \
  scripts/report_cifar_prelong_gate.py \
  scripts/run_cifar_prelong_gate.sh \
  tests/wireless/test_paired_r11_probe.py \
  tests/wireless/test_prelong_report.py \
  docs/wireless/CIFAR_PAIRED_PRELONG_GATE.md \
  docs/wireless/CHAT_HANDOFF_PAIRED_R11.md
  do
  if [ -e "$f" ]; then echo "STOP: Existing file: $f"; exit 1; fi
done
unzip -n ../FedKLS_CIFAR10_Paired_R11_PreLong_Gate_v1.zip
```

Read only the new logic tests (optional):

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m pytest -q -rs tests/wireless/test_paired_r11_probe.py tests/wireless/test_prelong_report.py
```

## One command to execute **all required pre-long checks once**

First inspect `pgrep -af 'bayesfl.main|_run_gpu_queue.sh|_run_wireless_queue.sh'` and `nvidia-smi`, and choose a free GPU (example GPU 0):

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
bash scripts/run_cifar_prelong_gate.sh 0
```

This executes `pytest tests/wireless`, full `pytest`, `export_handoff.py --check`, the fixed-channel resource preflight, original R10 checkpoint validation, **one** real local training pass per client, both branch uploads and their R11 test evaluations, and a fail-closed readiness report. It saves a new sibling folder/ZIP:

```
~/DungNDH/micl/bayesfl_cifar_prelong_gate_<timestamp>_<pid>/
~/DungNDH/micl/bayesfl_cifar_prelong_gate_<timestamp>_<pid>.zip
```

**Upload the final ZIP** to this chat. No dataset, raw posterior weights, checkpoints, or client KL score arrays are exported. The experiment's original files stay untouched.

The scientific decision on full R100/R150 training should be made **after inspecting** the measured paired evaluation and the remaining assumptions. A successful gate is not an automatic authorization for publication-grade claims.

## Optional manual path

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$PY" scripts/probe_cifar_paired_r11.py --preflight --checkpoint-round 10
OUT="$PWD/../bayesfl_paired_r11_manual_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/probe_cifar_paired_r11.py \
  --execute --checkpoint-round 10 --require-cuda --output-dir "$OUT"
```

Manual mode tests paired evaluation only, without the full pre-long bundle.

## Checks and limits

- The previous scalar-only R11 allocation probe (2026-11-09) reported a 97.2281% reduction in weighted discarded KL for the same R10 source. This is **reference evidence only**, not a guaranteed result of the new run.
- The 20-round pilot endpoints were equal-bandwidth 38.68%, joint 37.58%, both at 3,600 seconds modeled. They were **independent learning trajectories**, so their reported KL objectives were not directly counterfactual.
- This is a **single R10 starting point from the equal-bandwidth trajectory**; a symmetric experiment starting from joint R10 would make a stronger evaluation.
- This diagnostic uses the **local sequential transport** and exact model-array packet codec, **not Flower/Ray network traffic**. Flower/Ray was previously smoke-tested on the actual server. The wrapper reruns the wireless test suite.
- Real CIFAR mean-only centralized evaluation is performed but the computed R11 accuracy is not predicted in advance.
- Radio scenario remains `EXAMPLE_NOT_LOCKED`, with fixed channels, assumed compute/controls, fixed-PSD transmission and nominal 180 s slot. It is an engineering scenario, not a measured device deployment.
- A solver gap is a numerical diagnostic; not proof of global integer optimality.
- Even if joint wins on KL, **do not automatically conclude higher test accuracy**.

See `docs/wireless/CIFAR_PAIRED_PRELONG_GATE.md` for protocol/math/fairness details and `docs/wireless/CHAT_HANDOFF_PAIRED_R11.md` for continuity.
