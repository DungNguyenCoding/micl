# CIFAR-10 joint-only 6-case wireless resource sensitivity study

## Research objective

Investigate how joint FedKLS CIFAR-10 learning and resource diagnostics change as one of two factors is varied: total UL bandwidth B or the shared upload window U. **No equal-bandwidth training comparator**. The already verified R150 joint run at 1 MHz and U=58.97274041844295s is the reference.

### Controlled scenarios

| Case | Uplink B (MHz) | Round slot (s) | Upload U (s) | B×U relative to baseline |
|---|---:|---:|---:|---:|
| `u170` | 1.00 | 170 | 48.972740418 | 0.8304 |
| `u160` | 1.00 | 160 | 38.972740418 | 0.6609 |
| `u150` | 1.00 | 150 | 28.972740418 | 0.4913 |
| `b0550` | 0.55 | 180 | 58.972740418 | 0.5500 |
| `b0750` | 0.75 | 180 | 58.972740418 | 0.7500 |
| `b0950` | 0.95 | 180 | 58.972740418 | 0.9500 |

All cases: 20/20 CIFAR-10 seed-0 clients, FOLA learning locked, same fixed gains, noise PSD, uplink PSD (2.51188643150958e-7 W/Hz per client), downlink bandwidth/PSD 1 MHz / 3.981e-5 W/Hz, 30s control reserve, 1Mb/s control rate, 0.001s exchange latency, same modeled compute values, same solver. PSD×B<=24dBm cap because B never exceeds 1MHz. **Do not** scale both UL and DL bandwidth simultaneously in B cases; doing so changes U. 

The shared upload window comes from `round_target_s - 3.867259582 - 87.16 - 30`. `U` is a computed quantity, **not an independent YAML field**. U values above are computed analytical expectations from the pinned fixed scenario. Use source-backed preflight to confirm on server.

### Why B×U is important

With unchanged fixed uplink PSD and path gains, `c_k=log2(1+PSD*g_k/N0)` is fixed and each candidate count requires `B_k = L(m_k)/(U*c_k)`. Thus feasibility depends on B×U, and joint selection may be identical for conditions with equal B×U (up to numerical precision). The six points are deliberately separated along this effective resource axis. It is **not guaranteed** that test accuracy decreases monotonically: compression can alter training in nontrivial ways.

### Safe installation

Save `FedKLS_CIFAR10_Joint_Sensitivity_SixScenarios_v1.zip` in `~/DungNDH/micl/`, then:

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
unzip -n ../FedKLS_CIFAR10_Joint_Sensitivity_SixScenarios_v1.zip
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -B scripts/check_joint_sensitivity_six_v1.py
```

All package paths are newly added; do not overwrite existing files. The checker is read-only and requires the existing `scripts/check_cifar_wireless_scenario_v1.py` installed as part of the earlier CIFAR pilot.

### Two-round gate (most severe U and B)

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
for case in u150 b0550; do
  "$PY" scripts/prepare_wireless_study.py \
    --profile cifar10_a0p01_f1 \
    --scenario "scripts/configs/wireless/cifar10_joint_sensitivity6_v1/${case}.yaml" \
    --tag "cifar_joint_${case}_smoke_v1" --rounds 2 --smoke --allow-example \
    --backend flower --policies joint --prepare-partition || exit 1
done
```

Run each queue individually after verifying GPU occupancy and worker availability:

```bash
bash scripts/_run_wireless_queue.sh 0 scripts/queues/wireless_cifar_joint_u150_smoke_v1_gpu0.txt
bash scripts/_run_wireless_queue.sh 0 scripts/queues/wireless_cifar_joint_b0550_smoke_v1_gpu0.txt
```

Both must report `SUCCESS` and non-failed evaluations. Then generate all R150 configs.

### Generate six independent 150-round studies

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
for case in u170 u160 u150 b0550 b0750 b0950; do
  "$PY" scripts/prepare_wireless_study.py \
    --profile cifar10_a0p01_f1 \
    --scenario "scripts/configs/wireless/cifar10_joint_sensitivity6_v1/${case}.yaml" \
    --tag "cifar_joint_${case}_r150_v1" --rounds 150 --allow-example \
    --backend flower --policies joint --prepare-partition || exit 1
done
```

Inspect `study_manifest.json` and exact frozen learner/wireless settings. Generator writes one non-empty GPU0 queue per study (and an empty GPU1 queue). **Do not launch long runs until the two short smokes and preflight pass.** If any target study already exists, the generator refuses to overwrite; inspect rather than delete it.

Run sequentially on one GPU to avoid untested concurrent Flower/Ray resource contention:

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
pgrep -af 'bayesfl.main|_run_gpu_queue.sh|_run_wireless_queue.sh' || true
nvidia-smi
```

After confirming GPU0 availability, you may run each generated GPU0 queue in order. A wrapper/combined queue may be used only after previewing its lines; it stops on the first failed job and should not duplicate any existing run.

### Collect results (after each success)

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
for case in u170 u160 u150 b0550 b0750 b0950; do
  echo "===== ${case} ====="
  "$PY" scripts/summarize_wireless_study.py --study-dir "scripts/configs/wireless/cifar_joint_${case}_r150_v1" || exit 1
done
```

Report accuracy/NLL versus round and versus modeled time, true retained count and fractions, weighted discarded KL, power and energy, control message bytes/seconds, price-query count and allocation statuses, feasible max upload time and total granted Hz. Do not claim any new run has completed until observed. The different `round_target_s` values make R150 modeled-time horizons different across the U sweep. Keep the source R150 and all original paper experiments unchanged.

This is a **joint-only, single-seed, fixed-channel sensitivity study**, not a realistic fading validation or publication-grade significance test. After inspecting this sweep, independent seeds and/or a 2D B×U grid can test reproducibility and interactions.
