# BayesFL / FedKLS — new-chat handoff

**State date: 2026-10-26, Asia/Seoul.** This is a continuation record, not a new request to tune, launch training, clean the repository or implement wireless communication.

## 1. Start here and respect evidence limits

The user is preparing a sparse Bayesian federated-learning paper and wants future chats to continue without reconstructing the long preceding conversation. Current paper simulations are sufficient in the user's judgment. Cleanup is finished for now; plotting was reported to work normally.

Read this file, then the current `LOCKED_EXPERIMENTS.md`, `COMMAND_RULES.md`, `REPOSITORY_MAP.md`, `EXPERIMENT_REGISTRY.json` and `EVIDENCE_AND_GAPS.md`. The documentation package preserves all supplied old documents under `docs/history/pre_refresh_20261026/`; those are **historical**, not current defaults.

This handoff combines uploaded documentation, user terminal results, the October 26 registry/cleanup reports and limited inspection of an older source ZIP. The old ZIP predates the current MNIST CNN and strict-FedAvg changes. It must not replace the working source. Actual current source and raw metrics remain on the user's Linux machine; request/read them before source edits when they are not available in the new chat. Do not claim a current GPU test suite passed based on older reports.

## 2. Environment and interaction rules

```text
Repository: /home/micl/DungNDH/micl/bayesfl
Shell shorthand: ~/DungNDH/micl/bayesfl
Python: /home/micl/anaconda3/envs/dungndh/bin/python
GPU queue runner: scripts/_run_gpu_queue.sh
GPU pattern: one serial queue on GPU0, one serial queue on GPU1
Archive: /home/micl/DungNDH/micl/bayesfl_historical_archive
```

Use the explicit interpreter and `PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"`. Never silently use base Conda Python, upgrade a working CUDA stack or modify dependencies. Supplied pyproject declares Python >=3.10,<3.11, Flower 1.29.0, Bayesian-Torch 0.5.0; requirements intentionally excludes installation of the working Torch pair. These are recorded declarations, not fresh live version checks.

**E <= 10** is a hard user limit unless the user explicitly changes it. Clone locked YAMLs; modify only requested fields. New locked/recovery outputs must begin `run_`, not `debug_`; smoke and exploratory jobs need explicit distinct names. Never overwrite or delete existing outputs/partitions or launch concurrent writers. Do not restart a run when its checkpoint can be safely resumed.

When asked for simulation commands, give all five stages: (1) config/two-queue generation with exact job counts, (2) validation with current `load_config`, (3) pre-generate/inspect each shared realized partition, (4) nohup serial-per-GPU launch with timestamped logs/PIDs, (5) matching terminal-only result extractor **in the same response**. No automatic `.txt`, summary CSV or tee for numerical results unless requested. Documentation/audit files and normal logs are exceptions. Provide full copy-to-terminal blocks when requested.

Queue records are either a config path or:

```text
RESUME|/absolute/existing/run|TOTAL_ROUND_CAP|TOTAL_BYTE_BUDGET|EXPECTED_ENDPOINT
```

The last field is informational in the inspected runner; check the actual endpoint afterward. A blank queue can correctly mean all requested runs are already complete. Do not infer success from `GPU QUEUE FINISHED` alone. No broad `pkill python`/Ray shutdown on a shared machine.

## 3. Locked main experiments

| Field | MNIST primary | CIFAR-10 primary |
| --- | --- | --- |
| Model | CNN 16–32–FC128, d=80,202 | `paper_basiccnn`, d=878,538 |
| Model identifier | `mnist_cnn_16_32_128` (validate current factory) | `paper_basiccnn` |
| Clients / participation | 20 / 20 | 20 / 20 |
| Data | fixed_labels, 10 samples/client, one class/client | paper_dirichlet, alpha=.01, local_data_fraction=1.0 |
| Seed | 0 | 0 |
| E / B | 5 / 10 | 10 / 32 |
| Optimizer / momentum / WD | SGD / 0 / 0 | SGD / 0 / 0 |
| FOLA LR | .10 constant | .02 cosine, lr_min=.0001, H=400 |
| FedAvg LR | .10 constant | .01 constant |
| FOLA lambda / mode | .01 / paper_reference | .01 / paper_reference |
| Dense and sparse FOLA clip | Disabled | Norm10 |
| Selected FedAvg clip | Norm10 (Stage-23 recovery) | Norm10 |
| Dense reference / FedAvg endpoint | R150 / R300 | R150 / R300 |
| Budget bytes | 2,021,090,400 | 22,139,157,600 |

Selected dense MNIST is **unclipped Stage-18**, not clipped Stage-24. Its run reaches R200 but plot stops at R150; recovered FedAvg reaches R400 but plot stops at R300. Stage-16 unclipped FedAvg remains a separate tighter-control reference. MNIST's primary baseline clipping difference and CIFAR's LR/schedule difference must be disclosed.

Sparse methods: `kl_global_local` and `random`, r in {.75,.50,.25,.10,.05}; same full local model and settings, only uploaded coordinates differ. FOLA remains `paper_reference`, mean-only global evaluation; no hidden full-client-posterior snapshots in sparse runs. `local_data_fraction` is not participation. Fixed-label MNIST is not Dirichlet.

Historical MNIST s5/seed3, N20, E5/B5/LR.10 is preserved as an alternative. N100, old/new MLP, ResNet, alpha .1, half-data and reverse KL are distinct historical experiments. Do not reintroduce their defaults.

Current CIFAR numerical policy is documented as zero initial raw omega with score-only `floor_for_score` (floor resolved from precision_min=1e-12); epsilon=1e-5, precision_max=1e12. Additional MNIST numerical/CNN-layer/runtime details require current source/resolved YAML inspection. Do not invent them from the parameter count.

## 4. Algorithm and payload contract

FOLA uses a deterministic mean network plus raw float32 diagonal precision/omega. In the documented paper_reference branch, initialize local omega from the broadcast; at each minibatch add `(batch_size/client_size)*task_gradient**2`; train with CE plus `lambda*(0.5/t)*sum(omega_global*(theta-mu_global)**2)`. Accumulation uses squared minibatch-mean task gradients before objective-gradient clipping, not an explicit per-example Fisher.

After the full local fit, select m=ceil(r*d) model-wide coordinates. KL compares the **incoming global** and **outgoing local** posterior in that order. Random selects the same count uniformly without replacement. Wire packet: packed uint8 bitmap, selected float32 means, selected float32 raw precisions, values in ascending coordinate order. No extra dense posterior or score upload.

Server fills omissions from the exact broadcast, uses dataset-size-normalized weights and the common FOLA precision-weighted reducer. Touched means use the baseline `weighted_precision+epsilon` denominator. Coordinates selected by nobody are restored exactly to broadcast mean and precision. Score-only precision flooring is not a change to transmitted/trained raw omega.

Current modeled bytes:

```text
m = ceil(r*d); bitmap = ceil(d/8); K=20
C_FOLA   = 8*d*(K+1)
C_FedAvg = 4*d*(K+1)
C_sparse = 8*d + K*(8*m+bitmap)
B = 150*C_FOLA
R_sparse = floor(B/C_sparse)
```

Sparse target rounds at r=.75,.50,.25,.10,.05:

```text
MNIST:   193, 278, 498, 950, 1362
CIFAR:   193, 278, 499, 950, 1362
```

The **498 vs 499** distinction is intentional. Use raw `cumulative_all_array_bytes` and the last complete affordable checkpoint. Budget is decimal GB, not disk GiB. Do not reuse old unicast counts or old 197-round keep=.5 examples. Dense full-posterior FOLA has no bitmap; sparse r=1 does.

The old R120 alternative is MNIST B=1,616,872,320 and CIFAR B=17,711,326,080, FedAvg R240, targets154/222/399/760/1089. It is not the current default. Stage-22 directory names retain original R150 caps even when reading an earlier checkpoint.

## 5. Results and plotting

Main script:
`./scripts/plot_ieee_multicast_results_mnist_s10_stage22.py --mnist-fedavg clipped`

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
/home/micl/anaconda3/envs/dungndh/bin/python \
  scripts/plot_ieee_multicast_results_mnist_s10_stage22.py \
  --mnist-fedavg clipped
```

Outputs: `plots/Acc_Com.{pdf,png}` and `plots/Acc_KeepRatio.{pdf,png}`. Two panels; smoothed-only seven-point moving average; MNIST final-stage zoom. No seed/clipping info in labels by user request, but preserve disclosure elsewhere. Requested legend: five FedKLS ratios in row1, Full-posterior BayesFL and FedAvg centered in row2. Second plot uses **sparsity** 0/25/50/75/90/95%, with dense as shared0% endpoint. Do not patch this already-working plotter merely to update docs.

Reported MNIST s10 current R150-budget endpoints:

```text
Dense FOLA R150 = 89.44%; recovered clipped FedAvg R300 = 88.23%
r       KL       Random    KL-Random pp
.75     88.86    88.59     +0.27
.50     88.19    88.08     +0.11
.25     89.01    86.96     +2.05
.10     88.44    86.54     +1.90
.05     88.58    86.07     +2.51
```

These are user-reported raw checkpoints. No sparse KL endpoint exceeds dense89.44 in this s10 case. The old deleted clipped FedAvg was88.12, not the current recovered88.23. Stage-16 unclipped FedAvg R300 was88.64. Keep run provenance distinct.

CIFAR R150 completion is known, **but final R150-budget accuracies were not supplied in this documentation task**. Historical R120 values (dense51.24, FedAvg46.04, best KL56.10 at r=.25/R399) must not be relabeled R150/R499. Run the bundled `docs/tools/export_handoff.py --check` on the actual repository to print current raw endpoints; the JSON registry keeps missing values null.

Single selected seeds do not prove universal superiority, and budget-matched extra rounds do not establish compute/latency/energy savings or reduced overfitting. Interpret overall trends and keep-ratio effects, not just individual favorable values.

## 6. Source, scripts and generated files

Entry path: YAML -> `main.py` -> partition preparation -> `server.py` / `budget_server.py` / `strategies/research_strategy.py` -> `client.py` -> `training/{deterministic,fola}.py` -> `posterior/{sparse,aggregation,packing}.py` -> central `evaluation.py` and `logging_utils.py`. `communication.py` and `experiment_state.py` own byte/accounting/resume integration. Models include current `models/mnist_cnn.py`, `models/mnist_mlp.py`, `models/paper_cnn.py`; inspect factory before using identifiers.

Core/user-facing scripts: `_run_gpu_queue.sh`, `mnist_stage18_unclipped_fola.py`, `mnist_stage22_sparse_s10_seed0.py`, `mnist_stage21_sparse_s5_seed3.py`, `prepare_cifar_r150_sparse_resume.py`, `recover_cifar_plot.py`, main plotter. Stage-23 recovery config/queue exists; do not assume a previously downloaded standalone generator was installed if the registry does not list it.

Current MNIST config families: `mnist_stage18_fola_noclip_cnn/`, `mnist_stage22_sparse_cnn_s10_seed0/`, `mnist_stage23_recovery/`. Current CIFAR templates occur in `sparse_main_seed0/`, `comm_budget_extension_seed0/`, `confirm_newsrc_dense/`, `cifar_plot_recovery/`; some share run_names but differ in caps. The exact run's saved resolved config outranks a guessed template.

Per-run essentials: `metrics/global_metrics.csv`, `resolved_config.yaml`, `source_config.yaml`, partition metadata, protocol/layout metadata, run summary. For resume also preserve `communication/events.jsonl`, `resume/latest.json` and the matching state JSON/NPZ. Partition NPZ/JSON files live under `outputs/partitions/`. Weights alone do not restore accounting/RNG/config consistency; metrics alone do not resume training.

## 7. Cleanup and Git state

Stage2: 121 historical folders archived, 42 protected original folders verified; same-SSD organization, no independent backup or space saving. Stage3B: user reports 13 historical scripts/queues archived. Keep configs and other scripts in original locations; static refs=0 does not prove unused.

Cleanup managers depend on `../bayesfl_cleanup_DRY_RUN_20261026_194253.csv` and archive manifests/journals. Two incomplete N100/s10 unclipped FOLA runs are protected at last reported R152 seed3 and R172 seed4. Do not delete or relocate them.

`.gitignore` changes were discussed but **not confirmed applied**. A clone must include required metrics (24 primary trajectories, plus optional controls) to plot; source alone is insufficient. Archives/weights can be excluded from Git while preserving local full outputs. Do not globally ignore all NPZ/CSV, lose partition/test fixtures, or ignore a ledger and then assume the clone can resume. Current code, configs, metadata, docs and selected metrics need explicit transfer. The old source CHANGE_MANIFEST.json and PACKAGE_MANIFEST.txt are immutable historical artifacts, not current live inventories.

## 8. Next work

The present task is documentation/handoff. No new training or wireless patch is implied. An optional live export captures current source, configs, primary raw metrics, hashes and package versions, without model arrays/datasets or changing existing files:

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
/home/micl/anaconda3/envs/dungndh/bin/python docs/tools/export_handoff.py --check
/home/micl/anaconda3/envs/dungndh/bin/python docs/tools/export_handoff.py --export
```

The export is a lightweight chat/plot snapshot, **not a full training backup**. Attach it with this handoff for a code-editing task. The proposed next research topic is wireless/AirComp uplink and condition-dependent client keep ratios; none is implemented or certified by these docs. Read current code and relevant project papers before that design.

## 9. Exact primary run paths

All paths below are relative to the repository. Required CSV path is `<run>/metrics/global_metrics.csv`. These explicit timestamps are from the recorded user evidence; do not silently replace a missing run with another seed/method.

### mnist_s10

**dense_fola — R150**

```text
outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s10_n20_seed0_e5_b10_lr0p10_lam0p01_r200_20261009_154443
```

**kl_sparse r=0.05 — R1362**

```text
outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p05_budget150_r1362_20261011_023714
```

**kl_sparse r=0.10 — R950**

```text
outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p10_budget150_r950_20261011_011043
```

**kl_sparse r=0.25 — R498**

```text
outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p25_budget150_r498_20261011_003912
```

**kl_sparse r=0.50 — R278**

```text
outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p50_budget150_r278_20261011_002442
```

**kl_sparse r=0.75 — R193**

```text
outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p75_budget150_r193_20261011_001511
```

**random_sparse r=0.05 — R1362**

```text
outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p05_budget150_r1362_20261011_023914
```

**random_sparse r=0.10 — R950**

```text
outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p10_budget150_r950_20261011_011143
```

**random_sparse r=0.25 — R498**

```text
outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p25_budget150_r498_20261011_003942
```

**random_sparse r=0.50 — R278**

```text
outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p50_budget150_r278_20261011_002512
```

**random_sparse r=0.75 — R193**

```text
outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p75_budget150_r193_20261011_001511
```

**fedavg_baseline — R300**

```text
outputs/run_mnist_stage23_cnn16_32_fixed1c_s10_n20_fedavg_clip_seed0_e5_b10_lr0p10_r400_20261011_091458
```


### cifar10_a0p01_f1

**fedavg_baseline — R300**

```text
outputs/run_budgetmatch_cifar_basiccnn_a0p01_fedavg_n20_f1_seed0_e10_b32_lr001_budget150densefola_20260927_103154
```

**dense_fola — R150**

```text
outputs/run_confirm_newsrc_cifar_basiccnn_a0p01_fola_cos400_n20_f1_seed0_e10_b32_lr002_lam0p01_r150_20261010_195905
```

**kl_sparse r=0.05 — R1362**

```text
outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p05_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260927_193202
```

**kl_sparse r=0.10 — R950**

```text
outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p1_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_071300
```

**kl_sparse r=0.25 — R499**

```text
outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p25_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_090138
```

**kl_sparse r=0.50 — R278**

```text
outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p5_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260924_194221
```

**kl_sparse r=0.75 — R193**

```text
outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p75_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260924_194221
```

**random_sparse r=0.05 — R1362**

```text
outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p05_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260928_024119
```

**random_sparse r=0.10 — R950**

```text
outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p1_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_192526
```

**random_sparse r=0.25 — R499**

```text
outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p25_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_194658
```

**random_sparse r=0.50 — R278**

```text
outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p5_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_043532
```

**random_sparse r=0.75 — R193**

```text
outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p75_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_032226
```
