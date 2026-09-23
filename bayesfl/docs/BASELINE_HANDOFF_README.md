# BayesFL — source guide, experiment protocol, and project handoff

**Snapshot documented:** 23 September 2026  
**Package:** `bayesfl-base`, version `0.1.0`  
**Current research focus:** FedAvg and FOLA on MNIST and CIFAR-10  
**Purpose:** Place this README beside the supplied source, or upload both to a new project/chat, to continue the work without reconstructing the earlier conversation.

> **Read this first.** The current official experiments are **150-round, seed-0 runs**. MNIST uses the MLP and constant learning rates. Official CIFAR-10 uses **BasicCNN**, constant-LR FedAvg, and cosine-LR FOLA. The earlier **ResNet-56, 50-round, seeds 1–4 study is a different experiment family**. Do not mix those results, schedules, initial precisions, or output directories.

## Contents

1. [Evidence and snapshot boundaries](#1-evidence-and-snapshot-boundaries)
2. [Decisions to preserve in a new chat](#2-decisions-to-preserve-in-a-new-chat)
3. [Capabilities and limits](#3-capabilities-and-limits)
4. [Source-code architecture](#4-source-code-architecture)
5. [One federated simulation, step by step](#5-one-federated-simulation-step-by-step)
6. [FedAvg, FOLA, and BBB implementations](#6-fedavg-fola-and-bbb-implementations)
7. [Models](#7-models)
8. [Data, augmentation, and partitioning](#8-data-augmentation-and-partitioning)
9. [Locked hyperparameters](#9-locked-hyperparameters)
10. [Installation and environment](#10-installation-and-environment)
11. [Configure, validate, and run simulations](#11-configure-validate-and-run-simulations)
12. [Outputs, naming, and checkpoints](#12-outputs-naming-and-checkpoints)
13. [Monitoring and terminal-only result extraction](#13-monitoring-and-terminal-only-result-extraction)
14. [Plot generation](#14-plot-generation)
15. [Metrics and multi-seed interpretation](#15-metrics-and-multi-seed-interpretation)
16. [Optional posterior snapshots and compression research](#16-optional-posterior-snapshots-and-compression-research)
17. [Experiment results carried forward](#17-experiment-results-carried-forward)
18. [Exact official config and run registry](#18-exact-official-config-and-run-registry)
19. [Tests and validation status](#19-tests-and-validation-status)
20. [Known limitations and troubleshooting](#20-known-limitations-and-troubleshooting)
21. [Handoff and preservation checklist](#21-handoff-and-preservation-checklist)

## 1. Evidence and snapshot boundaries

This document distinguishes two evidence sources:

| Label | Meaning |
|---|---|
| **Source-verified** | Inspected in the uploaded `bayesfl.zip`: Python modules, YAML configs, shell scripts, plotting scripts, and tests. File paths and function names below are the primary references. |
| **Conversation-reported** | Numerical experiment results and research decisions supplied in the preceding conversation. These results are preserved here for continuity, **not independently recomputed from experiment CSVs**. |

The uploaded archive contains **398 YAML configs**, **32 package Python files**, **16 test files**, **44 queue files**, and plotting/launcher scripts. It contains **no `outputs/` experiment tree, no downloaded datasets, and no actual training logs**; `logs/` contains only `.gitkeep`. Consequently, the source archive alone cannot regenerate historical curves or verify reported accuracies. Keep the actual run directories and partition files separately.

Archive SHA-256:

```text
39e6782c00d21bf7fd83708f64b6e4f30fd9d7f657f284d83746bfeb3f36d802
```

The previously bundled `README.md`, `VALIDATION.md`, `PACKAGE_MANIFEST.txt`, and parts of `docs/` describe earlier snapshots. They are historical notes, not the current experimental contract. Examples of discrepancies found during this review are documented in Section 20. **For new runs, the chosen YAML and its saved `resolved_config.yaml` are authoritative.**

This is documentation of the uploaded implementation, not a fresh certification that it exactly reproduces a paper. No production source was modified to prepare this README.

## 2. Decisions to preserve in a new chat

- **Do not resume hyperparameter tuning by default.** The project moved from tuning to official runs and replication. Propose new conditions explicitly rather than silently changing a lock.
- **Maximum local epochs: `E <= 10`.** Current official experiments use `E=10`. This is a user-imposed compute limit, **not an upper-bound check enforced by `ExperimentConfig.validate()`**. Historical E20/E40 configs remain in the archive.
- Current priority is **FedAvg versus FOLA**. BBB is still implemented and tested, but is not the active comparison target.
- Preserve both `paper_dirichlet` and `fixed_labels`. The new fixed-label path is additive; it does not replace Dirichlet.
- Give execution commands **and a corresponding result-print command**. Numerical summaries should print to the terminal, not automatically create `.txt` files or use `tee`. Plots, configs, manifests, and normal simulation logs are intentionally saved.
- Use the same client-index manifest for a paired comparison. Preserve seed, preprocessing, architecture, and parameter ordering. Do not infer matching from similar folder names alone.
- Official CIFAR comparisons are **training-configuration comparisons**, not an isolated test of the aggregation rule: FOLA has a cosine schedule, and at alpha 0.01 it also has a different initial LR from FedAvg.
- A single positive seed does not establish universal superiority. Do not select only favorable fractions or retrospectively rename a metric to claim success. Preserve unsuccessful and inconclusive results.
- Keep snapshot saving **off** in the current official/tuning configs unless the user explicitly starts the compression study.
- Never regenerate a missing accuracy curve from a handful of reported checkpoints, or present an interpolated curve as actual training history.

## 3. Capabilities and limits

| Component | What this source supports | Important boundary |
|---|---|---|
| Federated execution | Flower client/server apps with Ray-backed virtual clients | Local simulation; not a provided multi-machine deployment/orchestration system |
| FedAvg | Deterministic local SGD, example-count-weighted parameter averaging | SGD only; no built-in FedProx, SCAFFOLD, or FedNova implementation |
| FOLA | Deterministic networks with custom diagonal curvature/precision state; two explicit modes | `paper_reference` and `online_recurrence` differ materially |
| BBB | Bayesian-Torch Conv/Linear reparameterization layers, custom variational loss, federated posterior aggregation | Retained extension; not the current official research focus |
| Datasets | Torchvision MNIST and CIFAR-10 | No other dataset dispatch implemented |
| Models | MNIST MLP; CIFAR BasicCNN; CIFAR ResNet-56 GN8 | Exact accepted model identifiers are listed in Section 7 |
| Dirichlet partitions | Paper-style for both datasets; legacy MNIST lognormal and CIFAR sparse variants | The variants do not mean the same distribution |
| User-set partition | MNIST `fixed_labels`, configurable sample count, exactly one class/client | **Only `labels_per_client=1`; clients must divide evenly among classes** |
| Data reduction | Deterministic per-client nested fractions of the base Dirichlet/legacy partition | This changes class coverage and local step counts as well as sample count |
| Scheduling | Constant/none and one-based round cosine schedule | Cosine horizon is independent of total rounds |
| Evaluation | Central global test accuracy, NLL, Brier, ECE/MCE, confidence, entropy, MI | Local `evaluate()` is disabled; train accuracy is not global test accuracy |
| Persistence | Configs, environment, partitions, CSVs, reliability arrays, global checkpoints | No training-resume CLI or saved optimizer state |
| Snapshots | Optional incoming global, outgoing clients, and uncompressed outgoing global states | No implemented compression-score comparison/evaluator |
| Plotting | Single-run diagnostics, multi-run comparisons, bespoke official figures, older multi-seed script | Generic `utils.py` comparisons do **not** average seeds automatically |
| Fixed optimizer-step diagnostic | Not present | The proposed `max_local_steps_per_round` change was declined and is absent from this snapshot |

**Library naming:** the installed distribution is **`bayesian-torch`**, imported as **`bayesian_torch`**. It is not a separate package named “bayes-torch.” In this project it supplies BBB layers; **FOLA does not convert its model into a Bayesian-Torch model**.

## 4. Source-code architecture

```text
bayesfl repository root/
├── pyproject.toml                    # package metadata, dependencies, entry points
├── requirements.txt                 # declared runtime/test requirements
├── README.md                        # this handoff document
├── docs/                            # historical algorithm/profile notes
├── scripts/
│   ├── configs/                     # 398 stored experiment YAML files
│   ├── queues/                      # newline-delimited config paths
│   ├── _run_nohup.sh                 # one detached simulation
│   ├── _run_gpu_queue.sh             # sequential jobs on one selected GPU
│   ├── check_environment.sh
│   ├── validate_install.sh
│   ├── generate_plots.sh
│   ├── plot_official_accuracy_curves.py
│   ├── plot_accuracy_rounds.py
│   ├── plot_basiccnn_lr_schedule_compare.py
│   └── run_*.sh                     # legacy wrappers / smoke / sweep launchers
├── src/bayesfl/
│   ├── config.py                    # dataclasses, YAML load/validation, LR
│   ├── main.py                      # CLI, run setup, partition preparation
│   ├── server.py                    # Flower/Ray assembly
│   ├── client.py                    # NumPyClient and local fit dispatch
│   ├── evaluation.py                # global evaluation and checkpoints
│   ├── metrics.py                   # numerical calibration/uncertainty metrics
│   ├── logging_utils.py             # run paths, environment, CSV recorder
│   ├── runtime_utils.py             # seeds, devices, CUDA cleanup
│   ├── utils.py                     # post-training plot APIs and CLI
│   ├── data/
│   │   ├── datasets.py              # loaders, partition dispatch/cache
│   │   ├── partition.py             # all partition generators + persistence
│   │   └── transforms.py            # normalization, augmentation, Cutout
│   ├── models/
│   │   ├── factory.py               # model selection and shared initialization
│   │   ├── mnist_mlp.py
│   │   ├── paper_cnn.py
│   │   ├── cifar_resnet.py
│   │   └── bayesian_layers.py        # BBB-only Bayesian-Torch constructors
│   ├── training/
│   │   ├── deterministic.py         # FedAvg local training
│   │   ├── fola.py                  # two FOLA local-training modes
│   │   └── bbb.py                   # BBB local training and variance floor
│   ├── strategies/
│   │   ├── common.py                # example weights and metric aggregation
│   │   └── research_strategy.py     # aggregation + optional snapshots
│   └── posterior/
│       ├── packing.py               # named-parameter transport layout
│       ├── gaussian.py              # Gaussian products/precision recurrence
│       ├── diagnostics.py           # posterior and update summaries
│       └── scale_mixture.py         # BBB prior densities/complexity
└── tests/                           # offline unit/import tests
```

`data/`, `outputs/`, and real `logs/*.log` are runtime artifacts and are not supplied in this source archive. Several `*.before_matched_init` files are archival backups, not active modules.

The dependency direction is intentionally simple:

```text
YAML -> main -> partition preparation -> server/strategy
                                      -> virtual client -> local training
                                      -> central evaluator -> CSV/checkpoint
saved artifacts -> utils.py / plotting scripts
```

Training modules do not import plotting utilities. This keeps Matplotlib out of the local-training execution path.

## 5. One federated simulation, step by step

Source: `main.py`, `server.py`, `client.py`, `strategies/research_strategy.py`, and `evaluation.py`.

1. **Resolve configuration.** `main.py` loads the explicit YAML (or a legacy default), applies the supported CLI overrides, and validates the config.
2. **Create a fresh run.** Save `resolved_config.yaml`, the original `source_config.yaml`, and `environment.json` under a timestamped run directory. Each launch starts from initialization; it does not resume an older run.
3. **Prepare client data.** `prepare_partition()` reuses a cached NPZ/JSON pair, or downloads the training dataset and builds the manifest. Copy partition statistics into the run's `partition_metadata.json`.
4. **Initialize server state.** `initialize_model()` seeds model initialization. `ParameterLayout` fixes the order and shape of `named_parameters()`. FOLA carries one mean and one precision/omega array per parameter tensor.
5. **Create Flower apps.** `server.py` constructs `ClientApp`, `ServerApp`, `ResearchStrategy`, the test loader, and `CentralEvaluator`, then invokes `flwr.simulation.run_simulation(..., backend_name="ray")`.
6. **Evaluate round 0.** The initial model is evaluated and checkpointed before local training.
7. **Dispatch a round.** The strategy requests `clients_per_round` fits, waits for the configured minimum available clients, and sends the global arrays plus a round number. All official configs use full participation.
8. **Train each selected client.** The client loads its saved indices, builds a fresh model, loads incoming parameters, creates a fresh optimizer, and dispatches to FedAvg, FOLA, or BBB. Optimizer momentum is **not persisted across communication rounds**.
9. **Return local state.** The client sends parameter arrays, its unique assigned example count, and scalar diagnostics. Each fit finally releases the model and invokes CUDA cache cleanup.
10. **Aggregate.** Results are weighted by `client_size / sum(selected_client_sizes)`. The strategy raises on failures or a result-count mismatch; it does not silently average a partial set.
11. **Evaluate and save.** The server evaluates the aggregated model on the common test set, records metrics/reliability/posterior summaries, and saves scheduled global checkpoints. Repeat through the configured final round.
12. **Finish.** The application logs `Simulation finished successfully: <run_dir>`.

**Seed handling.** A client's fit seed is:

```text
runtime.seed + 1_000_003 * client_id + 10_007 * server_round
```

Central evaluation uses `runtime.seed + 99_991 * server_round`. Python, NumPy, Torch, and CUDA generators are seeded. No strict deterministic-algorithm enforcement is enabled, so seeds do not guarantee bitwise identity across hardware, software, or asynchronous execution. The effective partition seed is **`runtime.seed`**, not an arbitrary `seed` key added inside the partition dictionary.

**Participation is not parallelism.** `100/100` means 100 client updates are required in the round; Ray may execute them in batches of actors. Resource fractions control scheduling, not the number of participants.

## 6. FedAvg, FOLA, and BBB implementations

### 6.1 FedAvg

Source: `training/deterministic.py::train_fedavg`, `strategies/common.py::weighted_average_arrays`.

Each client minimizes mean cross-entropy using SGD for `local_epochs` complete local passes. Optional gradient-norm clipping happens before `optimizer.step()`. The server computes an example-weighted arithmetic parameter mean:

```text
w_k = n_k / sum_j n_j
mu_global = sum_k w_k * theta_k
```

The local batch-weighted loss and accuracy are recorded across all samples visited in all local epochs. `local_steps` counts actual optimizer updates. `num_examples` is the assigned client dataset size, not epochs times dataset size.

### 6.2 FOLA: `paper_reference` — current official mode

Source: `training/fola.py::_train_fola_paper_reference` and `ResearchStrategy._aggregate_fola`.

FOLA uses an ordinary deterministic network as its mean model, plus one diagonal **omega** tensor for every named parameter. The code stores omega in arrays/fields called `precision`; in this mode it is an operational accumulated-curvature state, not a guarantee of calibrated Bayesian uncertainty.

At round `r`, the client starts from the global means and copies global omega into local omega. For each local minibatch `b`:

```text
L_task = mean cross_entropy(logits, labels)
g_b = gradient of L_task with respect to model parameters
omega_local += (batch_size_actual / client_size) * square(g_b)

L_prior = (0.5 / r) * sum_j omega_global[j] * (theta[j] - mu_global[j])^2
L_train = L_task + prior_lambda * L_prior
```

It obtains task-only gradients using `torch.autograd.grad`, accumulates their squares **before gradient clipping**, and separately backpropagates the regularized objective for SGD. The prior is centered on the incoming global model, not the changing local mean. The client returns its trained mean and the accumulated local omega.

Server aggregation is elementwise:

```text
omega_global_new = sum_k w_k * omega_local_k
mu_global_new = sum_k w_k * omega_local_k * mu_local_k
                / (omega_global_new + aggregation_epsilon)
```

Important implementation details:

- This is **squared minibatch-mean task gradients**, not a computed per-example Fisher/Hessian. Do not substitute a per-example formula when explaining these runs.
- The local omega increment is accumulated across all E epochs; there is no division by E at the end of this branch.
- `lambda_scale_by_size` is not used in this branch. The current configs set it false.
- The local variance-floor function and precision clipping used by the other mode are **not applied to paper-reference training/aggregation**. The reported local `variance_floor_fraction` is zero in this branch.
- The `0.5/r` prior factor is part of the recorded `prior_loss`; the displayed value has **not** already been multiplied by lambda.
- Zero `prior_lambda` removes the local prior penalty, **not** Gaussian-product aggregation. It is not equivalent to FedAvg.
- `aggregation_epsilon` changes the actual aggregated mean when omega is small; it is not only a display safeguard.
- Initialization is supplied by the YAML. **Official MNIST starts omega at 1.0; official CIFAR starts at 0.0**, despite the mode docstring discussing zero initialization. Preserve this distinction when reproducing results.

Do not silently replace this branch with the paper equation implemented by the other mode. They have different operational semantics.

### 6.3 FOLA: `online_recurrence` — retained alternative

Source: `training/fola.py::_train_fola_online_recurrence`, `posterior/gaussian.py`.

This branch forms a prior penalty without the paper-reference `1/r` multiplier, optionally scales lambda by `client_size / average_client_size`, and accumulates `batch_size_actual * square(task_gradient)`. It divides that accumulator by the number of optimizer steps, then calls the precision helper:

```text
P_local = gamma + F_local/r + ((r-1)/r) * (P_global - gamma)
```

Here `gamma=initial_precision` must be positive for the helper. Precision is clipped to configured bounds, and a variance floor can limit local precision to `P_global / variance_floor_ratio^2`. Server aggregation uses the clipped Gaussian-product utility.

This alternative is covered by recurrence/variance-floor tests. Those tests **do not, by themselves, validate the paper-reference branch**. The current official runs do not select `online_recurrence`.

### 6.4 Bayes by Backprop (BBB) — implemented but not active in the official study

Source: `models/bayesian_layers.py`, `training/bbb.py`, `posterior/scale_mixture.py`, `ResearchStrategy._aggregate_bbb`.

Bayesian-Torch supplies trainable `mu` and `rho` tensors and stochastic Conv/Linear forward passes:

```text
sigma = softplus(rho)
weight = mu + sigma * epsilon
```

The project calls these layers with `return_kl=False`. It reconstructs the actual sampled weights from their epsilon buffers and computes its own Monte Carlo `log q - log p` complexity term. Supported priors are `scale_mixture`, `standard_normal`, and the alias `normal`.

Local training averages `mc_train` stochastic objectives. The complexity multiplier includes resolved KL weight, minibatch weighting (`equal_minibatch` or `blundell_geometric`), optional size scaling, and optional round warmup. A null `kl_weight` resolves to `1/d`, or `1/kl_reference_dimension` when explicitly overridden. `rho_lr_multiplier` scales the LR for rho parameters. A post-training floor enforces `sigma_local >= ratio * sigma_global`.

`bbb.aggregation` selects either precision-weighted Gaussian-product posterior aggregation or direct averaging of all variational parameters (`fedavg_variational`). Deterministic parameters are averaged normally. **Federated BBB aggregation is a project extension, not something supplied by the original BBB paper.** `match_deterministic_init` optionally copies the deterministic model's initialization into posterior means.

## 7. Models

Source: `models/factory.py`, `mnist_mlp.py`, `paper_cnn.py`, and `cifar_resnet.py`.

| YAML `model.name` | Dataset | Architecture | Deterministic parameter count |
|---|---|---|---:|
| `mlp_784_500_300_10` | MNIST | Flatten 28×28 -> Linear 784–500 -> ReLU -> Linear 500–300 -> ReLU -> Linear 300–10 | 545,810 |
| `paper_basiccnn` | CIFAR-10 | Conv 3–32, 5×5 -> ReLU/MaxPool2; Conv 32–64, 5×5 -> ReLU/MaxPool2; flatten 1600 -> Linear 512 -> ReLU -> Linear 10 | 878,538 |
| `resnet56_gn8` | CIFAR-10 | CIFAR 6n+2 network, n=9; stages 16/32/64 channels, 9 residual blocks each; projection shortcuts; GroupNorm-8; global average pooling; Linear 64–10 | 855,578 |

Counts above were computed from the uploaded deterministic model builders. BasicCNN's Conv/Linear weights use Xavier-uniform initialization and zero biases. The MLP and ResNet constructors retain their layer constructors' initialization; there is no shared BasicCNN-style Xavier pass for them.

BasicCNN does **not** contain GroupNorm. An inherited `group_norm_groups: 8` field in its YAML is inert. ResNet explicitly requires eight groups.

For BBB ResNet, **851,514** variables are stochastic Conv/Linear weight/bias elements; the remaining **4,064** GroupNorm affine parameters remain deterministic. The total deterministic/FOLA ResNet dimension is therefore **not** 851,514. FOLA tracks every named parameter, including GroupNorm affine parameters. BBB BasicCNN's stochastic dimension is 878,538.

`ParameterLayout`/transport use `named_parameters()`, not a complete state dict including buffers. Adding a model with meaningful persistent buffers requires an explicit transport/evaluation audit.

## 8. Data, augmentation, and partitioning

### 8.1 Loaders and evaluation data

Source: `data/datasets.py`.

Training uses Torchvision datasets and `Subset` index lists from a shared NPZ manifest. Each client DataLoader has `shuffle=True`, `num_workers=0`, `drop_last=False`, the configured batch size, and a seeded Torch generator. GPU pinning is enabled when CUDA is available.

The central test loader uses the full test dataset, batch size 512, `shuffle=False`, and no training augmentation. `prepare_partition()` downloads the **training** dataset when it must create a manifest. A cache hit returns before this download; transformed training/test loaders use `download=False`. A fresh machine therefore needs both train and test data even when reusing a partition cache. See Section 10.

### 8.2 Current preprocessing

**MNIST official configs:** no augmentation; tensor conversion; mean `[0.1307]`, standard deviation `[0.3081]` normalization.

**CIFAR official configs:** `augment: true`, with this actual code order:

```text
RandomCrop(32, padding=4, fill=128)
-> RandomHorizontalFlip
-> torchvision CIFAR10 AutoAugment
-> ToTensor
-> Cutout (1 hole, length 16; tensor region filled with zero)
-> Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
```

Test transforms are tensor conversion plus normalization. Setting `augment: false` disables the crop, flip, AutoAugment, and Cutout operations regardless of their retained fields.

**Correction to earlier informal discussion:** the current official CIFAR configs **do include paper-style augmentation**. Do not carry forward the claim that this snapshot lacks it. The implementation uses Torchvision's CIFAR policy, not a vendored copy of the authors' augmentation helpers.

Source: `data/transforms.py` and the official YAML files.

### 8.3 Partition modes and fields

| `data.partition.type` | Dataset dispatch | Controls and actual behavior |
|---|---|---|
| `paper_dirichlet` | MNIST and CIFAR-10 | `dirichlet_alpha`; class-wise allocations over clients with integer flooring; no fixed per-client label cap or equal-size guarantee |
| `dirichlet_lognormal` | Legacy MNIST | `dirichlet_alpha`, `lognormal_sigma`, `min_samples_per_client`; class allocation with lognormal size bias; minimum-size repair moves examples from donors |
| `sparse_dirichlet` | Legacy CIFAR-10 | `dirichlet_alpha`, `avg_samples_per_client`, `classes_per_client`, `min_samples_per_client`, optional `target_total_samples`; Poisson sizes and active-class allocation |
| `fixed_labels` | MNIST only | `samples_per_client`, `labels_per_client=1`; equal-sized, single-class clients, balanced client counts across digits, disjoint examples |

Source: `data/partition.py` and `data/datasets.py::prepare_partition`.

#### Paper-style Dirichlet

For each of K classes, the generator draws a length-N vector from `Dirichlet([alpha] * N)`. The resulting matrix has client-by-class orientation. It allocates `floor(class_size * proportion)` examples per client/class using class indices in their existing dataset order, without an additional class-pool shuffle in this implementation.

Two reference-compatible edge cases matter:

- Integer rounding leaves some examples unassigned, even at `local_data_fraction=1.0`.
- An empty client is assigned the **first class-0 example**. This can duplicate a training index across clients. “Total samples used” counts assignments, which may differ from unique examples.

Always inspect `total_unique_samples_used`, backfills, and the actual manifest. Do not describe every Dirichlet partition as disjoint or perfectly balanced. The reported full-data seed-0 means in the earlier study were approximately 2,497 CIFAR examples/client at alpha 0.1 and 2,499.4 at alpha 0.01; those are realized counts, not user-entered per-client sizes.

Some legacy fields remain in MNIST paper-Dirichlet configs (`lognormal_sigma`, `min_samples_per_client`). They are **not used by the paper branch**. In particular, a retained minimum of 100 does not prevent one-example clients in that mode.

#### Local data fraction

After making the base non-fixed partition, `apply_local_data_fraction()` keeps:

```text
n_retained[k] = max(1, floor(fraction * n_base[k]))
```

Its permutation stream is seeded from `(runtime.seed + 104729) % (2**32 - 1)` and is independent of fraction. For the same base manifest and seed, smaller fractions are nested per-client subsets. Fraction 1 returns the original partition unchanged.

`local_data_fraction` is **not** client participation. It reduces each client's available unique data; all official clients still participate in every round. It also changes label coverage, effective aggregation weights through rounding, and optimizer steps at fixed E/B. Exact mean sample counts must come from the generated manifest, not `fraction * nominal_mean` alone.

The official MNIST approximately-10-example setting stores **`1/60 = 0.016666666666666666`**, not a truncated decimal, in YAML. The run-name label `f0p016667` is only a human-readable tag.

#### Fixed-label distribution

The active implementation is deliberately narrower than a general K-label allocator:

```yaml
data:
  partition:
    type: fixed_labels
    samples_per_client: 10
    labels_per_client: 1
```

For N=100 and K=10 it assigns exactly 10 clients per digit, shuffles client-to-digit assignment using the seed, shuffles each class pool, and allocates 10 disjoint examples/client. It produces **1,000 distinct training examples, 100 examples per digit, mean=min=max=10, one class/client**.

Restrictions enforced by the generator:

- Positive client, class, and sample counts.
- `labels_per_client` must equal **1**; 2 or more is not supported.
- `num_clients % num_classes == 0`.
- Each class must have enough examples for its assigned clients; otherwise it raises.
- Only MNIST is enabled in dataset dispatch.

The fixed-label branch returns before Dirichlet generation or fraction subsampling. Use no alpha/fraction keys for this experiment. Changing `samples_per_client` across fixed-label runs does **not** come with a nested-subset guarantee. It is a different mechanism from `local_data_fraction`.

This is not “alpha 0.01 with a different description.” It changes label assignment, sample-count balance, and possibly global class frequencies relative to the previous Dirichlet experiment.

#### Legacy sparse mode

The sparse generator reserves at least one example for each selected active class, then draws remaining class counts from a Dirichlet allocation. Without class-pool shortages this preserves `classes_per_client`. If pools are exhausted, it may fill from other classes, so the active-class count is not unconditional. The historical synthetic test checks the original non-exhausted 100-client, 10,046-assignment case, not the current official profile.

### 8.4 Partition files and cache safety

`prepare_partition()` writes to `<output.outputs_dir>/partitions/`. Common stems are:

```text
mnist_paper_dirichlet_a0.01_n100_seed0
mnist_paper_dirichlet_a0.01_f0.125_n100_seed0
mnist_paper_dirichlet_a0.01_f0.0166667_n100_seed0
cifar10_paper_dirichlet_a0.1_f0.5_n20_seed0
mnist_fixed_labels_s10_c1_n100_seed0
```

Each stem has `.npz` and `.json` files. NPZ keys are `client_0000`, `client_0001`, etc.; values are int64 dataset indices. The metadata SHA-256 covers client IDs, lengths, and index order. Use `partition_stem(cfg)` to obtain the exact name instead of hand-formatting floats.

**Cache limitations:** an existing NPZ+JSON pair is returned without validating every config field or recomputing its hash. Cache names do not encode the source revision, dataset root/content, or every legacy partition parameter; fraction text uses `:g` formatting. Back up old manifests and use a new output/cache root for deliberately changed partition semantics. Do not overwrite a manifest underlying published results. Generate each shared partition before parallel paired runs, because this cache writer has no lock/atomic-write protocol.

## 9. Locked hyperparameters

### 9.1 Current official profiles — source-verified

All profiles below use **SGD, E=10, B=32, momentum=0, weight decay=0, gradient clip norm=10**, full client participation, seed 0, and **150 communication rounds**. These are explicit official YAML settings, not dataclass defaults.

| Experiment | Clients | Partition/data amount | FedAvg LR | FOLA LR | FOLA lambda |
|---|---:|---|---|---|---:|
| MNIST MLP, Dirichlet alpha 0.01 | 100/100 | Fraction 0.125, approximately 75/client | 0.01 constant | 0.01 constant | 0.01 |
| MNIST MLP, Dirichlet alpha 0.01 | 100/100 | Fraction 1/60, approximately 10/client | 0.01 constant | 0.01 constant | 0.01 |
| MNIST MLP, fixed labels | 100/100 | Exactly 10/client, exactly 1 class/client | 0.01 constant | 0.01 constant | 0.01 |
| CIFAR BasicCNN, Dirichlet alpha 0.1 | 20/20 | Fractions 1.0 and 0.5 | 0.02 constant | 0.02 cosine -> 0.0001, horizon 400 | 0.1 |
| CIFAR BasicCNN, Dirichlet alpha 0.01 | 20/20 | Fractions 1.0 and 0.5 | 0.01 constant | 0.02 cosine -> 0.0001, horizon 400 | 0.01 |

Additional official FOLA and runtime settings:

| Field | MNIST official FOLA | CIFAR official FOLA |
|---|---:|---:|
| `mode` | `paper_reference` | `paper_reference` |
| `lambda_scale_by_size` | false | false |
| `paper_mean_only_eval` | true | true |
| `initial_precision` | **1.0** | **0.0** |
| `aggregation_epsilon` | **1e-5** (resolved default) | **1e-5** |
| `precision_min` | 1e-8 | 1e-12 |
| `precision_max` | 1e8 | 1e12 |
| `variance_floor_ratio` | 0.5; inactive in paper-reference local training | 0.5; inactive in paper-reference local training |
| `mc_eval` | 5; sampling disabled by mean-only evaluation | 5; sampling disabled by mean-only evaluation |
| `runtime.client_num_cpus` | 2.0 | 2.0 |
| `runtime.client_num_gpus` | 0.125 | 0.25 |
| `runtime.torch_num_threads` | 1 | 1 |
| `runtime.central_eval_device` | `auto` | `auto` |
| `output.checkpoint_every` | 10 | 10 |
| `output.save_full_client_posteriors` | false | false |

`precision_min` is still used when converting raw FOLA omega to diagnostic sigma and for optional sampling; it does not imply clipping of the paper-reference aggregation omega. Some FedAvg YAMLs contain inherited `fola.prior_lambda: 100` or other inactive BBB/FOLA fields. **They have no effect when `method: fedavg`**; do not report them as FedAvg regularization.

Two BasicCNN alpha-0.1 fraction pairs remain diagnostic R50 runs: fractions **0.25 and 0.125**, FedAvg constant LR 0.02 versus FOLA cosine base LR 0.02, lambda 0.1, horizon 400.

### 9.2 Earlier locks — retain as separate experiment families

| Family | Model | Rounds / seeds | Shared constant LR | FOLA lambda |
|---|---|---|---:|---:|
| MNIST alpha 0.1 tuning/validation | MLP | R50; prior five-seed 0–4 study | 0.01 | 0.0001 |
| MNIST alpha 0.01 tuning/validation | MLP | R50; prior five-seed 0–4 study | 0.01 | 0.01 |
| CIFAR alpha 0.1 matched-LR replication | ResNet56-GN8 | R50; seed 0 exploration, seeds 1–4 replication | 0.02 | 0.1 |
| CIFAR alpha 0.01 matched-LR replication | ResNet56-GN8 | R50; seed 0 exploration, seeds 1–4 replication | 0.01 | 0.01 |

The ResNet alpha-0.1 independently screened FedAvg LR was 0.05; the user subsequently selected the matched-LR 0.02 comparison. Do not substitute 0.05 into that matched study. These locks were not newly reoptimized for every reduced-data fraction or for every architecture.

The older MNIST output directories were not found in the user's inspected repository during the conversation; configs and summary evidence remained. This does not prove permanent deletion. The newer official MNIST R150 experiments were rerun and reported. The present ZIP carries configs, not either set of raw results.

### 9.3 Exact cosine schedule

Source: `config.py::round_learning_rate`.

```text
H = lr_decay_rounds
r0 = min(server_round - 1, H - 1)
lr(r) = lr_min + 0.5 * (lr_base - lr_min) * [1 + cos(pi * r0/(H-1))]
```

`server_round` is one-based. At H=1, the code returns `lr_min`; at/after H it stays at the minimum. Constant and `none` schedules return the configured base LR.

For the official CIFAR FOLA schedule (base 0.02, minimum 0.0001, H=400):

| Round | LR computed from this source |
|---:|---:|
| 1 | 0.02000000 |
| 50 | 0.01926862 |
| 100 | 0.01712714 |
| 150 | 0.01390289 |
| 300 | 0.00302815 |
| 400+ | 0.00010000 |

**R150 does not mean the cosine has decayed to its minimum.** Changing `training.rounds` leaves the schedule horizon unchanged. LR is computed for each federated round and is constant during that round's local epochs.

## 10. Installation and environment

Source: `pyproject.toml`, `requirements.txt`, and environment scripts.

| Item | Declared/recorded configuration |
|---|---|
| Python support in package metadata | `>=3.10,<3.11` |
| User's established environment | Python 3.10.20, Linux, Conda environment `dungndh` |
| Declared Flower dependency | **`flwr[simulation]==1.29.0`** |
| Declared Bayesian-Torch dependency | **`bayesian-torch==0.5.0`** |
| NumPy | `>=1.26,<3` |
| PyYAML | `>=6.0,<7` |
| Matplotlib | `>=3.8,<4` |
| Pytest | `>=8,<9` |
| Ray | Brought in by the Flower simulation dependency; no explicit independent pin in these manifests |
| Recorded target Torch stack | Existing CUDA-enabled Torch 2.7.0+cu128 and compatible Torchvision 0.22.x |

These are the **uploaded project's declarations**, not a claim about today's newest supported versions. Torch/Torchvision are not explicitly pinned by this package. Preserve the working matching CUDA pair and inspect any installation's dependency changes rather than assuming the entire environment is locked by this file.

The user's previous machine paths were:

```text
Repository: /home/micl/DungNDH/micl/bayesfl
Python:     /home/micl/anaconda3/envs/dungndh/bin/python
```

They are conveniences for that machine, not required installation paths.

```bash
cd ~/DungNDH/micl/bayesfl
conda activate dungndh

python -c 'import sys; print(sys.executable)'
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

bash scripts/check_environment.sh
python -m pytest -q
```

An editable installation is recommended for new development. `PYTHONPATH="$PWD/src"` is an alternative for importing the local package, but it does not install dependencies or select the right interpreter.

**Background jobs use `python` from PATH in this archive.** The supplied `_run_nohup.sh` and `_run_gpu_queue.sh` do not hardcode or activate `dungndh`. This differs from an earlier proposed troubleshooting patch. To use the established interpreter for shell-launched jobs:

```bash
export PATH="/home/micl/anaconda3/envs/dungndh/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import sys, bayesfl; print(sys.executable); print(bayesfl.__file__)'
```

For a fresh data directory, explicitly download both splits before a background run (safe to repeat with an existing download):

```bash
python - <<'PYCODE'
from torchvision.datasets import MNIST, CIFAR10
for dataset in (MNIST, CIFAR10):
    for train in (True, False):
        dataset(root='./data', train=train, download=True)
print('Training and test datasets are available.')
PYCODE
```

Use the selected config's `data.root` instead of `./data` when different. Do not accidentally train on the test split.

`bash scripts/validate_install.sh` also checks BBB's ResNet dimension and therefore requires Bayesian-Torch. The fast pytest suite is not a full Flower training smoke run.

## 11. Configure, validate, and run simulations

### 11.1 CLI input arguments

Source: `main.py::parse_args`; installed entry points in `pyproject.toml`.

```bash
python -m bayesfl.main --config scripts/configs/<experiment>.yaml
# Equivalent installed console entry:
bayesfl --config scripts/configs/<experiment>.yaml
```

| Argument | Effect |
|---|---|
| `--config PATH` | Load a YAML configuration |
| `--dataset {mnist,cifar10}` | Override dataset; without a config helps select the legacy default YAML |
| `--method {fedavg,bbb,fola}` | Override algorithm |
| `--rounds INT` | Override number of communication rounds |
| `--seed INT` | Override `runtime.seed` |

There is **no CLI `--lr`, `--alpha`, `--fraction`, `--model`, `--resume`, or `--gpu`**. Set those experiment fields in YAML; control physical GPU visibility through the environment.

Dataset/method overrides do not rebuild all other compatible settings. For example, changing dataset does not automatically change model, normalization, or partition fields. Likewise `--seed` and `--rounds` do **not** rewrite `run_name`; clone YAML with a matching name for official replication rather than producing a `seed0_r150` folder that actually used another seed/budget.

Without `--config`, defaults resolve to `scripts/configs/{method}_{dataset}.yaml`. These are **legacy starting profiles**, not the current official locks. Some wrapper scripts select those legacy configs; `run_fola_cifar10.sh` uses `fola_cifar10_selected.yaml` from an old sweep. Prefer explicit official config paths.

### 11.2 YAML sections

| Section | Principal fields |
|---|---|
| Top level | `run_name`, `method` |
| `data` | `dataset`, `root`, `num_classes`, transform settings, `partition` dictionary |
| `federation` | `num_clients`, `clients_per_round` |
| `model` | `name`, `group_norm_groups` |
| `training` | `optimizer`, `lr`, `momentum`, `weight_decay`, `batch_size`, `local_epochs`, `rounds`, `lr_schedule`, `lr_min`, `lr_decay_rounds`, `grad_clip_norm` |
| `fola` | `mode`, `prior_lambda`, `initial_precision`, `aggregation_epsilon`, precision bounds, size-scaling, MC evaluation, variance floor, mean-only evaluation |
| `bbb` | Prior type/parameters, posterior initialization, KL normalization/scheme/warmup, MC counts, rho LR, variance floor, aggregation, matched initialization |
| `runtime` | Client CPU/GPU resources, central evaluation device, seed, Torch threads, Flower verbosity |
| `output` | Output/log roots, checkpoint interval, optional full-posterior snapshot controls |

Dataclass sections reject unsupported keyword fields. Partition is an untyped dictionary and not all of its keys are validated at config-load time. Adding `max_local_steps_per_round` to `training` in this snapshot raises an unexpected-keyword error; the field is not implemented.

### 11.3 Clone a locked config for a new seed

The following is a **documentation recipe**, not an additional simulation already executed. It preserves a locked configuration and writes a new name rather than editing the old one:

```bash
python - <<'PYCODE'
from pathlib import Path
import copy
import yaml
from bayesfl.config import load_config

source = Path('scripts/configs/run_mnist_fixed1c_s10_official_fola_n100_seed0_e10_b32_lr001_lam0p01_r150.yaml')
seed = 1
cfg = copy.deepcopy(yaml.safe_load(source.read_text()))
cfg['runtime']['seed'] = seed
cfg['run_name'] = cfg['run_name'].replace('_seed0_', f'_seed{seed}_')
assert cfg['training']['local_epochs'] <= 10

target = source.with_name(cfg['run_name'] + '.yaml')
if target.exists():
    raise FileExistsError(f'Refusing to overwrite {target}')
target.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
checked = load_config(target)
print(target)
print('seed=', checked.runtime.seed, 'rounds=', checked.training.rounds)
PYCODE
```

Changing `runtime.seed` changes both partitioning and model/training randomness. There is no separate first-class partition-seed versus initialization-seed experiment control in the current execution path.

### 11.4 Validate and pre-generate the shared partition

For the already-supplied fixed-label official pair:

```bash
python - <<'PYCODE'
from pathlib import Path
from bayesfl.config import load_config
from bayesfl.data.datasets import prepare_partition

paths = [
    Path('scripts/configs/run_mnist_fixed1c_s10_official_fedavg_n100_seed0_e10_b32_lr001_r150.yaml'),
    Path('scripts/configs/run_mnist_fixed1c_s10_official_fola_n100_seed0_e10_b32_lr001_lam0p01_r150.yaml'),
]
previous = None
for path in paths:
    cfg = load_config(path)
    assert cfg.training.local_epochs <= 10
    manifest, meta = prepare_partition(cfg)
    signature = (str(manifest), meta['sha256'])
    if previous is not None:
        assert signature == previous, 'Paired methods must share the partition'
    previous = signature
    print(cfg.method, 'partition=', manifest)
    print('mean/client=', meta['mean_size'],
          'classes/client=', meta['mean_classes_per_client'],
          'SHA256=', meta['sha256'])
PYCODE
```

For a new experiment family, also check the expected dataset, model, schedule, initial precision, LR, lambda, seed, and sample counts. Loading successfully only establishes schema acceptance.

### 11.5 One foreground run

```bash
CUDA_VISIBLE_DEVICES=0 python -m bayesfl.main \
  --config scripts/configs/run_mnist_fixed1c_s10_official_fedavg_n100_seed0_e10_b32_lr001_r150.yaml
```

Run a reduced-round **newly named smoke config** before a new code change consumes a long training budget. A `--rounds` override is useful for a disposable smoke test, but its unchanged name must not be mistaken for a completed official run.

### 11.6 One detached run

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/_run_nohup.sh \
  scripts/configs/run_mnist_fixed1c_s10_official_fedavg_n100_seed0_e10_b32_lr001_r150.yaml
```

`_run_nohup.sh CONFIG [extra bayesfl args...]` changes to repository root, creates `logs/`, starts `nohup python -m bayesfl.main`, and prints a PID and outer log path. It returns immediately; a printed PID does not prove successful training.

### 11.7 Sequential GPU queues

Each queue is a newline-terminated list of config paths, usually relative to repository root. Blank lines and lines starting exactly with `#` are skipped. Avoid leading whitespace, inline comments, and names different from config stems.

Already-supplied queues:

| Queue files | Jobs |
|---|---|
| `official_seed0_gpu0.txt` | 4 official CIFAR alpha-0.1 R150 runs + 4 CIFAR alpha-0.1 R50 debug runs |
| `official_seed0_gpu1.txt` | 4 official MNIST R150 runs + 4 official CIFAR alpha-0.01 R150 runs |
| `mnist_fixed1c_s10_official_gpu0.txt` | Fixed-label MNIST FedAvg, one R150 job |
| `mnist_fixed1c_s10_official_gpu1.txt` | Fixed-label MNIST FOLA, one R150 job |
| `cifar_r56_multiseed_final_gpu0.txt` / `gpu1.txt` | Earlier R50 ResNet replication; 28 jobs each, not the new official BasicCNN study |

Run the fixed-label pair on two GPUs:

```bash
cd ~/DungNDH/micl/bayesfl
export PATH="/home/micl/anaconda3/envs/dungndh/bin:$PATH"
mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"

nohup bash scripts/_run_gpu_queue.sh 0 \
  scripts/queues/mnist_fixed1c_s10_official_gpu0.txt \
  > "logs/queue_mnist_fixed1c_s10_official_gpu0_${TS}.log" 2>&1 &
PID0=$!

nohup bash scripts/_run_gpu_queue.sh 1 \
  scripts/queues/mnist_fixed1c_s10_official_gpu1.txt \
  > "logs/queue_mnist_fixed1c_s10_official_gpu1_${TS}.log" 2>&1 &
PID1=$!

printf 'GPU0 queue PID=%s\nGPU1 queue PID=%s\n' "$PID0" "$PID1"
```

Do not launch these again merely because they are listed here; they rerun completed conditions if launched. The queue does not skip completed configs or resume partial runs. It logs `SUCCESS` or `FAILED_OR_STOPPED`, pauses, and continues to the next config even after a failure. A finished queue is not proof that every job succeeded.

The queue is sequential per GPU, but each simulation can have multiple concurrently scheduled Ray client actors. Physical GPU 1 exposed through `CUDA_VISIBLE_DEVICES=1` appears as local `cuda:0` inside that process. `client_num_gpus` is a Ray resource fraction, not a VRAM/compute utilization quota. At 0.25 or 0.125, one exposed GPU can permit roughly four or eight actors respectively, subject to available CPU/resources. Central evaluation is a separate process-side allocation, and unrelated users can also consume GPU memory.

## 12. Outputs, naming, and checkpoints

Source: `logging_utils.py`, `main.py`, `evaluation.py`, and `posterior/packing.py`.

### 12.1 Naming contract

Every normal launch creates:

```text
run_id = cfg.run_name + '_' + YYYYMMDD_HHMMSS
outputs/<run_id>/
logs/<run_id>.log
```

The timestamp is local process time and has one-second resolution; two identical run names launched within the same second can collide. Use unique names.

Project naming is a convention, not something the loader infers:

```text
run_...       official/retained experiment family
debug_...     tuning or exploratory condition
smoke_...     execution check, not a performance benchmark

a0p1         alpha 0.1
a0p01        alpha 0.01
f1/f0p5      data fractions 1.0 / 0.5
s10/s75      approximate or targeted samples/client; consult partition metadata
fixed1c      fixed one-class/client partition, no Dirichlet alpha
n20/n100     total clients
seed0        intended seed
e10_b32      local epochs 10, batch size 32
lr002        LR 0.02
lr001        LR 0.01
lam0p01      lambda 0.01
cos400       cosine horizon 400, not 400 training rounds
r150         intended run length 150 rounds
```

Not all older LR tags follow a uniform encoding: for example `lr0005` was used for 0.005, and `lr0050` for 0.05. Always read `resolved_config.yaml` rather than parse a decimal from the name.

`_run_nohup.sh` names its **outer shell log from the config filename stem**; the Python application names its own log/output from **`cfg.run_name`**. Keep those equal. Startup timestamps can differ, so one launch can have two nearby log names. The inner `Run directory:` line and the saved resolved config establish the actual output.

### 12.2 Per-run artifacts

```text
outputs/<run_name>_<timestamp>/
├── resolved_config.yaml              # actual validated settings, including defaults/overrides
├── source_config.yaml                # copy of original YAML before CLI overrides
├── environment.json                  # Python/Torch/CUDA/GPU/package versions
├── partition_metadata.json           # manifest statistics and hash used for this run
├── metrics/
│   ├── global_metrics.csv            # central test results, round 0 onward
│   ├── client_metrics.csv            # one row per fit/client/round
│   └── round_train_metrics.csv       # sample-weighted aggregate client diagnostics
├── posterior/
│   └── posterior_summary.csv         # BBB/FOLA per-tensor global summaries
├── reliability/
│   └── round_0000.npz, ...           # bin edges, accuracy, confidence, counts
├── checkpoints/
│   └── global_round_0000.npz, ...    # weights/posterior arrays
├── plots/                            # populated by post-training plotting
└── compression_snapshots/            # only if enabled for FOLA
```

The shared partition indices live outside each run, under `outputs/partitions/`. Copying only `partition_metadata.json` preserves statistics/hash, **not** the actual indices.

Checkpoint round 0 is always saved. Other checkpoints are saved when `checkpoint_every > 0` and the round is divisible by that interval. The final round is **not specially guaranteed** to get a checkpoint if it is not divisible by the interval. For the official R150/interval-10 settings it is saved.

FedAvg/BBB checkpoints use named-parameter keys. FOLA checkpoints use `mean__<parameter_name>` and `precision__<parameter_name>`. Loading FOLA's mean for evaluation is different from resuming FOLA: continuation would also require the precision history and round index. There is no implemented CLI resume path and no optimizer-state checkpoint.

`CsvRecorder` keeps rows in memory and **rewrites the whole CSV on every append**. It is not append-only on disk. Long runs can spend increasing time in CSV writes, especially one row per client. A reader can temporarily encounter a partially written file; re-read after completion. Do not let two jobs write into the same run directory.

## 13. Monitoring and terminal-only result extraction

### 13.1 Monitor running jobs

```bash
pgrep -af '_run_gpu_queue.sh|bayesfl.main'
nvidia-smi
free -h
```

For a particular newly launched queue, keep and inspect the printed log path:

```bash
tail -n 35 "logs/queue_mnist_fixed1c_s10_official_gpu0_${TS}.log"
```

For an individual run, inspect its real application log without relying only on the queue's filtered error display:

```bash
LOG='logs/REPLACE_WITH_THE_ACTUAL_RUN_LOG.log'
tail -n 60 "$LOG"
grep -F 'Simulation finished successfully' "$LOG"
grep -iE 'traceback|exception|error|ModuleNotFoundError|killed|cuda out of memory|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)' "$LOG"
```

Do not count all old `SUCCESS` messages to decide whether a new run set is complete. Check the exact intended run/config set and all expected metric rounds.

A compact resource view:

```bash
watch -n 2 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv; echo; free -h'
```

Those are **device-wide** totals, not proof of one simulation's utilization. The server in the conversation is shared. Do not kill unrelated Python/Ray jobs or issue a blanket `ray stop --force` to diagnose your own process. Stopping a detached queue alone may leave its separately nohup-launched simulation running; identify both owned processes before stopping anything.

### 13.2 Robust terminal summary of the supplied official configs

The following recipe reads all 14 `*official*.yaml` configs supplied in this archive, selects the latest matching run, and prints actual CSV values. It does not generate a summary file, guess missing accuracies, or silently average incomplete rounds. It can be run after the simulations or to see which are incomplete.

```bash
python - <<'PYCODE'
from pathlib import Path
import csv
import math
import re
import statistics
from bayesfl.config import load_config

configs = sorted(Path('scripts/configs').glob('run_*official*.yaml'))
if not configs:
    raise SystemExit('No official configs found in scripts/configs')

def output_for(cfg):
    root = Path(cfg.output.outputs_dir)
    pattern = re.compile(re.escape(cfg.run_name) + r'(?:_\d{8}_\d{6})?$')
    candidates = [p for p in root.glob(cfg.run_name + '*')
                  if p.is_dir() and pattern.fullmatch(p.name)]
    if not candidates:
        raise FileNotFoundError('no matching output directory')
    if len(candidates) > 1:
        print(f'NOTE: {cfg.run_name}: {len(candidates)} matching runs; choosing latest mtime')
    return max(candidates, key=lambda p: p.stat().st_mtime)

def curve_for(run, method):
    path = run / 'metrics' / 'global_metrics.csv'
    if not path.exists():
        path = run / 'global_metrics.csv'
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        key = ('fola_mean_accuracy' if method == 'fola'
               and 'fola_mean_accuracy' in fields else 'accuracy')
        if key not in fields or 'round' not in fields:
            raise ValueError(f'required metric columns not found: {path}')
        values = {}
        for row in reader:
            raw_round = float(row['round'])
            if not math.isfinite(raw_round) or not raw_round.is_integer():
                raise ValueError(f'invalid round: {row["round"]!r}')
            r = int(raw_round)
            a = float(row[key])
            if not math.isfinite(a) or not 0 <= a <= 1:
                raise ValueError(f'invalid fractional accuracy at R{r}: {a}')
            if r in values:
                raise ValueError(f'duplicate round {r}')
            values[r] = a
    return values

complete = 0
for config_path in configs:
    cfg = load_config(config_path)
    print('\n' + '=' * 100)
    print(cfg.run_name)
    try:
        run = output_for(cfg)
        print('Output:', run)
        values = curve_for(run, cfg.method)
        R = cfg.training.rounds
        missing = sorted(set(range(1, R + 1)) - set(values))
        if missing:
            raise ValueError(f'incomplete: missing {len(missing)} rounds; first={missing[:8]}')
        checkpoints = [r for r in (20, 50, 100, 150) if r <= R]
        if R not in checkpoints:
            checkpoints.append(R)
        for r in checkpoints:
            print(f'R{r:<4}: {100*values[r]:.2f}%')
        tail = range(max(1, R - 9), R + 1)
        best_r = max(range(1, R + 1), key=lambda r: values[r])
        print(f'Last-10   : {100*statistics.mean(values[r] for r in tail):.2f}%')
        print(f'Mean R1-{R}: {100*statistics.mean(values[r] for r in range(1, R+1)):.2f}%')
        print(f'Best      : {100*values[best_r]:.2f}% at R{best_r}')
        complete += 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('NOT READY:', exc)
print(f'\nComplete metric sets: {complete}/{len(configs)}')
PYCODE
```

“Complete metric set” means all expected rounds are present; confirm the success log as well when auditing a run. If a newer retry is incomplete, the recipe reports that rather than hiding it by silently choosing an older completed output. Freeze exact timestamps for formal reporting.

## 14. Plot generation

### 14.1 General plotting in `src/bayesfl/utils.py`

The installed entry point is `bayesfl-plots`; the module form is `python -m bayesfl.utils`. The module contains **plotting**, not training logic.

Single-run diagnostics:

```bash
MPLBACKEND=Agg python -m bayesfl.utils --run-dir outputs/<actual_run_directory>
# Equivalent wrapper:
bash scripts/generate_plots.sh outputs/<actual_run_directory>
```

This makes per-run global accuracy/NLL/ECE/Brier/MI plots, selected training/prior/LR plots, posterior-summary plots, and the latest available reliability diagram. These use the stored CSV/NPZ artifacts; missing inputs may produce no plot rather than a failure. Posterior aggregate plots average **tensor means equally**, not all parameter elements equally; use raw summaries for an element-weighted analysis.

Multi-run accuracy comparison:

```bash
MPLBACKEND=Agg python -m bayesfl.utils \
  --compare-accuracy \
  --run-dir outputs/<actual_FedAvg_run_directory> \
  --run-dir outputs/<actual_FOLA_run_directory> \
  --label 'FedAvg' --label 'FOLA' \
  --output-dir outputs/plots/my_comparison \
  --output-stem accuracy_vs_round \
  --title 'MNIST | fixed 10 samples/client, 1 class/client'
```

Comparison options:

| Option | Meaning |
|---|---|
| `--run-dir` | Repeat once per actual run directory |
| `--latest-glob` | Repeat quoted globs; choose newest matching directory by modification time |
| `--compare-accuracy` | Produce a multi-run accuracy comparison |
| `--label` | Optional custom label per run, in supplied order |
| `--output-dir` | Destination; default `outputs/plots` |
| `--output-stem` | Basename; default `accuracy_comparison` |
| `--title` | Figure title |

For FOLA, the primary comparison metric is `fola_mean_accuracy`, falling back to `accuracy`; other methods use `accuracy`. Comparisons render percentages and save **PNG, PDF, and a long-form merged CSV**. The API is `plot_accuracy_comparison(...)`; `latest_matching_run(...)` and `generate_all_plots(...)` are also available.

**No automatic multi-seed averaging occurs in this API.** Giving it four seeds creates four separate curves. Latest-glob selection does not verify completion or scientific equivalence; explicit frozen directories are preferable.

### 14.2 Current official accuracy-round figures

The current user-preferred presentation is implemented separately in:

```bash
MPLBACKEND=Agg python scripts/plot_official_accuracy_curves.py
```

It explicitly selects the **seven official seed-0 FedAvg/FOLA pairs**: four CIFAR BasicCNN Dirichlet cases, two MNIST Dirichlet cases, and one fixed-label MNIST case. It plots rounds 1–150, labels the y-axis as percent global accuracy, includes final R150 accuracies in the legend and endpoint labels, and shows estimated mean samples/client rather than fractions in the title. Fixed distribution is labeled without alpha.

Destination:

```text
outputs/plots/official_accuracy_round/
```

Its filenames combine case tags, rounded sample-count tags, and `_accuracy_vs_round.png`/`.pdf`. For fixed-label MNIST the current filename repeats the sample tag: `mnist_fixed1c_s10_s10_accuracy_vs_round.png`.

Read these important behaviors before running it:

- **It deletes existing `*_accuracy_vs_round.*` files in that directory before regenerating.** Copy any figures you need to retain to a different directory first. A later failure can otherwise leave the old plots gone.
- It uses hardcoded official run stems and chooses the latest matching output. It does not average seeds or automatically discover new official experiments.
- It expects all required outputs and, for Dirichlet titles, the corresponding `outputs/partitions/` metadata/indices. Fixed-label title quantities currently come from the hardcoded case, not a fresh manifest verification.
- It intersects available round indices and checks the maximum round, but does not strictly validate that all 150 interior rounds exist or that all values are finite. Use the terminal validation recipe first.
- The footer still says **“GENERATED 6 FIGURES”**, but `CASES` now contains **7** comparisons. That is a cosmetic stale message, not a missing fixed-label case.
- It is an R150 script; do not use it unchanged to label other run lengths or replication means.

Source: `scripts/plot_official_accuracy_curves.py`.

### 14.3 Older plot scripts

| Script | Existing purpose | Do not assume |
|---|---|---|
| `plot_accuracy_rounds.py` | Four R50 full-data figures; CIFAR explicitly selects ResNet seeds 1–4, MNIST uses heuristic families; mean and sample-SD bands | That it selects current BasicCNN R150 runs, preserves paired seeds, or identifies all historical MNIST families correctly |
| `plot_basiccnn_lr_schedule_compare.py` | Two alpha-specific BasicCNN R50 schedule-comparison figures, with explicit historical stems | That alpha-0.01 cosine versus constant is schedule-only; base LRs also differ |
| `plot_official_accuracy_curves.py` | Seven official R150 seed-0 pairs, final labels and sample-count titles | That it performs automatic multi-seed aggregation |

For the older discovery script, `--list-only` prints recognized run metadata. Missing files, a renamed root, or absent resolved metadata cannot be fixed by inventing output directories. The previous broad folder matcher failed because official CIFAR folder names use `cifar`, not always literal `cifar10`; the official script now uses exact stems.

### 14.4 Future multi-seed figures

When actual replication CSVs exist, use a documented explicit run list with one completed run per seed and method. Verify identical dataset, model, partition semantics/fraction, schedule, and all locked fields. At each round compute the unweighted mean **across seeds** and, when requested, sample standard deviation (`ddof=1`). Check that the same seed IDs and round range exist for both methods. Do not change the number of contributing seeds halfway through a curve, average already averaged runs twice, or fill missing rounds by interpolation without labeling it.

The user previously requested mean multi-seed curves whenever multi-seed results are available. The current official R150 reports are seed 0 only; the completed earlier four-seed ResNet study must not be used as replication for these BasicCNN R150 figures.

## 15. Metrics and multi-seed interpretation

Source: `metrics.py`, `evaluation.py`, `strategies/common.py`.

### 15.1 Evaluation and training metrics

Global CSV accuracy values are fractions in `[0,1]`. Plot/report percentages are 100 times those values. A difference of 0.026 means **2.60 percentage points**, not a 2.60% relative gain.

| Metric/key | Meaning in this source |
|---|---|
| `accuracy` | Global test argmax accuracy; for FOLA this is posterior-mean model accuracy |
| `fola_mean_accuracy` | Explicit FOLA posterior-mean accuracy, equal to generic accuracy in the current evaluator |
| `fola_mc_accuracy` | Optional posterior-sampled predictive accuracy; separate from the official mean metric |
| `nll` / log `loss` | Mean negative log probability of the correct test class, not local training objective |
| `brier` | Mean sum of squared probability errors across classes |
| `ece`, `mce` | Expected/maximum calibration gap, using 15 equal-width confidence bins |
| `predictive_entropy` | Entropy of predictive probabilities, averaged over test examples |
| `expected_entropy` | MC-averaged entropy when sampled predictions are present |
| `mutual_information` | Nonnegative difference of predictive and expected entropy; zero when no MC samples are supplied |
| `train_loss` | Sample-visit-weighted local objective; FOLA includes lambda-weighted prior |
| `task_loss`, `prior_loss` | Separate local terms; `prior_loss` itself is not multiplied by lambda |
| `train_accuracy` | Training-batch prediction accuracy accumulated while parameters change, not final local-model test accuracy |
| `local_steps` | Actual optimizer updates in the fit call |
| `update_l2`, `update_rms` | Changes in returned parameters/means relative to incoming arrays |

The row in `round_train_metrics.csv` is weighted by returned client example counts. Min/max-type diagnostic fields in that averaged row are weighted averages of client summaries, **not necessarily global extrema**. Inspect per-client/per-parameter rows for true outliers.

**Count-field caveat:** the strategy constructs the round row with a total `num_examples`, then expands the weighted client-metric dictionary. Because client metrics also contain `num_examples`, that expansion can overwrite the total with an example-weighted mean client size. This affects the reported round count field, not the aggregation weights. Use partition metadata or sum the raw per-client example counts when you need the federation total.

With `paper_mean_only_eval: true`, `fola.mc_eval` is inactive and official FOLA uncertainty fields are deterministic point-prediction statistics. Zero MI in that path does not demonstrate an absence of epistemic uncertainty.

### 15.2 Round summaries used in the conversation

For a run of R rounds and accuracy `A(r)`:

```text
R50 / R100 / R150 = A(50) / A(100) / A(150)
Final              = A(R), not the highest accuracy reached
Last-10            = mean A(r) for r = R-9 ... R
Mean R1-R          = mean A(r) for r = 1 ... R
Best               = maximum A(r) in the reporting range
BestR              = round attaining Best (state tie convention)
Gap                = FOLA - FedAvg, in percentage points
```

Round 0 is initialization, excluded from official trajectory averages and post-training “best” in the recommended extractor. Some older scripts included it, so use the reporting range explicitly. For R150, Last-10 means rounds **141–150**; for R50 it means **41–50**.

`Mean` is ambiguous without its axes: it may mean across communication rounds, across seeds at a fixed round, or across client metrics. Always label it.

Mean accuracy over a fixed round budget describes the observed learning trajectory. It is **not a measured mathematical convergence rate**, and it does not establish wall-clock or byte-level communication efficiency. Higher trajectory mean plus lower final accuracy does not identify exact crossing rounds. Last-10 reduces sensitivity to a single endpoint, but stability needs a dispersion measure as well as a mean.

### 15.3 Replication and comparisons

A fair paired seed difference is `FOLA(seed, setting) - FedAvg(seed, setting)`. Summarize those paired differences rather than treating 150 autocorrelated rounds as 150 independent experiments. The earlier ResNet study used seed 0 to choose conditions and seeds 1–4 as its primary replication set; keep that split visible.

For data-scarcity robustness at the same seed:

```text
robustness(f) = [FOLA(f) - FedAvg(f)] - [FOLA(full) - FedAvg(full)]
```

Positive robustness means the relative FOLA advantage increased from full data. It is not equivalent to FOLA winning at the reduced fraction. Validate full-data and reduced runs for the same seeds before making this comparison.

The present official CIFAR protocol changes algorithm **and schedule**, and for alpha 0.01 also base LR. Attribute its results to the compared locked configurations. A method-only conclusion would need the relevant matched-schedule/base-LR controls; do not add those runs or retune them silently.

## 16. Optional posterior snapshots and compression research

Source: `OutputConfig` and `ResearchStrategy` snapshot methods.

The implemented feature is **snapshot collection**, controlled by:

```yaml
output:
  save_full_client_posteriors: false
  full_client_posterior_rounds: [1, 5, 10, 25, 40, 50]
```

Only FOLA uses it. To collect early/middle/late snapshots in a future R150 study, explicitly choose appropriate rounds, for example `[1, 5, 10, 50, 100, 150]`; the default list does not auto-rescale with run length. This is not the same as `checkpoint_every`.

When enabled:

```text
compression_snapshots/round_0050/
├── incoming_global.npz
├── outgoing_global_uncompressed.npz
├── clients/
│   ├── client_0000.npz
│   └── ...
└── manifest.json
```

The NPZs contain `mean__name`/`precision__name`; client files also contain client ID, example count, and aggregation weight. The JSON describes mode, round, parameter names, aggregation epsilon, and saved client files. It records `training_trajectory_modified: false`.

The strategy copies states without applying a compression/reconstruction result to the next round. It therefore leaves the intended aggregation math unchanged, but enabled saving has copy, storage, and I/O overhead; the code flag is not a proof of bitwise-identical trajectories under all asynchronous hardware execution.

**Not implemented in this archive:** random coordinate compression, absolute-change ranking, change-based SNR selection, KL omission-score ranking, reconstruction/budget accounting, omitted-divergence measurements, global Gaussian KL evaluation, or immediate prediction-change comparisons. Those were proposed for a later **offline, no-feedback compression experiment**. The existing `posterior_snr_mean` diagnostic is not that change-based compression score.

For any future KL analysis, specify whether `precision` means raw paper-reference omega or another normalized positive Gaussian precision; zero/near-zero entries need an explicit, common numerical policy. Do not silently equate accumulated omega with a validated posterior covariance.

## 17. Experiment results carried forward

**Evidence status for this entire section:** conversation-reported results from the user's terminal, not CSVs included in `bayesfl.zip`. Config identities can be verified against the archive. Recompute from the preserved actual run CSVs before publication. Percentage-point differences rounded from full CSV precision may differ by 0.01 from subtraction of displayed percentages.

### 17.1 Official seed-0, R150 results

CIFAR uses **BasicCNN**; MNIST uses the MLP. “Cosine” means the official horizon-400 schedule. Every row below is a distinct method run, not a seed average.

| Data / distribution | Method | LR regime | R50 | R100 | R150 | Last-10 | Mean R1–150 |
|---|---|---|---:|---:|---:|---:|---:|
| CIFAR alpha .01, full | FedAvg | .01 constant | 35.87% | 41.96% | 44.56% | 44.01% | 37.41% |
| CIFAR alpha .01, full | FOLA | .02 cosine | 44.37% | 50.34% | 52.62% | 52.46% | 45.85% |
| CIFAR alpha .01, fraction .5 | FedAvg | .01 constant | 33.83% | 37.19% | 40.89% | 40.65% | 34.57% |
| CIFAR alpha .01, fraction .5 | FOLA | .02 cosine | 40.90% | 45.38% | 47.79% | 47.96% | 41.70% |
| CIFAR alpha .1, full | FedAvg | .02 constant | 65.23% | 71.41% | 73.82% | 73.45% | 66.29% |
| CIFAR alpha .1, full | FOLA | .02 cosine | 68.32% | 73.51% | 75.25% | 74.94% | 68.37% |
| CIFAR alpha .1, fraction .5 | FedAvg | .02 constant | 59.57% | 65.27% | 68.48% | 68.42% | 60.24% |
| CIFAR alpha .1, fraction .5 | FOLA | .02 cosine | 61.13% | 66.37% | 68.75% | 69.06% | 61.54% |
| MNIST alpha .01, fraction .125 (~75/client) | FedAvg | .01 constant | 78.95% | 82.22% | 83.82% | 83.86% | 77.55% |
| MNIST alpha .01, fraction .125 (~75/client) | FOLA | .01 constant | 80.29% | 82.73% | 84.43% | 84.40% | 78.85% |
| MNIST alpha .01, fraction 1/60 (~10/client) | FedAvg | .01 constant | 74.90% | 80.12% | 82.82% | 82.65% | 74.57% |
| MNIST alpha .01, fraction 1/60 (~10/client) | FOLA | .01 constant | 78.70% | 81.96% | 84.05% | 83.98% | 77.66% |
| MNIST fixed 10/client, 1 class/client | FedAvg | .01 constant | 74.01% | 76.79% | 78.64% | 78.51% | 73.69% |
| MNIST fixed 10/client, 1 class/client | FOLA | .01 constant | 77.64% | 79.81% | 81.24% | 81.13% | 76.99% |

Corresponding reported official configuration gaps:

| Condition | R150 FOLA−FedAvg | Last-10 gap | Mean R1–150 gap |
|---|---:|---:|---:|
| CIFAR alpha .01, full | +8.06 pp | +8.45 pp | +8.44 pp |
| CIFAR alpha .01, fraction .5 | +6.90 pp | +7.31 pp | +7.13 pp |
| CIFAR alpha .1, full | +1.43 pp | +1.48 pp | +2.08 pp |
| CIFAR alpha .1, fraction .5 | +0.27 pp | +0.64 pp | +1.30 pp |
| MNIST Dirichlet ~75/client | +0.61 pp | +0.54 pp | +1.30 pp |
| MNIST Dirichlet ~10/client | +1.23 pp | +1.33 pp | +3.09 pp |
| MNIST fixed 10/client, 1 class/client | +2.60 pp | +2.62 pp | approximately +3.30 pp from displayed means |

The fixed-label pair also reported R20 **70.64% FedAvg versus 75.02% FOLA**. Its sample count and label distribution differ from the Dirichlet ~10 condition; the two are not identical-data ablations.

These are favorable **single-seed observations**. The archive contains official seed-0 configs and queues, not evidence that official R150 seeds 1–4 have already completed. Replication was discussed as a possible next stage, not established here as completed.

### 17.2 BasicCNN official-adjacent diagnostics, alpha 0.1, R50

| Fraction | FedAvg R50 | FOLA R50 | Last-10 gap | Mean R1–50 gap |
|---:|---:|---:|---:|---:|
| 0.25 | 54.28% | 54.83% | +1.03 pp | +1.10 pp |
| 0.125 | 47.96% | 47.65% | −0.34 pp | +0.49 pp |

FedAvg is constant LR .02; FOLA has cosine base .02 and lambda .1. These four jobs remain diagnostics, not R150 results.

### 17.3 Earlier ResNet-56 matched-LR replication, R50, seeds 1–4

These results must remain separate from the BasicCNN official study.

| Alpha | Fraction | FedAvg mean R50 | FOLA mean R50 | R50 gap | Last-10 gap | Mean R1–50 gap |
|---:|---:|---:|---:|---:|---:|---:|
| .1 | 1.0 | 72.09% | 70.76% | −1.32 pp | −1.34 pp | +0.45 pp |
| .1 | .5 | 58.55% | 58.68% | +0.12 pp | +0.22 pp | +0.76 pp |
| .1 | .25 | 46.81% | 45.08% | −1.73 pp | −1.59 pp | −0.31 pp |
| .1 | .125 | 35.98% | 36.14% | +0.16 pp | +1.14 pp | +0.85 pp |
| .1 | .03 | 21.98% | 20.09% | −1.89 pp | −1.42 pp | −1.37 pp |
| .01 | 1.0 | 39.06% | 38.52% | −0.54 pp | −0.54 pp | +0.69 pp |
| .01 | .5 | 31.87% | 34.27% | +2.40 pp | +2.68 pp | +2.39 pp |

All reported four-seed paired 95% intervals for these metrics crossed zero. The alpha-.01/fraction-.5 result favored FOLA in 3/4 seeds; the alpha-.1/fraction-.125 trajectory mean favored FOLA in 4/4 seeds. The curves **did not establish a monotonic increase in FOLA advantage as sample count decreased**. Positive means or win counts do not imply every seed/round was better.

An older seed-0 ResNet scarcity sweep included fractions `.02`, `.01`, `.006`, and `.004` (roughly 50, 25, 15, and 10 examples/client). Its low-data accuracy was often near the 10-class chance level and the gap changed signs. Those exploratory points were not all replicated.

### 17.4 Earlier MNIST tuning and data-scarcity study

The earlier R50 study used N=100, E=10, B=32, constant LR .01 and five seeds 0–4. Its locked lambda was .0001 for alpha .1 and .01 for alpha .01.

| Fraction | MNIST alpha .1 last-10 gap | Alpha .1 trajectory-mean gap | MNIST alpha .01 last-10 gap | Alpha .01 trajectory-mean gap |
|---:|---:|---:|---:|---:|
| 1.0 | +0.01 pp | +0.03 pp | +0.36 pp | +0.50 pp |
| .5 | +0.01 pp | +0.05 pp | +0.70 pp | +1.03 pp |
| .25 | +0.03 pp | +0.11 pp | +1.14 pp | +1.78 pp |
| .125 | +0.05 pp | +0.23 pp | +1.56 pp | +2.67 pp |

This supported a scarcity-related relative benefit **within that tested MNIST setup**, especially alpha .01; it did not prove universality or the same behavior for CIFAR. Alpha .1 effects were very small in absolute terms. Exact old curves require old outputs; these averages cannot reconstruct 50 round-by-round values.

### 17.5 Historical R300 CIFAR context

Historical terminal summaries also reported the following paired final values. These are different configurations from the current official runs, and the old `default?` model label was not finally verified from a saved resolved config during that recovery.

| Historical family | Alpha | FedAvg R300 | FOLA R300 |
|---|---:|---:|---:|
| Old default/deeper-model family (reported as ResNet-era) | .1 | 81.20% | 75.80% |
| Old default/deeper-model family | .01 | 46.05% | 14.34% |
| BasicCNN, LR tag `lr001` | .1 | 77.54% | 78.98% |
| BasicCNN, LR tag `lr0005` (0.005) | .01 | 61.04% | 56.68% |

The corresponding historical FOLA names used lambda 1. Do not confuse `lr0005` with 0.0005, or use these old runs as schedule/architecture-only controls without comparing resolved configs and source versions. The attached-paper discussion is historical context; neither matching an endpoint percentage nor the name `paper_reference` proves faithful paper reproduction.

## 18. Exact official config and run registry

**Source-verified names.** Each stem below is both the intended `run_name` and the filename under `scripts/configs/` with `.yaml` appended. Outputs append a timestamp; **the actual timestamped output folders are not included in this ZIP**. No timestamp is invented here.

**CIFAR BasicCNN, alpha 0.01, fraction 0.5 — FedAvg, 150 rounds, seed 0**

```text
run_cifar_basiccnn_a0p01_official_fedavg_n20_f0p5_seed0_e10_b32_lr001_r150
```

**CIFAR BasicCNN, alpha 0.01, fraction 1 — FedAvg, 150 rounds, seed 0**

```text
run_cifar_basiccnn_a0p01_official_fedavg_n20_f1_seed0_e10_b32_lr001_r150
```

**CIFAR BasicCNN, alpha 0.01, fraction 0.5 — FOLA, 150 rounds, seed 0**

```text
run_cifar_basiccnn_a0p01_official_fola_cos400_n20_f0p5_seed0_e10_b32_lr002_lam0p01_r150
```

**CIFAR BasicCNN, alpha 0.01, fraction 1 — FOLA, 150 rounds, seed 0**

```text
run_cifar_basiccnn_a0p01_official_fola_cos400_n20_f1_seed0_e10_b32_lr002_lam0p01_r150
```

**CIFAR BasicCNN, alpha 0.1, fraction 0.5 — FedAvg, 150 rounds, seed 0**

```text
run_cifar_basiccnn_a0p1_official_fedavg_n20_f0p5_seed0_e10_b32_lr002_r150
```

**CIFAR BasicCNN, alpha 0.1, fraction 1 — FedAvg, 150 rounds, seed 0**

```text
run_cifar_basiccnn_a0p1_official_fedavg_n20_f1_seed0_e10_b32_lr002_r150
```

**CIFAR BasicCNN, alpha 0.1, fraction 0.5 — FOLA, 150 rounds, seed 0**

```text
run_cifar_basiccnn_a0p1_official_fola_cos400_n20_f0p5_seed0_e10_b32_lr002_lam0p1_r150
```

**CIFAR BasicCNN, alpha 0.1, fraction 1 — FOLA, 150 rounds, seed 0**

```text
run_cifar_basiccnn_a0p1_official_fola_cos400_n20_f1_seed0_e10_b32_lr002_lam0p1_r150
```

**MNIST, alpha 0.01, fraction 0.0166666667 — FedAvg, 150 rounds, seed 0**

```text
run_mnist_a0p01_official_fedavg_n100_s10_f0p016667_seed0_e10_b32_lr001_r150
```

**MNIST, alpha 0.01, fraction 0.125 — FedAvg, 150 rounds, seed 0**

```text
run_mnist_a0p01_official_fedavg_n100_s75_f0p125_seed0_e10_b32_lr001_r150
```

**MNIST, alpha 0.01, fraction 0.0166666667 — FOLA, 150 rounds, seed 0**

```text
run_mnist_a0p01_official_fola_n100_s10_f0p016667_seed0_e10_b32_lr001_lam0p01_r150
```

**MNIST, alpha 0.01, fraction 0.125 — FOLA, 150 rounds, seed 0**

```text
run_mnist_a0p01_official_fola_n100_s75_f0p125_seed0_e10_b32_lr001_lam0p01_r150
```

**MNIST fixed 10 samples/client, 1 class/client — FedAvg, 150 rounds, seed 0**

```text
run_mnist_fixed1c_s10_official_fedavg_n100_seed0_e10_b32_lr001_r150
```

**MNIST fixed 10 samples/client, 1 class/client — FOLA, 150 rounds, seed 0**

```text
run_mnist_fixed1c_s10_official_fola_n100_seed0_e10_b32_lr001_lam0p01_r150
```

Useful older full-data MNIST registry patterns, if locating archived runs:

```text
# Alpha .1, seed 0
 debug_mnist_paperpart_fedavg_a0p1_n100_lr001_b32_e10_r50
 debug_mnist_lock_a0p1_fola_n100_f1_lr001_b32_e10_lam0p0001_r50
# Alpha .1, seeds 1–4
 debug_mnist_a0p1_E10_fedavg_n100_f1_seed{S}_r50
 debug_mnist_a0p1_E10_fola_n100_f1_lam0p0001_seed{S}_r50
# Alpha .01, seed 0
 debug_mnist_paperpart_fedavg_a0p01_n100_lr001_b32_e10_r50
 debug_mnist_paperpart_fola_a0p01_n100_lr001_b32_e10_lam0p01_r50
# Alpha .01, seeds 1–4
 debug_mnist_paperpart_fedavg_a0p01_n100_lr001_b32_e10_seed{S}_r50
 debug_mnist_paperpart_fola_a0p01_n100_lr001_b32_e10_lam0p01_seed{S}_r50
```

Leading spaces above are visual only. `{S}` means an actual seed number. Config existence does not establish that the associated output is still present or complete.

## 19. Tests and validation status

### 19.1 Existing test coverage

Source: `tests/`.

The uploaded suite covers config loading, deterministic model shapes and parameter counts, parameter packing, Gaussian products, the alternative FOLA precision recurrence, variance floors, scale-mixture/normal densities, ECE, LR values, an old CIFAR profile, sparse and fixed-label partition properties, BBB initialization/rho behavior, and plotting helpers.

Important limits:

- `test_flower_smoke.py` is **import-level**, not a real federated training run.
- The fixed-label test checks 100 clients, 10 examples/client, one label/client, balanced client counts/class, and 1,000 distinct indices using synthetic labels.
- The precision recurrence tests exercise the `online_recurrence` math helper, not the full current paper-reference training/aggregation pipeline.
- `test_all_configs_load()` checks top-level YAML configs; separate documentation-time validation included nested configs too.
- No comprehensive integration test for snapshot trajectory equivalence or offline compression analysis was found.

### 19.2 Checks actually performed while preparing this README

Performed against the uploaded source without training or downloading data:

```text
Offline pytest:             23 passed, 4 skipped
All YAML load/validation:   398 configs loaded successfully
Deterministic model counts: MLP 545,810; BasicCNN 878,538; ResNet GN8 855,578
Official schedule values:  computed from round_learning_rate()
```

The documentation environment used Python **3.13.5**, not the project's declared 3.10 target. Flower/Ray/Bayesian-Torch were not installed in that environment. Three skips concerned Bayesian-Torch tests and one concerned Flower import. These checks **do not establish full dependency, CUDA, BBB, or Flower/Ray runtime compatibility**. No GPU simulation was run to create this documentation. Run the suite in the established 3.10 environment before changing production experiments.

On the actual server:

```bash
python -m pytest -q
# Full legacy install check, including BBB dependencies/model dimension:
bash scripts/validate_install.sh
```

The old `VALIDATION.md` report (`13 passed, 2 skipped`) belongs to an earlier build and should not be quoted as validation of this upload.

## 20. Known limitations and troubleshooting

### 20.1 Important source/documentation corrections

| Prior assumption or stale text | What the supplied source actually does |
|---|---|
| Flower is pinned to 1.30.0 | Both current dependency manifests pin **1.29.0** |
| The main/default YAML is the final tuned official config | Main MNIST/CIFAR defaults retain older optimizer/profile choices; select explicit `official` configs |
| All official FOLA omega starts at zero | **MNIST starts at 1.0; CIFAR starts at 0.0** |
| CIFAR does not use paper augmentation | Official CIFAR YAML enables crop, flip, AutoAugment, Cutout, and normalization |
| “BasicCNN” is the config identifier | Use **`paper_basiccnn`** |
| FOLA uses Bayesian-Torch layers | Only BBB uses those layers; FOLA has deterministic weights plus custom diagonal state |
| Fixed labels supports arbitrary K classes/client | The new mode permits **one class/client only**, MNIST only |
| A supplied fixed-step cap is active | No such field or loop cap is present |
| Queue launchers hardcode the corrected Conda Python | In this ZIP both use PATH-resolved **`python`** |
| `utils.py` already averages all supplied seeds | It draws separate runs; only the bespoke older script averages selected groups |
| The official plotting script generates six plots | It contains seven cases; only its final message is stale |
| A cache filename proves exact config equivalence | Cache lookup omits some fields and does not revalidate source/data provenance |
| Checkpoints support resume | There is no implemented resume entry point or optimizer-state restoration |

### 20.2 Frequent operational problems

**`ModuleNotFoundError: bayesfl`.** Check `sys.executable`, Conda activation, PATH in the shell launching the queue, and editable installation. An inline `PYTHONPATH=src python ...` command affects that command only, not a later background launch unless exported. Correcting import paths in base Conda is not the same as selecting the known working dependency environment.

**`FAILED_OR_STOPPED` with no useful error.** Read the actual simulation log. The queue's regex is incomplete and may omit errors such as a simple `ModuleNotFoundError`. A shell PID printed before import succeeds is not proof that training began.

**Unexpected data distribution.** Check `partition.type`, `runtime.seed`, realized class/size statistics, and NPZ hash. Review leftover inactive fields. A same-name old cache can be reused even after a generator change. Do not delete/rebuild old experiment partitions without preserving them.

**Unexpected speed or step count.** With `drop_last=False`, normal local steps are `E * ceil(n_client / B)`. For fixed 10 examples and B32/E10, there are 10 full-client-data optimizer steps per round. Counts vary across unbalanced clients; computing steps from mean client size is only approximate. Data fractions change this budget, not just information content.

**Unexpected late-run slowdown.** The CSV recorder rewrites accumulated rows repeatedly. Snapshot/checkpoint/reliability I/O and shared-machine contention are additional possible costs. GPU utilization alone cannot identify the bottleneck.

**Plots missing or selecting the wrong runs.** Inspect actual run directories and resolved configs. Do not filter R300 using an unanchored `300` substring, which can match a timestamp. Exact official stems are supplied above. The newest matching run might be incomplete; verify all expected rounds before plotting/reporting.

**Unknown MNIST partition type.** MNIST dispatch falls back to the legacy lognormal generator for types other than recognized `paper_dirichlet`/`fixed_labels`; a typo may silently select the wrong mode. CIFAR dispatch raises for an unknown generator type after a cache miss, but cache naming can still obscure intent. Use the exact spelling and inspect saved metadata.

**Changed model defaults.** A factory supports deterministic and BBB models but not arbitrary architecture names. GroupNorm buffers/affine handling and named-parameter packing must be audited before adding BatchNorm or custom models. Do not reuse a checkpoint across layouts.

**Reproducing the original paper.** This project has operational paper-reference choices, legacy stabilized alternatives, different models, explicit budget constraints, and nonidentical schedules. Keep “source-inspired implementation” separate from a strict reproduction claim. Numerical similarity to a published endpoint is not verification of equations, software semantics, or experimental parity.

### 20.3 Features not to claim yet

There is no implemented differential-privacy mechanism, secure aggregation, adversarial/privacy evaluation, general multi-label fixed partition, automatic checkpoint resume, dedicated validation-split hyperparameter selector, or end-to-end offline compression benchmark. Some scripts historically selected configs using the same global test accuracy later reported; describe that selection history when interpreting results.

Model-dimension counts and tests show structural properties, not predictive correctness. Low seed-0 accuracy does not prove a particular Fisher/precision bug; conversely, a favorable seed-0 official result does not validate every approximation or remove LR/schedule confounding.

## 21. Handoff and preservation checklist

### 21.1 What to transfer

Upload this README **and the source archive** to the next project/chat. For real result analysis, also transfer the exact selected run artifacts:

```text
resolved_config.yaml
source_config.yaml
environment.json
partition_metadata.json
metrics/global_metrics.csv
metrics/client_metrics.csv                 # when diagnosing clients/training
metrics/round_train_metrics.csv
posterior/posterior_summary.csv            # when diagnosing FOLA/BBB
checkpoints/                              # when doing evaluation/compression work
matching outputs/partitions/*.npz and *.json
relevant actual application and queue logs
```

Per-round `global_metrics.csv` is required to reproduce an accuracy curve. Final/last-10/trajectory means cannot reconstruct it. A partition JSON alone cannot reproduce a partition's exact membership without either the NPZ or a verified generator/data/seed combination.

The supplied `.gitignore` ignores most outputs by default, re-includes `outputs/plots`, `outputs/partitions`, and `outputs/run*`, and ignores `logs/*.log`. **Git tracking is not a backup.** Old `debug_*` directories may be omitted from a source export even when they were important tuning runs. The archive manifest also lists some historical paths not actually contained in this ZIP.

### 21.2 Safe continuation workflow

Before another simulation: read the intended official YAML, verify the model/partition/LR/mode/initial precision, preserve E<=10, make a unique name, validate config and manifest, check GPU availability, and launch through the known interpreter. For a paired new seed, ensure both methods share the manifest hash.

After a simulation: inspect success logs and every expected round, freeze actual output directories, summarize from CSVs, generate plots from those exact runs, and retain all results regardless of sign. Mark new exploratory experiments separately from the official cohort.

Before a code edit: read the current target files rather than applying a guessed string replacement from old chat history. Back up or commit, add a relevant test, run existing tests, and run an explicitly named small smoke simulation when needed. Do not modify active configs/source underneath ongoing experiments without coordinating the change.

### 21.3 Suggested first message in the new chat

> This is a continuation of the BayesFL project. Read this README and the uploaded source before proposing edits. Focus on FedAvg/FOLA; BBB remains implemented but inactive. Current official results use MNIST MLP and CIFAR BasicCNN, R150, seed 0. Preserve the locked settings in Section 9, including different MNIST/CIFAR initial FOLA precisions and the CIFAR schedule/base-LR caveat. Keep E<=10. Support both paper Dirichlet and the implemented one-class fixed-label MNIST partition. Use exact config/run identities and paired manifest hashes. Provide execution commands plus terminal-only summaries, and use actual per-round CSVs for accuracy plots. Do not silently retune, overwrite old artifacts, fabricate missing histories, or claim universal FOLA superiority. Section 17 contains conversation-reported results; the ZIP does not contain the underlying outputs.

---

**Source-of-truth order for future work:** actual run `resolved_config.yaml` + actual metrics/manifest + source revision used for that run; then the uploaded source/configs for planned runs; then this handoff's clearly labeled historical summaries. A directory name or old chat explanation is not a substitute for those artifacts.
