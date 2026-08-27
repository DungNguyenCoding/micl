# Changelog

## 1.7.0-priority-baseline

- Added opt-in SGD momentum and weight decay fields while preserving legacy defaults.
- Added logical-round cosine learning-rate scheduling with configurable minimum LR.
- Applied the same logical-round LR to FedAvg and to both Proposed Pyro phases.
- Logged optimizer/scheduler/effective-LR settings in server and client CSV metrics.
- Added CIFAR priority config: E=10, batch=256, SGD momentum=0.9, LR 0.05 -> 1e-4 cosine.
- Added schedule tests; full test suite passes.
- Bayesian backend remains Pyro for this stage; bayesian-torch comparison is deferred.

## v1.5.2 — Paired condition seeds

- Pair the stochastic realization seed across conditions within Figure 3, Figure 4, and Figure 5 sweeps.
- Remove the old condition-specific `+ 10_000 * condition_counter` seed offset.
- A given replication now uses `base_seed + realization` for every condition.
- Figure 5 therefore compares `P = 3/23/33 dBm` using the same client partition, model initialization, channel RNG seed, and noise RNG seed; only the configured transmit power changes.
- Keep the v1.5.1 process-local MNIST cache and all v1.5.0 Proposed/Hong-2023 learning and AirComp equations unchanged.
- Add regression tests for paired-condition seed semantics.

## v1.5.1 — MNIST host-memory cache

- Cache the MNIST train/test dataset objects once per Python process.
- Prevent native-Windows local runs from repeatedly loading the full MNIST tensor for every client and every Proposed physical phase.
- Preserve client subsets, deterministic shuffle seeds, transforms, and all learning/AirComp equations unchanged.
- Add a regression test for process-local MNIST caching.


## 1.5.0 — Reference-[13] benchmark power-control correction

- Inspected the actual Hong, Park, and Choi IEEE TWC 2023 reference [13].
- Preserved one shared KKT/QCQP magnitude optimizer for all AirComp payloads.
- Kept Proposed `Delta-rho`/`Delta-nu` power normalization unchanged.
- Changed FedAvg/FedProx/SCAFFOLD to Hong-2023 Eq. (8)/(10)/(20) scaling, including the `sigma_z^2` transmit factor and receiver de-scaling.
- Kept deterministic payloads as additive local updates `Delta-w`.
- Added `deterministic_reference_power_mode`, defaulting to the source-motivated `coordinated_aggregate` adaptation because the target paper has no BS dataset.
- Added exact Eq. (8), Eq. (10), reference-power, shared-KKT, and configuration tests.

## 1.4.2

- Restored FedAvg/FedProx wireless aggregation to additive local model updates: `Delta-w = w_local - w_global`.
- Kept the same `unified_kkt` QCQP/KKT AirComp solver for deterministic and Bayesian transmitted vectors.
- Changed `wireless.deterministic_payload_mode` from `model` to `update` in all bundled configurations.
- Added `global_model_update_l2`, `ideal_model_update_l2`, and `received_model_update_l2` to `metrics.csv`.
- Added regression tests for ideal no-wireless equivalence and deterministic update diagnostics.
- Proposed two-phase `Delta-rho` / `Delta-nu` training and aggregation are unchanged.

## 1.4.1

- Corrected FedAvg/FedProx AirComp payload semantics to match the paper's
  simulation description: clients transmit local **model weights** (d real
  values), not local-minus-global model deltas.
- The same `unified_kkt` QCQP/KKT magnitude-control solver is still used.
- Added `wireless.deterministic_payload_mode: model` and validation.
- Added payload semantics to `metrics.csv` for auditability.

# Changelog

## 1.4.0

- Made the shared AirComp power-control policy explicit with
  `wireless.power_control_mode: unified_kkt`; every method routes through the
  same QCQP/KKT solver.
- Preserved phase-1 client precision as float64 when reloading it for phase 2.
- Added posterior-mean accuracy/NLL/ECE next to paper-style posterior-predictive
  metrics.
- Added global mean/precision update norms.
- Added `proposed_debug` plotting mode.
- Added Proposed-first, strict no-clip, and FedAvg-comparison GPU configs.
- Added regression tests for the unified power-control contract and posterior
  mean diagnostic.

## 1.3.3

- Fixed the final float64 precision-state leak in server evaluation.
- Flower-decoded proposed state now remains `[float32 mean, float64 precision]`.
- Prevents evaluation from quantizing rho to float32 and feeding the rounded
  value into the next logical round.
- Adds posterior precision spread and offset diagnostics to `metrics.csv`.
- Adds a regression test for server-state dtype normalization at rho=400.

## 1.3.2

- Fixed the numerically frozen precision coordinate observed at rho=400.
- Keeps rho as a float64 master variable during Pyro VI, client state, server
  aggregation, and AirComp reconstruction.
- Casts sampled Bayesian weights to float32 only for the CNN forward pass.
- Preserves sub-float32-ULP precision updates in both ideal and wireless modes.
- Adds per-client precision delta and applied-gradient diagnostics.
- Adds a regression test for 1e-6 precision updates at rho=400.

## 1.3.1

- Optimizes the Bayesian precision coordinate `rho` directly with Pyro's
  differentiable ELBO and PyTorch SGD, matching Eq. (25).
- Keeps the phase-2 guide covariance at `Sigma_{t+1}` while regularizing
  against the round-start global prior `q_{theta_t}` from Eqs. (13) and (15).
- Scales mini-batch likelihoods to unbiased full-local-dataset estimates.
- Temporarily broadcasts `[mu_t, rho_{t+1}, rho_t]` during the natural-mean
  phase, then returns to the normal `[mu_{t+1}, rho_{t+1}]` server state.
- Adds source-contract tests for the corrected coordinates and phase-2 prior.

# Changelog

## 1.3.0

- Replaced the legacy single-client-call Bayesian update with the exact
  server-separated precision/natural-mean schedule from Algorithm 1.
- Added `bayesian_protocol.py` with physical/logical round mapping and the
  diagonal `rho <-> nu <-> mu` coordinate transforms.
- Added separate Pyro SVI entry points:
  `train_precision_phase` and `train_natural_mean_phase`.
- Proposed clients now return one `d`-value vector per phase and persist their
  phase-1 precision until the matching phase-2 call.
- Added separate AirComp aggregation functions for `Delta rho` and `Delta nu`.
- Both proposed phases reuse the same block-fading channel realization within a
  logical round, while additive noise samples remain independent.
- Flower/Ray runs now use two physical rounds for every logical proposed round;
  evaluation is skipped between the phases.
- Added phase-aware CSV fields, per-phase AirComp diagnostics, posterior
  precision summaries, and client `rho/nu` diagnostics.
- Added tests for phase order, the Eq. (33)/(34) coordinate round trip, ideal
  two-phase conflation, logging fields, AirComp, wireless normalization, and
  model size.

## 1.2.1

- Added missing wireless diagnostic fields to the CSV schema.

## 1.2.0

- Added explicit path-loss reference-distance normalization.

## 1.1.0

- Added stable native-Windows sequential CUDA backend.

## v1.6.0

- Added opt-in Proposed sparse posterior-evidence experiment (`--experiment sparse`).
- Added Bayesian update-SNR and same-budget random top-k coordinate selection.
- Added keep-ratio matrix 100/75/50/25/10/5/2% using paired-condition seeds.
- Added sparse communication accounting and reliability/accuracy/ECE plot suite.
- Dense paper `fig2`-`fig6` paths remain sparse-disabled by default.

## v1.6.1

- Removed redundant Bayesian/random 100% keep runs from `--experiment sparse`.
- Sparse matrix is now 12 runs: 75/50/25/10/5/2% for Bayesian and random.
- Added `utils.py --dense-baseline <fig2_result_dir>` to reuse an existing dense
  Proposed Figure-2 result as the shared 100% endpoint.
- Added `--dense-baseline-round`; when omitted, the sparse experiment's highest
  completed round is used automatically (normally 120).
- Dense reliability bins are reused at the exact target round; dense trajectory
  rows are clipped to that round for accuracy-vs-communication plots.
- Added seed-consistency validation between sparse runs and the reused baseline.
- Paper Figure 2–6 paths are unchanged.
