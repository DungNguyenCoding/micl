# Implementation report: BayesFL sparse posterior communication

## Delivered scope and evidence

This repository implements the requested post-training sparse FOLA extension, not just a proposed patch. It was developed against the actual uploaded `bayesfl.zip` and the supplied `guideline.md` and source-handoff `README.md`.

Original ZIP SHA-256:

```text
39e6782c00d21bf7fd83708f64b6e4f30fd9d7f657f284d83746bfeb3f36d802
```

The named `bayesian_sparse_survey.pdf` was not present in the listed conversation uploads and was not returned by available conversation/Library searches. It was not independently reviewed or substituted with a different survey. The implementation follows the concrete guideline, with the precision/zero-omega and epsilon adaptations disclosed below. The supplied paper-reference labels do not by themselves establish faithful reproduction of the source paper, and this work did not re-certify that reproduction.

**Observed validation:** the uploaded baseline produced 23 passed / 4 skipped; the modified available suite produced **110 passed / 5 skip reports**. Five synthetic CPU three-round runs completed through the unchanged trainers and production sparse/accounting code. Flower/Ray, CUDA, Bayesian-Torch, and full MNIST/CIFAR training were not exercised in this environment. See the exact logs in `validation/` and `VALIDATION_SPARSE.md`.

## 1. Baseline code audit

| Concern | Actual path and finding |
|---|---|
| Local FedAvg training | `src/bayesfl/training/deterministic.py::train_fedavg`; unchanged SGD trainer |
| Local FOLA training | `src/bayesfl/training/fola.py::train_fola`, `_train_fola_paper_reference`, `_train_fola_online_recurrence`; unchanged |
| Posterior transport | `src/bayesfl/posterior/packing.py`; one named-parameter mean array and one raw precision array per FOLA tensor, not variance arrays |
| Curvature/precision helpers | `src/bayesfl/posterior/gaussian.py`; unchanged |
| Original server reducer | `ResearchStrategy._aggregate_fola`; extracted without changing its arithmetic into `posterior/aggregation.py` |
| Client fit boundary | `src/bayesfl/client.py::BayesFLNumPyClient.fit`; training finishes before compression |
| Client schedule and fit replies | `src/bayesfl/strategies/research_strategy.py`; original sampling retained by default, separate deterministic logical-ID scheduling available for matched studies |
| Flower assembly | `src/bayesfl/server.py`; existing ClientApp/ServerApp/Ray path, using a custom legacy Server loop for real budget stopping |
| Evaluation/checkpoints | `src/bayesfl/evaluation.py::CentralEvaluator`; central evaluation, no extra model downloads |
| Configuration/CLI | `src/bayesfl/config.py`, `src/bayesfl/main.py`; additive compression/accounting/resume fields and external method aliases |
| Logging/plotting | `logging_utils.py`, original `utils.py` and scripts; new dedicated byte-aware comparison module added without replacing historical plot scripts |
| Parameters and buffers | Named parameters are transported. Existing MLP, BasicCNN, and GroupNorm ResNet have no persistent running-statistic buffers requiring additional transport. New sparse models with buffers or distinct aliased parameter storage are rejected pending an explicit adapter |

The uploaded paper-reference implementation accumulates squared minibatch-mean task gradients, not per-example Fisher/Hessian estimates. MNIST and CIFAR initialize omega differently. The alternative online-recurrence branch has separate stabilization semantics. These are preserved, not silently repaired as part of communication sparsification.

## 2. Implemented behavior

The five externally logged methods are `fedavg_dense`, `fola_dense`, `fola_sparse_kl_global_local`, `fola_sparse_kl_local_global`, and `fola_sparse_random`. Existing `fedavg`, `fola`, and `bbb` configuration names still work.

For every sparse fit, the client snapshots the incoming posterior, runs the unchanged full local trainer, and selects exactly `ceil(r*d)` model-wide coordinates. It uploads a single little-bit-order packed uint8 bitmap and the ascending-index float32 mean/precision pairs. Both KL argument orders reverse the complete coordinate KL formula. Ties use smaller global indices. Random selection is uniform without replacement with an independent stable seed/client/round generator.

The live sparse server never accesses a full untransmitted local posterior. It checks metadata, round/snapshot/layout/client identity, packed length, trailing padding, count, dtypes, finiteness and precision validity, reconstructs omissions from the exact broadcast, normalizes sample weights once over the accepted full result set, and calls the baseline FOLA reducer. Full downlinks remain dense.

Only masks differ among the three sparse controls. The same ratio, typed packet format, numerical policy, local loss, FOLA mode, client weighting and reconstruction are used. Full retention includes bitmap overhead. The baseline failure policy remains strict: missing/rejected client results fail the round instead of silently aggregating a partial set.

## 3. Explicit statistical adaptations

### Preserve raw precision on the wire

The actual source stores omega/precision. Dense and sparse FOLA therefore retain raw float32 precision communication, explicitly permitted as a shared protocol adaptation by the supplied guideline. There is no reciprocal conversion on the wire or during reconstruction. This avoids changing a dense full-retention result through reciprocal rounding and preserves valid baseline zero-omega state.

KL scoring uses true reciprocal precision under `precision_policy: strict`. Nonpositive scoring precision is rejected. The official zero-initial-omega CIFAR setup must explicitly use `floor_for_score`, computing `1/max(omega, floor)` for scoring only. A null floor uses the existing `fola.precision_min`. No trained, transmitted or aggregated omega is changed. Floored fractions and the exact policy/floor are logged. This is a surrogate-Gaussian KL, not a claimed finite KL for a zero-precision Gaussian or proof of calibrated uncertainty.

### Preserve epsilon except for coordinates omitted by everyone

The uploaded paper-reference reducer calculates:

```text
omega_new = sum_k pi_k * omega_effective_k
mu_new = sum_k pi_k * omega_effective_k * mu_effective_k
         / (omega_new + aggregation_epsilon)
```

That is not exactly the undamped product equation in the guideline. The same reducer is called for dense and sparse FOLA. After it returns, a coordinate selected by no client is restored exactly from the old broadcast, for both mean and precision. This prevents epsilon-induced shrinkage on a no-update coordinate. Touched coordinates keep the original epsilon behavior. At full retention no restoration applies, so aggregation remains identical to the original dense reducer on identical inputs.

Golden fixtures were obtained by extracting and executing the original uploaded `_aggregate_fola` method from the pristine baseline, for both modes. Provenance and the original method hash are recorded in `tests/fixtures/uploaded_fola_golden.json`. This checks implementation preservation, not agreement with a paper not independently audited here.

## 4. Concrete file changes

### Existing production files modified

| File | Change |
|---|---|
| `src/bayesfl/config.py` | Compression/communication dataclasses, five-way aliases, ratio/policy/budget validation, explicit zero-omega guard, compatible overrides |
| `src/bayesfl/client.py` | Scalar logical-ID query, immutable pre-training snapshot, post-training compression, sparse packet metadata |
| `src/bayesfl/strategies/research_strategy.py` | Logical client identity mapping, matched optional schedule, packet validation/reconstruction, event accounting, shared reducer invocation, total-count logging correction |
| `src/bayesfl/server.py` | RunState assembly, custom budget-aware Server, resumed initial state, sparse buffer/storage audit |
| `src/bayesfl/main.py` | New CLI flags, clean-resume setup/config checks, actual cached partition-content hash validation |
| `src/bayesfl/evaluation.py` | Cumulative byte/exposure fields, final checkpoint on stopping, evaluated-round resume commit |
| `src/bayesfl/logging_utils.py` | Existing-row loading for resume; atomic CSV replacement |
| `README.md` | New entry point; previous archive README and supplied handoff retained under `docs/` |

### New core files

| File | Responsibility |
|---|---|
| `src/bayesfl/posterior/sparse.py` | Layout/snapshot digests, variance adapter for scores, stable KL, exact-cardinality selectors, bitmap codec and validation |
| `src/bayesfl/posterior/aggregation.py` | Arithmetic-preserving extraction of the original two-mode FOLA reducer |
| `src/bayesfl/communication.py` | Integer byte counters, append-only transfer ledger, actual array and optional serialized-tensor sizes, rejection/unknown-size handling |
| `src/bayesfl/experiment_state.py` | Server-owned broadcast/current state, strict reservation, packet receive/finish lifecycle, omission invariants, resume consistency checks |
| `src/bayesfl/budget_server.py` | Actual true-round Flower loop with budget stop before dispatch; no fake empty rounds |
| `src/bayesfl/communication_plots.py` | Observed round/byte curves, common-budget lookup, terminal summaries, explicit seed averaging |
| `src/bayesfl/offline_smoke.py` | Synthetic CPU integration harness using the real trainers and new shared production logic |

### New scripts/configurations/documentation

`scripts/generate_sparse_configs.py` clones paired locked configs without retuning or overwriting originals. `scripts/summarize_sparse_study.py` validates actual completed run artifacts and prints common-budget results. `scripts/smoke_sparse_offline.py` invokes the synthetic integration harness.

Ten new YAMLs, two queues and two provenance/cost manifests are under `scripts/configs/bayesian_sparse/`: five methods each for fixed-one-label/10-sample MNIST and BasicCNN CIFAR alpha .1 with half the local data. Both use keep=.5 for sparse methods and one common budget equal to 150 dense FOLA rounds. Epoch/batch settings and inherited LR schedules are unchanged; the total safety cap is 1,000. Computed affordability is 300 FedAvg, 150 dense FOLA, and 197 sparse rounds in both studies.

The additional tests cover numerical directions/rankings, exact masks, bitmaps/padding, malformed packets, no-hidden-state reconstruction, both original reducer modes, all-omitted invariants, byte counts, strict budget stopping, clean resume, zero omega, config-generation preservation, real central evaluation, and byte-aware plotting/summary behavior. A separate real-Flower integration module is included for the target environment but was skipped here.

## 5. Accounting and fairness

The primary metric is aggregate logical client-server **array payload**, uplink plus downlink, summed at message boundaries. It is not model checkpoint size, Python memory usage, or measured on-wire traffic. Metadata/control/envelope bytes are excluded from the primary metric. Actual Flower serialized tensor sizes are kept separately when available; whole-message sizes remain null, not estimated.

For d float32 posterior-coordinate pairs, m retained coordinates and K recipients/replies:

```text
B_dense_FOLA = 16*K*d
B_sparse_FOLA = K*(8*d + 8*m + (d+7)//8)
B_FedAvg = 2*K*sum(actual uploaded FedAvg array.nbytes)
```

This last expression includes both directions and does not charge FedAvg for a covariance. Initialization is server-local; evaluation is central; the additional model-array cost is zero for both. An optional identity handshake carries only scalar/control information, outside the core-array counter.

Arrived but rejected array packets are charged. Undecodable tensor bytes are recorded at the serialized-storage level; raw array size is marked unknown, counters are explicitly incomplete/lower-bound, and exact budget comparison is refused for that failed run. Network partial messages/retries not observable at this boundary are not guessed.

The whole next-round array cost must fit before dispatch. The runner does not reduce participation or retention to fill a remainder. Global evaluation rows attach observed cumulative counters to the actual evaluated state. Common-budget lookup uses the latest evaluated model not exceeding the budget, never best test accuracy or an over-budget checkpoint. Round-capped runs are not extrapolated. Matched generated runs share logical client schedules and true-round seed prefixes; their original learning-rate horizons do not change with the communication budget.

## 6. Resume and operational boundaries

A completed evaluated round is committed with full global arrays, true round, layout and partition hashes, configuration fingerprint, ledger count/digest/totals, exposure counters, and seed/schedule policy. Resume appends to the same directory and cannot silently discard already observed communication. Only total round and byte limits may change. A deterministic client schedule is required from the start. The baseline already creates a fresh optimizer each fit; there is no persistent cross-round optimizer state to recover.

In-flight failure recovery, partial-message retries, rollback of consumed bytes, and whole-experiment power-loss atomicity are not provided. A mismatch after an interrupted round is rejected rather than hidden. Identity or other control events after the last committed state also remain part of the strict consistency check. Do not use concurrent writers in one run directory.

Current sparse models may not introduce unhandled buffers or distinct aliased parameter storage. Full-client server snapshot collection is deliberately disallowed for sparse configs because the server has not received those values. Dense snapshot behavior remains available. Per-evaluation resume snapshots add storage/I/O overhead but not communication bytes.

## 7. What is preserved and what is not claimed

All **398 original YAML configurations are byte-identical**. The original local training modules, posterior packing, Gaussian/precision helpers, model definitions, data partition generators, dependency manifests and historical plot scripts are retained. A machine-readable change manifest records original/current file hashes.

Legacy dense runs retain their original client-sampling setting by default. New matched studies intentionally enable deterministic logical-ID scheduling and stable aggregation order for all five methods. Different floating-point summation orders can affect trajectories relative to old asynchronous runs; this common new control is not portrayed as bitwise replay of historical runs. Training equations and golden reducer inputs/results are the preservation checks actually made.

No sparse-method MNIST/CIFAR accuracies, convergence advantage, publication claims, or claim of superior global-first KL are inferred from synthetic checks. No sparse downlink, quantization, error feedback, permanent pruning, model-subnet training, adaptive ratio, privacy mechanism, or secure aggregation was added. The retained original README's historical accuracies are not new results and were not reverified against absent training CSVs.

## 8. Run commands and next validation gate

`SPARSE_README.md` contains exact commands for installation in the working research environment, offline and real-Flower tests, a distinctly named two-round real Ray smoke, the five-way study queues, terminal summaries, explicit plots, new seeds/ratios, and clean resume. Before a long dataset experiment, run the optional Flower integration tests and the real Ray smoke in the user's original Python 3.10/CUDA environment; those are the execution boundaries not validated here.
