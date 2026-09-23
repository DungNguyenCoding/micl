# BayesFL: sparse posterior communication

This is the implemented extension to the uploaded FedAvg/FOLA repository. Read this file before the historical README. The original local trainers, posterior estimator, models, partition generators, dependency declarations, and 398 original experiment YAMLs are preserved. New studies live in a separate directory.

**Validation boundary:** 110 tests passed in the available CPU environment; five dependency-related skip reports remain. Flower/Ray and Bayesian-Torch were not installed here. The new Flower integration module is supplied but was not executed here. Synthetic checks are protocol/training integration tests, not MNIST/CIFAR performance results or CUDA/Ray certification. See `VALIDATION_SPARSE.md` and `IMPLEMENTATION_REPORT.md`.

## 1. Start in the existing research environment

Extract this archive to a new directory, not on top of an actively running experiment. Reuse the established Python 3.10 environment with its working Torch/CUDA pair and the original pinned dependencies. No dependency migration is part of this change.

```bash
cd /path/to/bayesfl_sparse
conda activate dungndh
python -m pip install -e . --no-deps
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
```

For a fresh environment, the original `requirements.txt` and installation notes remain applicable. The package still declares Python `>=3.10,<3.11`, `flwr[simulation]==1.29.0`, and `bayesian-torch==0.5.0`. Do not replace a working CUDA-enabled Torch installation merely to reproduce the CPU validation environment.

Run the actual Flower serializer/client/strategy/server tests in that environment:

```bash
python -m pytest -q tests/test_sparse_flower_integration.py
```

This module uses real Flower objects with an in-process transport adapter and synthetic data. It does not require Ray. A skipped module is not a successful Flower integration check.

A separate, dependency-light check runs unchanged local trainers through the same production codec, reconstruction, reducer, ledger, and resume logic, without Flower/Ray or dataset downloads:

```bash
python scripts/smoke_sparse_offline.py \
  --output-dir outputs/synthetic_sparse_check --rounds 3
```

The destination must be fresh. These are tiny synthetic binary-classification runs, not dataset benchmarks.

## 2. Implemented methods

| External identifier | What is uploaded after full local training |
|---|---|
| `fedavg_dense` | Original dense FedAvg arrays |
| `fola_dense` | Original dense FOLA means and raw precisions |
| `fola_sparse_kl_global_local` | Top-coordinate `KL(q_global_before || q_local_after)` |
| `fola_sparse_kl_local_global` | Top-coordinate `KL(q_local_after || q_global_before)` |
| `fola_sparse_random` | Uniform exact-cardinality random coordinate subset |

Legacy `fedavg`, `fola`, and `bbb` names remain accepted. Internally, configs normalize the FOLA names to `method: fola` plus `compression.selection_rule`; logs use the explicit five-way identifiers above. `fola_sparse` is also accepted when a non-dense selection rule is specified explicitly.

Compression happens **after the original full local FOLA training**. The mask never changes the loss, freezes weights, prunes a model, or applies dropout rescaling. Both mean and precision for a chosen coordinate are sent. Selection is model-wide, not layer-wise. `m = ceil(keep_ratio * d)` uses the same Python binary64 product and ceiling in the encoder, decoder, and budget predictor. Scores use float64; transmitted posterior values remain float32.

For variances `v0` and `vk` and `delta = muk - mu0`, the scores are:

```text
KL(global || local) = 0.5 * [log(vk/v0) + (v0 + delta^2)/vk - 1]
KL(local || global) = 0.5 * [log(v0/vk) + (vk + delta^2)/v0 - 1]
```

The implementation uses the stable `expm1(log(vP)-log(vQ))` form. Both the variance-change and mean-change terms are reversed, not just one denominator. Top-score ties choose smaller global coordinate indices. Random masks have exactly `m` distinct entries and a dedicated SHA-256-derived NumPy generator indexed by experiment seed, client ID, and true round. This does not advance the client-training or client-sampling RNG streams.

## 3. Important adaptation: raw precision, zero omega, and epsilon

### 3.1 The supplied FOLA code does not transmit variances

`training/fola.py` produces float32 raw precision/omega. Dense FOLA already transmits it and the server consumes it directly. The new sparse protocol therefore deliberately transmits **precision**, not variance:

```text
array 0: uint8[ceil(d/8)]   packed bitmap, little bit order
array 1: float32[m]        selected means, ascending coordinate order
array 2: float32[m]        selected raw precisions, ascending coordinate order
```

There are exactly three model arrays. No dense debug upload, score vector, separate index array, or hidden full local state reaches the sparse reducer. Sparse full retention still sends the bitmap. Dense FOLA does not. Mean/precision and mean/variance pairs have the same 8-byte payload size, but not the same statistical meaning.

Metadata records `covariance_representation: precision`, protocol version, layout digest, broadcast snapshot digest, round, client, counts, score policy, and score floor. The server reconstructs omissions from its immutable copy of the exact broadcast. Snapshot metadata is checked before aggregation.

### 3.2 Zero precision is not a finite Gaussian posterior

Official MNIST initializes omega at 1.0; official CIFAR initializes it at 0.0. Replacing zero with positive precision in the optimizer or upload would change the existing CIFAR baseline. Two **explicit** policies are implemented:

```yaml
compression:
  precision_policy: strict
  score_precision_floor: null
```

`strict` requires every scoring precision to be positive. It computes `variance = 1 / raw_precision` in float64 without clipping. A sparse configuration with zero initial omega is rejected before training.

```yaml
compression:
  precision_policy: floor_for_score
  score_precision_floor: null
```

`floor_for_score` computes only the selection surrogate

```text
v_score = 1 / max(raw_precision, score_precision_floor)
```

A null floor resolves to the already configured `fola.precision_min`: 1e-8 in the included MNIST profile and 1e-12 in the included CIFAR profile. It **does not modify the trained, transmitted, reconstructed, or aggregated raw precision**. The KL is consequently a KL between the stated positive surrogate Gaussians, not a claimed finite KL of a zero-precision Gaussian. The same policy applies to all three sparse controls; per-client floored fractions are logged. The included CIFAR study opts in explicitly. This is a reported numerical modeling choice, not an established calibration guarantee or a tuned hyperparameter improvement.

### 3.3 Preserve the supplied reducer, with the required all-omitted exception

The original paper-reference mean update divides by `weighted_precision + aggregation_epsilon`. It is not exactly the undamped Gaussian-product equation in the guideline. That epsilon remains unchanged for any coordinate selected by at least one client. We first reconstruct all client distributions and call the extracted original dense reducer.

Then a coordinate **omitted by every participating client** is copied exactly from the broadcast, for both mean and precision. Otherwise epsilon would shrink even a no-update mean. This explicit exception enforces the requested omission invariant. No client-weight renormalization per coordinate, extra global prior product, arithmetic variance averaging, or `1/keep_ratio` factor is introduced.

At full retention this exception never applies; the sparse and dense reducers receive identical local arrays. Golden tests compare both FOLA aggregation modes with values generated directly from the original uploaded method. Matched synthetic full-training tests establish exact equality in the tested CPU environment, not a general cross-device/async bitwise guarantee.

## 4. Ready-to-run paired studies

There are five new configs in each directory:

```text
scripts/configs/bayesian_sparse/mnist_fixed_s10/
scripts/configs/bayesian_sparse/cifar_a0p1_f0p5/
```

Each directory includes `queue.txt` and `study_manifest.json`. These are **new sparse-study runs**, not overwrites of the historical official runs.

Both studies use the two inherited locked baselines, three sparse controls at keep ratio 0.5, seed 0, E=10, the original batch size and training settings, and a common byte budget equal to 150 dense-FOLA rounds. The safety cap is 1,000 rounds, not a learning-rate horizon. The inherited CIFAR FOLA cosine horizon remains 400. The inherited FedAvg/FOLA LR/schedule differences remain visible; those comparisons are not isolated aggregation-rule ablations.

| Profile | d | Clients/round | Common array-byte budget | FedAvg rounds | Dense FOLA rounds | Each sparse rule rounds |
|---|---:|---:|---:|---:|---:|---:|
| MNIST fixed 10/client, one class/client | 545,810 | 100 | 130,994,400,000 | 300 | 150 | 197 |
| CIFAR BasicCNN, alpha .1, data fraction .5 | 878,538 | 20 | 42,169,824,000 | 300 | 150 | 197 |

These round counts are **payload affordability calculations**, not measured accuracy results. FedAvg also receives its affordable extra rounds; giving only sparse FOLA extra rounds would be an unfair budget comparison. Dense and sparse FOLA share the same true-round LR and prior schedules, including when sparse runs longer.

Prepare both training and test splits in the YAML's `data.root` before the first real simulation. The unchanged loader uses `download=False` after partition preparation. For the supplied default `./data`:

```bash
python - <<'PY'
from torchvision.datasets import MNIST, CIFAR10
for dataset in (MNIST, CIFAR10):
    for train in (True, False):
        dataset(root='./data', train=train, download=True)
print('Training and test splits are available.')
PY
```

Run one newly named two-round real Flower/Ray smoke before a long study. The changed name prevents a smoke override being mistaken for a completed study:

```bash
python - <<'PY'
from pathlib import Path
import yaml
source = Path('scripts/configs/bayesian_sparse/mnist_fixed_s10/sparse_mnist_fixed_s10_fola_sparse_kl_global_local_keep0p5_seed0.yaml')
cfg = yaml.safe_load(source.read_text())
cfg['run_name'] = 'smoke_mnist_sparse_global_local_r2'
cfg['training']['rounds'] = 2
cfg['communication']['max_communication_bytes'] = None
path = Path('scripts/configs/smoke_mnist_sparse_global_local_r2.yaml')
if path.exists():
    raise FileExistsError(path)
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(path)
PY
CUDA_VISIBLE_DEVICES=0 python -m bayesfl.main \
  --config scripts/configs/smoke_mnist_sparse_global_local_r2.yaml
```

The smoke retains E=10 and the actual dataset/profile; it is a real execution check, not a performance benchmark. The supplied new configs are not evidence that this real simulation has already run here.

Run all five MNIST methods sequentially, stopping on the first failure:

```bash
while IFS= read -r config; do
  [ -z "$config" ] && continue
  CUDA_VISIBLE_DEVICES=0 python -m bayesfl.main --config "$config" || break
done < scripts/configs/bayesian_sparse/mnist_fixed_s10/queue.txt
```

Use the CIFAR queue path for that study. Existing `_run_nohup.sh` and `_run_gpu_queue.sh` remain available with their original PATH-resolved interpreter behavior. Pre-generate/check the shared partition before parallel runs. New startup validation checks the actual NPZ content against its metadata SHA-256; it does not rebuild or change partition indices.

### Print results to the terminal and generate both plots

After a study completes:

```bash
MPLBACKEND=Agg python scripts/summarize_sparse_study.py \
  --study-dir scripts/configs/bayesian_sparse/mnist_fixed_s10 \
  --plot-dir outputs/plots/sparse_mnist_fixed_s10
```

This prints actual directories and common-budget accuracy, then saves separate PNGs for accuracy versus rounds and cumulative communication, observed-data CSV, and a comparison manifest. It does not create a terminal-summary `.txt` file. Multiple matching runs require an explicit `--latest`; the newest incomplete run is not silently replaced by an older successful one. Resolved configs and expected completed evaluation rows are checked.

For arbitrary explicit runs:

```bash
MPLBACKEND=Agg python -m bayesfl.communication_plots \
  --run-dir outputs/ACTUAL_DENSE_RUN \
  --run-dir outputs/ACTUAL_SPARSE_RUN \
  --label 'Dense FOLA' --label 'KL(global || local)' \
  --output-dir outputs/plots/explicit_sparse_comparison
```

Add `--summary-only` to print without creating plots. Add `--budget-bytes INTEGER` to query a particular jointly covered budget. The result is the latest actually evaluated checkpoint at or below that budget, not best test accuracy. Round-capped runs are not extrapolated beyond their last observed cost. A communication-budget-stopped run can report its final affordable checkpoint for its unspent sub-round remainder.

`--average-seeds` groups matching repeated labels and plots means plus sample-SD bands. It requires at least two distinct seeds per group, matching compared seed sets, equal evaluation/byte grids, and matching resolved data/model/training settings. Default plots keep runs separate.

## 5. Generate another keep ratio, seed, profile, or byte budget

Example: create a separate 10%-retention MNIST study without changing any existing config:

```bash
python scripts/generate_sparse_configs.py \
  --base-fola scripts/configs/run_mnist_fixed1c_s10_official_fola_n100_seed0_e10_b32_lr001_lam0p01_r150.yaml \
  --base-fedavg scripts/configs/run_mnist_fixed1c_s10_official_fedavg_n100_seed0_e10_b32_lr001_r150.yaml \
  --out-dir scripts/configs/bayesian_sparse/mnist_fixed_s10_keep01_seed1 \
  --tag mnist_fixed_s10_keep01_seed1 \
  --keep-ratios 0.1 --seed 1 --rounds 1000 --dense-budget-rounds 150
```

The generator clones both original locked configs, verifies shared data/model/partition settings and local E/B, preserves their respective LRs and schedules, and refuses overwriting existing study files. It rejects E>10 for newly generated studies but does not invalidate historical E20/E40 configs. Use `--precision-policy floor_for_score` for zero-initial-omega CIFAR. A user-specified `--max-communication-bytes INTEGER` is an alternative to `--dense-budget-rounds`.

For a round-limited comparison, omit both byte-budget options and set `--rounds 150`. For multiple keep ratios, use e.g. `--keep-ratios 0.5 0.1`; dense baselines appear once, with three sparse controls per ratio.

The new config sections are:

```yaml
method: fola_sparse_kl_global_local
compression:
  selection_rule: kl_global_local
  keep_ratio: 0.5
  covariance_representation: precision
  precision_policy: strict
  score_precision_floor: null
communication:
  deterministic_client_schedule: true
  max_communication_bytes: null
  budget_metric: cumulative_all_array_bytes
```

CLI additions: `--selection-rule`, `--keep-ratio`, `--precision-policy`, `--max-communication-bytes`, and `--resume RUN_DIR`. `training.rounds` remains the total safety cap (`--rounds` overrides it). There is no new `max_rounds` YAML key. The same logical client schedule prefix is derived for all paired methods, independently of mask and local-training randomness. Legacy configs default to the original client sampler; use the generated matched configs for the sparse comparison.

## 6. Communication scope and exact accounting

For `d` modeled weights, `m=ceil(r*d)`, and `K` participating clients, with no auxiliary/evaluation arrays:

```text
Dense FOLA/client: down 8*d, up 8*d bytes
Sparse FOLA/client: down 8*d, up 8*m + ceil(d/8) bytes
Dense FOLA/round: 16*K*d bytes
Sparse FOLA/round: K*(8*d + 8*m + ceil(d/8)) bytes
Dense float32 FedAvg/round: 8*K*d_F bytes
```

The ledger sums final typed array sizes at logical dispatch/arrival boundaries, not only round number times a theoretical constant. Uploads that arrive and are rejected still count. Non-arriving partial network packets/retries cannot be inferred by this API. An undecodable tensor reply records its known serialized tensor bytes and flags unknown array payload; such a failed run cannot be used as an exact byte comparison.

The primary counter is `cumulative_all_array_bytes`, including initialization and evaluation model arrays if any. These experiments initialize server-local and evaluate centrally, so both extra model-array costs are zero. Optional client identity queries carry scalar/control metadata, not model arrays. Metadata is not called physically free: **whole-message and network traffic are not measured**. `cumulative_serialized_tensor_bytes` counts actual Flower serialized tensor strings when available, including NumPy array headers, but excludes control/message/transport envelopes.

Each budget check reserves the whole next round, including full downloads to each recipient and expected replies. If it cannot fit, no clients train and no fake empty rounds are executed. The client count and keep ratio are not changed to use up a budget remainder. `stop_reason` distinguishes `communication_budget`, `max_rounds`, and failed runs. A safety-cap stop takes precedence if both limits are reached together.

At keep=1 the result matches dense aggregation but **costs more** because the bitmap is compulsory. Uplink-only savings must not be presented as total savings; the downlink remains dense.

## 7. Outputs and clean completed-round resume

New outputs alongside the baseline metrics/checkpoints:

```text
communication/layout_manifest.json
communication/protocol.json
communication/events.jsonl
metrics/global_metrics.csv       # existing file, now also includes byte/cost/step columns
metrics/round_train_metrics.csv  # actual total counts corrected; byte fields added
resume/latest.json
resume/round_XXXXXX.json         # protocol/config/partition/round/ledger and seed policy
resume/round_XXXXXX.npz          # complete global model and precision state
run_summary.json
```

The event ledger is append-only. CSV writes use atomic replacement and load existing rows when resuming. Latest and previous resume generations are retained; scheduled baseline checkpoints remain. A final checkpoint is saved even when the budget ends between checkpoint intervals. Saving full resume state every evaluated round adds disk I/O; it is not network communication.

Resume the **existing run directory**, retaining its ledger and actual round number:

```bash
CUDA_VISIBLE_DEVICES=0 python -m bayesfl.main \
  --resume outputs/ACTUAL_COMPLETED_RUN_DIRECTORY \
  --rounds 1000 --max-communication-bytes 200000000000
```

Only the round cap and byte limit may change; a new limit cannot be below already consumed bytes. Model, loss, LR schedule/horizon, seed, data, protocol, mask ratio, score policy, and run identity are checked. The original run must have enabled deterministic client scheduling from the start. All local optimizers were already recreated every fit in the supplied baseline, so no cross-round optimizer state is invented or restored. True-round reseeding and independent mask/schedule generators reproduce the same prefix.

This is **clean completed-and-evaluated-round resume**, not arbitrary mid-round fault recovery. Extra in-flight transfer events after the last committed state cause an explicit refusal rather than erasing spent communication or charging completed messages again. Missing/corrupt checkpoints and inconsistent CSV/ledger/partition state fail validation. Do not run two writers in the same directory. Power-loss atomicity of an entire multi-file experiment is not claimed.

## 8. Boundaries

No sparse downlink, permanent pruning, subnet training, quantization, error feedback, adaptive ratios, symmetric KL, shared-seed replacement of random bitmaps, differential privacy, or secure aggregation was added. Existing server-side dense-client snapshot saving is rejected for sparse configs because the server no longer receives full local posteriors. Old dense snapshot behavior remains available for dense FOLA.

The supplied guideline and source handoff govern this implementation. `bayesian_sparse_survey.pdf` was not returned by the available conversation/Library retrieval and is not claimed to have been inspected. No missing proposal text or missing historical accuracy histories were invented. This implementation does not establish that either KL direction outperforms the other or the random control; the generated studies are the experiments needed to measure that.
