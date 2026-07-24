# Bayesian Federated Learning Simulation Base Source

This source compares three training modes under the same grouped Flower simulation layout:

- `fedavg`: deterministic Federated Averaging baseline.
- `vi`: mean-field variational Bayesian FL using Pyro SVI and `AutoDiagonalNormal` over an MLP weight posterior.
- `ola`: Online Laplace Approximation / FOLA-style Bayesian FL using diagonal empirical Fisher precision, Gaussian-product server aggregation, and prior-iteration local regularization.

The main design choice is that **physical devices** and **Flower virtual clients** are separate. For example:

```bash
--num_devices 300 --num_virtual_clients 24 --client_fraction 0.1
```

selects about 30 physical devices per round, but launches at most 24 Flower/Ray client tasks per round. Each virtual client sequentially simulates the selected physical devices assigned to its group.

This version adds a **Bayes-FL observability schema**. Training still writes only `.csv`, `.log`, and `.pt` files. PNG plots are generated later with `utils.py`.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The plotting utilities intentionally avoid Pandas. This reduces NumPy/Pandas binary compatibility issues in shared Conda environments.

---

## Quick smoke test

```bash
python main.py \
  --method fedavg \
  --dataset mnist \
  --model mlp \
  --num_devices 10 \
  --num_virtual_clients 2 \
  --client_fraction 0.2 \
  --num_rounds 1 \
  --local_epochs 1 \
  --output_dir outputs/smoke_fedavg
```

---

## Example runs

### FedAvg, MNIST, IID balanced

```bash
python main.py \
  --method fedavg \
  --dataset mnist \
  --model mlp \
  --iid true \
  --balanced true \
  --num_devices 300 \
  --num_virtual_clients 24 \
  --client_fraction 0.1 \
  --num_rounds 20 \
  --local_epochs 1 \
  --batch_size 32 \
  --lr 0.05 \
  --output_dir outputs/fedavg_mnist_iid
```

### Variational Bayesian FL, MNIST, non-IID unbalanced

```bash
python main.py \
  --method vi \
  --dataset mnist \
  --model mlp \
  --iid false \
  --balanced false \
  --noniid_alpha 0.1 \
  --unbalanced_alpha 0.5 \
  --num_devices 300 \
  --num_virtual_clients 24 \
  --client_fraction 0.1 \
  --num_rounds 20 \
  --local_epochs 1 \
  --vi_lr 0.001 \
  --vi_prior_scale 0.05 \
  --bayes_aggregation product \
  --output_dir outputs/vi_mnist_noniid_unbalanced
```

### Online Laplace/FOLA-style Bayesian FL, CIFAR-10

```bash
python main.py \
  --method ola \
  --dataset cifar10 \
  --model mlp \
  --iid false \
  --balanced true \
  --noniid_alpha 0.3 \
  --num_devices 300 \
  --num_virtual_clients 24 \
  --client_fraction 0.1 \
  --num_rounds 50 \
  --local_epochs 1 \
  --batch_size 32 \
  --lr 0.02 \
  --ola_prior_lambda 1.0 \
  --output_dir outputs/ola_cifar10_noniid
```

---

## New observability arguments

These arguments control how much information is collected.

| Argument | Default | Meaning |
|---|---:|---|
| `--eval_every` | `1` | Evaluate global model/posterior every N rounds. |
| `--heavy_eval_every` | `5` | Save posterior summaries and SNR histograms every N rounds. |
| `--local_eval_every` | `0` | Evaluate selected clients on their local validation set every N rounds. `0` disables this. |
| `--val_ratio` | `0.0` | Fraction of each client's local data held out for local validation. Set this above 0 to populate `client_eval_metrics.csv`. |
| `--eval_mc_samples` | `1` | Number of posterior samples for Bayesian predictive evaluation. Use `>1` for VI/OLA uncertainty decomposition. |
| `--calibration_bins` | `15` | Number of bins used for ECE and reliability diagrams. |
| `--snr_hist_bins` | `80` | Number of SNR histogram bins. |
| `--save_posterior_every` | `0` | Save posterior snapshots every N rounds. Final snapshot is always saved. |
| `--metrics_level` | `bayes` | `basic`, `bayes`, or `full`; controls heavy Bayesian summaries. |

Recommended Bayesian evaluation run:

```bash
python main.py \
  --method ola \
  --dataset mnist \
  --model mlp \
  --iid false \
  --noniid_alpha 0.1 \
  --num_devices 100 \
  --num_virtual_clients 10 \
  --client_fraction 0.1 \
  --num_rounds 20 \
  --eval_mc_samples 5 \
  --heavy_eval_every 5 \
  --save_posterior_every 5 \
  --val_ratio 0.1 \
  --local_eval_every 5 \
  --output_dir outputs/ola_mnist_observable
```

---

# Output files

A run now produces the following files:

```text
outputs/<run_name>/
├── config.csv
├── run_summary.csv
├── metrics.csv
├── client_data_summary.csv
├── device_summary.csv
├── selection_summary.csv
├── selected_clients.csv
├── client_train_metrics.csv
├── client_eval_metrics.csv
├── calibration_bins.csv
├── posterior_summary.csv
├── snr_histograms.csv
├── aggregation_diagnostics.csv
├── communication_metrics.csv
├── pruning_eval.csv
├── run.log
├── final_model.pt
└── posterior_snapshots/
    ├── round_0005.pt
    ├── round_0010.pt
    └── final.pt
```

The schema version is stored as:

```text
bayesfl_observability_v1
```

Use blank/empty cells as `NaN`. A blank value means the metric is not applicable or was not computed. For example, FedAvg does not have posterior variance, so posterior uncertainty columns are blank.

---

# Main metric file: `metrics.csv`

`metrics.csv` contains **one row per evaluated communication round**. It is the main file for mixed plots across methods or hyperparameters.

## Identity/config columns

| Column | Meaning |
|---|---|
| `schema_version` | Metric schema identifier. |
| `run_id` | Output-folder based run name. |
| `round` | Communication round. |
| `method` | `fedavg`, `vi`, or `ola`. |
| `dataset` | `mnist` or `cifar10`. |
| `model` | `mlp` or `cnn`. |
| `iid`, `balanced` | Dataset split flags. |
| `noniid_alpha`, `unbalanced_alpha` | Heterogeneity controls. |
| `num_devices` | Number of simulated physical clients. |
| `num_virtual_clients` | Number of Flower/Ray virtual clients. |
| `client_fraction` | Physical client sampling fraction. |
| `selected_count` | Number of physical devices selected this round. |
| `selected_examples` | Number of examples owned by selected devices. |
| `total_examples` | Same as selected examples for the round aggregation. |
| `local_epochs`, `batch_size`, `lr`, `seed` | Training configuration. |

## Global performance submetrics

These are computed on the centralized test set using the current global model/posterior.

| Column | Meaning | Plot examples |
|---|---|---|
| `global_accuracy` | Global test accuracy. | accuracy/round, method comparison |
| `global_loss` | Global negative log likelihood / cross entropy. | loss/round |
| `global_error_rate` | `1 - global_accuracy`. | error/round |
| `global_nll` | Mean negative log likelihood. | calibration/uncertainty comparison |
| `global_brier` | Mean Brier score. | calibration comparison |
| `global_ece` | Expected calibration error. | ECE/round |
| `global_mce` | Maximum calibration error. | worst-bin calibration |
| `global_mean_confidence` | Mean max softmax probability. | confidence/round |
| `global_mean_entropy` | Mean predictive entropy. | predictive uncertainty/round |
| `global_num_eval_examples` | Test examples evaluated. | sanity check |

Backward-compatible aliases are also stored:

| Alias | Same as |
|---|---|
| `accuracy` | `global_accuracy` |
| `loss` | `global_loss` |
| `train_loss` | selected-client mean training objective |

## Bayesian predictive uncertainty submetrics

These are most useful when `--eval_mc_samples > 1` for VI/OLA.

| Column | Meaning |
|---|---|
| `global_mc_samples` | Number of posterior samples used for prediction. |
| `global_predictive_entropy` | Entropy of the averaged predictive distribution. |
| `global_expected_entropy` | Average entropy across sampled models. |
| `global_mutual_information` | `predictive_entropy - expected_entropy`; epistemic uncertainty proxy. |
| `global_aleatoric_uncertainty` | Same as expected entropy. |
| `global_epistemic_uncertainty` | Same as mutual information. |
| `global_predictive_variance_mean` | Mean probability variance across posterior samples. |
| `global_predictive_variance_std` | Std of probability variance across posterior samples. |

## Local evaluation submetrics

These summarize `client_eval_metrics.csv`. They are populated when `--local_eval_every > 0` and `--val_ratio > 0`.

| Column | Meaning |
|---|---|
| `local_eval_count` | Number of client models evaluated. |
| `local_accuracy_weighted` | Example-weighted local validation accuracy. |
| `local_accuracy_mean/std/min/p10/p25/p50/p75/p90/max` | Distribution of local validation accuracy. |
| `local_loss_weighted` | Example-weighted local validation loss. |
| `local_loss_mean/std/min/p50/max` | Distribution of local validation loss. |
| `local_ece_mean/std` | Local calibration summary. |
| `local_nll_mean` | Mean local NLL. |
| `local_brier_mean` | Mean local Brier score. |

## Local forgetting and drift proxy submetrics

| Column | Meaning |
|---|---|
| `client_update_l2_mean/std/min/max` | Norm of local update from global before training. |
| `client_update_cosine_mean` | Cosine between local parameter vector and incoming global vector. |
| `client_drift_from_global_l2_mean/std` | Parameter drift proxy. |
| `local_forgetting_proxy_mean/std/weighted` | Reserved for local/global performance-gap based forgetting. |
| `local_global_loss_gap_mean` | Reserved local-vs-global loss gap. |
| `local_global_acc_gap_mean` | Reserved local-vs-global accuracy gap. |

## Training decomposition submetrics

| Column | Meaning |
|---|---|
| `train_loss_mean/std` | Mean optimized local objective. |
| `task_loss_mean/std` | Mean data likelihood / cross-entropy component. |
| `prior_loss_mean/std` | Mean Bayesian prior/complexity regularizer. |
| `regularization_loss_mean/std` | Regularizer after multiplying by its weight. |

For FedAvg, `task_loss_mean` and `train_loss_mean` are the same. For OLA, `prior_loss_mean` is the prior-iteration quadratic penalty. For VI, `prior_loss_mean` records the local posterior KL/complexity proxy.

## VI-specific submetrics

| Column | Meaning |
|---|---|
| `vi_elbo_loss_mean/std` | Pyro SVI loss summary. |
| `vi_kl_loss_mean/std` | Diagonal Gaussian KL from local posterior to incoming global posterior/prior. |
| `vi_likelihood_loss_mean/std` | Reserved for explicit likelihood decomposition. |
| `vi_complexity_cost_mean` | Same role as Bayes-by-Backprop complexity cost. |
| `vi_scale_mean/std/p50/p90/max` | Posterior scale statistics. |
| `vi_prior_scale` | Initial/global prior scale argument. |
| `vi_min_scale` | Scale floor. |
| `vi_particles` | ELBO particles. |
| `vi_aggregation_mode` | `product` or `mean`. |

## OLA/FOLA-specific submetrics

| Column | Meaning |
|---|---|
| `ola_prior_lambda` | Prior-iteration regularization weight. |
| `ola_prior_loss_mean/std` | Local prior loss summary. |
| `ola_task_loss_mean/std` | Local task loss summary. |
| `ola_fisher_mean/std/min/p10/p50/p90/max` | Diagonal empirical Fisher statistics. |
| `ola_precision_mean/std/min/p10/p50/p90/max` | Posterior precision statistics. |
| `ola_sigma_mean/std/p50/p90/max` | Posterior standard deviation implied by precision. |
| `ola_gamma` | Initial precision constant. |
| `ola_online_weight_fisher`, `ola_online_weight_prior` | Reserved for explicit online weighting diagnostics. |

## Global posterior and SNR submetrics

For VI:

```text
mu = posterior loc
sigma = posterior scale
precision = 1 / sigma^2
```

For OLA:

```text
mu = global mean
precision = diagonal global precision
sigma = sqrt(1 / precision)
```

| Column | Meaning |
|---|---|
| `posterior_available` | `1` for VI/OLA posterior uncertainty, `0` for FedAvg. |
| `posterior_num_params` | Number of trainable parameters. |
| `posterior_mu_l2` | L2 norm of posterior mean. |
| `posterior_mu_abs_mean/std/p50/p90` | Absolute mean-weight statistics. |
| `posterior_sigma_mean/std/min/p10/p25/p50/p75/p90/p95/max` | Posterior standard-deviation statistics. |
| `posterior_precision_mean/std/min/p50/p90/max` | Diagonal precision statistics. |
| `posterior_var_trace` | Sum of diagonal variances. |
| `posterior_logdet_diag` | Sum of log diagonal variance. |
| `posterior_entropy_diag_gaussian` | Entropy of diagonal Gaussian posterior. |
| `posterior_snr_raw_*` | Signal-to-noise ratio statistics using `abs(mu) / sigma`. |
| `posterior_snr_db_*` | `20 * log10(SNR)` statistics. |
| `posterior_snr_frac_lt_1`, `posterior_snr_frac_gt_1`, etc. | Fraction of weights below/above SNR thresholds. |
| `effective_params_snr_gt_1`, `effective_params_snr_gt_2`, `effective_params_snr_gt_5` | Effective parameter count under SNR thresholding. |

## Aggregation submetrics

| Column | Meaning |
|---|---|
| `aggregation_delta_l2` | Norm of global update this round. |
| `aggregation_delta_linf` | Max absolute global parameter change. |
| `aggregation_weight_entropy` | Entropy of selected-client aggregation weights. |
| `aggregation_weight_min/max` | Min/max selected-client normalized aggregation weights. |
| `posterior_product_precision_mean/std` | Product-aggregation precision summary for VI/OLA. |
| `posterior_product_mu_norm` | Norm of product-aggregated mean. |
| `posterior_product_sigma_mean` | Mean sigma implied by product precision. |

## Dataset/selection and future wireless submetrics

| Column | Meaning |
|---|---|
| `selected_label_entropy_mean/std/min/max` | Label-distribution entropy among selected clients. |
| `selected_kl_to_global_label_mean/std` | Selected-client label KL to global distribution. |
| `selected_num_examples_mean/std/min/max` | Selected-client data-size summary. |
| `wireless_*`, `ota_*`, `digital_*` | Reserved placeholders for future analog OTA/digital communication experiments. |

---

# Supporting CSV files

## `run_summary.csv`

One row per run. Use this for hyperparameter sweeps and final-result bar plots.

Important columns:

```text
final_global_accuracy
final_global_loss
final_global_nll
final_global_ece
final_local_accuracy_weighted
best_global_accuracy
best_global_accuracy_round
best_global_ece
best_global_ece_round
final_posterior_sigma_mean
final_posterior_snr_raw_p50
final_posterior_snr_frac_gt_1
total_time_sec
mean_round_time_sec
final_model_path
```

## `client_data_summary.csv`

One row per physical device. Use this to analyze non-IID and imbalance.

Important columns:

```text
physical_client_id
virtual_client_id
num_examples
train_examples
val_examples
label_0_count ... label_9_count
label_entropy
dominant_label
dominant_label_fraction
kl_to_global_label_distribution
is_iid
is_balanced
noniid_alpha
unbalanced_alpha
```

## `device_summary.csv`

One row per physical device. Use this for radar/device plots and future wireless scheduling.

Important columns:

```text
physical_client_id / device_id
virtual_client_id
radius_m / distance_m
angle_rad
x_m, y_m
num_examples
label_entropy
dominant_label
kl_to_global_label_distribution
default_channel_gain
default_pathloss_db
default_noise_power
```

`device_id` is kept as a backward-compatible alias.

## `selection_summary.csv`

One row per round. Use this for selected-client-count and selected-client-heterogeneity plots.

Important columns:

```text
round
selection_policy
selected_count
available_count
selected_fraction
selected_examples
selected_examples_fraction
selected_label_entropy_mean
selected_label_entropy_std
selected_kl_to_global_label_mean
selected_distance_m_mean
selected_distance_m_std
```

## `selected_clients.csv`

One row per selected physical device per round.

Important columns:

```text
round
physical_client_id
virtual_client_id
selected_count
selection_policy
selection_probability
num_examples
label_entropy
dominant_label
kl_to_global_label_distribution
distance_m
angle_rad
channel_snr_db
pathloss_db
rate_mbps
delay_ms
energy_j
```

## `client_train_metrics.csv`

One row per selected physical device per round. Use this for local-training diagnostics.

Important columns:

```text
round
method
physical_client_id
virtual_client_id
num_examples
train_loss
task_loss
prior_loss
regularization_loss
accuracy_local_train_estimate
update_l2_norm
update_linf_norm
update_cosine_to_global
drift_from_global_before_l2
label_entropy
kl_to_global_label_distribution
vi_elbo_loss
vi_kl_loss
vi_scale_mean
vi_snr_raw_p50
ola_fisher_mean
ola_precision_mean
ola_sigma_mean
ola_snr_raw_p50
```

## `client_eval_metrics.csv`

One row per evaluated client per local-evaluation round. This file is populated only when:

```bash
--val_ratio > 0 --local_eval_every > 0
```

Important columns:

```text
round
physical_client_id
virtual_client_id
eval_scope
num_eval_examples
local_accuracy
local_loss
local_nll
local_brier
local_ece
local_mean_confidence
local_mean_entropy
local_mc_samples
num_examples_train
label_entropy
kl_to_global_label_distribution
```

## `calibration_bins.csv`

One row per calibration bin per evaluation round.

Important columns:

```text
round
eval_scope
bin_id
bin_left
bin_right
bin_count
bin_accuracy
bin_confidence
bin_gap
ece_contribution
nll_mean
brier_mean
```

Use this for reliability diagrams and ECE debugging.

## `posterior_summary.csv`

One row per round per layer/parameter group. Use this for layer-wise posterior uncertainty plots.

Important columns:

```text
round
scope
layer_name
num_params
mu_mean
mu_abs_mean
mu_l2
sigma_mean
sigma_p50
sigma_p90
precision_mean
precision_p90
snr_raw_mean
snr_raw_p50
snr_raw_p90
snr_db_mean
snr_db_p50
snr_frac_lt_1
snr_frac_gt_1
effective_params_snr_gt_1
```

## `snr_histograms.csv`

One row per SNR histogram bin. Use this to reproduce Bayes-by-Backprop-style SNR density and CDF plots.

Important columns:

```text
round
layer_name
value_space  # raw or db
bin_id
bin_left
bin_right
bin_center
count
density
cdf
total_count
```

## `aggregation_diagnostics.csv`

One row per round. Use this for aggregation-error and update-drift analysis.

Important columns:

```text
round
aggregation_mode
num_results_received
num_failures
total_selected_examples
aggregation_weight_entropy
aggregation_weight_min
aggregation_weight_max
global_before_l2
global_after_l2
aggregation_delta_l2
aggregation_delta_linf
aggregation_delta_cosine
client_update_l2_mean
client_update_l2_std
client_update_cosine_mean
```

## `communication_metrics.csv`

One row per selected device per round. Currently this stores placeholders for future wireless-aware selection.

Important future columns:

```text
channel_gain
channel_snr_db
pathloss_db
noise_power
tx_power
rate_mbps
delay_ms
energy_j
analog_ota_enabled
ota_noise_power
ota_distortion
ota_mse
digital_enabled
packet_error_rate
payload_bytes
communication_success
```

## `posterior_snapshots/*.pt`

Each snapshot stores full posterior arrays for post-hoc analysis.

```python
{
    "schema_version": "bayesfl_observability_v1",
    "run_id": str,
    "round": int,
    "method": str,
    "dataset": str,
    "model": str,
    "param_names": list[str],
    "param_shapes": list[tuple],
    "flat_slices": list[tuple],
    "global": {
        "mu_flat": Tensor,
        "sigma_flat": Tensor or None,
        "precision_flat": Tensor or None,
        "state_dict": dict,
    },
    "payload": list[np.ndarray],
    "summary": dict,
}
```

Use snapshots for post-hoc SNR thresholds, posterior-distance analysis, or pruning experiments that were not planned before training.

---

# Offline plotting

Training does not create PNG files. Generate plots separately.

## Single metric

```bash
python utils.py metric \
  --history outputs/ola_mnist_observable/metrics.csv \
  --metric global_accuracy \
  --output_dir plots/ola_mnist_observable
```

Backward-compatible names still work:

```bash
python utils.py metric --history outputs/ola_mnist_observable/metrics.csv --metric accuracy --output_dir plots/ola_mnist_observable
python utils.py metric --history outputs/ola_mnist_observable/metrics.csv --metric loss --output_dir plots/ola_mnist_observable
```

## Mixed method or hyperparameter comparison

```bash
python utils.py mix \
  --runs \
    fedavg=outputs/fedavg_mnist_iid \
    ola_lam1=outputs/ola_mnist_lam1 \
    ola_lam01=outputs/ola_mnist_lam01 \
    vi_prior005=outputs/vi_mnist_prior005 \
  --metrics global_accuracy global_loss global_ece posterior_sigma_mean posterior_snr_raw_p50 \
  --output_dir plots/compare_methods
```

The left side of `label=path` becomes the legend label.

## Selected clients per round

```bash
python utils.py selected \
  --selection outputs/ola_mnist_observable/selection_summary.csv \
  --output_dir plots/ola_mnist_observable
```

## Device radar plot

```bash
python utils.py radar \
  --device_summary outputs/ola_mnist_observable/device_summary.csv \
  --output_dir plots/ola_mnist_observable
```

## SNR density and CDF

```bash
python utils.py snr \
  --snr outputs/ola_mnist_observable/snr_histograms.csv \
  --round 20 \
  --layer all \
  --value_space db \
  --output_dir plots/ola_mnist_observable
```

This creates:

```text
snr_density_round_0020_all_db.png
snr_cdf_round_0020_all_db.png
```

## Reliability diagram

```bash
python utils.py calibration \
  --calibration outputs/ola_mnist_observable/calibration_bins.csv \
  --round 20 \
  --eval_scope global_test \
  --output_dir plots/ola_mnist_observable
```

---

# Method notes

## FedAvg

Each selected physical device starts from the global mean parameters, trains locally, and returns a deterministic model. Virtual clients average their assigned physical-device models before the server performs global weighted averaging.

## VI

Each selected physical device treats the incoming global `loc, scale` as the Gaussian prior over MLP weights. Pyro SVI learns a local diagonal Gaussian posterior. Server aggregation supports:

- `--bayes_aggregation product`: diagonal Gaussian product aggregation.
- `--bayes_aggregation mean`: moment-matched averaging.

VI-specific metrics include posterior scale, KL/complexity cost, ELBO loss, and SNR summaries.

## OLA/FOLA

Each selected physical device trains a deterministic model with prior-iteration regularization:

```text
0.5 * (theta - mu_global)^T Precision_global (theta - mu_global)
```

It also accumulates squared task-loss gradients as a diagonal empirical Fisher estimate. The local precision is updated online using the communication round index. The server uses diagonal Gaussian-product aggregation.

OLA-specific metrics include task/prior loss, empirical Fisher statistics, precision/sigma statistics, and SNR summaries.

---

# TODO: wireless-aware selection

The current active policy is:

```bash
--selector random
```

The placeholder `WirelessQualitySelector` in `selector.py` is the intended extension point for:

- analog over-the-air aggregation client selection,
- digital uplink-rate/deadline-aware selection,
- hybrid data-importance and channel-quality scheduling.

The source already saves `device_summary.csv`, `selected_clients.csv`, and `communication_metrics.csv` with stable columns so future wireless information can be added without redesigning the output schema.

---

## Shell script organization and nohup execution

The repository keeps reusable execution scripts under `scripts/` instead of the source-code root.

```text
scripts/
├── train/   # long-running training jobs that generate outputs/
├── tune/    # hyperparameter sweeps that generate outputs/tune_*
└── plot/    # offline plotting jobs that generate plots/
```

Each script name follows this pattern:

```text
<file_type>_<method>_<dataset>_<setting>_<hyperparameter_tag>[_seed].sh
```

Examples:

```text
scripts/train/train_fedavg_mnist_noniid_unbalanced_dense_seed42.sh
scripts/train/train_vi_mnist_noniid_unbalanced_sparse_update_snr_keep010_decay_seed42.sh
scripts/plot/plot_sparse_vi_mnist_noniid_unbalanced_update_snr_ratio_sweep_seed42.sh
```

All scripts automatically change directory to the repository root before running, so they can be launched from either the root folder or another working directory.

### Common nohup pattern

Long experiments should be launched with `nohup` so they continue after the terminal is closed.

```bash
mkdir -p logs
nohup bash <script_path> > logs/<log_name>.log 2>&1 &
```

Example:

```bash
nohup bash scripts/train/train_vi_mnist_noniid_unbalanced_sparse_update_snr_keep010_decay_seed42.sh \
  > logs/train_vi_sparse_keep010_decay_seed42.log 2>&1 &
```

Monitor the log:

```bash
tail -f logs/train_vi_sparse_keep010_decay_seed42.log
```

Check whether a job is still running:

```bash
pgrep -af "main.py|scripts/"
```

Check GPU usage:

```bash
watch -n 2 nvidia-smi
```

Stop a running job if necessary:

```bash
pkill -f "train_vi_mnist_noniid_unbalanced_sparse_update_snr_keep010_decay_seed42"
```

### Useful environment overrides

Most training scripts support these common overrides:

```bash
SEED=43 bash scripts/train/train_vi_mnist_noniid_unbalanced_dense_seed42.sh
FORCE_DEVICE=cpu bash scripts/train/train_vi_mnist_noniid_unbalanced_dense_seed42.sh
FORCE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 CLIENT_GPUS=0.5 bash scripts/train/train_vi_mnist_noniid_unbalanced_dense_seed42.sh
```

For sparse-ratio sweeps, the baseline dense run is usually reused for `drop000_keep100`, and the sparse runs write to:

```text
outputs/sparse_comm_mnist_noniid_unbalanced/
```

### Recommended reproduction order for the current validated MNIST stage

1. Train dense final comparison runs:

```bash
nohup bash scripts/train/train_fedavg_mnist_noniid_unbalanced_dense_seed42.sh > logs/train_fedavg_mnist_dense_seed42.log 2>&1 &
nohup bash scripts/train/train_ola_mnist_noniid_unbalanced_dense_seed42.sh > logs/train_ola_mnist_dense_seed42.log 2>&1 &
nohup bash scripts/train/train_vi_mnist_noniid_unbalanced_dense_seed42.sh > logs/train_vi_mnist_dense_seed42.log 2>&1 &
```

2. Train sparse communication sweeps:

```bash
nohup bash scripts/train/train_vi_mnist_noniid_unbalanced_sparse_update_snr_sweep_seed42.sh > logs/train_vi_sparse_update_snr_sweep_seed42.log 2>&1 &
nohup bash scripts/train/train_ola_mnist_noniid_unbalanced_sparse_precision_update_sweep_seed42.sh > logs/train_ola_sparse_precision_update_sweep_seed42.log 2>&1 &
```

3. Train stabilized VI variants:

```bash
nohup bash scripts/train/train_vi_mnist_noniid_unbalanced_decay_seed42.sh > logs/train_vi_decay_seed42.log 2>&1 &
nohup bash scripts/train/train_vi_mnist_noniid_unbalanced_sparse_update_snr_keep010_decay_seed42.sh > logs/train_vi_sparse_keep010_decay_seed42.log 2>&1 &
```

4. Generate plots:

```bash
nohup bash scripts/plot/plot_compare_fedavg_ola_mnist_noniid_unbalanced_seed42.sh > logs/plot_compare_fedavg_ola_seed42.log 2>&1 &
nohup bash scripts/plot/plot_compare_fedavg_vi_mnist_noniid_unbalanced_seed42.sh > logs/plot_compare_fedavg_vi_seed42.log 2>&1 &
nohup bash scripts/plot/plot_sparse_vi_mnist_noniid_unbalanced_update_snr_ratio_sweep_seed42.sh > logs/plot_sparse_vi_ratio_sweep_seed42.log 2>&1 &
nohup bash scripts/plot/plot_sparse_ola_mnist_noniid_unbalanced_precision_update_ratio_sweep_seed42.sh > logs/plot_sparse_ola_ratio_sweep_seed42.log 2>&1 &
nohup bash scripts/plot/plot_diagnostics_mnist_noniid_unbalanced_research_v1.sh > logs/plot_diagnostics_mnist_research_v1.log 2>&1 &
```


## Sparse communication ablation: Bayesian vs Random selection

The sparse communication module supports an additional ablation mode:

```bash
--sparse_selection bayesian
```

uses posterior-aware importance scores:

- VI: update-SNR
- OLA: precision-update

```bash
--sparse_selection random
```

uses random scores but keeps the same top-k keep ratio and aggregation pipeline.

This allows a fair comparison between Bayesian importance selection and random sparsification under identical communication budgets.
