# AirCompBayesFL

**Version 1.5.2 — Paired-condition seed + MNIST memory fixes**

A modular Pyro + Flower/Ray reproduction framework for:

> *Distribution-Level AirComp for Wireless Federated Learning under Data
> Scarcity and Heterogeneity* — Jun-Pyo Hong, Hyowoon Seo, Kisong Lee.

This is an independent reproduction, not the authors' original source. The
paper reports Blitz for Bayesian layers; this project intentionally uses Pyro.
Undisclosed details are exposed as configuration values instead of being hidden.

## v1.5.2 experiment-control fixes

- Keeps the v1.5.1 process-local MNIST cache, preventing repeated full-dataset reloads during long native-Windows runs.
- Uses `seed = base_seed + realization` for every condition in a figure sweep. Matching conditions therefore share the same realization seed instead of receiving the old condition-specific `+10,000` offset.
- In Figure 5, `P=3/23/33 dBm` now use the exact same client partition, model initialization, channel RNG seed, and noise RNG seed within each replication; only transmit power changes.
- No Bayesian VI, deterministic SGD, AirComp, KKT, wireless, or metric equation was changed from the v1.5.0 algorithmic implementation.

## v1.5.0 priorities

Development and experiments are intentionally ordered as:

1. make the **Proposed** method trustworthy and reproduce its learning curve;
2. run **FedAvg** under the same wireless/power-control implementation;
3. add FedProx and SCAFFOLD comparisons afterward.

## Paper-reference optimal power control

v1.5.0 corrects the main benchmark mismatch found after inspecting the actual
IEEE TWC 2023 reference [13]. The **KKT/QCQP magnitude optimizer is shared**,
but the transmit-power scale is not identical between the Bayesian method and
the conventional baselines.

```yaml
wireless:
  power_control_mode: paper_reference_kkt
  deterministic_payload_mode: update
  deterministic_reference_power_mode: coordinated_aggregate
```

For the Proposed method, `Delta-rho` and `Delta-nu` retain the target 2025
paper's Eqs. (27),(28),(31):

```text
delta_bar = sum_k pi_k ||Delta_k||^2 / d
u_k = pi_k^2 * gamma / delta_bar * |h_k|^-2
```

For FedAvg/FedProx/SCAFFOLD, local update vectors use Hong et al. (2023)
Eqs. (8),(10),(20):

```text
u_k = w_k^2 * gamma * sigma_z^2 / rho_ref * |h_k|^-2
receiver_scale = sqrt(rho_ref / (gamma * sigma_z^2))
```

Both paths call the same `_optimal_magnitude()` implementation of
`v = |Delta|` when feasible and `v = |Delta|/(1 + lambda*u)` otherwise.
Therefore v1.5.0 is source-faithful without maintaining two unrelated power
optimizers.

Reference [13] obtains `rho_ref` from a BS-local update, but the target 2025
paper has no BS dataset in the conventional benchmark setup and does not state
how it adapts this scalar. The default `coordinated_aggregate` mode follows
Remark 6 of [13], which describes conventional AirComp as coordinating power
with statistics of the aggregated update:

```text
rho_ref = ||sum_k pi_k Delta_k||^2 / d
```

`weighted_local` remains available as a sensitivity mode. This adaptation is
logged in every `metrics.csv` row and must be reported as a reproduction
assumption.

### Deterministic payload semantics

FedAvg and FedProx transmit `Delta-w = w_local - w_global` and the server
applies the recovered update additively. This matches the local-update
transmission in reference [13] and avoids replacing the global state with an
attenuated absolute model. Communication accounting remains `d` values per
round.

## Proposed Algorithm 1 implementation

Each Proposed logical round has a real server boundary between phases:

```text
rho phase
  client: rho_t,k <- rho_t and optimize rho_t,k
  server: AirComp aggregate Delta-rho -> rho_t+1
  server: broadcast rho_t+1

nu phase
  client: reload its own rho_t,k
  client: initialize/optimize nu_t,k using rho_t+1
  server: AirComp aggregate Delta-nu -> mu_t+1

server evaluates q(mu_t+1, rho_t+1)
```

Precision is preserved in float64 end-to-end because the direct Eq. (25) update
can be below one float32 ULP near rho=400. CNN forward computation remains
float32.

## New Proposed diagnostics

`metrics.csv` now records both forms of Bayesian evaluation:

```text
accuracy                         # paper-style posterior predictive accuracy
posterior_predictive_accuracy
posterior_predictive_nll
posterior_predictive_ece
posterior_mean_accuracy          # diagnostic at w = mu
posterior_mean_nll
posterior_mean_ece
```

It also records actual global coordinate movement:

```text
global_mean_update_l2
global_mean_update_max_abs
global_precision_update_l2
global_precision_update_max_abs
```

These fields answer the key debugging question:

- mean accuracy high, predictive accuracy low -> covariance/sampling issue;
- both low -> the `nu`/mean-learning path is the main issue.

The phase-1 precision state is reloaded as float64 in phase 2.

## Recommended Windows environment

The combination that worked on the RTX 3060 Laptop GPU during this project is:

```text
Python       3.12
PyTorch      2.5.1+cu121
Torchvision  0.20.1+cu121
Pyro         1.9.1
Flower       1.32.1
Ray          2.55.1
```

Native Windows CUDA uses the sequential `local` backend. WSL2/Linux can use
Flower/Ray virtual clients in parallel.

## Validation

```powershell
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m compileall .
.\.venv\Scripts\python.exe -m pytest -q
```

The package contains 33 tests in v1.5.0.

## Priority stage 1: Proposed only

Start with the validated learning settings, one realization, 60 logical rounds:

```powershell
.\.venv\Scripts\python.exe main.py `
  --config configs/proposed_gpu.yaml `
  --experiment fig2 `
  --methods proposed `
  --rounds 60 `
  --replications 1 `
  --path-loss-reference-m 1000 `
  --output results\proposed_v150_60
```

Inspect the important fields:

```powershell
Import-Csv .\results\proposed_v150_60\metrics.csv |
  Select-Object `
    round,channel_uses_cumulative,
    accuracy,posterior_mean_accuracy,
    nll,ece,
    global_mean_update_l2,global_precision_update_l2,
    posterior_precision_mean,posterior_variance,
    precision_aircomp_nmse,mean_aircomp_nmse |
  Format-Table -AutoSize
```

Plot posterior predictive vs posterior mean:

```powershell
.\.venv\Scripts\python.exe utils.py `
  --input results\proposed_v150_60 `
  --figure proposed_debug
```

The paper-style Figure 2 curve still uses `accuracy`, i.e. posterior predictive
accuracy.

### Full Proposed communication budget

With `d = 62,346` and `2d` transmitted values per logical Proposed round,
30,000,000 channel uses correspond to approximately 240 logical rounds. After
the 60-round trajectory is validated:

```powershell
.\.venv\Scripts\python.exe main.py `
  --config configs/proposed_gpu.yaml `
  --experiment fig2 `
  --methods proposed `
  --replications 1 `
  --output results\proposed_v150_full1
```

`num_rounds: null` derives the round count from the channel-use budget.
The paper-style final result should later be averaged over 10 independent
realizations.

## Optional strict-source optimizer diagnostic

The published training table does not list gradient clipping. The working
configuration retains the previously validated clip value (`10.0`) so v1.5.0
does not unexpectedly change the trajectory. To test a no-clipping interpretation:

```powershell
.\.venv\Scripts\python.exe main.py `
  --config configs/proposed_strict_gpu.yaml `
  --experiment fig2 `
  --methods proposed `
  --rounds 20 `
  --replications 1 `
  --no-wireless `
  --output results\proposed_v150_strict20
```

Treat this as a sensitivity experiment, not as a silent replacement for the
working configuration.

## Priority stage 2: FedAvg

Only after Proposed is validated, run FedAvg with exactly the same wireless
wireless channel settings, while using the Hong-2023 reference-[13] scaling:

```powershell
.\.venv\Scripts\python.exe main.py `
  --config configs/fedavg_compare_gpu.yaml `
  --experiment fig2 `
  --methods fedavg `
  --replications 1 `
  --path-loss-reference-m 1000 `
  --output results\fedavg_v150_compare
```

Then combine/plot matched communication budgets. FedProx and SCAFFOLD can be
added after Proposed-vs-FedAvg is credible.

## Reproduction assumptions that remain explicit

- `initial_prior_std: 0.05` is an implementation assumption; the paper does not
  publish its initial covariance value.
- `path_loss_reference_m: 1000` is a numerical path-loss normalization
  assumption; the paper states `r^-alpha` and a 200 m cell but does not publish
  the reference-distance constant used in code.
- `gradient_clip_norm: 10.0` is retained in the working configuration for
  continuity; `proposed_strict_gpu.yaml` disables it for sensitivity testing.


## v1.5.0 deterministic AirComp contract

FedAvg/FedProx/SCAFFOLD use the Hong-2023 Eq. (8)/(10)/(20) power scale and
receiver de-scaling on local update vectors. Proposed retains its target-2025
normalization. The shared KKT magnitude core is tested directly. See
`V150_REFERENCE13_POWER_CONTROL.md`.

The deterministic diagnostics `global_model_update_l2`,
`ideal_model_update_l2`, and `received_model_update_l2` remain in
`metrics.csv`, and `deterministic_reference_power_mode` records the `rho_ref`
adaptation used by each run.
