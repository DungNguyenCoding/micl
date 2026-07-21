# AirComp Bayesian FL Reproduction Module

This folder documents the new source-code extension for the paper:

> **Distribution-Level AirComp for Wireless Federated Learning under Data Scarcity and Heterogeneity**.

The new implementation is intentionally separate from the Flower simulator in `main.py`, because AirComp assumes simultaneous analog transmission and waveform superposition over a shared wireless channel. Flower's normal client/server API models digital request-response communication, while the paper models analog aggregation through the physical wireless channel.

## 1. New source files

| File | Purpose |
|---|---|
| `aircomp_bayesfl.py` | Standalone simulator implementing AirComp FedAvg, AirComp FedProx, AirComp SCAFFOLD, and the proposed distribution-level AirComp Bayesian FL. |
| `aircomp_plots.py` | Plot generator for Section VI-style figures: accuracy curves, power/label/dataset-size comparisons, reliability diagrams, and summary plots. |
| `scripts/train/train_aircomp_bayesfl_mnist_paper_smoke_seed42.sh` | Short end-to-end smoke test. |
| `scripts/train/train_aircomp_bayesfl_mnist_paper_default_seed42.sh` | Default scenario for paper Fig. 2 and Fig. 6. |
| `scripts/train/train_aircomp_bayesfl_mnist_paper_full_seed42.sh` | Full Section VI reproduction attempt: Fig. 2 to Fig. 6. |
| `scripts/plot/plot_aircomp_bayesfl_mnist_paper_default_seed42.sh` | Plot default scenario outputs. |
| `scripts/plot/plot_aircomp_bayesfl_mnist_paper_full_seed42.sh` | Plot full reproduction outputs. |

## 2. Mathematical mapping

### 2.1 Local VI objective

The paper defines the local Bayesian objective as:

\[
L_k(\theta)=L_{task,k}(\theta)+\lambda L_{reg,t}(\theta),
\]

where:

\[
L_{reg,t}(\theta)=KL[q_\theta(w)\|q_{\theta_t}(w)]
\]

and the task loss is approximated with Monte Carlo samples:

\[
L_{task,k}(\theta)=\frac{1}{M}\sum_{m=1}^{M}-\log p(D_k|w^{(m)}).
\]

Code mapping:

```text
aircomp_bayesfl.py::gaussian_kl_diag()
aircomp_bayesfl.py::vi_task_loss_mc()
aircomp_bayesfl.py::train_proposed_phase_rho()
aircomp_bayesfl.py::train_proposed_phase_mu()
```

### 2.2 Product-of-Gaussians posterior aggregation

For Gaussian variational posteriors:

\[
\Sigma^{-1}_{t+1}=\sum_k \pi_k\Sigma^{-1}_{t,k},
\]

\[
\mu_{t+1}=\Sigma_{t+1}\sum_k\pi_k\mu_{t,k}\Sigma^{-1}_{t,k}.
\]

The paper rewrites this into two AirComp phases:

```text
Phase 1: aggregate precision/covariance statistics rho = diag(Sigma^-1)
Phase 2: aggregate mean-related statistics nu
```

Code mapping:

```text
aircomp_bayesfl.py::run_method(method="proposed")
  - phase 1 calls train_proposed_phase_rho()
  - aggregates rho updates with aircomp_aggregate()
  - phase 2 calls train_proposed_phase_mu()
  - aggregates mean updates with aircomp_aggregate()
```

### 2.3 AirComp channel model

The paper models the received signal as:

\[
y_t^{(n)}=\sum_{k\in K} h_{t,k}\odot x_{t,k}^{(n)} + z_t^{(n)}.
\]

The code simulates Rayleigh fading with path loss:

\[
|h|^2\sim\text{Exponential}(r_k^{-\alpha}).
\]

Code mapping:

```text
aircomp_bayesfl.py::aircomp_aggregate()
```

### 2.4 Power control

The paper solves the constrained problem:

\[
\min_v \| |\Delta|-v\|^2
\quad
\text{s.t. } u^T(v\odot v)\le P,\quad v_f\ge 0.
\]

The KKT solution is:

\[
v=|\Delta|\oslash(1+\lambda u),
\]

when the unconstrained update violates the power budget.

Code mapping:

```text
aircomp_bayesfl.py::solve_power_control()
```

## 3. Experiment mapping to Section VI

| Paper figure | Script / option | Output plot |
|---|---|---|
| Fig. 2 default method comparison | `--experiment default` | `fig2_accuracy_default.png` |
| Fig. 3 label skewness | `--experiment label_skew` or `--experiment full` | `fig3_accuracy_label_skew.png` |
| Fig. 4 dataset size | `--experiment dataset_size` or `--experiment full` | `fig4_accuracy_dataset_size.png` |
| Fig. 5 power budget | `--experiment power` or `--experiment full` | `fig5_accuracy_power_budget.png` |
| Fig. 6 reliability diagrams | default scenario calibration output | `fig6_reliability_default.png` |

## 4. Running smoke test

```bash
cd ~/DungNDH/micl/bayesfl_base_source
mkdir -p logs
nohup bash scripts/train/train_aircomp_bayesfl_mnist_paper_smoke_seed42.sh \
  > logs/train_aircomp_smoke.log 2>&1 &
tail -f logs/train_aircomp_smoke.log
```

Plot smoke/default outputs:

```bash
RUN=outputs/aircomp_bayesfl_mnist_paper_smoke_seed42 \
OUT=plots/aircomp_bayesfl_mnist_paper_smoke_seed42 \
bash scripts/plot/plot_aircomp_bayesfl_mnist_paper_default_seed42.sh
```

## 5. Running default paper scenario

```bash
cd ~/DungNDH/micl/bayesfl_base_source
mkdir -p logs
nohup bash scripts/train/train_aircomp_bayesfl_mnist_paper_default_seed42.sh \
  > logs/train_aircomp_default_seed42.log 2>&1 &
```

Plot:

```bash
nohup bash scripts/plot/plot_aircomp_bayesfl_mnist_paper_default_seed42.sh \
  > logs/plot_aircomp_default_seed42.log 2>&1 &
```

## 6. Running full Section VI reproduction

This is expensive because it repeats multiple methods, scenarios, and stochastic data realizations.

```bash
cd ~/DungNDH/micl/bayesfl_base_source
mkdir -p logs
nohup bash scripts/train/train_aircomp_bayesfl_mnist_paper_full_seed42.sh \
  > logs/train_aircomp_full_seed42.log 2>&1 &
```

Plot:

```bash
nohup bash scripts/plot/plot_aircomp_bayesfl_mnist_paper_full_seed42.sh \
  > logs/plot_aircomp_full_seed42.log 2>&1 &
```

## 7. Faster development commands

Run fewer realizations and fewer channel uses:

```bash
REALIZATIONS=1 MAX_CHANNEL_USES=1000000 \
nohup bash scripts/train/train_aircomp_bayesfl_mnist_paper_full_seed42.sh \
  > logs/train_aircomp_full_fast.log 2>&1 &
```

Run only the proposed method:

```bash
METHODS=proposed REALIZATIONS=1 MAX_CHANNEL_USES=1000000 \
nohup bash scripts/train/train_aircomp_bayesfl_mnist_paper_default_seed42.sh \
  > logs/train_aircomp_proposed_fast.log 2>&1 &
```

## 8. Output files

A run writes:

```text
outputs/aircomp_bayesfl_mnist_paper_*/
├── config.csv
├── metrics.csv
├── calibration_bins.csv
├── client_data_summary.csv
└── run_summary.csv
```

`metrics.csv` contains method/round/channel-use accuracy, loss, ECE, AirComp distortion, posterior precision/sigma/SNR for the proposed Bayesian method, and runtime.

`calibration_bins.csv` contains bin-level reliability-diagram information.

`run_summary.csv` contains final/best accuracy and ECE per method, scenario, condition, and realization.

## 9. Important implementation note

The module is a faithful engineering implementation of the paper's algorithmic structure, but exact numerical reproduction may differ because some low-level choices are not fully specified in the paper, including random seeds, exact initialization, optimizer details for the variational parameters, and baseline hyperparameters. For paper-grade comparison, run multiple realizations and report mean/std.
