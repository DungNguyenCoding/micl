# AirCompBayesFL

A modular Flower/Ray + Pyro implementation of the simulation system described in:

> **Distribution-Level AirComp for Wireless Federated Learning under Data Scarcity and Heterogeneity**  
> Jun-Pyo Hong, Hyowoon Seo, and Kisong Lee, arXiv:2506.06090 (2025)

The package implements the disclosed MNIST/CNN environment, scarce non-IID partitions, Pyro mean-field variational inference, Gaussian posterior conflation through sufficient statistics, Rayleigh fading, additive noise, symbol-power constraints, KKT-based AirComp power control, FedAvg, FedProx, SCAFFOLD, CSV logging, and separate plotting.

## Important reproducibility note

This is an independent implementation, not the authors' original code. The paper states that its experiments used Blitz, while this package intentionally uses Pyro. The paper also does not disclose every implementation choice needed for bit-for-bit reproduction, including all random seeds, the FedProx coefficient, exact initialization, and the exact stopping-round schedule. Therefore this package is designed to reproduce the **method, environment, ablations, metrics, and plot structure**, but it cannot guarantee numerically identical curves.

The proposed method uses Pyro SVI with a diagonal Gaussian guide and optimizes:

`task loss + lambda * KL(q_local || q_global)`

The server transmits/aggregates two Gaussian sufficient statistics—precision and precision-weighted mean—corresponding to Eqs. (18)-(19). `two_phase` mode first optimizes local uncertainty with the mean fixed and then optimizes the mean with uncertainty fixed. This is the documented interpretation used to translate the paper's two-phase algorithm into Pyro.

## Project layout

```text
AirCompBayesFL/
├── main.py              Experiment runner
├── config.py            Dataclass/YAML configuration
├── dataset.py           MNIST download and scarce non-IID partitions
├── models.py            62,346-parameter paper CNN
├── serialization.py     Flat-vector parameter layout
├── bayes_vi.py          Pyro SVI local Bayesian training
├── deterministic.py     FedAvg/FedProx/SCAFFOLD local training
├── client.py            Flower NumPyClient and CUDA cleanup
├── server.py            Flower ServerApp and AirComp strategy
├── runtime_utils.py     CUDA/Ray preflight and safe device handling
├── gpu_check.py         Standalone GPU configuration checker
├── wireless.py          Rayleigh fading, path loss, and noise
├── aircomp.py           AirComp and KKT power control
├── aggregation.py       Model/posterior aggregation
├── metrics.py           Accuracy, NLL, ECE, reliability bins
├── logger.py            metrics.csv and reliability.csv
├── experiments.py       Figure 2-6 experiment matrices
├── utils.py             Standalone plotting
├── configs/
│   ├── smoke.yaml       Small CPU functional test
│   ├── paper.yaml       Tables I-II and 10 realizations on CPU
│   ├── smoke_gpu.yaml   RTX 3060/Windows GPU smoke test
│   └── paper_gpu.yaml   RTX 3060/Windows paper-scale starting point
└── tests/
```

## Supported platforms

- Linux x86-64
- Windows 11 x86-64
- Python 3.11 or Python 3.12 recommended

Ray's local Windows support is usable but may be slower and more memory intensive than Linux. Run commands from a normal terminal rather than from inside an interactive notebook on Windows.

## Installation

### Linux

```bash
cd AirCompBayesFL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows 11 PowerShell

```powershell
cd AirCompBayesFL
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA GPU clients, install the matching CUDA build of PyTorch first, then install the remaining requirements. Set `runtime.client_num_gpus` and `runtime.client_device` in YAML.

## RTX 3060 Laptop GPU on Windows 11

The native-Windows profiles deliberately use one GPU client at a time and keep the Flower ServerApp on CPU:

```yaml
runtime:
  client_num_gpus: 1.0
  client_device: cuda
  server_device: cpu
  cleanup_cuda_after_fit: true

data:
  num_workers: 0
  pin_memory: false
```

Run the preflight check:

```powershell
.\.venv\Scripts\python.exe gpu_check.py --config configs/smoke_gpu.yaml
```

Stop stale Ray workers after any previous crash, then run the GPU smoke test:

```powershell
.\.venv\Scripts\ray.exe stop --force
.\.venv\Scripts\python.exe main.py `
  --config configs/smoke_gpu.yaml `
  --experiment fig2 `
  --methods fedavg,proposed
```

The code now enforces device-aware DataLoader behavior: the CPU server never requests CUDA-pinned memory, while a CUDA client only uses pinned memory when explicitly enabled. Ray shutdown runs from a `finally` block, so failed simulations do not normally leave GPU-owning worker processes alive.

## Quick functional test

The smoke profile uses 4 clients, 2 rounds, one local epoch, and fewer Monte Carlo samples:

```bash
python main.py --config configs/smoke.yaml --experiment fig2 --methods fedavg,proposed
```

Check syntax and unit tests:

```bash
python -m compileall .
pytest -q
```

## Paper-scale experiments

`configs/paper.yaml` contains the disclosed values from Tables I-II:

- 40 clients in a 200 m cell
- Poisson local size with mean 10
- single-label local data by default
- `P = 23 dBm`
- noise `-74 dBm`
- 1,024 subchannels
- path-loss exponent 4
- local learning rate 0.1
- batch size 10
- 3 local epochs
- `gamma = 10 dB`
- `lambda = 1/50,000`
- 5 Monte Carlo samples
- 10 independent realizations

The paper-scale matrix is computationally expensive. Start with one replication and a limited number of rounds:

```bash
python main.py --config configs/paper.yaml --experiment fig2 --replications 1 --rounds 10
```

Then launch the complete disclosed experiment matrix:

```bash
python main.py --config configs/paper.yaml --experiment all
```

When `training.num_rounds` is `null`, the runner derives rounds from a 30-million channel-use budget. FedAvg/FedProx transmit `d` values per round; SCAFFOLD/Proposed transmit `2d`, where `d=62,346`.

### Individual figures

```bash
python main.py --config configs/paper.yaml --experiment fig2
python main.py --config configs/paper.yaml --experiment fig3
python main.py --config configs/paper.yaml --experiment fig4
python main.py --config configs/paper.yaml --experiment fig5
python main.py --config configs/paper.yaml --experiment fig6
```

Figure mapping:

- `fig2`: FedAvg, FedProx, SCAFFOLD, Proposed
- `fig3`: 1-, 2-, and 10-class local label support
- `fig4`: Poisson means 10, 20, and 50
- `fig5`: power budgets 3, 23, and 33 dBm
- `fig6`: reliability diagrams and ECE

Useful options:

```bash
python main.py --help
python main.py --config configs/paper.yaml --experiment fig2 --dry-run
python main.py --config configs/paper.yaml --experiment fig2 --resume
python main.py --config configs/paper.yaml --experiment fig2 --no-wireless
```

## Output files

The server appends all runs to:

```text
results/metrics.csv
results/reliability.csv
results/client_metrics.csv
results/checkpoints/*.npz
results/partitions/*.json
```

`metrics.csv` includes global accuracy, NLL, ECE, posterior variance, channel uses, OFDM-vector counts, AirComp NMSE, clipping fraction, power use, noise norm, and wall-clock time.

## Generate plots separately

```bash
python utils.py --input results --figure all
```

Or one figure:

```bash
python utils.py --input results --figure fig2
python utils.py --input results --figure fig6
```

Plots are written to `results/plots/`.

## Parallelism

Flower's Simulation Runtime uses Ray virtual clients. The degree of parallelism is controlled by:

```yaml
runtime:
  client_num_cpus: 1
  client_num_gpus: 0
```

For a machine with 16 CPU cores, assigning one CPU per client normally allows multiple clients to train concurrently. PyTorch threads are limited per actor to reduce CPU oversubscription.

## Wireless implementation

For each round, the server samples a block-fading channel:

`h_k ~ CN(0, r_k^(-alpha) I)`

For each transmitted chunk, the code computes the weighted average update power, channel-inversion coefficient, and the KKT power-control vector. If the unconstrained vector violates the symbol-power limit, the scalar multiplier in Eq. (43) is found by bisection. The receiver adds complex Gaussian noise and applies the paper's de-biasing scale.

The physical layer is simulated centrally in the Flower strategy after clients train in parallel. This is intentional: virtual clients perform local computation independently, while the strategy emulates their synchronized analog superposition over one shared channel.

## Known practical considerations

1. The full paper configuration can take many hours or days on CPU.
2. Pyro SVI is more expensive than deterministic local SGD.
3. Windows Ray processes use more memory than Linux processes.
4. Exact curves vary strongly with scarce-data partitions and assigned labels; use all 10 realizations before comparing averaged curves.
5. SCAFFOLD is included, but—as discussed in the paper—its control variate can become unstable under analog aggregation noise.
