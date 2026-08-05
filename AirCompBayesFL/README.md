# AirCompBayesFL

An independent, modular implementation of the simulation system described in:

> **Distribution-Level AirComp for Wireless Federated Learning under Data Scarcity and Heterogeneity**  
> Jun-Pyo Hong, Hyowoon Seo, and Kisong Lee, arXiv:2506.06090 (2025)

The project includes MNIST scarce/non-IID partitions, the 62,346-parameter CNN, Pyro mean-field variational inference, posterior sufficient-statistic aggregation, Rayleigh fading, additive noise, symbol-power constraints, KKT-based AirComp power control, FedAvg, FedProx, SCAFFOLD, CSV logging, and separate plotting.

## Reproducibility scope

This is not the authors' original source code. The paper reports Blitz, while this project intentionally uses Pyro. Some implementation details needed for bit-identical curves are not disclosed. The goal is therefore to reproduce the method, simulation environment, ablations, metrics, and plot structure rather than guarantee identical random curves.

## Execution backends

The same client, server, AirComp, aggregation, and metrics code can run through two execution backends:

- `ray`: Flower Simulation Runtime with Ray virtual clients. Use this on Linux or WSL2 for parallel client execution.
- `local`: virtual clients execute sequentially in the launcher process. This is the stable CUDA path for native Windows 11.
- `auto`: selects `local` for native Windows + CUDA and selects `ray` elsewhere.

Native Windows Ray support is experimental. Version 1.1.0 no longer sends CUDA work into Ray workers by default on native Windows because that combination can pass the launcher CUDA check but fail during a worker's first `model.to("cuda")` call. The local backend still performs all local training on the GPU; only client concurrency changes.

## Project layout

```text
AirCompBayesFL/
├── main.py              Experiment runner and backend dispatch
├── config.py            Dataclass/YAML configuration
├── dataset.py           MNIST download and scarce non-IID partitions
├── models.py            Paper CNN
├── serialization.py     Flat-vector parameter layout
├── bayes_vi.py          Pyro SVI local Bayesian training
├── deterministic.py     FedAvg/FedProx/SCAFFOLD local training
├── client.py            Backend-neutral virtual client
├── server.py            Flower/Ray backend + local CUDA backend
├── runtime_utils.py     CUDA/backend preflight and device handling
├── gpu_check.py         Standalone preflight checker
├── wireless.py          Rayleigh fading and path loss
├── aircomp.py           AirComp and KKT power control
├── aggregation.py       Deterministic/posterior aggregation
├── metrics.py           Accuracy, NLL, ECE, reliability bins
├── logger.py            CSV and checkpoint output
├── experiments.py       Figure 2–6 experiment matrices
├── utils.py             Standalone plotting
├── configs/
│   ├── smoke.yaml
│   ├── paper.yaml
│   ├── smoke_gpu.yaml
│   └── paper_gpu.yaml
└── tests/
```

## Installation

Python 3.11 or 3.12 is recommended.

### Windows 11 PowerShell

```powershell
cd C:\Users\Admin\Desktop\micl\AirCompBayesFL
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build before the remaining requirements when GPU support is needed. Verify it with:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Linux/WSL2

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## RTX 3060 Laptop GPU on native Windows

`configs/smoke_gpu.yaml` contains:

```yaml
runtime:
  backend: auto
  client_num_cpus: 1
  client_num_gpus: 1.0
  client_device: cuda
  server_device: cpu
  cleanup_cuda_after_fit: true
  fail_on_client_failure: true
```

On native Windows, `auto` resolves to `local`; on WSL2/Linux it resolves to `ray`.

Run the preflight:

```powershell
.\.venv\Scripts\python.exe gpu_check.py --config configs/smoke_gpu.yaml
```

Expected native-Windows output includes:

```text
Resolved backend: local
CUDA available: True
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
Client concurrency: sequential (stable native-Windows CUDA mode)
```

Remove results produced by the previous failed Ray/CUDA attempt because those rounds had zero successful client updates:

```powershell
Remove-Item -Recurse -Force .\results\gpu_smoke_windows -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\results\gpu_smoke_windows_v110 -ErrorAction SilentlyContinue
```

Run the corrected GPU smoke test:

```powershell
.\.venv\Scripts\python.exe main.py `
  --config configs/smoke_gpu.yaml `
  --experiment fig2 `
  --methods fedavg,proposed
```

The output should say `backend=local` and `Local backend: clients execute sequentially`. It should not start a Flower RayBackend for this native-Windows CUDA run.

## Parallel GPU clients

For parallel virtual-client GPU training, run the project in WSL2 or Linux and either keep `backend: auto` or set:

```yaml
runtime:
  backend: ray
  client_num_gpus: 1.0
  client_device: cuda
  server_device: cpu
```

With one RTX 3060 Laptop GPU, start with `client_num_gpus: 1.0`. Fractional GPU scheduling can be tested later, but Ray fractions are scheduling resources and do not enforce a VRAM partition.

## CPU smoke test

```powershell
.\.venv\Scripts\python.exe main.py `
  --config configs/smoke.yaml `
  --experiment fig2 `
  --methods fedavg,proposed
```

## Paper-scale starting point

The disclosed values from Tables I–II are represented in `configs/paper.yaml` and `configs/paper_gpu.yaml`, including 40 clients, Poisson mean 10, one label per client, 23 dBm, −74 dBm, 1,024 subchannels, path-loss exponent 4, learning rate 0.1, batch size 10, three local epochs, `lambda=1/50,000`, five MC samples, and ten realizations.

Start with one realization and ten rounds:

```powershell
.\.venv\Scripts\python.exe main.py `
  --config configs/paper_gpu.yaml `
  --experiment fig2 `
  --methods proposed `
  --replications 1 `
  --rounds 10
```

## Output

Each configured output directory contains:

```text
metrics.csv
reliability.csv
client_metrics.csv
checkpoints/*.npz
partitions/*.json
```

Version 1.1.0 fails immediately when a client job fails. It no longer treats a round with zero client results as a completed training round.

## Plot generation

```powershell
.\.venv\Scripts\python.exe utils.py `
  --input results\gpu_smoke_windows_v110 `
  --figure fig2
```

## Validation

```powershell
.\.venv\Scripts\python.exe -m compileall .
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m pytest -q
```

## Important performance note

The paper CNN and the very small per-client datasets can make GPU execution slower than expected because each client performs little work and CUDA setup/transfer overhead is significant. The native-Windows local backend is intended for correctness and stability. Use WSL2/Linux Ray mode for actual parallel virtual-client experiments.


## Wireless distance normalization

The channel law uses `(distance_m / path_loss_reference_m)^(-alpha)`. The
configs default to `path_loss_reference_m: 1000.0` because the paper does not
publish the numerical distance reference. See `WIRELESS_NORMALIZATION.md`.

Run a channel-normalization sensitivity test without editing YAML:

```powershell
python main.py --config configs/smoke_gpu.yaml --experiment fig2 --methods fedavg --rounds 10 --path-loss-reference-m 1500 --output results/ref1500
```
