# AirCompBayesFL 1.3.1

A modular Pyro + Flower/Ray simulator for the paper:

> **Distribution-Level AirComp for Wireless Federated Learning under Data
> Scarcity and Heterogeneity** — Jun-Pyo Hong, Hyowoon Seo, and Kisong Lee.

This is an independent reproduction, not the authors' original code. The paper
used Blitz; this project intentionally uses Pyro. Undisclosed implementation
details, including numerical path-loss normalization, are explicit configurable
assumptions.

## What changed in 1.3.1

Version 1.3.1 corrects the local optimization coordinates and the phase-2 KL prior:

- precision is optimized directly in `rho`, not in `log(rho)`;
- the phase-2 guide uses `rho_{t+1}`, while its KL prior remains the
  round-start posterior with `rho_t`;
- mini-batch likelihoods are scaled to estimate the full local-data objective.


The proposed method now follows Algorithm 1 with a real server boundary between
its two phases:

```text
logical round t
  precision clients: optimize rho_{t,k}
  server: AirComp aggregate Delta-rho -> rho_{t+1}
  server: broadcast rho_{t+1}
  mean clients: optimize nu_{t,k} using rho_{t+1}
  server: AirComp aggregate Delta-nu -> mu_{t+1}
  evaluate q(mu_{t+1}, rho_{t+1})
```

See `ALGORITHM1_TWO_PHASE.md` for the equations and Flower round mapping.

## Project layout

```text
AirCompBayesFL/
├── main.py                  experiment CLI
├── config.py                dataclass/YAML configuration
├── experiments.py           Figure 2–6 condition matrix
├── dataset.py               MNIST scarce/non-IID partitions
├── models.py                62,346-parameter paper CNN
├── serialization.py         stable flat parameter vectors
├── bayesian_protocol.py     rho/nu transforms and phase schedule
├── bayes_vi.py              Pyro precision and natural-mean SVI phases
├── deterministic.py         FedAvg/FedProx/SCAFFOLD local training
├── client.py                phase-aware virtual client
├── server.py                Flower/Ray + native-Windows local backends
├── aggregation.py           model, rho, and nu aggregation
├── wireless.py              Rayleigh fading and path loss
├── aircomp.py               AirComp/KKT power control
├── metrics.py               accuracy, NLL, ECE, reliability bins
├── logger.py                metrics/client/reliability CSVs
├── utils.py                 standalone paper-style plotting
├── configs/
└── tests/
```

## Recommended Windows environment

The tested working combination for the RTX 3060 Laptop GPU is:

```text
Python       3.12
PyTorch      2.5.1+cu121
Torchvision  0.20.1+cu121
Pyro         1.9.1
Flower       1.32.1
Ray          2.55.1
```

Keep the existing virtual environment when replacing an older package. For a
fresh environment, install CUDA PyTorch first:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install `
  torch==2.5.1 torchvision==0.20.1 `
  --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify a real CUDA allocation:

```powershell
.\.venv\Scripts\python.exe -c "import torch; torch.cuda.init(); x=torch.ones((64,64),device='cuda'); y=x@x; torch.cuda.synchronize(); print(torch.__version__, torch.version.cuda, float(y[0,0]))"
```

## Backends

- `runtime.backend: local`: clients execute sequentially in the launcher. This
  is the stable native-Windows CUDA path.
- `runtime.backend: ray`: Flower/Ray parallel virtual clients, recommended on
  Linux or WSL2.
- `runtime.backend: auto`: native Windows + CUDA selects `local`; otherwise
  selects `ray`.

For the proposed method, Ray executes two physical fit rounds per logical FL
round. Phase-1 client state is persisted locally between actor calls.

## Smoke test

```powershell
.\.venv\Scripts\python.exe main.py `
  --config configs/smoke_gpu.yaml `
  --experiment fig2 `
  --methods fedavg,proposed `
  --rounds 2 `
  --output results\v130_smoke
```

Expected proposed output includes a precision phase before every evaluated
round:

```text
Logical round 1/2, phase=precision: ...
Round 1/2: ...
```

## 40-client pilot

```powershell
.\.venv\Scripts\python.exe main.py `
  --config configs/paper_gpu.yaml `
  --experiment fig2 `
  --methods fedavg,proposed `
  --rounds 5 `
  --replications 1 `
  --path-loss-reference-m 1000 `
  --output results\paper40_v130_pilot
```

The `--rounds` value is a **logical** round count. Proposed internally executes
`2 * rounds` physical fit phases but still transmits `2d` values and records one
metrics row per logical round.

## Full Figure 2 starting command

```powershell
.\.venv\Scripts\python.exe main.py `
  --config configs/paper_gpu.yaml `
  --experiment fig2 `
  --methods fedavg,fedprox,scaffold,proposed
```

With `num_rounds: null`, the logical round count is derived from 30 million
channel uses: `d` per FedAvg/FedProx round and `2d` per Proposed/SCAFFOLD round.

## Outputs

```text
metrics.csv
client_metrics.csv
reliability.csv
checkpoints/*.npz
partitions/*.json
client_state/*        temporary phase state; normally removed after phase 2
```

Important new fields include:

```text
logical_round, physical_round, phase
phase1_train_loss, phase2_train_loss
precision_aircomp_*, mean_aircomp_*
posterior_precision_mean/min/max
local_precision_mean/min/max, local_nu_l2, local_implied_mean_l2
```

## Plotting

```powershell
.\.venv\Scripts\python.exe utils.py `
  --input results\paper40_v130_pilot `
  --figure fig2
```

## Validation

```powershell
.\.venv\Scripts\python.exe -m compileall .
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m pytest -q
```

## Reproduction assumptions

- The channel uses `(distance_m/path_loss_reference_m)^(-alpha)`; the default
  reference is 1000 m because the paper does not publish its numerical distance
  normalization.
- Precision positivity is enforced by optimizing log-precision inside Pyro.
- Native Windows CUDA uses sequential clients. Parallel GPU clients should use
  WSL2/Linux Ray mode.
