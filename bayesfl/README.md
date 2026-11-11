# BayesFL / FedKLS — legacy benchmarks and opt-in digital wireless extension

**Wireless v1 source release · 2026-11-09.** This release is built from the supplied
`baseline_src.zip`; it is not a replacement learner. The original FOLA/FedAvg trainers,
models, partition implementation, legacy sparse packets, learning configurations and
selected paper outputs are preserved.

Start with [docs/START_HERE.md](docs/START_HERE.md). For the new feature, use
[docs/wireless/README.md](docs/wireless/README.md); for a new conversation, attach a
current export and read [docs/CHAT_HANDOFF.md](docs/CHAT_HANDOFF.md).

## Two deliberately separate workflows

| Workflow | Status and resource metric |
|---|---|
| Existing MNIST/CIFAR-10 paper figures | Preserved fixed-ratio, single-multicast array-byte benchmark; FOLA R150 / FedAvg R300 |
| Wireless v2 packet/controller | Opt-in digital orthogonal links, exact integer per-client counts, current-round price queries and modeled fixed time slots |

Wireless is **not AirComp**, not model pruning and not noisy weight injection. Clients
train fully once, retain a client-owned prepared state, answer scalar queries and send
only their granted packet. Missing coordinates use the broadcast posterior. This is an
ideal successful-digital-delivery simulator, not a measured physical radio system.

The implementation includes local and Flower/Ray adapters, numerical and integration
tests, clean completed-round resume, two-GPU queue generation, summaries, new plots,
and explicit wireless-run handoff exports. **Flower/Ray and CUDA were not available in
the authoring environment; their runtime gates must pass on the target machine.** See
[validation](docs/wireless/VALIDATION.md) for exact tests and limitations.

## Existing primary plots (unchanged command)

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
/home/micl/anaconda3/envs/dungndh/bin/python \
  scripts/plot_ieee_multicast_results_mnist_s10_stage22.py \
  --mnist-fedavg clipped
```

The selected MNIST experiment remains s10/N20/seed0/unclipped FOLA and recovered
clipped FedAvg. CIFAR-10 remains BasicCNN/alpha .01/full local data/N20/seed0.
Do not interpret their 150:300 byte-budget relation as a wireless equal-time relation.

## First wireless checks, not a long experiment

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
/home/micl/anaconda3/envs/dungndh/bin/python -m pytest -q tests/wireless

PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
/home/micl/anaconda3/envs/dungndh/bin/python \
  scripts/smoke_wireless_offline.py \
  --output-dir "outputs/smoke_wireless_cpu_$(date +%Y%m%d_%H%M%S)" --rounds 3
```

The smoke uses synthetic data and original local trainers, not MNIST/CIFAR accuracy
benchmarks. It does not need Flower, Ray or downloads. Physical parameters in
`scripts/configs/wireless/scenario_example.yaml` are **illustrative, not locked**.
Choose and document the intended physical scenario before long simulations.

Dependencies and `.gitignore` are not changed. Reuse the existing Python 3.10/CUDA
research environment; do not install the author's CPU stack over it.
