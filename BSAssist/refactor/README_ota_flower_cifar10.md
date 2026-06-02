# BS Dataset-Assisted OTA-FL on CIFAR-10 with Flower

This refactors the provided `Fig1.ipynb` into a Flower-based Python simulation and maps the experiment from MNIST to CIFAR-10. It implements the paper's BS-dataset-assisted **superimposed update report** with **optimized power allocation** only. The TCI benchmark branch is intentionally not included.

## Key changes from the notebook

- Dataset: `torchvision.datasets.MNIST` -> `torchvision.datasets.CIFAR10`.
- Model: MNIST 6-layer CNN (`D=582,026`) -> CIFAR-10 7 trainable-layer CNN (`D=307,498`). With `F=1024`, the update report needs `N=ceil(D/F)=301` OFDM symbols per FL round.
- Network: default `K=300` devices uniformly distributed in a 550 m BS coverage disk.
- Data split: BS data is i.i.d.; edge devices are non-i.i.d. and imbalanced, each with samples from 3 of 10 classes.
- FL framework: local training runs inside Flower `NumPyClient` objects; the custom server `Strategy` performs the BS initial update and OTA aggregation.
- TCI: removed.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_ota_flower_cifar10.txt
```

## Quick smoke test

```bash
python ota_flower_cifar10.py --smoke-test --client-cpus 1
```

## Fig. 1-style CIFAR-10 run

This is computationally heavy because it trains 300 clients per round.

```bash
python ota_flower_cifar10.py \
  --rounds 1000 \
  --num-devices 300 \
  --coverage-m 550 \
  --m0-values 1600 160 20 \
  --mean-client-size 160 \
  --F 1024 \
  --P-dbm 20 \
  --sigma-z2-dbm -50 \
  --gamma-db -10 \
  --alpha 4 \
  --plot \
  --output-dir outputs/cifar10_fig1
```

Each run writes:

- `m0_<value>_history.csv`: round, symbol transmissions `Nt`, accuracy, loss, distortion, and `rho_ref`.
- `m0_<value>_model.pt`: final model state.
- `fig1_cifar10_ota_flower.png`: optional Fig. 1-style plot when `--plot` is passed.
- `data_split_summary.csv`: BS/client class distributions.

## Notes

The simulation uses Flower for FL orchestration. The analog wireless channel is simulated numerically: each client returns its noiseless received contribution `h*x` to the custom Flower strategy; the strategy sums those contributions, adds receiver noise, decodes the OTA aggregate, and updates the global model. This preserves the paper logic while remaining runnable on a single machine.
