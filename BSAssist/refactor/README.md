# Grouped Flower OTA-FL CIFAR-10

This version keeps the Flower framework, but follows the modular style of your
`fedavg_mnist` project:

- `main.py`: Hydra entry point and `fl.simulation.start_simulation(...)`
- `client.py`: `NumPyClient` and `gen_client_fn(context)`
- `dataset.py`: CIFAR-10 loading and non-IID/imbalanced device partitioning
- `model.py`: CIFAR-10 CNN, training, evaluation, parameter helpers
- `strategy.py`: custom BS-assisted OTA aggregation strategy
- `ota.py`: optimized power allocation and OTA PHY simulation
- `docs/conf/config.yaml`: tunable Hydra config

## Why grouped Flower clients?

The physical experiment still has `K=300` edge devices. However, launching one
Flower/Ray actor task per physical device was very slow in your tests. Here,
`num_flower_clients` controls how many Flower virtual clients are used as worker
actors. Each Flower client trains/simulates a group of physical devices and
returns one grouped OTA signal to the server.

For example, with `num_devices=300` and `num_flower_clients=48`, each Flower
client handles about 6 or 7 simulated edge devices. This keeps Flower in the
simulation while greatly reducing Ray scheduling and parameter-transfer overhead.

## Quick benchmark

```bash
cd ota_flower_cifar10_grouped
python main.py num_rounds=5 m0_values=[160] num_devices=300 num_flower_clients=48 eval_every=5
```

## 50-round test

```bash
cd ota_flower_cifar10_grouped
python main.py num_rounds=50 m0_values=[160] num_devices=300 num_flower_clients=48 eval_every=5
```

## Fig. 1-style run

```bash
cd ota_flower_cifar10_grouped
python main.py \
  num_rounds=1000 \
  m0_values='[1600,160,20]' \
  num_devices=300 \
  num_flower_clients=48 \
  mean_client_size=160 \
  coverage_m=550 \
  num_subchannels=1024 \
  eval_every=10 \
  output_dir=outputs/cifar10_fig1_grouped
```

For strict per-round evaluation, set `eval_every=1`. For faster runtime, keep
`eval_every=5` or `eval_every=10`.

## Notes

- TCI is intentionally not implemented.
- The CNN has exactly `D = 307,498` communicated trainable parameters.
- With `F = 1024`, each update round uses `N = ceil(D/F) = 301` OFDM symbols.
