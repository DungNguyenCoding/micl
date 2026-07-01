# Bayesian Federated Learning Simulation Base Source

This base source compares three training modes under the same grouped Flower simulation layout:

- `fedavg`: classic deterministic Federated Averaging.
- `vi`: mean-field variational Bayesian FL using Pyro SVI and `AutoDiagonalNormal` over an MLP weight posterior.
- `ola`: Online Laplace Approximation / FOLA-style Bayesian FL using diagonal empirical Fisher precision, Gaussian-product server aggregation, and prior-iteration local regularization.

The key design choice is that **physical devices** and **Flower virtual clients** are separate. For example, `--num_devices 300 --num_virtual_clients 24 --client_fraction 0.1` randomly selects about 30 physical devices per round, but launches at most 24 Flower/Ray client tasks per round. Each virtual client sequentially simulates the selected physical devices assigned to its group.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick runs

FedAvg, MNIST, IID balanced, 300 physical devices grouped into 24 virtual clients:

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

Variational Bayesian FL with non-IID and unbalanced clients:

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
  --output_dir outputs/vi_mnist_noniid_unbalanced
```

Online Laplace/FOLA-style Bayesian FL on CIFAR-10:

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

## Output files

`main.py` writes only experiment artifacts such as:

- `config.csv`
- `metrics.csv`
- `selected_clients.csv`
- `device_summary.csv`
- `run.log`
- `final_model.pt`

It does **not** write `.png` files during training.

## Offline plotting

Generate plots after training:

```bash
python utils.py metric --history outputs/fedavg_mnist_iid/metrics.csv --metric accuracy --output_dir plots/fedavg_mnist_iid
python utils.py metric --history outputs/fedavg_mnist_iid/metrics.csv --metric loss --output_dir plots/fedavg_mnist_iid
python utils.py selected --selection outputs/fedavg_mnist_iid/selected_clients.csv --output_dir plots/fedavg_mnist_iid
python utils.py radar --device_summary outputs/fedavg_mnist_iid/device_summary.csv --output_dir plots/fedavg_mnist_iid
```

## Dataset arguments

- `--dataset mnist|cifar10`
- `--iid true|false`
- `--balanced true|false`
- `--noniid_alpha`: smaller means stronger label skew.
- `--unbalanced_alpha`: smaller means stronger client sample-count imbalance.

`dataset.py` supports all four combinations:

- IID + balanced
- IID + unbalanced
- non-IID + balanced
- non-IID + unbalanced

## Method notes

### FedAvg

Each selected physical device starts from the global mean parameters, trains locally, and returns a deterministic update. Virtual clients average their assigned physical-device models before the server performs global weighted averaging.

### VI

Each selected physical device treats the incoming global `loc, scale` as the Gaussian prior over MLP weights. Pyro SVI learns a local diagonal Gaussian posterior. Server aggregation can be selected with:

- `--bayes_aggregation product`: diagonal Gaussian product aggregation.
- `--bayes_aggregation mean`: moment-matched averaging.

### OLA/FOLA

Each selected physical device trains a deterministic model with prior-iteration regularization:

```text
0.5 * (theta - mu_global)^T Precision_global (theta - mu_global)
```

It also accumulates squared task-loss gradients as a diagonal empirical Fisher estimate. The local precision is updated online using the communication round index, then the server uses diagonal Gaussian-product aggregation.

## TODO: wireless-aware selection

The current active policy is `--selector random`. The placeholder `WirelessQualitySelector` in `selector.py` is the intended extension point for:

- analog over-the-air aggregation client selection,
- digital uplink-rate/deadline-aware selection,
- hybrid data-importance and channel-quality scheduling.

The source already saves `device_summary.csv` with synthetic polar coordinates and group IDs so that future channel models can be joined onto a stable device metadata schema.
