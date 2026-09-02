# Baseline v2.0 migration notes

This is a deliberate algorithmic reset for future work, not an exact continuation
of the AirComp paper reproduction implementation.

Removed from the active source:

- Pyro
- AirComp/wireless channel simulation
- transmit-power/path-loss/noise configuration
- precision phase / natural-mean phase
- natural-coordinate communication
- sparse posterior communication code
- channel-use-derived round counts

Retained structurally:

- `config.py`
- `dataset.py`
- `models.py`
- `client.py`
- `server.py`
- `aggregation.py`
- `metrics.py`
- `logger.py`
- `training_schedule.py`
- `runtime_utils.py`
- `serialization.py`
- Flower `ClientApp` / `ServerApp` / Ray simulation structure

New canonical methods:

- `fedavg`
- `bayesavg`

New canonical CLI:

```bash
python main.py --dataset mnist --method fedavg
python main.py --dataset mnist --method bayesavg
python main.py --dataset cifar10 --method fedavg
python main.py --dataset cifar10 --method bayesavg
```

The Bayesian server/client state is a single flat vector in the order:

```text
[all Conv/Linear posterior mu,
 all Conv/Linear Bayesian-Torch rho,
 all deterministic GroupNorm affine parameters]
```

For CIFAR ResNet-56-GN this contains 1,707,284 scalars.
