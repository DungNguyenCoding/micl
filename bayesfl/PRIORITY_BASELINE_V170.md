# CIFAR-10 priority baseline upgrade (v1.7.0 research extension)

This extension leaves legacy defaults unchanged and adds opt-in training fields:

- SGD momentum (`training.momentum`)
- weight decay (`training.weight_decay`, default 0)
- logical-round LR scheduling (`training.lr_scheduler`: `constant` or `cosine`)
- cosine floor (`training.min_learning_rate`)

The priority CIFAR config is:

`configs/cifar_priority/cifar_baseline_e10_m09_cosine.yaml`

It uses E=10, batch=256, SGD momentum=0.9, initial LR=0.05, cosine decay to
1e-4 over the requested logical-round horizon, prior std=0.01, and 23 dBm.
Both physical Proposed phases use the same LR for the same logical round.
FedAvg uses the identical round-level LR schedule.

The Bayesian backend remains Pyro in this stage. A bayesian-torch comparison is
intentionally deferred until the Pyro-vs-FedAvg baseline experiment is complete.
