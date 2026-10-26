# Current CIFAR-10 profile — BasicCNN, alpha .01, full local data

This page supersedes the older ResNet-56 profile at this path. The old text is preserved unchanged in [history](history/pre_refresh_20261026/docs/PAPER_CIFAR10_PROFILE.md).

Current main profile: `paper_basiccnn`, d=878,538, N=20, participation=20/20, paper Dirichlet alpha=.01, local_data_fraction=1.0, seed0, SGD, E=10, B=32, momentum=0, weight_decay=0, gradient clipping norm10. Dense/sparse FOLA: `paper_reference`, lambda=.01, LR=.02 cosine to .0001 over H=400. FedAvg: LR=.01 constant. No silent matched-LR substitution.

The chosen comparison is dense FOLA R150 / FedAvg R300 at 22,139,157,600 modeled multicast bytes. KL/random sparse targets are 193,278,499,950,1362 at r=.75,.50,.25,.10,.05. The older R120 table is not the current R150 endpoint table.

Detailed settings/evidence: [LOCKED_EXPERIMENTS.md](LOCKED_EXPERIMENTS.md). Byte model: [COMMUNICATION_ACCOUNTING.md](COMMUNICATION_ACCOUNTING.md). Exact output/YAML registry: [RUN_REGISTRY.md](RUN_REGISTRY.md). The profile is a project experimental choice, not a claim to reproduce every original paper setting.
