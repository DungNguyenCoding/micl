# Repository, generated files, and source map

The path map combines the current source inventory and registry with earlier source/documentation inspection. It does not claim that every current function body has been freshly audited.

```text
~/DungNDH/micl/
├── bayesfl/
│   ├── README.md                  # current entry point
│   ├── AGENTS.md                  # continuation / command rules for assistants
│   ├── pyproject.toml             # runtime packaging; unchanged by docs refresh
│   ├── requirements.txt           # runtime dependency declarations; unchanged
│   ├── CHANGE_MANIFEST.json       # HISTORICAL sparse-release hashes, not live inventory
│   ├── PACKAGE_MANIFEST.txt       # HISTORICAL 2026-09-02 base-package list
│   ├── DOCS_UPDATE_MANIFEST.json   # this documentation package only
│   ├── docs/                     # current handbook, handoff, evidence, history
│   ├── src/bayesfl/               # training/runtime implementation
│   ├── scripts/                  # current and retained historical tools
│   │   ├── configs/              # keep source YAML paths
│   │   └── queues/               # keep executable path references intact
│   ├── tests/                    # retain source tests and small fixtures
│   ├── data/                     # local downloaded datasets
│   ├── outputs/
│   │   ├── partitions/           # protected realized client assignments
│   │   └── run_.../              # protected/current run artifacts
│   ├── plots/                    # current Acc_Com / Acc_KeepRatio figures
│   └── logs/                     # simulation and queue logs
├── bayesfl_historical_archive/    # same-SSD historical output/script archive
├── bayesfl_cleanup_DRY_RUN_20261026_194253.csv
└── bayesfl_*reports/logs...       # earlier audit evidence outside repository
```

## Training path

```text
YAML + CLI -> main.py -> partition preparation -> server.py / strategy
                       -> client.py -> local SGD/FOLA -> sparse packet
                       -> validation / reconstruction / aggregation
                       -> central evaluation -> metrics + checkpoint/resume
```

| Module | Responsibility |
| --- | --- |
| `src/bayesfl/main.py` | Entry point; config, run paths, partitions, new versus resume setup |
| `config.py` | Dataclasses, validation, overrides and round LR |
| `server.py` / `budget_server.py` | Flower/Ray assembly and budget-aware true-round loop |
| `client.py` | Local fit dispatch and post-fit compression boundary |
| `training/deterministic.py` | FedAvg SGD; later strict/unclipped changes require current-source inspection |
| `training/fola.py` | `paper_reference` and `online_recurrence` branches |
| `training/bbb.py` | Retained Bayesian-Torch variational baseline |
| `strategies/research_strategy.py` | Sampling, result checks, aggregation, accounting integration |
| `strategies/common.py` | Sample weights and common array reduction |
| `posterior/sparse.py` | Coordinate layout, scores, selectors, bitmap codec and packet checks |
| `posterior/aggregation.py` | Shared dense FOLA reducer |
| `posterior/packing.py` | Named-parameter layout / dense mean–precision packing |
| `posterior/gaussian.py` | Precision/Gaussian helpers; separate mode-specific behavior |
| `posterior/diagnostics.py`, `posterior/scale_mixture.py` | Diagnostics and BBB prior support |
| `communication.py` | Payload counters/ledger; `core_round_bytes` supports multicast in inspected snapshot |
| `experiment_state.py` | Broadcast state, budget reservation, omission invariants and resume consistency |
| `evaluation.py`, `metrics.py`, `logging_utils.py` | Central metrics, persistence and logging |
| `data/datasets.py`, `data/partition.py`, `data/transforms.py` | Data loading, realized client assignment, transforms |
| `models/factory.py` | Model identifier dispatch / initialization |
| `models/mnist_cnn.py` | Current CNN 16–32 implementation; path confirmed, current body not in older ZIP |
| `models/mnist_mlp.py` | Original and later MLP controls |
| `models/paper_cnn.py` | CIFAR BasicCNN |
| `models/cifar_resnet.py`, `models/bayesian_layers.py` | Retained ResNet / BBB support |
| `communication_plots.py`, `utils.py` | General analysis helpers; not the chosen bespoke paper plotter |
| `offline_smoke.py` | Synthetic integration helper, not measured dataset performance |

## Artifacts inside one run

| Artifact | Use / preservation |
| --- | --- |
| `resolved_config.yaml` | What configuration actually ran; preserve |
| `source_config.yaml` | Original submitted config; preserve |
| `partition_metadata.json` | Actual partition metadata/hash; preserve |
| environment metadata | Runtime provenance; inspect actual filename/version |
| `metrics/global_metrics.csv` | Global point-estimate trajectory and cumulative bytes; essential for current figures |
| `metrics/round_train_metrics.csv`, client metrics | Diagnostics / compute exposure; not substitutes for test accuracy |
| `communication/layout_manifest.json`, `communication/protocol.json` | Coordinate/protocol evidence; preserve |
| `communication/events.jsonl` | Ledger; required by clean-resume consistency, not just expendable logs |
| `resume/latest.json`, referenced `.json` and `.npz` | Consistent committed global state; required together for resume |
| `checkpoints/`, posterior/reliability arrays | Evaluation/model analysis depending on consumer; not normally needed by metrics-only plotter |
| `run_summary.json` | Completion/stop reason metadata when available |
| `*.before_resume_*`, migration backups | Historical configuration/accounting provenance; do not discard blindly |

A lightweight Git/plot package and a full resume backup are different artifacts. Ignoring a file in Git must not be confused with deleting the local copy. The historic `outputs/plots/` folder was archived; current paper figures are in root `plots/`.

Code references a repository layout. Moving a Python script can change `Path(__file__).resolve().parents[...]`; moving a YAML can break queue references. This documentation refresh adds new docs without relocating runtime code, configs or outputs.
