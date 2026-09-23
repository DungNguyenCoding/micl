# Validation actually performed

## Results

| Check | Observed result |
|---|---|
| Uploaded baseline pytest | 23 passed, 4 skipped |
| Modified repository pytest | **110 passed, 5 skip reports**, 5.36 seconds in the final captured run |
| Original YAML preservation | All 398 original YAML files byte-identical |
| Config load and validation | 408 YAMLs loaded: original 398 plus 10 new study configs |
| Original trainer/reducer-helper/dependency preservation | Hash/byte comparisons passed for local FOLA and FedAvg trainers, original packing/Gaussian helpers, `requirements.txt`, and `pyproject.toml` |
| Python syntax compatibility check | Package, scripts and test source parsed using Python 3.10 grammar; this is not Python 3.10 runtime validation |
| Package/scripts compile | `compileall` completed |
| CLI | Main help and config generator commands executed successfully without Flower import at CLI-help time |
| Synthetic local-training integration | All five methods completed three CPU rounds |
| Actual central evaluator | Existing MLP evaluator ran on synthetic image tensors and wrote byte-aware metrics, checkpoint and resume state |
| Plots | Two separate PNGs rendered from recorded synthetic histories; observed CSV export checked; communication figure visually inspected |

Exact logs: `validation/baseline_pytest.txt`, `validation/modified_pytest.txt`, `validation/environment_checks.json`, `validation/synthetic_smoke.txt`, and `validation/cli_help.txt`.

## Test coverage added

The added executable checks include both KL directions and the supplied numerical/ranking examples; identical/equal-variance behavior; tied-score ordering; global exact-cardinality selection; independent random-mask RNG; correct little-bit-order bitmap padding and lengths; dtype/shape/count/protocol/layout/snapshot/corrupt-message rejection; ascending value ordering; no hidden local-state reconstruction; full-retention agreement; all-omitted preservation; mixed-mask/sample-weight behavior; and both FOLA modes against original-code golden fixtures.

Accounting tests cover d=4/m=2, million-coordinate formulas at keep=.5/.1, compulsory full-retention bitmap overhead, explicit initialization/evaluation arrays, rejected arrivals without double-counting, unknown raw payload on undecodable messages, and strict byte limits including zero-budget/no-training cases. Resume tests compare all five methods with uninterrupted synthetic training and check no repeated metric rows or reset costs, along with rejection of changed LR or extra in-flight communication.

Config tests preserve locked MNIST/CIFAR settings and explicit zero-omega policy. Plot tests enforce latest-under-budget rather than best accuracy, actual integer byte counters, no round-cap extrapolation, exhausted-budget remainder semantics, observed-data exports, and distinct matched seed checks. The study-summary lookup refuses to invent missing run directories.

## Five skip reports

Three pre-existing BBB tests skipped because `bayesian_torch` was absent. The original Flower/Ray import-level test skipped because `flwr` was absent. The new `tests/test_sparse_flower_integration.py` module also skipped at import because Flower was absent; its real serializer test and five client/strategy/server method cases were not executed. A skipped module is not counted as successful Flower compatibility validation.

That new module uses actual Flower NumPyClient adapters, typed serialization, ResearchStrategy, BudgetServer and ServerAppComponents with in-process client proxies and synthetic data. It is not a fake replacement Flower package and does not require Ray. Run it in the original target environment:

```bash
python -m pytest -q tests/test_sparse_flower_integration.py
```

Then run the separately named real Flower/Ray dataset smoke from `SPARSE_README.md` before a long study.

## Execution environment versus target environment

Available authoring environment: Python 3.13.5, NumPy 2.3.5, Torch 2.10.0+cpu, Torchvision 0.25.0+cpu, pytest 9.0.2, Matplotlib 3.10.8. Flower, Ray, Bayesian-Torch and CUDA were absent. Dependency retrieval attempts did not make Flower available. No dependency manifests were changed to match this environment.

The supplied project continues to declare Python `>=3.10,<3.11`, Flower 1.29.0, Bayesian-Torch 0.5.0 and pytest `>=8,<9`, with the user's original working CUDA Torch pair. The executed checks do not establish runtime compatibility of that complete target stack. The target-environment commands remain a required execution check, not an assumed result.

## Synthetic results: communication only

The tiny model has 37 scalar parameters and each round selects two of three clients. Every fit runs the unchanged local trainer with synthetic binary-classification data. At keep=.5, m=19 and bitmap length=5 bytes:

| Method | Completed rounds | Cumulative model-array bytes |
|---|---:|---:|
| FedAvg dense | 3 | 1,776 |
| FOLA dense | 3 | 3,552 |
| FOLA sparse KL(global || local) | 3 | 2,718 |
| FOLA sparse KL(local || global) | 3 | 2,718 |
| FOLA sparse random | 3 | 2,718 |

These totals were counted by the production ledger on serialized-array round trips and match the exact padded formulas. This harness does not run Flower or Ray. The toy accuracies are deliberately not presented as comparative research evidence. Full MNIST/CIFAR training, multi-seed empirical accuracy comparisons, and GPU profiling were not run or fabricated.
