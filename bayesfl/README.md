# BayesFL — sparse posterior communication implementation

The uploaded FedAvg/FOLA baseline has been extended with model-wide packed-bitmap posterior communication, both KL argument orders, an exact-size random control, communication-budget-aware execution, cumulative accuracy-versus-byte logging, and clean completed-round resume.

**Start here: [SPARSE_README.md](SPARSE_README.md).** It contains runnable commands, paired MNIST/CIFAR configurations, terminal-only result summaries, plotting, and the important precision/zero-omega policies.

- [Implementation report](IMPLEMENTATION_REPORT.md): code audit, concrete changes, baseline preservation, and known boundaries.
- [Validation report](VALIDATION_SPARSE.md): checks actually run and optional dependencies not exercised.
- [Original supplied source handoff](docs/BASELINE_HANDOFF_README.md): original locked experiments and historical results, not new sparse results.
- [Implementation guideline](docs/SPARSE_IMPLEMENTATION_GUIDELINE.md): requested extension specification.
- [Old README from the source archive](docs/BASELINE_ARCHIVE_README.md): historical documentation, not the new sparse contract.

## Quick validation

Use the established Python 3.10 research environment and original dependencies:

```bash
python -m pip install -e . --no-deps
python -m pytest -q
python -m pytest -q tests/test_sparse_flower_integration.py
```

A skipped Flower integration module does not validate Flower/Ray execution. The available authoring environment produced **110 passed, 5 dependency-related skip reports** using a different CPU/Python stack; details are in the validation report. No new MNIST/CIFAR performance results are claimed.

The original 398 experiment YAMLs, local training modules, models, partition generation code, and dependency manifests are preserved. The 10 new paired configs are under `scripts/configs/bayesian_sparse/`.

**Critical representation choice:** this baseline stores raw float32 precision/omega, not variance. Sparse and dense FOLA retain that wire representation. Zero-initialized CIFAR uses an explicit score-only positive-precision surrogate; local training and transmitted omega are not floored. See the sparse README before running or interpreting those experiments.
