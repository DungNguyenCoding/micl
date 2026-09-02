# Validation status

The package was statically compiled and its offline unit tests were run in the artifact-building environment.

Build-time result:

```text
13 passed, 2 skipped
```

The two skipped tests require packages not installed in the artifact builder (`flwr` and `bayesian_torch`). The requested runtime should execute them after dependency installation. Run:

```bash
bash scripts/validate_install.sh
```

That command checks the environment, executes pytest, constructs the real Bayesian-Torch CIFAR ResNet-56, and asserts the requested Bayesian random-variable dimension of `851,514`.

The CIFAR sparse-partition unit test is fully offline and verifies the requested seed-0 realized size statistics: total `10046`, mean `100.46`, min `79`, max `127`, exactly four realized classes/client, no empty client, no class-pool exhaustion, and no duplicate sample index.
