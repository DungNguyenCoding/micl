# Validation artifacts, not benchmark results

These files record checks actually executed on the uploaded and modified code.
`synthetic_smoke.txt` reports communication totals from tiny synthetic CPU runs,
not MNIST/CIFAR accuracy. It is not a Flower/Ray simulation result.

`modified_pytest.txt` identifies dependency-related skips explicitly. The optional
Flower integration module was not executed in the available environment. No
historical or proposed sparse-method performance curves are included here.
