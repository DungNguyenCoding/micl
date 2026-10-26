# Documentation package validation

Date: **2026-10-26**. These checks concern this documentation bundle, installer and read-only endpoint/export helper—not the live BayesFL training implementation.

## Checks actually run during authoring

1. Installer and helper parse with Python 3.10 grammar.
2. Current Markdown relative links resolve inside the bundle.
3. All 14 supplied original files preserved byte-for-byte.
4. Registry 24 primary/41 protected runs and all exact budget bounds checked.
5. Installer preview leaves a simulated repository unchanged.
6. Installer apply backs up replacements and leaves runtime/dependencies unchanged.
7. Installed doc verification and repeat-install idempotence.
8. Installer refuses conflicting live documentation without overwriting it.
9. Installer refuses symlink destinations.
10. Terminal helper reads 24 synthetic endpoints and writes no repository files.
11. Live-export helper creates hash-checked ZIP, excludes resume arrays, preserves originals.
12. Endpoint checker refuses missing target rounds.
13. Endpoint checker refuses byte mismatches instead of estimating costs.
14. Accuracy units are explicit; percent histories require the percent option.

**14 check groups passed** in a temporary simulated repository. The executing Python version was 3.13.5; the code also parsed under Python 3.10 grammar. This does not establish a separate Python 3.10 runtime test.

The helper fixtures contained 24 synthetic endpoint CSVs with known byte totals. They tested file selection, validation, output formatting and export, not measured MNIST/CIFAR performance. Temporary fixtures were removed after testing and are not distributed as experiment data.

## Not performed here

No live-server installation, current full pytest suite, Flower/Ray training, GPU run, fresh current-source audit or actual new CIFAR R150 accuracy extraction was performed. The current production code and raw CSVs remain on the user's machine. Installation preview/apply/verify and the optional endpoint check must run there.

The final package includes exact hashes for every payload file and preserves 14 supplied original files byte-for-byte. It deliberately leaves runtime source, existing scripts/configs/queues, tests, dependency declarations, .gitignore, data, outputs and the original historical code/package manifests outside its write scope.
