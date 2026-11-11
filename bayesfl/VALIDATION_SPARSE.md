# Sparse / wireless validation boundary

The original sparse-v1 implementation and original tests are retained. Current executed
results for the legacy-plus-wireless tree are in
[docs/wireless/VALIDATION.md](docs/wireless/VALIDATION.md).

New tests exercise integer payloads, threshold/dense alternatives, feasible incumbents,
packets/prior fill, phase caching, actual local trainers, clean resume, failure accounting,
actual CNN central evaluation on synthetic data, generator preservation and dynamic exports.
The optional real Flower adapter test is not a fake local replacement and was skipped
when Flower was absent. It must pass on the target runtime before dataset queues.

No new wireless accuracy, convergence superiority, global integer optimum or GPU
throughput measurement is inferred from synthetic checks. Original historical sparse
validation text is archived under docs/history/pre_refresh_20261026/.
