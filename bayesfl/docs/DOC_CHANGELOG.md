# Documentation change log

## 2026-11-09 — wireless v1 source and documentation

Added digital orthogonal wireless modules, strict opt-in routing, exact integer grant
protocol, resource accounting/resume, study/queue/results/export helpers and tests.
Updated current documentation/handoff to distinguish legacy byte-budget studies from
wireless fixed-slot studies and physical settings still awaiting approval.

Closed old documentation gaps using the supplied current baseline source/configs and
primary raw CSVs, including CIFAR R150 results. Preserved old docs under
history/pre_wireless_v1/ and original proposal under wireless/proposal/.
Retained preimplementation review as historical material. No new dataset wireless
accuracy or target GPU/Flower/Ray validation was fabricated.

## Earlier documentation/cleanup

October26/27 documentation installation, export helper/readme, primary result capture
and approved cleanup history remain preserved. Source installation supersedes old
hashes for intentionally changed documentation; do not rerun an old docs installer to
restore stale status. Original release manifests stay historical, with a separate
WIRELESS_CHANGE_MANIFEST.json for this source release.
