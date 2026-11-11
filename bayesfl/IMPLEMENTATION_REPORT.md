# Implementation report entry point

The current extension is wireless v1 (2026-11-09), described in
[WIRELESS_IMPLEMENTATION_REPORT.md](WIRELESS_IMPLEMENTATION_REPORT.md) and
[docs/wireless/IMPLEMENTATION.md](docs/wireless/IMPLEMENTATION.md).

It adds an opt-in digital orthogonal controller around the existing learner. It does
not replace FOLA/FedAvg local equations, the legacy sparse protocol, original partitions,
old plotting scripts or dependencies. Exact modified/added paths and prior hashes are
recorded in WIRELESS_CHANGE_MANIFEST.json. Validation scope is explicitly limited to
what was executed; see docs/wireless/VALIDATION.md.

The earlier implementation/report and October documentation versions are preserved in
docs/history/. Their counts, fixed-ratio studies and test environments are historical.
They are not current wireless performance evidence or GPU/Ray certification.
