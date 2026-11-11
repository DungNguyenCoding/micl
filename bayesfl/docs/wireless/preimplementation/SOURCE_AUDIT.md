# Source audit and citation map

All local-source references below refer to the uploaded ZIP members, not to an
assumed live server revision. The `bayesfl/` prefix is omitted for baseline members.
Numbered excerpts are available under `evidence/` for direct inspection.

## Archive identities

```text
baseline_src.zip
SHA256 36343d890cd5313d74ceb953f46d9ce81af2d90a7588d1538cd9fce712a2b41b

FedKLS_Wireless_Develop.zip
SHA256 3622dbf89fc0984eb48388d710175248b31122aebb67d8c2c9681e45e6b2599f
```

## Proposal locators

| ID | Member and lines | Supports |
|---|---|---|
| P1 | `wireless_model.tex:3-43` | Full-model learning, orthogonal links, path gain, fading, PSD and fixed-power caveat |
| P2 | `wireless_model.tex:45-76` | Arbitrary integer counts, ideal sparse/dense payload, multicast and reliable delivery |
| P3 | `wireless_model.tex:78-124` | Local preparation, synchronization barrier, omissions and prescribed completion |
| P4 | `joint_problem.tex` sections Exact Local Reconstruction Cost / Joint Problem and Exact Reduction | Prior-fill KL identity and separable count optimization |
| P5 | `joint_problem.tex` section Feasibility and Endpoint Cases | Minimum and dense feasibility, non-monotonic dense-format exception |
| P6 | `method1.tex:1-81` | Local price, threshold, dense comparison, demand and feasible search |
| P7 | `method1.tex:83-135` | Dual bounds, integer gap, qualified certificate and KKT scope |
| P8 | `method1.tex:137-184` | Procedure, bounded refinement and complexity |
| P9 | `implementation.tex:24-48` | No retraining during queries, restricted server information, original aggregation, numerical/control/power safeguards |
| P10 | `implementation.tex:50-74` | Version A, feasible-incumbent comparison, matched baselines and fixed-round claims |
| P11 | `IMPLEMENTATION.md`, `README.md`, `CHANGELOG.md` | Integration-only status, interfaces, exact count and single-method scope |

`paper_section.tex` is the one-section manuscript form; the split body files are
used above for stable reference. `main.tex`, `preamble.tex` and `references.tex`
provide wrapper/macros/bibliography, not an implemented Python runtime.

## Baseline locators

| ID | Member and lines | Supports |
|---|---|---|
| B1 | `src/bayesfl/client.py:56-146` | Full fit trains/compresses/returns; client+round RNG; no staged allocation |
| B2 | `src/bayesfl/server.py:91-105` | Client reconstructed from Context; existing BudgetServer assembly |
| B3 | `src/bayesfl/config.py:140-189,215-243,342-361` | Fixed-ratio settings, raw precision policy, root section parser |
| B4 | `src/bayesfl/posterior/sparse.py:88-131` | ceil ratio count, stable KL, strict versus score-floor adapter |
| B5 | `src/bayesfl/posterior/sparse.py:140-244` | Ties, random masks, bitmap always, static m metadata and reconstruction |
| B6 | `src/bayesfl/communication.py:34-85` | Existing exact byte formulas and component accounting |
| B7 | `src/bayesfl/experiment_state.py:163-243` | Pre-dispatch fixed-cost reservation and one modeled multicast event |
| B8 | `src/bayesfl/experiment_state.py:245-359` | Received-byte charging, fixed decode m, all-omitted restore, cost equality, old timing fields |
| B9 | `src/bayesfl/experiment_state.py:26-41,361-394` | Configuration fingerprint and completed-round commit schema |
| B10 | `src/bayesfl/posterior/aggregation.py:10-50` | Actual epsilon reducer, not silently idealized product |
| B11 | `src/bayesfl/strategies/research_strategy.py:115-207` | Dispatch/aggregation contract and normalized example weights |
| B12 | `src/bayesfl/evaluation.py:175-224` | Reliability bins, unchanged predictor outputs and resume commit |
| B13 | `docs/tools/export_handoff.py:160-205,250-269` | Text suffix/selected-output allowlist and exactly24 primary runs |
| B14 | Selected `outputs/<exact-run>/resolved_config.yaml` and `partition_metadata.json` | Actual locked settings, precision policies, scheduling and realized data sizes |
| B15 | Selected `outputs/<exact-run>/metrics/global_metrics.csv` | Verified raw endpoint accuracies and byte totals |
| B16 | `.gitignore` | Archive/checkpoint rules exist in this uploaded snapshot; not proof of tracked-file status |

B14 and B15 exact paths/hashes are retained in
`evidence/primary_configuration_audit.json` and the captured endpoint report.
B1-B13/B16 excerpts and inspected source hashes are also retained.

## Narrow external verification

Flower's official *Design stateful ClientApps* documentation explains ephemeral
ClientApp instances and the use of local `Context.state` records. This was used
only to check the persistence design; the uploaded baseline remains the basis for
all actual API/call-site conclusions.

Reference consulted:
https://flower.ai/docs/framework/1.25/en/how-to-design-stateful-clients.html

The version1.29 documentation URL did not resolve through the browsing tool. The
actual pinned1.29 runtime was not available in the review environment. Persistence
integration is therefore a required runtime gate, not asserted as tested. No new
Flower version or dependency migration is recommended.

No outside literature is used to silently replace the supplied wireless model.
The mathematical/protocol adaptations in the review are labeled recommendations.
