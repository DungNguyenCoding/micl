# Source and artifact map — wireless v1

| Path | Responsibility |
|---|---|
| src/bayesfl/main.py | Original CLI plus explicit opt-in wireless routing/time-cap flag |
| src/bayesfl/config.py | Legacy configs plus strictly parsed wireless section; disabled fingerprint compatible |
| src/bayesfl/server.py | Original Flower path; rejects an accidentally misrouted enabled wireless config |
| src/bayesfl/client.py | Original one-fit legacy client, unchanged |
| src/bayesfl/training/ | Original FOLA/FedAvg/BBB learning code, unchanged |
| src/bayesfl/models/ | Original model definitions, including current MNIST CNN |
| src/bayesfl/data/ | Original preprocessing/partition/cache logic |
| src/bayesfl/posterior/ | Original packing/scoring/reducer and fixed-ratio sparse protocol v1 |
| src/bayesfl/communication.py, experiment_state.py, budget_server.py | Original byte-budget controller/accounting/resume |
| src/bayesfl/evaluation.py | Original centralized evaluator, also used by wireless runner |
| src/bayesfl/wireless/ | New isolated controller/protocol/channel/allocator modules |
| scripts/ | Existing tools preserved plus explicitly named wireless tools |
| docs/tools/export_handoff.py | Legacy24 plus opt-in explicit wireless-run exports |
| tests/wireless/ | New numerical, protocol, runtime, tooling and optional real Flower tests |
| validation/wireless/ | This release's actual CPU/compatibility/endpoint logs |
| outputs/ | Existing paper results stay in place; new wireless outputs use new identities |
| docs/history/pre_wireless_v1/ | Documentation immediately preceding this release |

Wireless modules: `payload.py` exact integer bytes; `oracle.py` client tails/thresholds;
`allocation.py` server price coordination/baselines; `channel.py` rates/traces/timing;
`protocol.py` grants/codec/prior fill; `client.py` preparation/persistent store/trainer calls;
`transport.py` local/actual Flower RPC adapters; `controller.py` phase orchestration;
`state.py` accounting/commit/strict resume; `runner.py` real data/model/evaluator assembly;
`reporting.py` standard-library dynamic metric checks; `smoke.py` synthetic integration.

Model-array counters are not output-file sizes. Lightweight source/metrics ZIPs are not
complete resume backups. Preserve partitions, checkpoint arrays and append-only ledgers
separately for continuation. The sibling `bayesfl_historical_archive/` and cleanup
manifests remain unchanged and are not part of the wireless installer.
