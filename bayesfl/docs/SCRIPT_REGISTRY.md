# Current scripts and generated artifacts

## Preserved current workflow

`scripts/plot_ieee_multicast_results_mnist_s10_stage22.py --mnist-fedavg clipped`
produces the selected legacy Acc_Com/Acc_KeepRatio figures. `_run_gpu_queue.sh` retains
its historical config/RESUME byte-budget grammar. Cleanup/registry/archive tools and
historical configs remain where the approved cleanup left them; do not rerun them.
Exact legacy outputs are in RUN_REGISTRY.md and EXPERIMENT_REGISTRY.json.

## Wireless v1 tools

| Script | Behavior / outputs |
|---|---|
| smoke_wireless_offline.py | Six synthetic CPU cases in a fresh explicitly chosen output parent |
| smoke_wireless_flower.py | Actual Flower/Ray CPU synthetic smoke, max 3 rounds; runtime gate |
| prepare_wireless_study.py | Clone exact saved learning configs, validate, write fresh study directory + two serial queues; no training |
| _run_wireless_queue.sh | Explicit interpreter/GPU, stop-on-failure, config or WRESUME records; normal per-job logs |
| summarize_wireless_study.py | Resolve exact study manifest run names, reject ambiguity, print validated endpoints |
| summarize_wireless_results.py | Explicit run directories, dynamic packet/resource checks, optional covered time cut; terminal only |
| plot_wireless_results.py | New destination; separate raw accuracy/time and accuracy/array-GB PNG/PDF |
| docs/tools/export_handoff.py | Default legacy24 check/export; add explicit wireless-run flags for new results |

Generated study paths: `scripts/configs/wireless/TAG/*.yaml`, `study_manifest.json`,
`scripts/queues/wireless_TAG_gpu0.txt`, `wireless_TAG_gpu1.txt`. Queue files have no
embedded Python; never paste shell prompt/terminal output into them. Config generation
can skip matching completed runs or emit clean WRESUME records, but refuses ambiguous
multiple output matches. No automatic latest-directory choice is made.

The source snapshot has 891 legacy YAML files; all load with identical serialized
settings and legacy fingerprint in the compatibility check. This count describes this
uploaded snapshot, not a promise that the user's live directory never changes.
