# Configuration and scenario decisions

The `wireless:` section is opt-in. Unknown wireless/nested keys fail validation instead
of being silently ignored. The original fixed-ratio format remains valid when it is
absent or disabled. Methods remain `fola` or `fedavg`; do not invent new CLI method names.

Use the complete example at `scripts/configs/wireless/scenario_example.yaml` as a schema
illustration. Its marker `EXAMPLE_NOT_LOCKED` requires `--allow-example` in the study
generator. These physical values were not optimized, supplied by the proposal, or
approved for the paper. Dense CIFAR may be infeasible under that example's common target.

| Field | Meaning |
|---|---|
| enabled | Boolean, default false |
| backend | `flower` (real Flower/Ray adapter) or `local` (serial client emulator) |
| policy | `joint`, `equal_bandwidth`, `uniform`, `dense` |
| uplink_bandwidth_hz / downlink_bandwidth_hz | Positive system resources, Hz |
| round_target_s | Positive common fixed slot duration, seconds |
| total_time_budget_s | Optional positive total modeled-time cap; no partial slots |
| control_reserve_s | Explicit nonnegative per-slot control allowance |
| control_rate_bps | Positive shared logical control rate |
| control_exchange_latency_s | Explicit nonnegative modeled delay per exchange |
| rpc_timeout_s | Wall-clock RPC wait limit, not a radio deadline; default 3600 |

`channel` fields: mode (`fixed`, `block_rayleigh`, `trace`), independent nonnegative seed,
positive `path_gains` (nontrace modes), positive `uplink_psd_w_hz` vector,
`downlink_psd_w_hz`, `noise_psd_w_hz`, boolean `reciprocal_fading`, optional `trace_csv`,
optional conservative `client_power_cap_w` vector. Vectors are logical-client ordered
and must have exactly N entries. No dBm conversion is inferred; inputs are linear W/Hz,
W and dimensionless power gains.

`compute` requires N explicit nonnegative entries for each of
`seconds_per_sample_epoch`, `fixed_seconds`, `score_seconds`. Zero is permitted only
as an explicit modeling choice. No equal sample-count approximation replaces the
realized client sizes.

`solver`: initial_price (positive, default 1), max_bracket_queries (48),
max_bisection_queries (64), price_atol (0), price_rtol (1e-12), gap_atol (1e-10),
gap_rtol (1e-6). These are implementation defaults, not trained hyperparameters or
proof-certified tolerances. They contribute to actual query overhead.

## Required learning/protocol constraints

All N clients participate; nonempty partitions; E<=10; no server full-client snapshots;
`communication.downlink_mode: multicast`; deterministic logical identities;
`communication.max_communication_bytes: null`. FOLA adaptive policies require
`compression.selection_rule: kl_global_local` or `random`. Dense policies require
`dense`, and FedAvg supports dense only. FOLA keep_ratio must be 1.0 as an inert
placeholder for exact integer grants. Keep existing score-precision policy and floor.

Do not silently combine the old total-byte cap with dynamic wireless allocation.
Use rounds and/or total modeled time; retain actual byte counters for reporting.
All method comparisons must declare the same physical scenario and slot policy.
LR, epochs, clipping, seed and scheduler remain explicit saved learning choices.

## Trace format

```csv
round,client_id,uplink_gain,downlink_gain
1,0,1e-12,1e-12
1,1,2e-12,2e-12
```

Provide one row for every required (round, client), no duplicates, positive finite gains.
Counts and IDs are integer decimal strings. The small example above is not a complete
20-client trace. Relative paths in a scenario are resolved relative to that scenario
file by the generator. The runner copies the trace into the new run and pins its SHA-256.
Resume reads the pinned copy; a changed/missing trace fails. Counter-seeded generated
channels are reproducible by round and do not consume trainer/mask RNG streams.

## New study versus continuation

Use `prepare_wireless_study.py --help`. It clones actual primary resolved learning
configs recorded in `docs/EXPERIMENT_REGISTRY.json`, not historical default YAMLs.
A new tag creates a new directory and never overwrites existing study files. Its output
manifest records inherited source hashes and wireless fingerprints. Re-run the existing
queue only after checking status; do not create duplicate writers.

Clean resume permits only a larger total round cap and/or total time cap. Changing the
backend, channel, control assumptions, seed, optimizer, count policy or trace identity
requires a new study/run. The old R150/R300 relation is not an equal-time rule.
