# Wireless outputs, resource fields and resume

A new real run retains baseline configuration/environment/partition metadata and central
evaluation artifacts. It adds:

```text
outputs/run_wireless_..._<timestamp>/
  resolved_config.yaml
  source_config.yaml
  partition_metadata.json
  environment.json
  run_summary.json
  metrics/global_metrics.csv
  metrics/wireless_rounds.csv
  metrics/wireless_clients.csv
  wireless/protocol.json
  wireless/layout_manifest.json
  wireless/channel_trace.csv             # trace mode only
  wireless/events.jsonl
  wireless/allocations/round_XXXXXX.json
  wireless_resume/latest.json
  wireless_resume/round_XXXXXX.json
  wireless_resume/round_XXXXXX.npz
  checkpoints/...                        # baseline evaluator's scheduled saves
```

Global accuracy columns retain the existing semantics: `fola_mean_accuracy` for FOLA,
`accuracy` for FedAvg, stored as fractions. Wireless CSVs are not smoothed. Each completed
round has every client's count and packet size, allocation status, reconstruction
objective diagnostics and separate modeled versus emulator times.

## Resource definitions

| Field family | Meaning |
|---|---|
| cumulative_all_array_bytes | One modeled multicast model plus each actual received upload packet; not ZIP/checkpoint size |
| cumulative_control_bytes | Canonical phase-instruction/response JSON byte model, separate from model arrays |
| cumulative_modeled_time_seconds | Completed fixed slots; partial attempts retain consumed modeled resources on failure |
| cumulative_modeled_tx_energy_j | UL+DL data-plane modeled transmit energy only |
| keep_count / realized_keep_ratio | Operative integer grant and descriptive m/d; no ratio-to-count reconstruction |
| bandwidth_hz / uplink_snr / gain | Per-client modeled radio values |
| discarded KL | Top-m oracle tail and actual omitted-score total; random masks can be worse |
| gap/status | Numerical optimization diagnostics, not statistical or globally certified optimality |
| wall/preparation/allocation wall seconds | Emulator runtime; never relabel as wireless latency |

Control metadata, network framing and radio traffic differ: the code does not measure
on-wire bytes. Startup identity discovery and CSI estimation are outside the reported
control phases. Data-plane energy excludes control, compute, circuit and idle energy.
There are no artificial sleeps; local and Flower backends both calculate modeled clocks.

`run_summary.json` records completed/pending rounds, stop reason, failure and counters.
Successful stops are `max_rounds` or `modeled_time_budget`; `failed` is not a completed
comparison input. The next whole slot must fit the total-time cap. No extra fictitious
round fills a residual budget. Known infeasibility does not drop clients or extend time.

## Clean evaluated-round continuation

State saves the global arrays, true round, counters, RNG policy, layout/broadcast digest,
partition-content digest, channel trace digest, configuration fingerprint, event-ledger
digest and exact metric-file digests. Latest and previous resume generations are kept.
Original evaluator checkpoint retention is unchanged. The run holds a filesystem lock.

Only `training.rounds` and `wireless.total_time_budget_s` are excluded from the wireless
configuration fingerprint. Limits cannot be reduced below consumption. Do not rename a
run directory casually: its name is part of the phase identity. No cross-protocol import
of a legacy checkpoint, mid-round recovery, ledger truncation or model-only warm start
is represented as a clean resume.

The terminal summary accepts complete recorded trajectories and selects the last
observed checkpoint <= a requested time budget. It refuses out-of-range extrapolation,
missing/duplicate rounds/clients, packet byte inconsistencies or failed runs. A new-chat
light export deliberately omits checkpoint arrays and full ledgers, so it supports
inspection/plots but is **not sufficient to resume training**.
