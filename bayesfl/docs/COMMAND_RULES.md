# Command-generation rules for this project

These rules preserve the user's execution preferences and supersede old example settings in the historical command guide. They are project workflow guidance; an explicit later user request can change a setting.

## Fixed execution context

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Use `"$PY"` for code/config/partition checks, tests, metrics and plotting. Do not fall back to `/home/micl/anaconda3/bin/python` or change a working CUDA environment. The recorded setup uses two GPUs, indices 0 and 1. Use one serial queue per GPU, not one independent Ray job per configuration.

## Before changing or launching anything

Read [START_HERE.md](START_HERE.md), [LOCKED_EXPERIMENTS.md](LOCKED_EXPERIMENTS.md) and the **current on-disk** relevant code/YAML. Do not reconstruct a configuration from old chat snippets when a selected resolved config exists. Do not fabricate CLI switches, helper functions, YAML fields or timestamps. Prefer exact registry paths; multiple candidate runs require explicit resolution, not an unannounced newest/best-result choice.

Preserve existing results and partitions. Check for completed or partially completed runs before preparing jobs. A run is complete only after validating its required metric round, configuration, byte counter and, for resume, committed checkpoint—not from its folder name or the queue's exit banner.

**Hard limit: local_epochs <= 10.** Main MNIST uses E=5, not the old common E=10 template. No silent LR, lambda, architecture, clipping, client count, partition, scheduler or precision-policy retuning. Normal runs do not impose diagnostic optimizer-step caps. Keep `output.save_full_client_posteriors=false` for sparse communication.

## Naming and output preferences

Use `run_...` for locked runs, baseline recovery and replications. Never recover a published/selected baseline under `debug_...`. Use `smoke_...` for explicitly requested short checks; reserve `debug_...` for explicitly requested exploratory tuning. Do not rename existing run directories just to improve a label.

Give complete copy-to-terminal blocks, not fragments with implied setup. On requests for scripts directly in chat, provide the full script rather than only a download. Every simulation queue response must include its **matching terminal-only result extraction command** in the same response. No automatic `tee`, .txt summary or summary CSV unless requested. Normal simulation logs, plot files, configs, manifests and explicitly requested documentation/audit exports are allowed outputs.

## Required five-stage format for new simulation requests

1. **Generate configs and two queues.** Clone the appropriate locked config; modify only requested fields. Use unique `run_name`s and refuse silent overwrite of a different config. Print exact job counts per GPU and assert the expected total. Empty queues may legitimately mean no remaining work.
2. **Validate.** Load actual queued YAMLs with `bayesfl.config.load_config`; validate model/dimension, data, participation, seed, E/B, LR/horizon, optimizer/clipping, FOLA coefficient/mode, compression, numerical policy, multicast and budget. Assert no duplicate jobs. Never launch on validation failure.
3. **Prepare/inspect partitions.** Use the current `prepare_partition` / `partition_stem` APIs after checking their signatures. Generate each unique partition once before parallel jobs. Inspect realized sizes, classes, unique assignments, shared hash and any backfills. Do not regenerate a publication partition in place.
4. **Launch.** Use the existing `_run_gpu_queue.sh`, sequential per GPU, with timestamped nohup logs and printed PIDs. Check GPU/process status first. No concurrent writers to one run directory. Do not broadly kill other Ray/Python jobs.
5. **Extract results after completion.** Use actual CSVs and exact expected checkpoints. Print FOLA-FedAvg and KL-random gaps in percentage points. State whether the comparison is same-round or equal-byte. Print config details only when requested or necessary to flag a mismatch.

For an existing-run continuation, replace new-config preparation with checkpoint/ledger preflight and resume entries. Do not restart R1 to work around a resume inconsistency.

## Queue contract

Observed runner: `scripts/_run_gpu_queue.sh GPU_ID QUEUE_FILE`.

```text
scripts/configs/actual_selected_config.yaml
RESUME|/absolute/path/to/existing/run|TOTAL_ROUND_CAP|TOTAL_BYTE_BUDGET|EXPECTED_ENDPOINT
```

A resume entry has five pipe-separated fields. The final expected-endpoint field is a label in the inspected runner, **not an automatic endpoint assertion**; the result extractor must check it. The total cap is not “additional rounds”. Keep the original seed, partition, optimizer, LR horizon, compression and precision policy.

The multicast migration was a separate accounting operation. Do not casually change `downlink_mode` on published runs; preserve prior bookkeeping/migration backups. Empty CIFAR R150 resume queues were confirmed to reflect already-complete runs, not a runner defect.

## Metrics and interpretation

Prefer `fola_mean_accuracy` for FOLA; use the documented point-estimate accuracy for FedAvg. Never substitute train accuracy or MC accuracy silently. Inspect units (normally fractions), display percent, and reject missing/nonfinite values. Do not infer a single row's unit merely from `value <= 1.5` when the data convention is unknown.

For a checkpoint R, read exactly that round. A missing endpoint is an error, not permission to use the closest, last or best value. Round 0 is initialization, not one of rounds 1...R. `last20` FOLA versus `last40` FedAvg spans the same bytes only for the recorded constant 2:1 dense costs. Sparse tail windows require byte matching, not an arbitrary same number of rounds.

A five-seed overview is not an obligation to rerun multiseed experiments. For requested replication, preserve all seeds, report paired gaps, and distinguish selected seed 0 from independent replications. Confidence intervals crossing zero do not establish equivalence. Do not pick a new favorable seed without disclosing selection.

Stop at the current evidence gate: do not schedule speculative follow-on simulations before interpreting the requested results. Proposed wireless/AirComp work must be a separately named change, not a retrofit into completed FedKLS runs.
