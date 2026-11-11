# Execution and response rules

Project root: `~/DungNDH/micl/bayesfl`. Interpreter:
`/home/micl/anaconda3/envs/dungndh/bin/python`. Use
`PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"`; do not assume base Conda Python.
Keep the working Python3.10/CUDA/dependency stack. Do not silently upgrade packages.

New simulation instructions must include configuration/two-queue generation, current
loader validation, shared realized-partition checks/preparation, nohup launch with one
serial queue/GPU and timestamps/PIDs, and matching terminal-only result extraction.
Do not automatically write numerical summaries to .txt/CSV or use tee unless requested.
Normal execution logs, requested plots and requested documentation/audit files are allowed.

Clone the selected saved configs. E<=10 unless explicitly changed. No silent LR/clipping/
scheduler/model/data changes, no overwrite of output/partition data, no concurrent writers,
no restarting completed experiments to fill a summary, no broad process-kill commands.
Names for normal/recovery experiments start `run_`; runtime smoke tests start `smoke_`.

## Two controllers / two queue grammars

Legacy: `scripts/_run_gpu_queue.sh` accepts config paths or
`RESUME|RUN_DIR|TOTAL_ROUNDS|TOTAL_BYTE_BUDGET|EXPECTED_ENDPOINT`.
Wireless: `scripts/_run_wireless_queue.sh` accepts config paths or
`WRESUME|ABSOLUTE_RUN_DIR|TOTAL_ROUNDS|TOTAL_MODELED_TIME_CAP_OR_none`.
The wireless `none` token leaves an existing cap unchanged, not unbounded.
Do not interchange runners or interpret total limits as additional rounds.

Before a queue, check actual active processes and GPU occupancy. Empty queue plus
SKIP_COMPLETE can be correct; QUEUE FINISHED alone is not experiment success.
Wireless failures/infeasibility must remain visible. Never drop a client, redraw a
channel, expand a deadline or delete spent accounting to force a run to finish.

## Change and documentation discipline

Use a preview/backup installer or a separate extracted copy; stop training before source
changes. Read actual current code/configs before claiming API support. New source changes
must include tests, relevant docs and CHAT_HANDOFF updates. Keep old run provenance.
Use real raw endpoints, not smoothed display values or invented missing measurements.

Legacy figures are equal bytes. Wireless fixed-slot studies are equal modeled time only
when the compared checkpoints/scenarios genuinely match. Numerical solver gaps are not
accuracy/significance/optimality certificates. A skipped Flower or CUDA test is not passed.
Physical example YAMLs require explicit consent and must not become paper locks by naming.
