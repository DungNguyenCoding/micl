# `export_handoff.py` — BayesFL / FedKLS handoff tool

**Location:** `bayesfl/docs/tools/export_handoff.py`  
**Run from:** the `bayesfl/` repository root  
**Purpose:** verify the current paper's saved numerical endpoints, or export a lightweight, timestamped copy of the code, documentation, experiment configurations, and selected results for another ChatGPT conversation or computer.

> **Important:** This tool **does not train models, change experiment results, import BayesFL, modify Git, or load model weights**. It is a standard-library Python utility. Its ZIP is a **research handoff**, **not** an independent backup of datasets, checkpoints, or resumable training state.

## 1. Quick start

Use the existing research environment on the Linux server:

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python

# Show usage and available options.
"$PY" docs/tools/export_handoff.py --help

# Check and print the saved MNIST/CIFAR-10 primary-paper results.
"$PY" docs/tools/export_handoff.py --check

# Check those results again, then create a new handoff ZIP.
"$PY" docs/tools/export_handoff.py --export
```

**Note:** `--check` is the default. Running `"$PY" docs/tools/export_handoff.py` without a mode option is equivalent to `--check`. You cannot use `--check` and `--export` together.

| Goal | Command / action | Creates files? |
| --- | --- | --- |
| Learn supported options | `"$PY" docs/tools/export_handoff.py --help` | No |
| Verify existing results | `"$PY" docs/tools/export_handoff.py --check` | **No** — terminal output only |
| Make a new chat-ready snapshot | `"$PY" docs/tools/export_handoff.py --export` | **Yes** — one new `.zip` **beside** `bayesfl/` |
| Read a repository at another location | Add `--root /absolute/path/to/bayesfl` | According to chosen mode |
| Read known percentage-valued CSV accuracies | Add `--accuracy-unit percent` | According to chosen mode |

`--accuracy-unit fraction` is the default and is **correct for the current recorded experiment CSVs**. Do not use `percent` unless you have verified that the CSV stores, for example, `89.44` instead of `0.8944`. The script does not guess the unit. **Be careful:** percent mode can accept small fraction-valued numbers while displaying them incorrectly, so verify the source column rather than relying on its range check alone.

## 2. What `--check` actually does

The tool reads `docs/EXPERIMENT_REGISTRY.json` and selects **exactly 24 runs** marked `primary_figure: true`:

- **MNIST:** 1 full-posterior/dense FOLA baseline, 1 FedAvg baseline, 5 FedKLS KL-selected sparse runs, and 5 random-selection controls.
- **CIFAR-10:** the same 12-run breakdown.

It reads each run's exact `outputs/<run_name_timestamp>/metrics/global_metrics.csv`. It checks the **registered target round**, rather than using the latest or highest-accuracy round. It also checks the stored cumulative model-array communication bytes against the accounting formula and the locked dataset communication budget. It refuses missing target rounds, missing costs, duplicated round IDs, malformed values, and inconsistent byte totals. For FOLA, the reader prefers the `fola_mean_accuracy` field when it is present; otherwise it uses the first recognized global accuracy field. It does not substitute training accuracy or load models. These checks do **not** test the training algorithm, all historical metric rows, saved checkpoints, or plot rendering.

### How to interpret the terminal results

A successful check prints two blocks: `MNIST_S10` and `CIFAR10_A0P01_F1`, followed by:

```text
PASS: 24 primary endpoints and exact byte totals read from current CSV files.
No simulation, model loading, metric modification, or inferred missing accuracy.
```

| Printed field | Meaning |
| --- | --- |
| `Full-posterior BayesFL` | Dense FOLA baseline, evaluated at **R150** |
| `FedAvg` | FedAvg baseline, evaluated at **R300** |
| `R...` / `Round` | The **exact** evaluated communication round from the saved metrics CSV |
| `Comm GB` | Accumulated logical **model-array** bytes at that checkpoint, divided by **1,000,000,000** (decimal GB, not GiB) |
| `Keep` | Fraction of model coordinates retained for sparse **uplink** transmission (`0.75`, `0.50`, `0.25`, `0.10`, `0.05`) |
| `KL %` | Test accuracy of KL-guided FedKLS at the displayed keep ratio |
| `Random %` | Test accuracy of random-coordinate sparse FOLA at the same keep ratio and affordable round |
| `KL-Random` | `KL % - Random %`, in **percentage points (pp)** |
| `KL-Dense` | `KL % - dense FOLA accuracy`, in **pp** |
| `KL-FedAvg` | `KL % - FedAvg accuracy`, in **pp** |
| `Dense minus FedAvg` | Dense FOLA accuracy minus FedAvg accuracy, in **pp** |

A positive gap means the first-named method has the higher accuracy. A difference of `+2.50 pp` is **not** a `+2.50%` relative improvement.

For example, from the server's **October 27, 2026** successful check (examples of recorded measurements, **not hardcoded outputs**):

| Dataset | Dense FOLA R150 | FedAvg R300 | KL-guided keep=0.25 | Random keep=0.25 |
| --- | ---: | ---: | ---: | ---: |
| MNIST s10 | 89.44% | 88.23% | 89.01% at R498 | 86.96% at R498 |
| CIFAR-10 α=0.01, fraction=1 | 53.10% | 47.28% | 57.53% at R499 | 44.76% at R499 |

The two common communication budgets are **2,021,090,400 bytes for MNIST** and **22,139,157,600 bytes for CIFAR-10**. Sparse checkpoints are the last affordable registered rounds; their actual totals may be slightly under budget. The code uses **single multicast downlink** accounting and per-client sparse uplink accounting. See [`../COMMUNICATION_ACCOUNTING.md`](../COMMUNICATION_ACCOUNTING.md) and [`../LOCKED_EXPERIMENTS.md`](../LOCKED_EXPERIMENTS.md).

**Not included in this check:** MNIST's five-sample alternative, other archived/ablation results, and the separate unclipped FedAvg control. This is deliberately a **24-primary-trajectory** check, not a check of every protected output folder.

## 3. What `--export` creates

`--export` first performs the same **24-run endpoint and byte checks**. If they pass, it writes a new ZIP in the **parent directory of `bayesfl/`**, for example:

```text
/home/micl/DungNDH/micl/
├── bayesfl/
│   ├── docs/tools/export_handoff.py
│   └── ...
├── bayesfl_historical_archive/
└── bayesfl_LIVE_HANDOFF_20261027_071451_587016.zip
```

The real filename includes a fresh local date, time, and microsecond suffix. Each export creates a **new file**; older exports are not overwritten. The terminal prints something like:

```text
PASS: 24 primary endpoints and exact byte totals read from current CSV files.
...
EXPORT CREATED: /home/micl/DungNDH/micl/bayesfl_LIVE_HANDOFF_<timestamp>.zip
Files copied: 1553
ZIP size: 3,937,317 bytes
Review before sharing. Existing repository files were not modified.
```

These numbers are an **example from the October 27 export**, not guaranteed constants. `Files copied` counts indexed project files; the ZIP additionally includes **three top-level handoff metadata files**.

### Archive layout and included contents

```text
bayesfl_LIVE_HANDOFF_<timestamp>.zip
├── READ_ME_FIRST.txt
├── HANDOFF_CAPTURE.json
├── HANDOFF_FILE_HASHES.json
└── bayesfl/
    ├── README.md, AGENTS.md, .gitignore, pyproject.toml, ...
    ├── src/bayesfl/                      # Python source files
    ├── scripts/                          # source, configs, queues, text files
    ├── tests/                            # source tests and small .npz fixtures
    ├── docs/                             # including docs/tools/README.md
    └── outputs/
        ├── partitions/*.json             # metadata only, not partition arrays
        └── <24 selected run folders>/
            ├── metrics/global_metrics.csv
            ├── resolved_config.yaml
            ├── source_config.yaml         # if present
            ├── partition_metadata.json    # if present
            ├── run_summary.json           # if present
            ├── environment.json/.txt       # if present
            └── communication/
                ├── layout_manifest.json   # if present
                └── protocol.json          # if present
```

**Selection rules in the current exporter:**

- Includes an explicit list of available root-level project/provenance files (such as `README.md`, `AGENTS.md`, `SPARSE_README.md`, `IMPLEMENTATION_REPORT.md`, `VALIDATION*.md`, `pyproject.toml`, `requirements.txt`, `.gitignore`, and manifest files).
- Recursively includes files with these extensions beneath `src/`, `scripts/`, `tests/`, and `docs/`: `.py`, `.sh`, `.yaml`, `.yml`, `.json`, `.md`, `.txt`, `.toml`, `.ini`, and `.cfg`. Small `.npz` files are additionally allowed **only** under `tests/fixtures/`.
- Requires each primary run's `metrics/global_metrics.csv` and `resolved_config.yaml`; a missing required file causes the export to fail. Copies the per-run supporting metadata above **when present**.
- Includes `outputs/partitions/*.json` files, **not** the corresponding `.npz` arrays.
- Excludes conventional cache/environment directories such as `__pycache__`, `.pytest_cache`, `.venv`, `venv`, `node_modules`, and `.git` during its source/docs scan. It refuses symbolic-link paths and unexpectedly large text/fixture files.

### Top-level metadata files and how to use them

**`READ_ME_FIRST.txt`** — A short entry point for the next conversation. It instructs the reader to open `bayesfl/docs/CHAT_HANDOFF.md` and `HANDOFF_CAPTURE.json` and warns that the ZIP is not a resume backup.

**`HANDOFF_CAPTURE.json`** — A record of the export, including:

| JSON field | What it tells you |
| --- | --- |
| `captured_at` | When this ZIP was generated (timestamp with timezone) |
| `repository` | Original server repository path; not a portable destination path |
| `python_executable`, `python_version` | Interpreter used to create the export |
| `installed_distribution_versions` | Detected packages (Torch, Flower, Ray, NumPy, etc.); `null` means a version wasn't found in that Python environment |
| `primary_endpoints` | One record per selected run: exact round, accuracy percentage, bytes, metric path and metric-file SHA-256 |
| `scope`, `excluded`, `snapshot_warning` | Explicit limitations and snapshot consistency caution |

In `primary_endpoints`, fields such as `accuracy_column` and `input_accuracy_unit` explain exactly which CSV column was read and how it was interpreted. `previous_documented_accuracy_percent` is historical reference information, not a replacement for the current CSV result.

**`HANDOFF_FILE_HASHES.json`** — A list of entries with `path`, `bytes`, and `sha256` for each indexed project file under `bayesfl/`. Use it to check that files extracted or read from the ZIP match the snapshot. It does not create an external off-disk backup.

## 4. What is **not** in the handoff ZIP

The snapshot intentionally excludes:

| Not exported | Why you might need it separately |
| --- | --- |
| Raw MNIST/CIFAR-10 datasets | Fresh dataset training needs them |
| Saved model weights and checkpoints | Reevaluating weights or restoring a model needs them |
| `resume/*.npz`, resume JSON and full transfer ledgers | Resuming training requires the **complete consistent saved run state** |
| Realized `outputs/partitions/*.npz` | Exact partition reproduction may require these arrays |
| Most `outputs/` experiment files and the 121 historical archived runs | These are outside the 24 primary selected trajectories |
| `logs/` and old training diagnostics | Useful for troubleshooting but not included |
| `.git/` history | The ZIP is not a Git clone or a commit history |
| Generated `plots/*.png`, `plots/*.pdf` | Figures must be regenerated from metrics if needed |

**Portability distinction:** A second computer can inspect code/docs and the bundled primary metrics. To **regenerate the paper plots**, it must also have compatible plotting dependencies and the plotting script's expected paths; inspect any hardcoded absolute paths before running on a different machine. To **train again or resume**, separately transfer/restore datasets, partitions, weights/resume state as appropriate, and verify the runtime configuration. For details, see [`../DATA_AND_GIT.md`](../DATA_AND_GIT.md).

## 5. How to inspect or verify a generated ZIP

After exporting on Linux:

```bash
cd ~/DungNDH/micl || exit 1

# Choose the newest export for review.
ZIP="$(ls -t bayesfl_LIVE_HANDOFF_*.zip | head -1)"
printf 'Inspecting: %s\n' "$ZIP"

# Show file names and sizes without extracting anything.
unzip -l "$ZIP" | less

# Check the ZIP's stored CRCs (not the full SHA-256 manifest).
unzip -t "$ZIP" | tail -2
```

You can also **verify every exported project file's SHA-256** against the included manifest, without extracting:

```bash
cd ~/DungNDH/micl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
ZIP="$(ls -t bayesfl_LIVE_HANDOFF_*.zip | head -1)"

"$PY" - "$ZIP" <<'PY'
import hashlib
import json
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    entries = json.loads(archive.read('HANDOFF_FILE_HASHES.json'))
    for entry in entries:
        data = archive.read(entry['path'])
        if len(data) != entry['bytes']:
            raise SystemExit(f"SIZE MISMATCH: {entry['path']}")
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise SystemExit(f"SHA-256 MISMATCH: {entry['path']}")
    print(f"PASS: {len(entries)} indexed project files match their SHA-256 hashes")
PY
```

To view the saved capture metadata directly:

```bash
unzip -p "$ZIP" HANDOFF_CAPTURE.json | less
```

To use the ZIP on another computer, upload it to the new ChatGPT conversation **without needing to extract it locally**. If you do extract it, extract into a **fresh directory**, never over an actively used repository:

```bash
mkdir -p ~/bayesfl_handoff_review
unzip -n "$ZIP" -d ~/bayesfl_handoff_review
```

## 6. When should I regenerate it?

A generated handoff ZIP is a **point-in-time snapshot**. It does **not** automatically synchronize with later filesystem or conversation changes.

| What changed since the last export? | Export a fresh ZIP? |
| --- | --- |
| Only the chat window changed; no code, data, or documentation changed | **No**, reuse the same ZIP |
| Source files or tests changed | **Yes** |
| Simulation YAMLs, scripts, queues, or selected run settings changed | **Yes** |
| New selected experiment results were created/extended | **Yes**; first update `EXPERIMENT_REGISTRY.json` if the selected run/target changed |
| Important research decisions, results, or next steps changed in conversation | **Yes, after writing them into `docs/`** |
| Starting a new major phase (e.g. wireless/AirComp + adaptive keep ratios) | **Yes**, once the new state is documented |

**Critical:** `--export` snapshots whatever is **currently on disk**. It **does not automatically summarize your ChatGPT conversation or rewrite** `docs/CHAT_HANDOFF.md`, `docs/LOCKED_EXPERIMENTS.md`, or other guides. Update those documents when decisions change. Also, the current exporter is fixed to the registered 24 primary endpoints: if you deliberately move to a new experiment set, its registry and/or tool must be updated and validated rather than assuming it automatically includes new runs.

### Recommended end-of-chat workflow

1. Save any code/configuration changes; verify your simulation results and plotter separately.
2. Update `docs/CHAT_HANDOFF.md`, `docs/DECISIONS_AND_NEXT_WORK.md`, `docs/LOCKED_EXPERIMENTS.md`, and any relevant run registry entries when the project state changes.
3. Run `export_handoff.py --check` from the repository root. Resolve any errors before exporting.
4. Run `export_handoff.py --export` and inspect the new ZIP (especially `HANDOFF_CAPTURE.json`).
5. Upload the **newest ZIP** to the next ChatGPT conversation within your project, and specify the current task.

Copy/paste the full commands for steps 3–4 (they do not train anything):

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
/home/micl/anaconda3/envs/dungndh/bin/python docs/tools/export_handoff.py --check
/home/micl/anaconda3/envs/dungndh/bin/python docs/tools/export_handoff.py --export
```

Suggested message for the next chat:

> Continue my FedKLS/BayesFL project from the attached lightweight handoff ZIP. First read `READ_ME_FIRST.txt`, `bayesfl/docs/CHAT_HANDOFF.md`, `bayesfl/docs/COMMAND_RULES.md`, `bayesfl/docs/LOCKED_EXPERIMENTS.md`, `bayesfl/docs/EXPERIMENT_REGISTRY.json`, and `HANDOFF_CAPTURE.json`. Treat locked simulation settings and completed results as fixed unless I request changes. Use the included current source rather than reconstructing it from older chat messages. My next task is: [describe it here].

## 7. Common errors and what to do

| Error / observation | Explanation and safe response |
| --- | --- |
| `Missing exact metric file` | A **registered** primary run's `global_metrics.csv` is absent at the exact path. Restore that file or explicitly revise the registry after checking provenance; do not silently substitute another run. |
| `INCOMPLETE/MISSING CHECKPOINT` | The required endpoint round was not found. Check real run completion; do not relabel a shorter run as complete. |
| `Byte mismatch` or `Dense endpoint does not equal budget` | The CSV's cumulative model-array bytes disagree with the registry/accounting. Check selected profiles, metrics and multicast rules before proceeding. |
| `No recognized ... accuracy field` | The selected CSV does not contain a supported accuracy column. Inspect the actual schema instead of guessing. |
| `Accuracy is invalid for explicit unit` | Confirm whether the CSV stores a fraction (`0.8944`) or percentage (`89.44`), and pass the correct `--accuracy-unit`. |
| `NOTE: ... differs from old documented endpoint` | A **warning** that current CSV and older recorded accuracy differ; examine provenance. The tool does not silently replace either value. |
| `File changed while reading` / `Metrics changed after checking` | Stop training/editing/writers and rerun. If export failed midway, **do not share the potentially incomplete ZIP**. |
| `Symbolic link not followed` / `Export refuses symlinked directory` | Paths were refused for snapshot safety. Inspect the filesystem; do not disable the check without reviewing the risk. |
| ZIP created but figures fail on another PC | The ZIP contains metrics, not plotting dependencies or every possible external absolute-path assumption. Check path resolution and install a compatible Python environment. |
| `--export` succeeds but the new chat lacks a recent decision | The decision was never saved into `docs/`; update the docs and export again. |

## 8. Scope and safety reminders

- **`--check`:** reads project CSVs and registry; **no file is written**.
- **`--export`:** reads and checks the project, writes a new ZIP **outside** the repository, then checks its ZIP CRC. It does **not** modify existing repository data or the Git index.
- The tool checks each source file's size/mtime/inode around reading, but **does not provide an atomic whole-repository snapshot**. Stop concurrent writers before export.
- The handoff may contain local paths, source/config details, and experiment metadata. **Review it before sending to third parties.**
- SHA-256 verification checks archive consistency, **not** correctness of the ML algorithm or fidelity of omitted checkpoints.
- Preserve complete original outputs and a separately verified backup for research reproducibility and training resumption. A same-SSD archive is not an independent backup.

### Related documents

- [`../START_HERE.md`](../START_HERE.md) — overall documentation map.
- [`../CHAT_HANDOFF.md`](../CHAT_HANDOFF.md) — new-chat continuation context.
- [`../COMMAND_RULES.md`](../COMMAND_RULES.md) — required command and experiment-execution conventions.
- [`../EXPERIMENT_REGISTRY.json`](../EXPERIMENT_REGISTRY.json) — exact selected run paths and target rounds.
- [`../RUNBOOK.md`](../RUNBOOK.md) — current plotting, result checks, and operational commands.
- [`../RESULTS_AND_PLOTTING.md`](../RESULTS_AND_PLOTTING.md) — experiment comparison rules and figure interpretation.
- [`../DATA_AND_GIT.md`](../DATA_AND_GIT.md) — cross-PC requirements and Git portability.
