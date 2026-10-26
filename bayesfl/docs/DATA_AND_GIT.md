# Data, Git and cross-PC reproduction

Status: a Git-ignore strategy was discussed; no current `.gitignore` or `git status` was supplied for this refresh. **No ignore rules, index entries, history or runtime files are changed by the documentation installer.**

## Separate three tasks

| Task on another PC | Required artifacts |
| --- | --- |
| Regenerate existing paper figures | Current plot script + plotting dependencies + required raw `metrics/global_metrics.csv` files in expected run paths; preserve run/config/protocol metadata for provenance |
| Start a new matching simulation | Current code/configs, compatible environment, datasets, actual partition indices/metadata (or a separately verified identical generation), seeds and runtime controls |
| Resume an existing simulation | All of the above plus consistent committed resume arrays/JSON, the transfer ledger, metrics and relevant run state; a weights-only file is not sufficient |

A source-only clone does **not** contain historical training trajectories unless they are explicitly tracked or transferred. Model weights cannot substitute for a missing CSV curve. Conversely a metrics-only plot does not need to load model weights, but future scripts that reevaluate a model do.

## Recommended tracking policy

Track source, tests (including small required test fixtures), configs, docs, useful queues, a selected lightweight paper-results bundle and provenance. Retain full outputs locally and make a separate backup for future resume. Do not blanket-delete data because it was excluded from Git.

The chosen primary plot has **24 trajectories**; the protected main-paper category has 25 because it additionally preserves the unclipped MNIST FedAvg control. Keep exact directory names initially to avoid rewriting the plotter. Paths are in [EXPERIMENT_REGISTRY.json](EXPERIMENT_REGISTRY.json).

Possible `.gitignore` additions, **to review against the actual existing rules before applying**:

```gitignore
# Generated archives (documentation/source/results remain unpacked when tracked)
*.zip
*.tar.gz

# Heavy model/resume arrays: retain local copies, not ordinary Git objects
/outputs/**/checkpoints/
/outputs/**/resume/
/outputs/**/posterior/
/outputs/**/*.pt
/outputs/**/*.pth
/outputs/**/*.ckpt

# Large diagnostics; not needed by the selected metrics-only paper plotter
/outputs/**/metrics/client_metrics.csv
/outputs/**/metrics/round_train_metrics.csv
/outputs/**/communication/events.jsonl
```

The ledger ignore rule is **only a Git-transfer choice**: preserve it locally/offline for resume. Do not globally ignore `*.npz`: realized partitions and test fixtures also use that extension. Do not blanket-ignore all CSVs; global metrics are essential. Existing parent-directory exclusions such as `/outputs/` may still exclude metrics even when a child exception is added. Check the actual rule hierarchy and tracked file list before claiming a clean clone will plot.

A review command for one exact required CSV is:

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
git check-ignore -v -- outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s10_n20_seed0_e5_b10_lr0p10_lam0p01_r200_20261009_154443/metrics/global_metrics.csv
git ls-files -- outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s10_n20_seed0_e5_b10_lr0p10_lam0p01_r200_20261009_154443/metrics/global_metrics.csv
```

These require a Git repository. No output from the first command does not by itself prove the file is tracked. `.gitignore` does not remove already-tracked content or old Git-history blobs. Untracking with `git rm --cached` is an explicit separate index change; never use broad removal commands without reviewing the file list. Do not rewrite history as part of documentation cleanup.

## Portable paths and export

Commands currently name `/home/micl/...`; another PC must choose its own interpreter and repository root. The inspected queue runner also contains that interpreter path. Inspect the current plotter for absolute paths; moving metrics to the same relative `outputs/` layout is necessary but may not resolve every external dependency automatically.

`docs/tools/export_handoff.py --export` provides a lightweight current-code + primary-metrics ZIP outside the repo. It includes selected docs/config metadata and hashes, excludes raw datasets and weights, and is intended for new-chat/code review and figure reproduction. It is **not** a full training backup. Shared partition NPZs and complete resume trees must be separately preserved when training reproducibility/resumption is needed.

Same-SSD archives are reversible organization, not disaster-recovery backups. Consider a separately verified copy of the protected paper outputs and source on independent storage before any permanent deletion.
