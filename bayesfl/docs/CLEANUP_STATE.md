# Cleanup and archive state

## Stage 2 — verified in supplied terminal output

121 historical experiment/plot folders were moved to
`~/DungNDH/micl/bayesfl_historical_archive/`, grouped by classification. All 42 protected original folders were retained; no unknown folders were reported. Same-SSD rename/verification passed. The reported disk sizes were bayesfl 6.3G and archive 8.8G; the logical archived size was 8.613 GiB. Those are different measurement/rounding conventions, not proof of missing data. No SSD space was freed.

Manifest dependency: `../bayesfl_cleanup_DRY_RUN_20261026_194253.csv`.
Manager: `scripts/bayesfl_same_ssd_organizer.py`.
Keep the manifest, archive journal and exact paths; missing archived outputs need restoration or explicit path-aware analysis, not retraining by default.

42 protected folders = 25 main-paper category + 13 MNIST s5 alternative + one clipped-FOLA ablation + two incomplete N100 runs + one shared partition directory. The primary plot itself uses 24 trajectories.

Incomplete N100 s10 unclipped FOLA runs remain protected at reported R152 (seed3) and R172 (seed4). Do not relocate them or launch a new run when the intent is continuation.

## Stage 3 — audit, not a dependency proof

The supplied registry is a pre-Stage-3B read-only snapshot. Static references do not detect every runtime import, generated filename or direct terminal use. No file was declared dispensable merely because it had zero references. All source, test and config files remain important until reviewed.

## Stage 3B — completion reported by user

The user said Stage 3B was done. No separate post-Stage-3B transcript was uploaded. The approved organizer moves exactly the following 13 files under
`../bayesfl_historical_archive/STAGE3_LEGACY_SCRIPTS_QUEUES/<CATEGORY>/scripts/...`:

### HISTORICAL_VERSIONS

- `scripts/_run_gpu_queue.before_resume_support.sh`
- `scripts/migrate_multicast_accounting.before_legacy_mnist.py`
- `scripts/plot_ieee_multicast_results.py.before_cifar_f1_patch`
- `scripts/plot_ieee_multicast_results_updated_old.py`
- `scripts/plot_ieee_multicast_results_with_fedavg_curve.py.before_cifar_f1_patch`

### OBSOLETE_EMPTY_QUEUES

- `scripts/queues/cifar_plot_recovery_gpu1.txt`
- `scripts/queues/mnist_densegate_stageB2_gpu1_EMPTY.txt`

### HISTORICAL_RESUME_QUEUES

- `scripts/queues/mnist_newmlp_final_dense_confirm_seed0_gpu0.txt`
- `scripts/queues/mnist_stage11_r200400_seed0_gpu1.txt`
- `scripts/queues/multicast_resume_keep005_seed0_gpu0.txt`
- `scripts/queues/multicast_resume_keep005_seed0_gpu1.txt`
- `scripts/queues/multicast_resume_no005_seed0_gpu0.txt`
- `scripts/queues/multicast_resume_no005_seed0_gpu1.txt`

Manifest: `stage3b_13_files_manifest.json` in that archive directory. Manager: `scripts/bayesfl_stage3b_organizer.py`. Both current `cifar_sparse_r150_resume_*` queues remain in place even when empty. No bulk movement of 898 config files or other scripts was authorized.

## Verification and restore

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
"$PY" scripts/bayesfl_same_ssd_organizer.py --verify
"$PY" scripts/bayesfl_stage3b_organizer.py --verify
```

Restore is a deliberate change, not a routine verification step. Existing manager confirmation tokens are `RESTORE_121_ARCHIVE_CANDIDATES` for Stage 2 and `RESTORE_13_HISTORY_FILES` for Stage 3B. Inspect current manager help and destination collisions before use.

The user confirmed plotting still works. That supports the selected figure workflow, not a blanket claim that every training path is bug-free. No more cleanup is needed before new research. Avoid deleting caches/logs/configs “for completeness”; preserve reproducibility and focus on the requested work.
