# Joint FedKLS: R150 side-by-side sensitivity plotting runbook

## Prerequisites

- Seven completed R150 **joint** studies in the live `bayesfl` repository, with their original `run_summary.json`, `resolved_config.yaml`, `metrics/global_metrics.csv` and `metrics/wireless_rounds.csv` retained.
- Existing Miniconda environment `dungndh`, containing `matplotlib` and `PyYAML` (the same environment used for BayesFL).
- The plotting script installed at `scripts/plot_joint_cifar_sensitivity_v1.py`.
- No training is needed; **this is read-only against all seven source runs**. It refuses an existing output directory rather than overwriting prior plots.

## Validate source data (does not create files)

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" scripts/plot_joint_cifar_sensitivity_v1.py --root "$PWD" --check-only
```

Expected: `PASS: 7 matched joint runs` and a seven-row numeric table. If any run is absent, has duplicate output directories, or differs unexpectedly in learned configuration, the command refuses to plot. The `reference` case is the **joint** portion of `cifar_eqb_joint150_fixed_v1`; the equal-bandwidth trajectory is intentionally excluded.

## Generate all figures / CSVs once

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
OUT="plots/wireless/joint_cifar_sensitivity_r150_$(date +%Y%m%d_%H%M%S)"
"$PY" scripts/plot_joint_cifar_sensitivity_v1.py --root "$PWD" --output-dir "$OUT"
echo "PLOTS: $PWD/$OUT"
ls -lh "$OUT"
```

Output (each chart in `.png` and vector `.pdf`):

1. `joint_sensitivity_side_by_side`: **2 columns × 4 rows**: accuracy vs rounds, NLL vs rounds, mean keep ratio vs rounds, and cumulative recorded **array + control** GB vs rounds. Left: U=48.97, 38.97, 28.97 s at Bmax=1 MHz. Right: Bmax=0.55,0.75,0.95 MHz at U=58.97 s. Both include dashed original 1-MHz/U58.97 **joint** reference.
2. `joint_accuracy_vs_modeled_time`: same two sweeps vs modeled elapsed hours (critical because U cases have different round durations).
3. `joint_discarded_kl_by_round`: log-scale KL trajectory from R2, excluding the enormous R1 initial-precision-flooring artifact. **These curves come from different training trajectories**, not an identical-posterior test.
4. `joint_resource_endpoints`: R150 accuracy and retention versus normalized Bmax × U resource.

Two CSVs (`joint_sensitivity_endpoints.csv`, `joint_sensitivity_trajectories.csv`) and `joint_sensitivity_provenance.json` provide source attribution and exact numeric values. `README_PLOTS.md` describes assumptions.

## Interpretation rules

- Comparisons at equal **round** do not imply equal modeled **time** for different U settings.
- This is a single-seed, fixed-path-loss, model-only communication study with hypothetical compute/control timing.
- Joint solution status `price_tolerance` is not a failed round; `numerical_gap_target` is not a proof of global integer optimality. Status counts appear in the endpoint CSV.
- Report model arrays and control payload separately; do not label their sum *measured wireless RF traffic*.
- The initial high KL value from precision scoring is excluded from KL plots; source CSV retains R1.
- Do not edit original experiment CSVs, YAMLs, checkpoints, or existing plot files.

## Optional: Reproduce figures from the seven audited ZIPs (off-server)

```bash
python scripts/plot_joint_cifar_sensitivity_v1.py \
  --audit-dir /path/with/all/seven/audit/zips \
  --output-dir /a/new/plot/folder
```

This mode checks **every source SHA-256 digest** in the ZIPs. It requires exactly the six `bayesfl_joint_R150_audit_20261112_*.zip` archives plus the original `bayesfl_joint_R150_audit_20261110_200059.zip`, accessible in one directory. It never writes into those ZIPs.
