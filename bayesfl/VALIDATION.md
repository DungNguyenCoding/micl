# Validation status — evidence rather than a blanket PASS

As of 2026-10-26:

| Check | Evidence / status |
| --- | --- |
| Stage-2 organization | Uploaded terminal transcript: 121 archive folders validated, 42 protected originals retained |
| Stage-3 registry | Uploaded read-only report; paths and static references examined |
| Stage-3B organization | User reports completion; no new verify transcript supplied |
| Main plots after cleanup | User reports normal generation |
| Current complete training test suite | Not freshly run or supplied for this documentation task |
| Current source/metrics audit | Partial historical source and current inventory/terminal evidence; live export recommended |
| Documentation package | See `docs/DOCUMENTATION_VALIDATION.md` for authoring checks; not training certification |

The original baseline `13 passed, 2 skipped` report is at [history/VALIDATION.md](docs/history/pre_refresh_20261026/VALIDATION.md). It concerns an earlier baseline and ResNet/sparse-partition test profile, not the final MNIST CNN / CIFAR BasicCNN study.

Use the working interpreter for current tests:

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
/home/micl/anaconda3/envs/dungndh/bin/python -m pytest -q
```

Record actual pass/fail/skip counts with source/environment provenance. A skipped dependency module is not successful Flower/Ray validation. Running a plot validates only its selected data-reading/rendering path. See [VALIDATION_SPARSE.md](VALIDATION_SPARSE.md) for historical sparse-test scope and [docs/RUNBOOK.md](docs/RUNBOOK.md) for current operational checks.
