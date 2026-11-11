# Current runbook

## Existing primary figures and results

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" docs/tools/export_handoff.py --check
"$PY" scripts/plot_ieee_multicast_results_mnist_s10_stage22.py --mnist-fedavg clipped
```

The endpoint check is read-only. Plotting overwrites its selected generated figure files,
not experiment metrics. Selected profiles remain MNIST s10 and CIFAR alpha.01/f1 with
FOLA R150/FedAvg R300 byte endpoints. Existing training/clean-resume commands are unchanged.

## New wireless path

Use the complete [wireless runbook](wireless/RUNBOOK.md): CPU tests, actual Flower/Ray
runtime gates, reviewed physical scenario, fresh study configs, partition preparation,
two serial GPU queues, terminal summaries, separate figures and clean wireless resume.
No long queue is required simply to install this source or create a handoff.

## New-chat snapshots

```bash
"$PY" docs/tools/export_handoff.py --export
```

This default includes current code/docs plus the 24 legacy primary results. For new
wireless trajectories, repeat `--wireless-run-dir outputs/ACTUAL_RUN`. Add
`--wireless-only` only when intentionally skipping the legacy-primary requirement.
See [tools/README.md](tools/README.md). Update decisions/docs first; regenerate after
meaningful changes; reuse the ZIP only while the project state is unchanged.

## Installation/hash records

The wireless overlay's verifier checks its release files against its manifest. Historical
`install_docs.py --verify` from the old documentation package will no longer report all
old hashes identical after intentional updates. Do not rerun that old installer to
"repair" current docs. Read `WIRELESS_CHANGE_MANIFEST.json` and the wireless installation
receipt instead. No manifest alone proves the training algorithm is bug-free.
