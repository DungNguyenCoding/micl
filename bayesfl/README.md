# BayesFL / FedKLS

**Project documentation updated: 2026-10-26.** Research code for dense FedAvg, full-posterior FOLA/BayesFL and KL-guided sparse posterior uplinks. The current final figures use MNIST CNN s10/N20/seed0 and CIFAR-10 BasicCNN alpha=.01/full-data/N20/seed0 at **dense FOLA R150 multicast communication budgets**.

## Read first

- [Current project entry point](docs/START_HERE.md)
- [New-chat handoff](docs/CHAT_HANDOFF.md)
- [Locked simulation settings](docs/LOCKED_EXPERIMENTS.md)
- [Command rules and execution preferences](docs/COMMAND_RULES.md)
- [Full documentation index](docs/README.md)

## Regenerate the chosen figures

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
/home/micl/anaconda3/envs/dungndh/bin/python \
  scripts/plot_ieee_multicast_results_mnist_s10_stage22.py \
  --mnist-fedavg clipped
```

This reads existing result CSVs; it does not retrain. Expected current filenames are `plots/Acc_Com.{pdf,png}` and `plots/Acc_KeepRatio.{pdf,png}`. Code alone is not sufficient on another PC: transfer the required metrics at their expected paths. See [data/Git policy](docs/DATA_AND_GIT.md).

## Important distinctions

The selected MNIST dense FOLA and sparse runs are **unclipped**; its primary FedAvg is the recovered **clipped Stage23** run. CIFAR uses clipped methods but different FOLA/FedAvg LR schedules. These choices must be disclosed, not hidden by generic figure labels.

Sparsity applies to post-fit uploads, not local network training. Raw precision is transmitted with means. The downlink is a modeled single multicast broadcast. No physical wireless/AirComp simulation, adaptive ratio or compute-saving claim is implied.

The [current results page](docs/RESULTS_AND_PLOTTING.md) separates known MNIST endpoints, older CIFAR R120 numbers and still-unextracted CIFAR R150 accuracies. The [evidence page](docs/EVIDENCE_AND_GAPS.md) distinguishes reported runs from fresh code/test validation.

## Preservation

Runtime code, configs, outputs and dependency files are not changed by this docs refresh. The original uploaded docs are preserved in [docs/history](docs/history/README.md). Root `CHANGE_MANIFEST.json` and `PACKAGE_MANIFEST.txt` are **historical release records**, not a current inventory. The new `DOCS_UPDATE_MANIFEST.json` covers the documentation package only.

Repository organization: Stage2 archived121 historical folders while retaining42 protected folders; Stage3B completion of the13-file historical cleanup was user-reported. Do not delete or relocate additional configs/scripts without dependency review. See [cleanup state](docs/CLEANUP_STATE.md).
