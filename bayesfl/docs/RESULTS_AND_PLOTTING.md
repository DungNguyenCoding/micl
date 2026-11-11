# Results and plotting — preserved legacy records / separate wireless figures

The following are raw endpoints read from the supplied baseline CSVs, not smoothed
plot values and not wireless experiments. Primary data/model/seed choices remain unchanged.

## MNIST s10

Dense FOLA R150 **89.44%**; FedAvg R300 **88.23%**.

| Keep | Sparse round | KL % | Random % | KL − Random pp | KL − Dense pp |
|---:|---:|---:|---:|---:|---:|
| 0.75 | 193 | 88.86 | 88.59 | +0.27 | -0.58 |
| 0.50 | 278 | 88.19 | 88.08 | +0.11 | -1.25 |
| 0.25 | 498 | 89.01 | 86.96 | +2.05 | -0.43 |
| 0.10 | 950 | 88.44 | 86.54 | +1.90 | -1.00 |
| 0.05 | 1362 | 88.58 | 86.07 | +2.51 | -0.86 |
## CIFAR-10 alpha .01 / f1

Dense FOLA R150 **53.10%**; FedAvg R300 **47.28%**.

| Keep | Sparse round | KL % | Random % | KL − Random pp | KL − Dense pp |
|---:|---:|---:|---:|---:|---:|
| 0.75 | 193 | 53.70 | 52.73 | +0.97 | +0.60 |
| 0.50 | 278 | 53.58 | 50.07 | +3.51 | +0.48 |
| 0.25 | 499 | 57.53 | 44.76 | +12.77 | +4.43 |
| 0.10 | 950 | 53.42 | 38.73 | +14.69 | +0.32 |
| 0.05 | 1362 | 51.98 | 33.63 | +18.35 | -1.12 |

## Figure and inference boundaries

Existing `plot_ieee_multicast_results_mnist_s10_stage22.py --mnist-fedavg clipped`
is unchanged; it produces Acc_Com and Acc_KeepRatio using the chosen legacy profiles.
Communication curves use the existing 7-point display smoothing, no raw faded overlay,
and the MNIST zoom. Raw tables are independent of smoothing. Legend positioning remains
whatever is implemented in the supplied current plot script; this release does not
redesign it. Do not merge the old deleted MNIST FedAvg88.12% with recovered88.23%.

These are selected single-seed results, not an across-seed statistical claim. MNIST
baseline clipping and CIFAR LR/schedule differences must be disclosed. Extra affordable
rounds contribute to legacy gains; the curves do not establish lower computation or
reduced overfitting.

Wireless results do not exist yet for real datasets in this release. Use
`summarize_wireless_results.py` or `summarize_wireless_study.py` for dynamic bytes and
modeled times. `plot_wireless_results.py` produces separate raw figures at a fresh
destination and does not touch existing Acc_Com/Acc_KeepRatio. Verify common physical
scenario, slot policy, seed/partition/learning settings and covered time checkpoints
before claiming equal-time comparisons. The 150:300 legacy relation is not a wireless rule.
