# Results provenance and paper plotting

Current main plotter: `scripts/plot_ieee_multicast_results_mnist_s10_stage22.py`.
The user confirmed plotting works after repository organization. This documentation refresh does **not** replace or re-execute that plotter.

## Current figure contract

Both panels use a **dense FOLA R150** byte reference and **FedAvg R300**. MNIST is s10/seed0; CIFAR is alpha=.01/local-data-fraction 1.0/seed0. Labels identify dense FOLA as **Full-posterior BayesFL** and KL sparse FOLA as **FedKLS**.

Accuracy–communication: two dataset panels, decimal GB x-axis, global test accuracy (%) y-axis; seven-point moving-average trajectories; no faded raw lines; a final-stage MNIST zoom inset. The requested shared legend has two centered rows:

```text
FedKLS (r = 0.75) | FedKLS (r = 0.5) | FedKLS (r = 0.25) | FedKLS (r = 0.10) | FedKLS (r = 0.05)
                         Full-posterior BayesFL | FedAvg
```

Seed and clipping details are omitted from figure labels **by request**, not from scientific disclosure. Preserve them in the experiment setup/caption or accompanying text. Do not silently change legend, smoothing, inset or model/run choices while editing docs. A seven-point window spans different byte widths for different keep ratios; the smoothed line is a visualization, not the raw endpoint evidence or a confidence interval.

Accuracy–keep-ratio figure: x-axis is **sparsity** `(1-r)*100`, with 0,25,50,75,90,95%; dense FOLA is the shared 0% sparsity endpoint of KL/random curves. Do not relabel the axis as retention while retaining these tick values.

Files: `plots/Acc_Com.pdf`, `plots/Acc_Com.png`, `plots/Acc_KeepRatio.pdf`, `plots/Acc_KeepRatio.png` (latest requested names). Some older script versions used long `fig_accuracy_vs_...` names; the current on-disk save calls decide the actual outputs.

## MNIST s10: reported raw endpoints at current R150 budget

Source: user's actual plot-preflight table in this conversation; not recalculated from raw CSV in the documentation environment.

Dense FOLA Stage-18 R150: **89.44%**. Selected recovered clipped FedAvg Stage-23 R300: **88.23%**. Gap: **+1.21 percentage points**.

| Keep r | Sparse round | KL % | Random % | KL−Random pp | KL−Dense pp | KL−FedAvg pp |
| --- | --- | --- | --- | --- | --- | --- |
| 0.75 | 193 | 88.86 | 88.59 | +0.27 | -0.58 | +0.63 |
| 0.50 | 278 | 88.19 | 88.08 | +0.11 | -1.25 | -0.04 |
| 0.25 | 498 | 89.01 | 86.96 | +2.05 | -0.43 | +0.78 |
| 0.10 | 950 | 88.44 | 86.54 | +1.90 | -1.00 | +0.21 |
| 0.05 | 1362 | 88.58 | 86.07 | +2.51 | -0.86 | +0.35 |

KL exceeds random at every tested ratio in this selected case. Its best sparse endpoint is r=.25 at 89.01%; none exceeds dense FOLA 89.44%. This does not establish a universal optimum or statistical significance.

The earlier deleted Stage-12 clipped FedAvg run was reported at **88.12%** at R300. The recovered Stage-23 run is **88.23%**; use the latter for the selected current figure. An unclipped Stage-16 control was reported at 88.64% R300 (+0.80 pp gap versus dense FOLA). These are separate records, not interchangeable versions of one number.

## CIFAR-10: current R150 completion is known, final numbers need extraction

Current budget is **22,139,157,600 bytes**. Required checkpoints are dense R150, FedAvg R300 and sparse **193,278,499,950,1362**. The cleanup/status reports show these endpoints are available. **No R150-budget accuracy table was supplied for this refresh.** Do not fill it with R120 values.

Historical R120 budget = 17,711,326,080 bytes. At that budget dense FOLA was 51.24% R120 and FedAvg 46.04% R240; KL/random were:

| r | Round | KL % | Random % |
| --- | --- | --- | --- |
| .75 | 154 | 51.83 | 51.17 |
| .50 | 222 | 52.34 | 48.31 |
| .25 | 399 | 56.10 | 44.58 |
| .10 | 760 | 52.19 | 38.99 |
| .05 | 1089 | 53.27 | 32.14 |

These are **historical R120-budget results only**; source: [saved evidence](evidence/CIFAR_R120_RESULTS_HISTORICAL.txt). The JSON registry stores null, not guesses, for missing current CIFAR accuracy values. Use `docs/tools/export_handoff.py --check` to print the actual current table on the server.

## MNIST s5 alternative: preserve, do not substitute into s10

Seed3, dense R150=84.85%, clipped FedAvg R300=83.55%. At keep .75/.50/.25/.10/.05, KL endpoints are 84.92/84.66/84.38/83.94/85.06%; random endpoints are 83.73/83.85/82.53/82.13/82.29%. These were user-reported measurements. They belong to the **s5 alternative**, not the primary s10 plot. Paths are retained in the registry.

## Interpretation constraints for the paper

MNIST's primary FOLA/sparse methods are unclipped while its selected FedAvg is clipped. CIFAR uses clipping for both, but FOLA and FedAvg have different LRs/schedules. Disclose these differences; neither comparison isolates the server aggregation rule.

Equal communication permits extra sparse training rounds; it does not imply equal compute, data exposure, latency or energy. Full local models still train. Greater accuracy does not by itself prove reduced overfitting or improved uncertainty calibration. Do not claim universal superiority from selected seed0/seed3 experiments.

The earlier five-seed unclipped dense overview at R150/R300 reported mean paired gaps +0.26 pp (s5, 95% interval [-0.36,+0.89]) and +0.12 pp (s10, [-0.68,+0.92]). This is conversation-reported descriptive context, not a new calculation and not the selected single-seed figure result. The intervals do not establish superiority or equivalence.

When drafting result text, describe the overall trajectory, then how decreasing retention changes KL/random behavior and the trade-off between information retained and extra rounds. Use a few supporting numbers; do not replace the interpretation with a list of endpoints.
