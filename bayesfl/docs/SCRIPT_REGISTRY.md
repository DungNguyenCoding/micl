# Script and queue map

Registry observations are from the uploaded Stage-3 report; Stage-3B completion is user-reported. This is not an executable scheduler.

| Path (under scripts/) | Role / handling |
| --- | --- |
| _run_gpu_queue.sh | Core serial queue executor; keep original path. |
| _run_nohup.sh | General launcher; registry shows other scripts reference it. |
| plot_ieee_multicast_results_mnist_s10_stage22.py | Current primary two-panel plotter; keep path. |
| mnist_stage18_unclipped_fola.py | Selected dense-FOLA family generator; do not blindly rerun. |
| mnist_stage22_sparse_s10_seed0.py | Primary MNIST sparse generator. |
| mnist_stage21_sparse_s5_seed3.py | Preserved s5 alternative generator. |
| prepare_cifar_r150_sparse_resume.py | CIFAR budget continuation/status tool; all targets previously complete. |
| recover_cifar_plot.py | CIFAR baseline recovery provenance/tool. |
| mnist_stage24_recover_clipped_fola_s10.py | Clipped FOLA ablation recovery; not primary dense curve. |
| extract_cifar_sparse_results.py | Result helper; inspect code/targets before treating as R150 current. |
| print_main_sparse_summary.py | Result helper; inspect chosen paths/targets. |
| generate_sparse_configs.py | Generic original generator; old default profiles/budgets are NOT current locks. |
| migrate_multicast_accounting.py | Historical accounting migration; do not reapply casually. |
| bayesfl_cleanup_dry_run.py | Original classification audit; preserve manifest dependencies. |
| bayesfl_same_ssd_organizer.py | Stage-2 verify/restore manager. |
| bayesfl_stage3_registry.py | Static dependency/report audit, not an unused-code detector. |
| bayesfl_stage3b_organizer.py | Stage-3B 13-file organizer/verify/restore; user reports completion. |

## Current queue locations

```text
scripts/queues/mnist_stage22_cnn_s10_n20_seed0_gpu0_kl_global_local.txt
scripts/queues/mnist_stage22_cnn_s10_n20_seed0_gpu1_random.txt
scripts/queues/mnist_stage21_cnn_s5_n20_seed3_gpu0_kl_global_local.txt
scripts/queues/mnist_stage21_cnn_s5_n20_seed3_gpu1_random.txt
scripts/queues/mnist_stage18_fola_noclip_gpu0.txt
scripts/queues/mnist_stage18_fola_noclip_gpu1.txt
scripts/queues/mnist_stage23_recovery_gpu0.txt
scripts/queues/cifar_sparse_r150_resume_gpu0_kl_global_local.txt
scripts/queues/cifar_sparse_r150_resume_gpu1_random.txt
```

Current CIFAR source YAML families include `sparse_main_seed0/`, `comm_budget_extension_seed0/`, `confirm_newsrc_dense/` and `cifar_plot_recovery/`. These folders mix current, historical and duplicate-name templates. Do not move a whole family just because some runs were archived. Current MNIST config families include `mnist_stage18_fola_noclip_cnn/`, `mnist_stage22_sparse_cnn_s10_seed0/`, `mnist_stage23_recovery/`, plus s5 `mnist_stage21_sparse_cnn_s5_seed3/`.

Six historical resume/mixed queues and two obsolete empty queues were selected for Stage-3B archival. The two empty **current CIFAR R150** queues were intentionally retained. A queue can contain both valid and stale jobs; do not blindly rewrite all references to the archive location and launch it.

## Historical scripts still worth preserving

Stage-14/15/16/20 generators and analysis helpers, old MLP comparisons, BBB launchers, alternate plotters, smoke scripts and environment validators retain research/reproduction value. `refs=0` only means no detected literal reference in the audit. A direct terminal call or dynamically built path may still use them.

The pre-Stage-3B audit recorded 73 top-level script/tool files, 898 files under configs, 131 queue .txt files and 839 job records. Of configs, 43 matched active run names, 120 archived names, 728 no static run match, and seven had no literal run_name. These are **historical audit counts**, not hard-coded expectations for a changing repository. Stage-3B moved five top-level versions and eight queues, and adding the organizer itself also changes counts.

See [evidence/STAGE3_REGISTRY_20261026_211438.txt](evidence/STAGE3_REGISTRY_20261026_211438.txt) for the full pre-move list, [CLEANUP_STATE.md](CLEANUP_STATE.md) for the exact 13-file move, and [RUN_REGISTRY.md](RUN_REGISTRY.md) for run-to-YAML candidates.
