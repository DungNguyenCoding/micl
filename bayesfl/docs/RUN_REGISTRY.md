# Exact output and configuration registry

As of 2026-10-26. These names come from user preflight and the cleanup/Stage-3 reports, not guessed timestamps. For automation use [EXPERIMENT_REGISTRY.json](EXPERIMENT_REGISTRY.json). The primary two-panel figure uses 24 runs (12 per dataset); a protected extra unclipped MNIST FedAvg reference makes the cleanup “main paper” category 25. All protected folders total 42, including partitions.

Candidate configuration matches are by literal run name only. `resolved_config.yaml` inside the exact run is authoritative for what ran; `source_config.yaml` and migration/resume backups preserve history. Multiple candidate YAMLs are not automatically interchangeable.

## mnist_s10

### dense_fola_clipped_ablation

- Primary figure: false; plotted round: 150; last metric round in audit: 200.
- Exact output: `outputs/run_mnist_stage24_cnn16_32_fixed1c_s10_n20_fola_clip_seed0_e5_b10_lr0p10_lam0p01_r200_20261011_094131`
- Metrics: `outputs/run_mnist_stage24_cnn16_32_fixed1c_s10_n20_fola_clip_seed0_e5_b10_lr0p10_lam0p01_r200_20261011_094131/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage24_recover_clipped_fola_s10/run_mnist_stage24_cnn16_32_fixed1c_s10_n20_fola_clip_seed0_e5_b10_lr0p10_lam0p01_r200.yaml`

### dense_fola

- Primary figure: true; plotted round: 150; last metric round in audit: 200.
- Exact output: `outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s10_n20_seed0_e5_b10_lr0p10_lam0p01_r200_20261009_154443`
- Metrics: `outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s10_n20_seed0_e5_b10_lr0p10_lam0p01_r200_20261009_154443/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage18_fola_noclip_cnn/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s10_n20_seed0_e5_b10_lr0p10_lam0p01_r200.yaml`

### kl_sparse; r=0.05

- Primary figure: true; plotted round: 1362; last metric round in audit: 1362.
- Exact output: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p05_budget150_r1362_20261011_023714`
- Metrics: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p05_budget150_r1362_20261011_023714/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage22_sparse_cnn_s10_seed0/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p05_budget150_r1362.yaml`

### kl_sparse; r=0.10

- Primary figure: true; plotted round: 950; last metric round in audit: 950.
- Exact output: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p10_budget150_r950_20261011_011043`
- Metrics: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p10_budget150_r950_20261011_011043/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage22_sparse_cnn_s10_seed0/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p10_budget150_r950.yaml`

### kl_sparse; r=0.25

- Primary figure: true; plotted round: 498; last metric round in audit: 498.
- Exact output: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p25_budget150_r498_20261011_003912`
- Metrics: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p25_budget150_r498_20261011_003912/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage22_sparse_cnn_s10_seed0/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p25_budget150_r498.yaml`

### kl_sparse; r=0.50

- Primary figure: true; plotted round: 278; last metric round in audit: 278.
- Exact output: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p50_budget150_r278_20261011_002442`
- Metrics: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p50_budget150_r278_20261011_002442/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage22_sparse_cnn_s10_seed0/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p50_budget150_r278.yaml`

### kl_sparse; r=0.75

- Primary figure: true; plotted round: 193; last metric round in audit: 193.
- Exact output: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p75_budget150_r193_20261011_001511`
- Metrics: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p75_budget150_r193_20261011_001511/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage22_sparse_cnn_s10_seed0/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p75_budget150_r193.yaml`

### random_sparse; r=0.05

- Primary figure: true; plotted round: 1362; last metric round in audit: 1362.
- Exact output: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p05_budget150_r1362_20261011_023914`
- Metrics: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p05_budget150_r1362_20261011_023914/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage22_sparse_cnn_s10_seed0/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p05_budget150_r1362.yaml`

### random_sparse; r=0.10

- Primary figure: true; plotted round: 950; last metric round in audit: 950.
- Exact output: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p10_budget150_r950_20261011_011143`
- Metrics: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p10_budget150_r950_20261011_011143/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage22_sparse_cnn_s10_seed0/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p10_budget150_r950.yaml`

### random_sparse; r=0.25

- Primary figure: true; plotted round: 498; last metric round in audit: 498.
- Exact output: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p25_budget150_r498_20261011_003942`
- Metrics: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p25_budget150_r498_20261011_003942/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage22_sparse_cnn_s10_seed0/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p25_budget150_r498.yaml`

### random_sparse; r=0.50

- Primary figure: true; plotted round: 278; last metric round in audit: 278.
- Exact output: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p50_budget150_r278_20261011_002512`
- Metrics: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p50_budget150_r278_20261011_002512/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage22_sparse_cnn_s10_seed0/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p50_budget150_r278.yaml`

### random_sparse; r=0.75

- Primary figure: true; plotted round: 193; last metric round in audit: 193.
- Exact output: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p75_budget150_r193_20261011_001511`
- Metrics: `outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p75_budget150_r193_20261011_001511/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage22_sparse_cnn_s10_seed0/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p75_budget150_r193.yaml`

### fedavg_baseline

- Primary figure: true; plotted round: 300; last metric round in audit: 400.
- Exact output: `outputs/run_mnist_stage23_cnn16_32_fixed1c_s10_n20_fedavg_clip_seed0_e5_b10_lr0p10_r400_20261011_091458`
- Metrics: `outputs/run_mnist_stage23_cnn16_32_fixed1c_s10_n20_fedavg_clip_seed0_e5_b10_lr0p10_r400_20261011_091458/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage23_recovery/run_mnist_stage23_cnn16_32_fixed1c_s10_n20_fedavg_clip_seed0_e5_b10_lr0p10_r400.yaml`

### fedavg_unclipped_reference

- Primary figure: false; plotted round: 300; last metric round in audit: 400.
- Exact output: `outputs/run_stage16_paper00_fedavg_cnn16_32_s10_n20_seed0_e5_b10_lr0p10_r400_20261009_115708`
- Metrics: `outputs/run_stage16_paper00_fedavg_cnn16_32_s10_n20_seed0_e5_b10_lr0p10_r400_20261009_115708/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage16_paper00_fedavg/run_stage16_paper00_fedavg_cnn16_32_s10_n20_seed0_e5_b10_lr0p10_r400.yaml`

## cifar10_a0p01_f1

### fedavg_baseline

- Primary figure: true; plotted round: 300; last metric round in audit: 300.
- Exact output: `outputs/run_budgetmatch_cifar_basiccnn_a0p01_fedavg_n20_f1_seed0_e10_b32_lr001_budget150densefola_20260927_103154`
- Metrics: `outputs/run_budgetmatch_cifar_basiccnn_a0p01_fedavg_n20_f1_seed0_e10_b32_lr001_budget150densefola_20260927_103154/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/cifar_plot_recovery/run_budgetmatch_cifar_basiccnn_a0p01_fedavg_n20_f1_seed0_e10_b32_lr001_budget150densefola.yaml`
  - `scripts/configs/comm_budget_extension_seed0/run_budgetmatch_cifar_basiccnn_a0p01_fedavg_n20_f1_seed0_e10_b32_lr001_budget150densefola.yaml`

### dense_fola

- Primary figure: true; plotted round: 150; last metric round in audit: 150.
- Exact output: `outputs/run_confirm_newsrc_cifar_basiccnn_a0p01_fola_cos400_n20_f1_seed0_e10_b32_lr002_lam0p01_r150_20261010_195905`
- Metrics: `outputs/run_confirm_newsrc_cifar_basiccnn_a0p01_fola_cos400_n20_f1_seed0_e10_b32_lr002_lam0p01_r150_20261010_195905/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/cifar_plot_recovery/run_confirm_newsrc_cifar_basiccnn_a0p01_fola_cos400_n20_f1_seed0_e10_b32_lr002_lam0p01_r150.yaml`
  - `scripts/configs/confirm_newsrc_dense/run_confirm_newsrc_cifar_basiccnn_a0p01_fola_cos400_n20_f1_seed0_e10_b32_lr002_lam0p01_r150.yaml`

### kl_sparse; r=0.05

- Primary figure: true; plotted round: 1362; last metric round in audit: 1362.
- Exact output: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p05_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260927_193202`
- Metrics: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p05_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260927_193202/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/comm_budget_extension_seed0/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p05_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense.yaml`

### kl_sparse; r=0.10

- Primary figure: true; plotted round: 950; last metric round in audit: 950.
- Exact output: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p1_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_071300`
- Metrics: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p1_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_071300/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/sparse_main_seed0/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p1_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense.yaml`

### kl_sparse; r=0.25

- Primary figure: true; plotted round: 499; last metric round in audit: 499.
- Exact output: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p25_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_090138`
- Metrics: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p25_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_090138/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/sparse_main_seed0/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p25_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense.yaml`

### kl_sparse; r=0.50

- Primary figure: true; plotted round: 278; last metric round in audit: 278.
- Exact output: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p5_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260924_194221`
- Metrics: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p5_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260924_194221/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/sparse_main_seed0/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p5_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense.yaml`

### kl_sparse; r=0.75

- Primary figure: true; plotted round: 193; last metric round in audit: 193.
- Exact output: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p75_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260924_194221`
- Metrics: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p75_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260924_194221/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/sparse_main_seed0/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p75_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense.yaml`

### random_sparse; r=0.05

- Primary figure: true; plotted round: 1362; last metric round in audit: 1362.
- Exact output: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p05_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260928_024119`
- Metrics: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p05_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260928_024119/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/comm_budget_extension_seed0/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p05_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense.yaml`

### random_sparse; r=0.10

- Primary figure: true; plotted round: 950; last metric round in audit: 950.
- Exact output: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p1_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_192526`
- Metrics: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p1_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_192526/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/sparse_main_seed0/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p1_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense.yaml`

### random_sparse; r=0.25

- Primary figure: true; plotted round: 499; last metric round in audit: 499.
- Exact output: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p25_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_194658`
- Metrics: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p25_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_194658/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/sparse_main_seed0/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p25_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense.yaml`

### random_sparse; r=0.50

- Primary figure: true; plotted round: 278; last metric round in audit: 278.
- Exact output: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p5_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_043532`
- Metrics: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p5_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_043532/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/sparse_main_seed0/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p5_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense.yaml`

### random_sparse; r=0.75

- Primary figure: true; plotted round: 193; last metric round in audit: 193.
- Exact output: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p75_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_032226`
- Metrics: `outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p75_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_032226/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/sparse_main_seed0/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p75_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense.yaml`

## mnist_s5_alternative

### fedavg_baseline

- Primary figure: false; plotted round: 300; last metric round in audit: 400.
- Exact output: `outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fedavg_seed3_e5_b5_lr0p10_r300_20261008_171748`
- Metrics: `outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fedavg_seed3_e5_b5_lr0p10_r300_20261008_171748/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage15_cnn_multiseed/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fedavg_seed3_e5_b5_lr0p10_r300.yaml`

### dense_fola

- Primary figure: false; plotted round: 150; last metric round in audit: 200.
- Exact output: `outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s5_n20_seed3_e5_b5_lr0p10_lam0p01_r200_20261009_153613`
- Metrics: `outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s5_n20_seed3_e5_b5_lr0p10_lam0p01_r200_20261009_153613/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage18_fola_noclip_cnn/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s5_n20_seed3_e5_b5_lr0p10_lam0p01_r200.yaml`

### kl_sparse; r=0.05

- Primary figure: false; plotted round: 1362; last metric round in audit: 1362.
- Exact output: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p05_budget150_r1362_20261010_165629`
- Metrics: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p05_budget150_r1362_20261010_165629/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage21_sparse_cnn_s5_seed3/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p05_budget150_r1362.yaml`

### kl_sparse; r=0.10

- Primary figure: false; plotted round: 950; last metric round in audit: 950.
- Exact output: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p10_budget150_r950_20261010_153027`
- Metrics: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p10_budget150_r950_20261010_153027/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage21_sparse_cnn_s5_seed3/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p10_budget150_r950.yaml`

### kl_sparse; r=0.25

- Primary figure: false; plotted round: 498; last metric round in audit: 498.
- Exact output: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p25_budget150_r498_20261010_145857`
- Metrics: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p25_budget150_r498_20261010_145857/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage21_sparse_cnn_s5_seed3/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p25_budget150_r498.yaml`

### kl_sparse; r=0.50

- Primary figure: false; plotted round: 278; last metric round in audit: 278.
- Exact output: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p50_budget150_r278_20261010_144426`
- Metrics: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p50_budget150_r278_20261010_144426/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage21_sparse_cnn_s5_seed3/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p50_budget150_r278.yaml`

### kl_sparse; r=0.75

- Primary figure: false; plotted round: 193; last metric round in audit: 193.
- Exact output: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p75_budget150_r193_20261010_143526`
- Metrics: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p75_budget150_r193_20261010_143526/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage21_sparse_cnn_s5_seed3/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p75_budget150_r193.yaml`

### random_sparse; r=0.05

- Primary figure: false; plotted round: 1362; last metric round in audit: 1362.
- Exact output: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p05_budget150_r1362_20261010_165559`
- Metrics: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p05_budget150_r1362_20261010_165559/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage21_sparse_cnn_s5_seed3/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p05_budget150_r1362.yaml`

### random_sparse; r=0.10

- Primary figure: false; plotted round: 950; last metric round in audit: 950.
- Exact output: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p10_budget150_r950_20261010_153027`
- Metrics: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p10_budget150_r950_20261010_153027/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage21_sparse_cnn_s5_seed3/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p10_budget150_r950.yaml`

### random_sparse; r=0.25

- Primary figure: false; plotted round: 498; last metric round in audit: 498.
- Exact output: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p25_budget150_r498_20261010_145826`
- Metrics: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p25_budget150_r498_20261010_145826/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage21_sparse_cnn_s5_seed3/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p25_budget150_r498.yaml`

### random_sparse; r=0.50

- Primary figure: false; plotted round: 278; last metric round in audit: 278.
- Exact output: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p50_budget150_r278_20261010_144426`
- Metrics: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p50_budget150_r278_20261010_144426/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage21_sparse_cnn_s5_seed3/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p50_budget150_r278.yaml`

### random_sparse; r=0.75

- Primary figure: false; plotted round: 193; last metric round in audit: 193.
- Exact output: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p75_budget150_r193_20261010_143526`
- Metrics: `outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p75_budget150_r193_20261010_143526/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage21_sparse_cnn_s5_seed3/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p75_budget150_r193.yaml`

### fedavg_unclipped_reference

- Primary figure: false; plotted round: 300; last metric round in audit: 400.
- Exact output: `outputs/run_stage16_paper00_fedavg_cnn16_32_s5_n20_seed3_e5_b5_lr0p10_r400_20261009_142030`
- Metrics: `outputs/run_stage16_paper00_fedavg_cnn16_32_s5_n20_seed3_e5_b5_lr0p10_r400_20261009_142030/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage16_paper00_fedavg/run_stage16_paper00_fedavg_cnn16_32_s5_n20_seed3_e5_b5_lr0p10_r400.yaml`

## n100_incomplete

### resume_sensitive

- Primary figure: false; plotted round: not selected; last metric round in audit: 152.
- Exact output: `outputs/run_mnist_stage20_n100_cnn_s10_seed3_fola_noclip_r200_20261010_134606`
- Metrics: `outputs/run_mnist_stage20_n100_cnn_s10_seed3_fola_noclip_r200_20261010_134606/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage20_n100_cnn/run_mnist_stage20_n100_cnn_s10_seed3_fola_noclip_r200.yaml`

### resume_sensitive

- Primary figure: false; plotted round: not selected; last metric round in audit: 172.
- Exact output: `outputs/run_mnist_stage20_n100_cnn_s10_seed4_fola_noclip_r200_20261010_133836`
- Metrics: `outputs/run_mnist_stage20_n100_cnn_s10_seed4_fola_noclip_r200_20261010_133836/metrics/global_metrics.csv`
- Candidate source configs:
  - `scripts/configs/mnist_stage20_n100_cnn/run_mnist_stage20_n100_cnn_s10_seed4_fola_noclip_r200.yaml`

<!-- BEGIN JOINT_FEDKLS_CIFAR10_R150_VERIFIED_20261110 -->
### 2026-11-10 — Joint-only CIFAR-10 R150 audit (supersedes earlier pending CIFAR/Flower statuses)

- **Primary research method:** FedKLS joint KL-guided bandwidth and integer coordinate retention. Equal bandwidth is a historical engineering pilot, **not the focus** of the current research.
- **Verified run:** `run_wireless_cifar_eqb_joint150_fixed_v1_cifar10_a0p01_f1_joint_seed0_20261109_230207`; **R150 accuracy 52.63%**, NLL 1.40449; best 52.76% at R148; final-20-round mean 52.1115%.
- **Source audit:** `PASS_ENGINEERING_RECORD_INTEGRITY`; 150 rounds, 3,000 client-round rows, 5,579 price queries, 11,438,601,248 model-array bytes and 111,770,677 control bytes. All 161 included exported-evidence hashes independently verified.
- **Limits:** One CIFAR-10 seed 0 and fixed wireless path-loss scenario; no globally certified integer solution; power/control/compute times remain modeled, not physical measurements; no independent checkpoint-tensor examination.
- **Evidence & detailed findings:** [CIFAR-10 Joint R150 Review](wireless/r150_joint_20261110/JOINT_R150_REVIEW.md) (includes control breakdown, retention, learning curves and normalized KL contribution correction).
- **Derived audit caveat:** Previously generated per-client `weighted_discarded_kl` audit CSV contains `n_k*tail` rather than `(n_k/N)*tail`; the production round objective is correct. Use the corrected per-client CSV included in the R150 review package.

<!-- END JOINT_FEDKLS_CIFAR10_R150_VERIFIED_20261110 -->

<!-- BEGIN FEDKLS_JOINT_SENSITIVITY_R150_V1 -->
## Joint CIFAR-10 wireless sensitivity R150 — verified 2026-11-12

- Main method: **joint-KL** bandwidth/retention; 20 clients, CIFAR BasicCNN, seed 0, fixed channel gains, 150 rounds. The original joint reference (Bmax=1 MHz, U=58.973 s) reached **52.63%**. Six additional fixed-channel R150 cases passed engineering record-integrity audits.
- **U sweep** at Bmax=1 MHz: U=48.973 s → 50.25%; U=38.973 s → 52.47%; U=28.973 s → 43.43%.
- **B sweep** at U=58.973 s: Bmax=0.55 MHz → 47.29%; 0.75 MHz → 52.53%; 0.95 MHz → 54.01%.
- All seven source-audit ZIPs independently matched their **161 SHA-256 evidence hashes**. Configurations match except run_name, uplink bandwidth and round target. The R150 learning outcomes are **non-monotone**, and the two harshest cases show pronounced learning oscillations.
- Price-search `price_tolerance` in U150 **21/150** and B0550 **27/150** rounds. This is a solver stopping caution, not an audit failure; global integer optimality is not certified.
- This is a **single-seed engineering simulation**. The reference scenario is `EXAMPLE_NOT_LOCKED`, not physically calibrated. Shorter U also means shorter modeled round duration; compare both accuracy vs rounds and modeled time.

**Full verified audit and results:** `docs/wireless/JOINT_CIFAR10_SENSITIVITY_R150_20261112.md`  
**Reproducible plotting guide:** `docs/wireless/JOINT_SENSITIVITY_PLOTS_RUNBOOK_v1.md`  
**Plot script:** `scripts/plot_joint_cifar_sensitivity_v1.py` (generates 4 side-by-side PNG/PDF figures + CSVs without retraining).
<!-- END FEDKLS_JOINT_SENSITIVITY_R150_V1 -->
