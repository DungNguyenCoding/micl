# Execution Scripts

This folder keeps reproducible shell entry points outside the source root.
All scripts automatically `cd` to the repository root, so they can be launched from any working directory.

## Layout

```text
scripts/
├── train/   # full training runs that create outputs/
├── tune/    # hyperparameter sweeps that create outputs/tune_*
└── plot/    # offline plotting commands that create plots/
```

## Naming convention

```text
<file_type>_<method>_<dataset>_<setting>_<hyperparameter_tag>[_seed].sh
```

Examples:

```text
train_vi_mnist_noniid_unbalanced_sparse_update_snr_keep010_decay_seed42.sh
plot_sparse_vi_mnist_noniid_unbalanced_update_snr_ratio_sweep_seed42.sh
tune_ola_mnist_noniid_unbalanced_fixed_cf005.sh
```

## Main training scripts

| Script | Output folder |
|---|---|
| `train/train_fedavg_mnist_noniid_unbalanced_dense_seed42.sh` | `outputs/final_compare_mnist_noniid_unbalanced/fedavg_seed42` |
| `train/train_ola_mnist_noniid_unbalanced_dense_seed42.sh` | `outputs/final_compare_mnist_noniid_unbalanced/ola_seed42` |
| `train/train_vi_mnist_noniid_unbalanced_dense_seed42.sh` | `outputs/final_compare_mnist_noniid_unbalanced/vi_seed42` |
| `train/train_vi_mnist_noniid_unbalanced_decay_seed42.sh` | `outputs/vi_mnist_stabilized_decay_seed42` |
| `train/train_vi_mnist_noniid_unbalanced_sparse_update_snr_keep010_decay_seed42.sh` | `outputs/vi_sparse_keep010_decay_seed42` |
| `train/train_vi_mnist_noniid_unbalanced_sparse_update_snr_sweep_seed42.sh` | `outputs/sparse_comm_mnist_noniid_unbalanced/vi_*` |
| `train/train_ola_mnist_noniid_unbalanced_sparse_precision_update_sweep_seed42.sh` | `outputs/sparse_comm_mnist_noniid_unbalanced/ola_*` |

## Main plotting scripts

| Script | Output folder |
|---|---|
| `plot/plot_compare_fedavg_ola_mnist_noniid_unbalanced_seed42.sh` | `plots/final_compare_mnist_noniid_unbalanced/fa_vs_ola` |
| `plot/plot_compare_fedavg_vi_mnist_noniid_unbalanced_seed42.sh` | `plots/final_compare_mnist_noniid_unbalanced/fa_vs_vi` |
| `plot/plot_sparse_vi_mnist_noniid_unbalanced_update_snr_ratio_sweep_seed42.sh` | `plots/sparse_comm_mnist_noniid_unbalanced/vi_ratio_sweep` |
| `plot/plot_sparse_ola_mnist_noniid_unbalanced_precision_update_ratio_sweep_seed42.sh` | `plots/sparse_comm_mnist_noniid_unbalanced/ola_ratio_sweep` |
| `plot/plot_diagnostics_mnist_noniid_unbalanced_research_v1.sh` | `plots/research_diagnostics_mnist_noniid_unbalanced` |

## Running with nohup

From the repository root:

```bash
mkdir -p logs
nohup bash scripts/train/train_vi_mnist_noniid_unbalanced_sparse_update_snr_keep010_decay_seed42.sh \
  > logs/train_vi_sparse_keep010_decay_seed42.log 2>&1 &
```

Monitor:

```bash
tail -f logs/train_vi_sparse_keep010_decay_seed42.log
```

Check running jobs:

```bash
pgrep -af "main.py|scripts/"
```
