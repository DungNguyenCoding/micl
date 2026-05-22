"""
Diagnostic plot/evaluation script for reproducing the pattern of Figure 4 and
Table 2 from Section 5.1 of Blundell et al., "Weight Uncertainty in Neural
Networks".

This version is designed to work with the modified BBB.py used in this project.
It keeps the original Table-2 behavior, but adds diagnostics that are important
for debugging failed reproduction attempts:

  1. Prints global SNR percentiles.
  2. Prints layer-wise SNR percentiles.
  3. Prints layer-wise pruning counts for each pruning level.
  4. Saves SNR statistics to CSV.
  5. Saves raw SNR values to .npy.
  6. Derives the default output directory from the checkpoint directory.
  7. Accepts --dataset mnist for compatibility with older commands.

Typical usage:

    python plot.py \
      --hidden 1200 \
      --model Results_paper/BBB_mnist_1200_0.0001_samples1_ID0.pth \
      --device cuda

Optional:

    python plot.py \
      --hidden 1200 \
      --model Results_paper/BBB_mnist_1200_0.0001_samples1_ID0_best.pth \
      --device cuda \
      --output-dir Results_paper/Table2_Figure4_best
"""

import argparse
import copy
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import BBB as bbbmod


# -----------------------------------------------------------------------------
# Hyperparameters and checkpoint loading
# -----------------------------------------------------------------------------


def build_hyper(hidden_units: int) -> bbbmod.BBB_Hyper:
    hyper = bbbmod.BBB_Hyper()
    hyper.dataset = "mnist"
    hyper.hidden_units = hidden_units
    bbbmod.apply_default_prior_for_hidden_units(hyper)
    return hyper


def extract_state_dict(state):
    """Accept either a raw state_dict or a checkpoint dictionary."""
    if isinstance(state, dict):
        if "state_dict" in state:
            return state["state_dict"]
        if "model_state_dict" in state:
            return state["model_state_dict"]
    return state


def default_output_dir_from_model(model_path: str) -> str:
    model_dir = os.path.dirname(os.path.abspath(model_path))
    if not model_dir:
        model_dir = os.getcwd()
    return os.path.join(model_dir, "Table2_Figure4")


# -----------------------------------------------------------------------------
# SNR helpers
# -----------------------------------------------------------------------------


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def tensor_snr(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    sigma = softplus(rho)
    return torch.abs(mu) / (sigma + 1e-12)


def collect_weight_snr(model, include_bias: bool = False) -> np.ndarray:
    """
    Compute global signal-to-noise ratio for every weight:

        SNR_i = |mu_i| / sigma_i

    where:

        sigma_i = log(1 + exp(rho_i))

    By default, biases are ignored because Table 2 reports network weights.
    """
    snrs = []

    for layer in model.layers:
        weight_snr = tensor_snr(layer.weight_mu.detach().cpu(), layer.weight_rho.detach().cpu())
        snrs.append(weight_snr.reshape(-1))

        if include_bias:
            bias_snr = tensor_snr(layer.bias_mu.detach().cpu(), layer.bias_rho.detach().cpu())
            snrs.append(bias_snr.reshape(-1))

    return torch.cat(snrs).numpy()


def collect_layerwise_snr(model, include_bias: bool = False) -> Dict[str, np.ndarray]:
    """Return SNR values separately for each layer."""
    out = {}

    for idx, layer in enumerate(model.layers):
        weight_snr = tensor_snr(layer.weight_mu.detach().cpu(), layer.weight_rho.detach().cpu())
        out[f"layer{idx}.weight"] = weight_snr.reshape(-1).numpy()

        if include_bias:
            bias_snr = tensor_snr(layer.bias_mu.detach().cpu(), layer.bias_rho.detach().cpu())
            out[f"layer{idx}.bias"] = bias_snr.reshape(-1).numpy()

    return out


def count_total_weights(model, include_bias: bool = False) -> int:
    total = 0
    for layer in model.layers:
        total += layer.weight_mu.numel()
        if include_bias:
            total += layer.bias_mu.numel()
    return total


def snr_to_db(snr: np.ndarray, db_factor: float = 20.0) -> np.ndarray:
    snr = np.asarray(snr, dtype=np.float64)
    snr = snr[np.isfinite(snr)]
    snr = np.maximum(snr, 1e-12)
    return db_factor * np.log10(snr)


def summarize_snr_array(name: str, snr: np.ndarray, db_factor: float = 20.0) -> List[Dict[str, float]]:
    snr = np.asarray(snr, dtype=np.float64)
    snr = snr[np.isfinite(snr)]
    snr = np.maximum(snr, 1e-12)
    snr_db = snr_to_db(snr, db_factor=db_factor)

    rows = []
    for q in [0, 1, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100]:
        rows.append(
            {
                "name": name,
                "percentile": q,
                "snr": float(np.percentile(snr, q)),
                f"snr_db_{int(db_factor)}log10": float(np.percentile(snr_db, q)),
            }
        )
    return rows


def print_snr_summary(name: str, snr: np.ndarray, db_factor: float = 20.0):
    print()
    print(f"SNR statistics: {name}")
    print("Percentile |          SNR |      SNR dB")
    print("-" * 42)
    for row in summarize_snr_array(name, snr, db_factor=db_factor):
        print(
            f"q{row['percentile']:>3.0f}%      | "
            f"{row['snr']:>12.6g} | "
            f"{row[f'snr_db_{int(db_factor)}log10']:>11.3f}"
        )


def save_snr_statistics(
    global_snr: np.ndarray,
    layerwise_snr: Dict[str, np.ndarray],
    output_dir: str,
    db_factor: float = 20.0,
):
    rows = []
    rows.extend(summarize_snr_array("global", global_snr, db_factor=db_factor))
    for name, values in layerwise_snr.items():
        rows.extend(summarize_snr_array(name, values, db_factor=db_factor))

    path = os.path.join(output_dir, "SNR_statistics.csv")
    fieldnames = ["name", "percentile", "snr", f"snr_db_{int(db_factor)}log10"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    np.save(os.path.join(output_dir, "SNR_values_global.npy"), global_snr)

    return path


# -----------------------------------------------------------------------------
# Figure 4
# -----------------------------------------------------------------------------


def plot_figure4(snr: np.ndarray, output_dir: str, db_factor: float = 20.0):
    """
    Generate Figure-4-style density and CDF plots of SNR in dB.

    The 75% pruning threshold is shown as a red dashed vertical line.
    db_factor is usually:
        20.0 for amplitude ratio SNR = |mu| / sigma
        10.0 if treating SNR as a power ratio
    """
    os.makedirs(output_dir, exist_ok=True)

    snr = np.asarray(snr)
    snr = snr[np.isfinite(snr)]
    snr = np.maximum(snr, 1e-12)

    cutoff_75 = float(np.percentile(snr, 75))
    snr_db = db_factor * np.log10(snr)
    cutoff_75_db = db_factor * np.log10(cutoff_75 + 1e-12)

    threshold_style = {
        "color": "red",
        "linestyle": "--",
        "linewidth": 2.5,
        "zorder": 20,
        "label": "75% pruning threshold",
    }

    # ------------------------------------------------------------------
    # Density plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))

    plt.hist(
        snr_db,
        bins=120,
        density=True,
        alpha=0.75,
        zorder=1,
    )

    plt.axvline(
        cutoff_75_db,
        **threshold_style,
    )

    plt.xlabel("Signal-to-Noise Ratio (dB)")
    plt.ylabel("Density")
    plt.title("Figure 4-style SNR Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure4_density.png"), dpi=300)
    plt.close()

    # ------------------------------------------------------------------
    # CDF plot
    # ------------------------------------------------------------------
    xs = np.sort(snr_db)
    ys = np.arange(1, len(xs) + 1) / len(xs)

    plt.figure(figsize=(6, 4))

    plt.plot(
        xs,
        ys,
        linewidth=2,
        zorder=1,
    )

    plt.axvline(
        cutoff_75_db,
        **threshold_style,
    )

    plt.axhline(
        0.75,
        color="red",
        linestyle=":",
        linewidth=1.8,
        alpha=0.8,
        zorder=15,
        label="CDF = 0.75",
    )

    plt.xlabel("Signal-to-Noise Ratio (dB)")
    plt.ylabel("CDF")
    plt.title("Figure 4-style SNR CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure4_cdf.png"), dpi=300)
    plt.close()

    # ------------------------------------------------------------------
    # Combined figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(
        snr_db,
        bins=120,
        density=True,
        alpha=0.75,
        zorder=1,
    )

    axes[0].axvline(
        cutoff_75_db,
        color="red",
        linestyle="--",
        linewidth=2.5,
        zorder=20,
        label="75% pruning threshold",
    )

    axes[0].set_xlabel("Signal-to-Noise Ratio (dB)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Density")
    axes[0].legend()

    axes[1].plot(
        xs,
        ys,
        linewidth=2,
        zorder=1,
    )

    axes[1].axvline(
        cutoff_75_db,
        color="red",
        linestyle="--",
        linewidth=2.5,
        zorder=20,
        label="75% pruning threshold",
    )

    axes[1].axhline(
        0.75,
        color="red",
        linestyle=":",
        linewidth=1.8,
        alpha=0.8,
        zorder=15,
        label="CDF = 0.75",
    )

    axes[1].set_xlabel("Signal-to-Noise Ratio (dB)")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("CDF")
    axes[1].legend()

    fig.suptitle("Figure 4-style Signal-to-Noise Ratio Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "Figure4_density_cdf.png"), dpi=300)
    plt.close(fig)

    return cutoff_75, cutoff_75_db


# -----------------------------------------------------------------------------
# Table 2 pruning
# -----------------------------------------------------------------------------


def pruning_layer_report(model, threshold: float, include_bias: bool = False) -> List[Dict[str, float]]:
    """Count how many parameters would be pruned per layer."""
    rows = []
    for idx, layer in enumerate(model.layers):
        weight_snr = tensor_snr(layer.weight_mu.detach(), layer.weight_rho.detach())
        weight_mask = weight_snr <= threshold
        weight_total = layer.weight_mu.numel()
        weight_pruned = int(weight_mask.sum().item())
        rows.append(
            {
                "name": f"layer{idx}.weight",
                "total": weight_total,
                "pruned": weight_pruned,
                "pruned_percent": 100.0 * weight_pruned / max(weight_total, 1),
            }
        )

        if include_bias:
            bias_snr = tensor_snr(layer.bias_mu.detach(), layer.bias_rho.detach())
            bias_mask = bias_snr <= threshold
            bias_total = layer.bias_mu.numel()
            bias_pruned = int(bias_mask.sum().item())
            rows.append(
                {
                    "name": f"layer{idx}.bias",
                    "total": bias_total,
                    "pruned": bias_pruned,
                    "pruned_percent": 100.0 * bias_pruned / max(bias_total, 1),
                }
            )
    return rows


def prune_by_snr(
    model,
    percentile: float,
    include_bias: bool = False,
    prune_rho: bool = True,
    rho_prune_value: float = -20.0,
):
    """
    Prune the lowest-SNR weights globally.

    For percentile=0, nothing is pruned. This fixes the common bug where the
    minimum-SNR weight is accidentally removed at 0% pruning.

    If prune_rho=True, pruned weights have:
        mu = 0
        rho = rho_prune_value

    For deterministic posterior-mean evaluation, only mu matters. Setting rho
    is useful if samples>1, because it prevents pruned weights from sampling
    nonzero noise.
    """
    pruned_model = copy.deepcopy(model)
    total = count_total_weights(model, include_bias=include_bias)

    if percentile <= 0:
        return pruned_model, float("-inf"), 0, total, []

    snr_all = collect_weight_snr(model, include_bias=include_bias)
    threshold = float(np.percentile(snr_all, percentile))
    layer_rows = pruning_layer_report(model, threshold, include_bias=include_bias)

    pruned_count = 0

    with torch.no_grad():
        for layer in pruned_model.layers:
            weight_snr = tensor_snr(layer.weight_mu, layer.weight_rho)
            weight_mask = weight_snr <= threshold
            pruned_count += int(weight_mask.sum().item())
            layer.weight_mu[weight_mask] = 0.0
            if prune_rho:
                layer.weight_rho[weight_mask] = rho_prune_value

            if include_bias:
                bias_snr = tensor_snr(layer.bias_mu, layer.bias_rho)
                bias_mask = bias_snr <= threshold
                pruned_count += int(bias_mask.sum().item())
                layer.bias_mu[bias_mask] = 0.0
                if prune_rho:
                    layer.bias_rho[bias_mask] = rho_prune_value

    active_count = total - pruned_count
    return pruned_model, threshold, pruned_count, active_count, layer_rows


@torch.no_grad()
def evaluate_accuracy(model, loader, eval_device: torch.device, samples: int = 1) -> float:
    """
    Evaluate test accuracy.

    Recommended for Table 2:
        samples = 1

    samples=1 uses the deterministic posterior-mean network.
    samples>1 averages sampled softmax probabilities.
    """
    model.eval()

    correct = 0
    total = 0

    for data, target in loader:
        data = data.to(eval_device, non_blocking=True)
        target = target.to(eval_device, non_blocking=True)

        if samples == 1:
            logits = model(data, infer=True)
            pred = logits.argmax(dim=1)
        else:
            probs_sum = None
            for _ in range(samples):
                logits = model(data, infer=False)
                probs_sample = F.softmax(logits, dim=1)
                probs_sum = probs_sample if probs_sum is None else probs_sum + probs_sample
            pred = (probs_sum / samples).argmax(dim=1)

        correct += pred.eq(target).sum().item()
        total += target.numel()

    return correct / total


def make_test_loader(batch_size: int = 1000, num_workers: int = 0, device: torch.device = torch.device("cpu")):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0 / 126.0),
        ]
    )

    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    return test_loader


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce Table 2 and Figure 4-style outputs.")

    parser.add_argument("--hidden", type=int, default=1200, help="Hidden units per layer. Use 1200 for Table 2.")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist"], help="Accepted for compatibility. Only MNIST is supported.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained BBB .pth checkpoint.")
    parser.add_argument("--output-dir", type=str, default=None, help="Default: <checkpoint_directory>/Table2_Figure4")
    parser.add_argument("--samples", type=int, default=1, help="Use 1 for deterministic Table 2 evaluation.")
    parser.add_argument("--eval-batch-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--include-bias", action="store_true", help="Also include biases in SNR/pruning. Default: weights only.")
    parser.add_argument(
        "--keep-rho-after-pruning",
        action="store_true",
        help="Only zero mu for pruned weights. Default also sets rho to --rho-prune-value.",
    )
    parser.add_argument("--rho-prune-value", type=float, default=-20.0)
    parser.add_argument("--db-factor", type=float, default=20.0, choices=[10.0, 20.0], help="dB conversion factor for Figure 4 x-axis. Pruning is unaffected.")
    parser.add_argument("--no-save-pruned", action="store_true", help="Do not save Pruned_*.pth checkpoints.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = default_output_dir_from_model(args.model)

    if args.device == "auto":
        eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        eval_device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Using device:", eval_device)
    print("Output directory:", args.output_dir)

    hyper = build_hyper(args.hidden)
    hyper.eval_batch_size = args.eval_batch_size
    hyper.num_workers = args.num_workers

    print("Loaded hyperparameters:")
    print("  hidden_units =", hyper.hidden_units)
    print("  pi =", hyper.pi)
    print("  sigma1/wide =", hyper.s1)
    print("  sigma2/narrow =", hyper.s2)
    print("  rho_init =", hyper.rho_init)

    model = bbbmod.BBB(28 * 28, 10, hyper).to(eval_device)

    state = torch.load(args.model, map_location=eval_device)
    model.load_state_dict(extract_state_dict(state))
    model.eval()

    # ------------------------------------------------------------------
    # Figure 4 and SNR diagnostics
    # ------------------------------------------------------------------
    snr = collect_weight_snr(model, include_bias=args.include_bias)
    layerwise_snr = collect_layerwise_snr(model, include_bias=args.include_bias)

    print_snr_summary("global", snr, db_factor=args.db_factor)
    for name, values in layerwise_snr.items():
        print_snr_summary(name, values, db_factor=args.db_factor)

    snr_stats_path = save_snr_statistics(
        global_snr=snr,
        layerwise_snr=layerwise_snr,
        output_dir=args.output_dir,
        db_factor=args.db_factor,
    )

    cutoff_75, cutoff_75_db = plot_figure4(snr, args.output_dir, db_factor=args.db_factor)

    print()
    print("Figure 4-style plots saved:")
    print(" ", os.path.join(args.output_dir, "Figure4_density.png"))
    print(" ", os.path.join(args.output_dir, "Figure4_cdf.png"))
    print(" ", os.path.join(args.output_dir, "Figure4_density_cdf.png"))
    print("SNR statistics saved:")
    print(" ", snr_stats_path)
    print(" ", os.path.join(args.output_dir, "SNR_values_global.npy"))
    print("75% cutoff SNR:", cutoff_75)
    print(f"75% cutoff SNR dB ({int(args.db_factor)} log10):", cutoff_75_db)

    # ------------------------------------------------------------------
    # Table 2
    # ------------------------------------------------------------------
    if args.samples != 1:
        print()
        print("WARNING: For Table 2, --samples 1 is recommended.")
        print("Since this script prunes rho by default, MC evaluation is allowed, but it is slower.")

    test_loader = make_test_loader(
        batch_size=hyper.eval_batch_size,
        num_workers=hyper.num_workers,
        device=eval_device,
    )

    prune_levels = [0, 50, 75, 95, 98]
    total_weights = count_total_weights(model, include_bias=args.include_bias)
    prune_rho = not args.keep_rho_after_pruning

    csv_path = os.path.join(args.output_dir, "Table2_results.csv")
    layer_csv_path = os.path.join(args.output_dir, "Table2_layerwise_pruning.csv")
    rows = []
    layer_rows_all = []

    print()
    print("Table 2-style pruning evaluation")
    print("Target removed | Actual removed | Weights left | Test error (%) | SNR threshold")
    print("-" * 86)

    for p in prune_levels:
        pruned_model, threshold, pruned_count, active_count, layer_rows = prune_by_snr(
            model=model,
            percentile=p,
            include_bias=args.include_bias,
            prune_rho=prune_rho,
            rho_prune_value=args.rho_prune_value,
        )
        pruned_model = pruned_model.to(eval_device)

        accuracy = evaluate_accuracy(
            model=pruned_model,
            loader=test_loader,
            eval_device=eval_device,
            samples=args.samples,
        )

        test_error = 100.0 * (1.0 - accuracy)
        actual_removed = 100.0 * pruned_count / total_weights

        print(
            f"{p:>13.0f}% | "
            f"{actual_removed:>13.3f}% | "
            f"{active_count:>12,d} | "
            f"{test_error:>14.3f} | "
            f"{threshold:.8g}"
        )

        rows.append(
            {
                "target_removed_percent": p,
                "actual_removed_percent": actual_removed,
                "weights_left": active_count,
                "test_error_percent": test_error,
                "snr_threshold": threshold,
                "include_bias": args.include_bias,
                "prune_rho": prune_rho,
                "rho_prune_value": args.rho_prune_value,
                "samples": args.samples,
            }
        )

        for lr in layer_rows:
            layer_rows_all.append(
                {
                    "target_removed_percent": p,
                    "snr_threshold": threshold,
                    **lr,
                }
            )

        if not args.no_save_pruned:
            torch.save(pruned_model.state_dict(), os.path.join(args.output_dir, f"Pruned_{p}.pth"))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_removed_percent",
                "actual_removed_percent",
                "weights_left",
                "test_error_percent",
                "snr_threshold",
                "include_bias",
                "prune_rho",
                "rho_prune_value",
                "samples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(layer_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_removed_percent",
                "snr_threshold",
                "name",
                "total",
                "pruned",
                "pruned_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(layer_rows_all)

    print()
    print("Table 2 CSV saved:")
    print(" ", csv_path)
    print("Layer-wise pruning CSV saved:")
    print(" ", layer_csv_path)
    if not args.no_save_pruned:
        print("Pruned checkpoints saved in:")
        print(" ", args.output_dir)


if __name__ == "__main__":
    main()
