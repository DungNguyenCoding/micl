"""
Bayes by Backprop MNIST training script for reproducing the pattern of
Figure 4 and Table 2 in Section 5.1 of:

    Blundell et al., "Weight Uncertainty in Neural Networks"

This version is meant to replace BBB.py.  It keeps the original command style

    python BBB.py 1200 mnist

but adds diagnostics and knobs that are important for reproducing the pruning
behavior in Table 2:

  1. Correct scale-mixture prior order:
       pi * N(0, sigma1^2) + (1-pi) * N(0, sigma2^2), sigma1 > sigma2.
  2. Stable log-mixture prior using logsumexp.
  3. Network returns logits; likelihood uses cross_entropy.
  4. Optional small posterior-mean initialization.  This is important because
     Kaiming mu with rho=-7 starts with extremely high SNR and often gives a
     deterministic posterior that prunes badly.
  5. Optional KL scale for debugging posterior sparsity.
  6. Deterministic posterior-mean validation can be used for selecting the
     best checkpoint, while MC validation/test is still printed.
  7. SNR diagnostics during training.

Recommended reproduction/debugging runs:

    # First stable sanity run with lower initial SNR
    python BBB.py 1200 mnist \
      --kl-weighting equal \
      --mu-init small \
      --mu-init-scale 0.01 \
      --rho-init -5 \
      --epochs 600 \
      --batch-size 128 \
      --lr 1e-4 \
      --device cuda \
      --output-dir Results_equal_smallinit \
      --id 0

    # If pruning is still too destructive, increase KL pressure
    python BBB.py 1200 mnist \
      --kl-weighting paper \
      --mu-init small \
      --mu-init-scale 0.01 \
      --rho-init -5 \
      --lr 1e-5 \
      --epochs 600 \
      --batch-size 128 \
      --lr 1e-4 \
      --device cuda \
      --output-dir Results_paper_smallinit_kl2 \
      --id 0

After training, run plot.py on the final checkpoint and the best checkpoint.
"""

import argparse
import csv
import math
import os
import random
from copy import deepcopy
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets


# -----------------------------------------------------------------------------
# Reproducibility and device
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# A module-level default is useful when this file is imported by plot.py.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------


class BBB_Hyper(object):
    def __init__(self):
        self.dataset = "mnist"  # mnist || fmnist || cifar10

        self.lr = 1e-4
        self.momentum = 0.95
        self.hidden_units = 400

        self.mixture = True

        # These are overwritten by apply_paper_hyperparams().
        # The paper's scale mixture is:
        #   pi * N(0, sigma1^2) + (1-pi) * N(0, sigma2^2)
        # with sigma1 > sigma2.
        self.pi = 0.25
        self.s1 = float(np.exp(-1))   # wide prior component, sigma1
        self.s2 = float(np.exp(-6))   # narrow prior component, sigma2
        self.rho_init = -8.0

        # Important: Kaiming mu + rho=-7 starts with a very high SNR posterior.
        # For Table-2-style pruning, small mu initialization is usually better.
        self.mu_init = "small"        # small || kaiming || uniform
        self.mu_init_scale = 0.01
        self.rho_init_std = 0.05

        self.multiplier = 1.0

        self.max_epoch = 600
        self.n_samples = 1
        self.n_test_samples = 10
        self.batch_size = 128
        self.eval_batch_size = 1000

        # "paper" implements the exponentially decaying minibatch weights.
        # "equal" is useful as a debugging baseline.
        self.kl_weighting = "equal"  # equal || paper
        self.kl_scale = 1.0

        # Which validation metric should choose *_best.pth?
        # mean: deterministic posterior-mean network, stable for checkpointing.
        # mc:   stochastic posterior predictive average.
        self.best_by = "mean"  # mean || mc

        self.seed = 0
        self.valid_fraction = 1.0 / 6.0
        self.num_workers = 0
        self.output_dir = "Results"

        # Diagnostic frequency. Set to 0 to disable.
        self.snr_log_every = 25


def apply_paper_hyperparams(hyper: BBB_Hyper) -> BBB_Hyper:
    """
    Hyperparameter choices used by the original code family for this experiment.

    Format:
        hidden_units: (pi, rho_init, s1_code, s2_code)

    Then:
        sigma1 = exp(-s1_code)   # wide component
        sigma2 = exp(-s2_code)   # narrow component

    Important:
        sigma1 must be larger than sigma2.
    """
    top = {
        400: (0.25, -8.0, 1, 6),
        800: (0.25, -7.0, 0, 8),
        1200: (0.25, -7.0, 1, 8),
    }

    if hyper.hidden_units not in top:
        raise ValueError(
            "Unsupported hidden_units. Use one of: "
            + ", ".join(str(k) for k in sorted(top.keys()))
        )

    pi, rho, s1_code, s2_code = top[hyper.hidden_units]

    hyper.pi = float(pi)
    hyper.rho_init = float(rho)
    hyper.s1 = float(np.exp(-s1_code))
    hyper.s2 = float(np.exp(-s2_code))

    if not hyper.s1 > hyper.s2:
        raise ValueError(
            f"Prior scale order is wrong: sigma1={hyper.s1}, sigma2={hyper.s2}. "
            "Expected sigma1 > sigma2."
        )

    return hyper


# -----------------------------------------------------------------------------
# Probability helpers
# -----------------------------------------------------------------------------


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def log_gaussian(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Elementwise log N(x | mu, sigma^2)."""
    sigma_t = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    mu_t = torch.as_tensor(mu, dtype=x.dtype, device=x.device)

    return (
        -0.5 * math.log(2.0 * math.pi)
        - torch.log(sigma_t)
        - ((x - mu_t) ** 2) / (2.0 * sigma_t ** 2)
    )


def log_gaussian_rho(x: torch.Tensor, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """
    Elementwise log q(w | theta), where:

        sigma = log(1 + exp(rho))
        w = mu + sigma * epsilon
    """
    sigma = softplus(rho)

    return (
        -0.5 * math.log(2.0 * math.pi)
        - torch.log(sigma + 1e-12)
        - ((x - mu) ** 2) / (2.0 * sigma ** 2 + 1e-12)
    )


def log_mixture_prior(input_tensor: torch.Tensor, pi: float, sigma1: float, sigma2: float) -> torch.Tensor:
    """
    Numerically stable log scale-mixture prior:

        log[ pi * N(0, sigma1^2) + (1-pi) * N(0, sigma2^2) ]

    sigma1 is the wide component and sigma2 is the narrow component.
    """
    if not sigma1 > sigma2:
        raise ValueError(f"Expected sigma1 > sigma2, got sigma1={sigma1}, sigma2={sigma2}")

    log_prob1 = math.log(pi) + log_gaussian(input_tensor, 0.0, sigma1)
    log_prob2 = math.log(1.0 - pi) + log_gaussian(input_tensor, 0.0, sigma2)

    return torch.logsumexp(torch.stack([log_prob1, log_prob2], dim=0), dim=0)


def minibatch_beta(batch_id: int, num_batches: int, mode: str = "equal") -> float:
    """
    KL weighting for minibatch training.

    equal:
        beta = 1 / num_batches

    paper:
        beta_i = 2^(M-i) / (2^M - 1), i = 1, ..., M
        implemented stably as 2^(-i) / (1 - 2^(-M)), where i starts at 1.
    """
    if mode == "equal":
        return 1.0 / float(num_batches)

    if mode == "paper":
        denominator = 1.0 - 2.0 ** (-num_batches)
        return (2.0 ** (-(batch_id + 1))) / denominator

    raise ValueError("Unknown kl_weighting mode: " + str(mode))


# -----------------------------------------------------------------------------
# Bayesian neural network
# -----------------------------------------------------------------------------


class BBBLayer(nn.Module):
    def __init__(self, n_input: int, n_output: int, hyper: BBB_Hyper):
        super(BBBLayer, self).__init__()

        self.n_input = n_input
        self.n_output = n_output

        self.s1 = hyper.s1
        self.s2 = hyper.s2
        self.pi = hyper.pi
        self.mixture = hyper.mixture

        self.weight_mu = nn.Parameter(torch.empty(n_output, n_input))
        self.bias_mu = nn.Parameter(torch.empty(n_output))

        self.reset_mu_parameters(hyper)

        self.weight_rho = nn.Parameter(
            torch.empty(n_output, n_input).normal_(hyper.rho_init, hyper.rho_init_std)
        )
        self.bias_rho = nn.Parameter(
            torch.empty(n_output).normal_(hyper.rho_init, hyper.rho_init_std)
        )

        self.lpw = torch.tensor(0.0)
        self.lqw = torch.tensor(0.0)

    def reset_mu_parameters(self, hyper: BBB_Hyper) -> None:
        if hyper.mu_init == "kaiming":
            nn.init.kaiming_uniform_(self.weight_mu, nonlinearity="relu")
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)
        elif hyper.mu_init == "small":
            nn.init.normal_(self.weight_mu, mean=0.0, std=hyper.mu_init_scale)
            nn.init.zeros_(self.bias_mu)
        elif hyper.mu_init == "uniform":
            nn.init.uniform_(self.weight_mu, -hyper.mu_init_scale, hyper.mu_init_scale)
            nn.init.uniform_(self.bias_mu, -hyper.mu_init_scale, hyper.mu_init_scale)
        else:
            raise ValueError("Unknown mu_init: " + str(hyper.mu_init))

    def forward(self, data: torch.Tensor, infer: bool = False) -> torch.Tensor:
        """
        infer=True:
            use posterior means only.

        infer=False:
            sample weights using the reparameterization trick and update lpw/lqw.
        """
        if infer:
            return F.linear(data, self.weight_mu, self.bias_mu)

        weight_sigma = softplus(self.weight_rho)
        bias_sigma = softplus(self.bias_rho)

        epsilon_w = torch.randn_like(self.weight_mu)
        epsilon_b = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + weight_sigma * epsilon_w
        bias = self.bias_mu + bias_sigma * epsilon_b

        output = F.linear(data, weight, bias)

        self.lqw = (
            log_gaussian_rho(weight, self.weight_mu, self.weight_rho).sum()
            + log_gaussian_rho(bias, self.bias_mu, self.bias_rho).sum()
        )

        if self.mixture:
            # Correct paper order: pi * wide + (1-pi) * narrow.
            self.lpw = (
                log_mixture_prior(weight, self.pi, self.s1, self.s2).sum()
                + log_mixture_prior(bias, self.pi, self.s1, self.s2).sum()
            )
        else:
            self.lpw = (
                log_gaussian(weight, 0.0, self.s1).sum()
                + log_gaussian(bias, 0.0, self.s1).sum()
            )

        return output


class BBB(nn.Module):
    def __init__(self, n_input: int, n_output: int, hyper: BBB_Hyper):
        super(BBB, self).__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.hidden_units = hyper.hidden_units

        self.layers = nn.ModuleList(
            [
                BBBLayer(n_input, hyper.hidden_units, hyper),
                BBBLayer(hyper.hidden_units, hyper.hidden_units, hyper),
                BBBLayer(hyper.hidden_units, n_output, hyper),
            ]
        )

    def forward(self, data: torch.Tensor, infer: bool = False) -> torch.Tensor:
        """Return logits, not softmax probabilities."""
        output = data.view(-1, self.n_input)
        output = F.relu(self.layers[0](output, infer=infer))
        output = F.relu(self.layers[1](output, infer=infer))
        output = self.layers[2](output, infer=infer)
        return output

    def get_lpw_lqw(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lpw = self.layers[0].lpw + self.layers[1].lpw + self.layers[2].lpw
        lqw = self.layers[0].lqw + self.layers[1].lqw + self.layers[2].lqw
        return lpw, lqw


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


@torch.no_grad()
def collect_weight_snr_tensor(model: BBB, include_bias: bool = False) -> torch.Tensor:
    snrs = []
    for layer in model.layers:
        weight_sigma = softplus(layer.weight_rho.detach())
        weight_snr = torch.abs(layer.weight_mu.detach()) / (weight_sigma + 1e-12)
        snrs.append(weight_snr.reshape(-1).float().cpu())
        if include_bias:
            bias_sigma = softplus(layer.bias_rho.detach())
            bias_snr = torch.abs(layer.bias_mu.detach()) / (bias_sigma + 1e-12)
            snrs.append(bias_snr.reshape(-1).float().cpu())
    return torch.cat(snrs)


@torch.no_grad()
def posterior_snr_summary(model: BBB) -> Dict[str, float]:
    snr = collect_weight_snr_tensor(model, include_bias=False).numpy()
    snr = np.maximum(snr[np.isfinite(snr)], 1e-12)
    snr_db = 20.0 * np.log10(snr)
    return {
        "snr_q1_db": float(np.percentile(snr_db, 1)),
        "snr_q25_db": float(np.percentile(snr_db, 25)),
        "snr_q50_db": float(np.percentile(snr_db, 50)),
        "snr_q75_db": float(np.percentile(snr_db, 75)),
        "snr_q95_db": float(np.percentile(snr_db, 95)),
    }


def format_snr_summary(summary: Dict[str, float]) -> str:
    return (
        f"SNRdB q1/q25/q50/q75/q95 = "
        f"{summary['snr_q1_db']:.2f}/"
        f"{summary['snr_q25_db']:.2f}/"
        f"{summary['snr_q50_db']:.2f}/"
        f"{summary['snr_q75_db']:.2f}/"
        f"{summary['snr_q95_db']:.2f}"
    )


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


def sample_elbo_terms(model: BBB, hyper: BBB_Hyper, data: torch.Tensor, target: torch.Tensor):
    """Monte Carlo estimate of log prior, log posterior, and log likelihood."""
    s_log_pw = 0.0
    s_log_qw = 0.0
    s_log_likelihood = 0.0

    for _ in range(hyper.n_samples):
        logits = model(data, infer=False)
        sample_log_pw, sample_log_qw = model.get_lpw_lqw()
        sample_log_likelihood = -F.cross_entropy(logits, target, reduction="sum")

        s_log_pw = s_log_pw + sample_log_pw / hyper.n_samples
        s_log_qw = s_log_qw + sample_log_qw / hyper.n_samples
        s_log_likelihood = s_log_likelihood + sample_log_likelihood / hyper.n_samples

    # Retained for compatibility with the base code.
    s_log_likelihood = s_log_likelihood * hyper.multiplier

    return s_log_pw, s_log_qw, s_log_likelihood


# Backwards-compatible alias for code that calls probs().
def probs(model: BBB, hyper: BBB_Hyper, data: torch.Tensor, target: torch.Tensor):
    return sample_elbo_terms(model, hyper, data, target)


def ELBO(
    l_pw: torch.Tensor,
    l_qw: torch.Tensor,
    l_likelihood: torch.Tensor,
    beta: float,
    kl_scale: float = 1.0,
) -> torch.Tensor:
    kl = kl_scale * beta * (l_qw - l_pw)
    return kl - l_likelihood


def train_epoch(model: BBB, optimizer, loader, hyper: BBB_Hyper, train_device: torch.device, do_train: bool = True):
    if do_train:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    kl_sum = 0.0
    nll_sum = 0.0
    num_batches = len(loader)

    for batch_id, (data, target) in enumerate(loader):
        data = data.to(train_device, non_blocking=True)
        target = target.to(train_device, non_blocking=True)

        if do_train:
            optimizer.zero_grad(set_to_none=True)

        beta = minibatch_beta(batch_id=batch_id, num_batches=num_batches, mode=hyper.kl_weighting)

        with torch.set_grad_enabled(do_train):
            l_pw, l_qw, l_likelihood = sample_elbo_terms(model, hyper, data, target)
            loss = ELBO(l_pw, l_qw, l_likelihood, beta, kl_scale=hyper.kl_scale)

            if do_train:
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss detected at batch {batch_id}: "
                        f"loss={loss.item()}, "
                        f"l_pw={l_pw.item()}, "
                        f"l_qw={l_qw.item()}, "
                        f"l_likelihood={l_likelihood.item()}"
                    )

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0, error_if_nonfinite=True
                )

                optimizer.step()

                # Keep posterior standard deviations in a safe numerical range.
                # rho=-12 gives sigma≈6e-6; rho=2 gives sigma≈2.13.
                with torch.no_grad():
                    for layer in model.layers:
                        layer.weight_rho.clamp_(-12.0, 2.0)
                        layer.bias_rho.clamp_(-12.0, 2.0)

        loss_sum += loss.item() / num_batches
        kl_sum += (l_qw - l_pw).item() / num_batches
        nll_sum += (-l_likelihood).item() / num_batches

    if do_train:
        return loss_sum, kl_sum, nll_sum

    return kl_sum


@torch.no_grad()
def evaluate(model: BBB, loader, eval_device: torch.device, infer: bool = True, samples: int = 1) -> float:
    """
    Return accuracy in [0, 1].

    samples == 1:
        deterministic posterior mean if infer=True.

    samples > 1:
        Monte Carlo average over softmax probabilities.
    """
    model.eval()

    correct = 0
    total = 0

    for data, target in loader:
        data = data.to(eval_device, non_blocking=True)
        target = target.to(eval_device, non_blocking=True)

        if samples == 1:
            logits = model(data, infer=infer)
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


def clone_state_dict_to_cpu(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def BBB_run(hyper: BBB_Hyper, train_loader, valid_loader, test_loader, n_input: int, n_output: int, run_id: int = 0):
    print("Using device:", device)
    print("Training configuration:")
    for key, value in sorted(hyper.__dict__.items()):
        print(f"  {key} = {value}")

    model = BBB(n_input, n_output, hyper).to(device)

    print("Initial posterior diagnostic:")
    print(" ", format_snr_summary(posterior_snr_summary(model)))

    optimizer = torch.optim.SGD(model.parameters(), lr=hyper.lr, momentum=hyper.momentum)

    train_losses = np.zeros(hyper.max_epoch)
    train_kls = np.zeros(hyper.max_epoch)
    train_nlls = np.zeros(hyper.max_epoch)
    valid_mean_accs = np.zeros(hyper.max_epoch)
    test_mean_accs = np.zeros(hyper.max_epoch)
    valid_mc_accs = np.zeros(hyper.max_epoch)
    test_mc_accs = np.zeros(hyper.max_epoch)
    snr_q50_db = np.zeros(hyper.max_epoch)
    snr_q75_db = np.zeros(hyper.max_epoch)

    best_score = -1.0
    best_state = None

    for epoch in range(hyper.max_epoch):
        train_loss, train_kl, train_nll = train_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            hyper=hyper,
            train_device=device,
            do_train=True,
        )

        # Stable deterministic posterior-mean evaluation.
        valid_mean_acc = evaluate(model, valid_loader, device, infer=True, samples=1)
        test_mean_acc = evaluate(model, test_loader, device, infer=True, samples=1)

        # Stochastic posterior predictive evaluation, closer to Bayesian testing.
        valid_mc_acc = evaluate(model, valid_loader, device, infer=False, samples=hyper.n_test_samples)
        test_mc_acc = evaluate(model, test_loader, device, infer=False, samples=hyper.n_test_samples)

        score = valid_mean_acc if hyper.best_by == "mean" else valid_mc_acc
        if score > best_score:
            best_score = score
            best_state = clone_state_dict_to_cpu(model)

        summary = posterior_snr_summary(model)
        snr_q50_db[epoch] = summary["snr_q50_db"]
        snr_q75_db[epoch] = summary["snr_q75_db"]

        msg = (
            f"Epoch {epoch + 1:03d}/{hyper.max_epoch} | "
            f"Loss {train_loss:.4f} | KLraw {train_kl:.2f} | NLL {train_nll:.2f} | "
            f"ValidMeanErr {100.0 * (1.0 - valid_mean_acc):.3f}% | "
            f"TestMeanErr {100.0 * (1.0 - test_mean_acc):.3f}% | "
            f"ValidMCErr {100.0 * (1.0 - valid_mc_acc):.3f}% | "
            f"TestMCErr {100.0 * (1.0 - test_mc_acc):.3f}%"
        )

        if hyper.snr_log_every and ((epoch + 1) % hyper.snr_log_every == 0 or epoch == 0):
            msg += " | " + format_snr_summary(summary)

        print(msg)

        train_losses[epoch] = train_loss
        train_kls[epoch] = train_kl
        train_nlls[epoch] = train_nll
        valid_mean_accs[epoch] = valid_mean_acc
        test_mean_accs[epoch] = test_mean_acc
        valid_mc_accs[epoch] = valid_mc_acc
        test_mc_accs[epoch] = test_mc_acc

    os.makedirs(hyper.output_dir, exist_ok=True)

    path = os.path.join(
        hyper.output_dir,
        "BBB_"
        + hyper.dataset
        + "_"
        + str(hyper.hidden_units)
        + "_"
        + str(hyper.lr)
        + "_samples"
        + str(hyper.n_samples)
        + "_ID"
        + str(run_id),
    )

    with open(path + ".csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", lineterminator="\n")
        writer.writerow(
            [
                "epoch",
                "valid_mean_error",
                "test_mean_error",
                "valid_mc_error",
                "test_mc_error",
                "train_loss",
                "train_kl_raw",
                "train_nll",
                "snr_q50_db",
                "snr_q75_db",
            ]
        )
        for i in range(hyper.max_epoch):
            writer.writerow(
                (
                    i + 1,
                    1.0 - valid_mean_accs[i],
                    1.0 - test_mean_accs[i],
                    1.0 - valid_mc_accs[i],
                    1.0 - test_mc_accs[i],
                    train_losses[i],
                    train_kls[i],
                    train_nlls[i],
                    snr_q50_db[i],
                    snr_q75_db[i],
                )
            )

    torch.save(model.state_dict(), path + ".pth")
    if best_state is not None:
        torch.save(best_state, path + "_best.pth")

    print("Saved final model to:", path + ".pth")
    print("Saved best-valid model to:", path + "_best.pth")
    print("Saved log to:", path + ".csv")

    return model


# -----------------------------------------------------------------------------
# Dataset and CLI
# -----------------------------------------------------------------------------


def make_data_loaders(hyper: BBB_Hyper):
    if hyper.dataset in ["mnist", "fmnist"]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255.0 / 126.0),
            ]
        )

        if hyper.dataset == "mnist":
            train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
            test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
        else:
            train_data = datasets.FashionMNIST(root="data2", train=True, download=True, transform=transform)
            test_data = datasets.FashionMNIST(root="data2", train=False, download=True, transform=transform)

        n_input = 28 * 28
        n_output = 10

    elif hyper.dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
        n_input = 32 * 32 * 3
        n_output = 10

    else:
        raise ValueError("Unknown dataset: " + str(hyper.dataset))

    num_train = len(train_data)
    indices = np.arange(num_train)
    rng = np.random.default_rng(hyper.seed)
    rng.shuffle(indices)

    split = int(hyper.valid_fraction * num_train)
    valid_idx = indices[:split]
    train_idx = indices[split:]

    train_sampler = SubsetRandomSampler(train_idx.tolist())
    valid_sampler = SubsetRandomSampler(valid_idx.tolist())

    pin_memory = device.type == "cuda"

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=hyper.batch_size,
        sampler=train_sampler,
        num_workers=hyper.num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=hyper.eval_batch_size,
        sampler=valid_sampler,
        num_workers=hyper.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=hyper.eval_batch_size,
        shuffle=False,
        num_workers=hyper.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader, n_input, n_output


def parse_args():
    parser = argparse.ArgumentParser(description="Train Bayes by Backprop on MNIST/FashionMNIST/CIFAR10.")

    # Backwards-compatible positional arguments.
    parser.add_argument("pos_hidden", nargs="?", type=int, help="Old-style hidden units positional argument.")
    parser.add_argument("pos_dataset", nargs="?", type=str, help="Old-style dataset positional argument.")

    parser.add_argument("--hidden", type=int, default=None, help="Hidden units per layer: 400, 800, or 1200.")
    parser.add_argument("--dataset", type=str, default=None, choices=["mnist", "fmnist", "cifar10"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None, help="MC samples for the training objective.")
    parser.add_argument("--n-test-samples", type=int, default=None, help="MC samples for validation/test during training.")
    parser.add_argument("--kl-weighting", type=str, default=None, choices=["equal", "paper"])
    parser.add_argument("--kl-scale", type=float, default=None, help="Multiplier on the KL term. Try 2 or 5 if SNR remains too high.")
    parser.add_argument("--mu-init", type=str, default=None, choices=["small", "kaiming", "uniform"])
    parser.add_argument("--mu-init-scale", type=float, default=None)
    parser.add_argument("--rho-init", type=float, default=None, help="Override rho_init after paper hyperparameters are applied.")
    parser.add_argument("--rho-init-std", type=float, default=None)
    parser.add_argument("--best-by", type=str, default=None, choices=["mean", "mc"])
    parser.add_argument("--snr-log-every", type=int, default=None, help="Print SNR diagnostics every N epochs. Use 0 to disable.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--id", type=int, default=0, help="Run ID used in saved filenames.")

    return parser.parse_args()


def main():
    global device

    args = parse_args()
    hyper = BBB_Hyper()

    if args.pos_hidden is not None:
        hyper.hidden_units = args.pos_hidden
    if args.pos_dataset is not None:
        hyper.dataset = args.pos_dataset

    if args.hidden is not None:
        hyper.hidden_units = args.hidden
    if args.dataset is not None:
        hyper.dataset = args.dataset
    if args.epochs is not None:
        hyper.max_epoch = args.epochs
    if args.lr is not None:
        hyper.lr = args.lr
    if args.momentum is not None:
        hyper.momentum = args.momentum
    if args.batch_size is not None:
        hyper.batch_size = args.batch_size
    if args.eval_batch_size is not None:
        hyper.eval_batch_size = args.eval_batch_size
    if args.n_samples is not None:
        hyper.n_samples = args.n_samples
    if args.n_test_samples is not None:
        hyper.n_test_samples = args.n_test_samples
    if args.kl_weighting is not None:
        hyper.kl_weighting = args.kl_weighting
    if args.kl_scale is not None:
        hyper.kl_scale = args.kl_scale
    if args.mu_init is not None:
        hyper.mu_init = args.mu_init
    if args.mu_init_scale is not None:
        hyper.mu_init_scale = args.mu_init_scale
    if args.rho_init_std is not None:
        hyper.rho_init_std = args.rho_init_std
    if args.best_by is not None:
        hyper.best_by = args.best_by
    if args.snr_log_every is not None:
        hyper.snr_log_every = args.snr_log_every
    if args.seed is not None:
        hyper.seed = args.seed
    if args.output_dir is not None:
        hyper.output_dir = args.output_dir
    if args.num_workers is not None:
        hyper.num_workers = args.num_workers

    if args.device is not None and args.device != "auto":
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(hyper.seed)
    apply_paper_hyperparams(hyper)

    # rho override must happen after apply_paper_hyperparams(), because that
    # function sets the paper-family rho value for each hidden size.
    if args.rho_init is not None:
        hyper.rho_init = float(args.rho_init)

    train_loader, valid_loader, test_loader, n_input, n_output = make_data_loaders(hyper)

    BBB_run(
        hyper=hyper,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        n_input=n_input,
        n_output=n_output,
        run_id=args.id,
    )


if __name__ == "__main__":
    main()
