import argparse
import csv
import math
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO


# =============================================================================
# Reproducibility / device
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Hyperparameters
# =============================================================================

class BBB_Hyper(object):
    def __init__(self):
        self.dataset = "mnist"  # mnist || fmnist || cifar10

        self.lr = 3e-4
        self.momentum = 0.95
        self.hidden_units = 1200

        # Prior:
        # p(w) = pi * N(0, sigma1^2) + (1 - pi) * N(0, sigma2^2)
        # with sigma1 > sigma2.
        self.pi = 0.25
        self.s1 = float(np.exp(-1))   # sigma1, wide component
        self.s2 = float(np.exp(-8))   # sigma2, narrow component

        # Variational posterior initialization.
        self.rho_init = -5.0
        self.rho_init_std = 0.05

        # Posterior mean initialization.
        self.mu_init = "small"        # small || kaiming || uniform
        self.mu_init_scale = 0.01

        # Training setup.
        self.max_epoch = 600
        self.batch_size = 128
        self.eval_batch_size = 1000

        # MC samples.
        self.n_samples = 1
        self.n_test_samples = 10

        # Equal KL weighting only:
        # beta = kl_scale / num_batches
        #
        # If kl_scale = 1.0, this is the paper-style equal minibatch KL weighting.
        # Smaller kl_scale weakens KL pressure.
        self.kl_scale = 1.0

        # Checkpoint selection.
        self.best_by = "mean"  # mean || mc

        # Misc.
        self.seed = 0
        self.valid_fraction = 1.0 / 6.0
        self.num_workers = 0
        self.output_dir = "Results_pyro"
        self.snr_log_every = 25


def apply_default_prior_for_hidden_units(hyper: BBB_Hyper) -> BBB_Hyper:
    """
    Defaults inspired by the original repo / paper-style experiments.

    Format:
        hidden_units: (pi, rho_init, s1_code, s2_code)

    Then:
        sigma1 = exp(-s1_code)
        sigma2 = exp(-s2_code)

    sigma1 is the wide component.
    sigma2 is the narrow component.
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
            f"Expected sigma1 > sigma2, but got sigma1={hyper.s1}, sigma2={hyper.s2}"
        )

    return hyper


# =============================================================================
# Probability utilities
# =============================================================================

def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def make_scale_mixture_prior(
    shape: Tuple[int, ...],
    pi: float,
    sigma1: float,
    sigma2: float,
    ref_tensor: torch.Tensor,
):
    """
    Pyro distribution for:

        pi * N(0, sigma1^2) + (1 - pi) * N(0, sigma2^2)

    Event shape equals `shape`.
    """
    if not sigma1 > sigma2:
        raise ValueError(f"Expected sigma1 > sigma2, got {sigma1}, {sigma2}")

    probs = torch.tensor(
        [pi, 1.0 - pi],
        dtype=ref_tensor.dtype,
        device=ref_tensor.device,
    )

    mixture_distribution = dist.Categorical(
        probs=probs.expand(*shape, 2)
    )

    component_locs = torch.zeros(
        *shape,
        2,
        dtype=ref_tensor.dtype,
        device=ref_tensor.device,
    )

    component_scales = torch.tensor(
        [sigma1, sigma2],
        dtype=ref_tensor.dtype,
        device=ref_tensor.device,
    ).expand(*shape, 2)

    component_distribution = dist.Normal(component_locs, component_scales)

    return dist.MixtureSameFamily(
        mixture_distribution,
        component_distribution,
    ).to_event(len(shape))


def log_gaussian(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    sigma_t = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    mu_t = torch.as_tensor(mu, dtype=x.dtype, device=x.device)

    return (
        -0.5 * math.log(2.0 * math.pi)
        - torch.log(sigma_t + 1e-12)
        - ((x - mu_t) ** 2) / (2.0 * sigma_t ** 2 + 1e-12)
    )


def log_mixture_prior(
    x: torch.Tensor,
    pi: float,
    sigma1: float,
    sigma2: float,
) -> torch.Tensor:
    """
    Diagnostic-only log scale-mixture prior.
    Pyro training uses make_scale_mixture_prior().
    """
    log_prob1 = math.log(pi) + log_gaussian(x, 0.0, sigma1)
    log_prob2 = math.log(1.0 - pi) + log_gaussian(x, 0.0, sigma2)

    return torch.logsumexp(torch.stack([log_prob1, log_prob2], dim=0), dim=0)


# =============================================================================
# Bayesian layer
# =============================================================================

class BBBLayer(nn.Module):
    def __init__(self, n_input: int, n_output: int, hyper: BBB_Hyper):
        super(BBBLayer, self).__init__()

        self.n_input = n_input
        self.n_output = n_output

        self.pi = hyper.pi
        self.s1 = hyper.s1
        self.s2 = hyper.s2

        self.weight_mu = nn.Parameter(torch.empty(n_output, n_input))
        self.bias_mu = nn.Parameter(torch.empty(n_output))

        self.reset_mu_parameters(hyper)

        self.weight_rho = nn.Parameter(
            torch.empty(n_output, n_input).normal_(
                hyper.rho_init,
                hyper.rho_init_std,
            )
        )

        self.bias_rho = nn.Parameter(
            torch.empty(n_output).normal_(
                hyper.rho_init,
                hyper.rho_init_std,
            )
        )

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
            nn.init.uniform_(
                self.weight_mu,
                -hyper.mu_init_scale,
                hyper.mu_init_scale,
            )
            nn.init.uniform_(
                self.bias_mu,
                -hyper.mu_init_scale,
                hyper.mu_init_scale,
            )

        else:
            raise ValueError("Unknown mu_init: " + str(hyper.mu_init))

    def posterior_weight_dist(self):
        return dist.Normal(
            self.weight_mu,
            softplus(self.weight_rho),
        ).to_event(2)

    def posterior_bias_dist(self):
        return dist.Normal(
            self.bias_mu,
            softplus(self.bias_rho),
        ).to_event(1)

    def prior_weight_dist(self):
        return make_scale_mixture_prior(
            shape=tuple(self.weight_mu.shape),
            pi=self.pi,
            sigma1=self.s1,
            sigma2=self.s2,
            ref_tensor=self.weight_mu,
        )

    def prior_bias_dist(self):
        return make_scale_mixture_prior(
            shape=tuple(self.bias_mu.shape),
            pi=self.pi,
            sigma1=self.s1,
            sigma2=self.s2,
            ref_tensor=self.bias_mu,
        )

    def forward_mean(self, data: torch.Tensor) -> torch.Tensor:
        return F.linear(data, self.weight_mu, self.bias_mu)

    def forward_with_weight(
        self,
        data: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return F.linear(data, weight, bias)

    def sample_posterior(self):
        weight = self.posterior_weight_dist().rsample()
        bias = self.posterior_bias_dist().rsample()

        return weight, bias


# =============================================================================
# Bayesian neural network
# =============================================================================

class BBB(nn.Module):
    def __init__(self, n_input: int, n_output: int, hyper: BBB_Hyper):
        super(BBB, self).__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.hidden_units = hyper.hidden_units
        self.hyper = hyper

        self.layers = nn.ModuleList(
            [
                BBBLayer(n_input, hyper.hidden_units, hyper),
                BBBLayer(hyper.hidden_units, hyper.hidden_units, hyper),
                BBBLayer(hyper.hidden_units, n_output, hyper),
            ]
        )

    # -------------------------------------------------------------------------
    # Standard forward path.
    # This keeps plot.py compatibility.
    # -------------------------------------------------------------------------
    def forward(self, data: torch.Tensor, infer: bool = False) -> torch.Tensor:
        if infer:
            return self.forward_mean(data)

        weights = [layer.sample_posterior() for layer in self.layers]

        return self.forward_given_weights(data, weights)

    def forward_mean(self, data: torch.Tensor) -> torch.Tensor:
        output = data.view(-1, self.n_input)

        output = F.relu(self.layers[0].forward_mean(output))
        output = F.relu(self.layers[1].forward_mean(output))
        output = self.layers[2].forward_mean(output)

        return output

    def forward_given_weights(self, data: torch.Tensor, weights) -> torch.Tensor:
        output = data.view(-1, self.n_input)

        w0, b0 = weights[0]
        w1, b1 = weights[1]
        w2, b2 = weights[2]

        output = F.relu(self.layers[0].forward_with_weight(output, w0, b0))
        output = F.relu(self.layers[1].forward_with_weight(output, w1, b1))
        output = self.layers[2].forward_with_weight(output, w2, b2)

        return output

    # -------------------------------------------------------------------------
    # Pyro model and guide.
    # -------------------------------------------------------------------------
    def pyro_model(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        beta: float,
    ) -> None:
        """
        Model:
            p(w) p(y | x, w)

        The global latent weight prior terms are scaled by beta.

        Equal minibatch KL:
            beta = kl_scale / num_batches

        If kl_scale = 1:
            beta = 1 / num_batches
        """
        weights = []

        with pyro.poutine.scale(scale=beta):
            for idx, layer in enumerate(self.layers):
                weight = pyro.sample(
                    f"layer_{idx}_weight",
                    layer.prior_weight_dist(),
                )

                bias = pyro.sample(
                    f"layer_{idx}_bias",
                    layer.prior_bias_dist(),
                )

                weights.append((weight, bias))

        logits = self.forward_given_weights(data, weights)

        with pyro.plate("data", data.shape[0]):
            pyro.sample(
                "obs",
                dist.Categorical(logits=logits),
                obs=target,
            )

    def pyro_guide(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        beta: float,
    ) -> None:
        """
        Guide:
            q(w | theta)

        The posterior log-probability terms are scaled by the same beta
        used for the prior terms.
        """
        with pyro.poutine.scale(scale=beta):
            for idx, layer in enumerate(self.layers):
                pyro.sample(
                    f"layer_{idx}_weight",
                    layer.posterior_weight_dist(),
                )

                pyro.sample(
                    f"layer_{idx}_bias",
                    layer.posterior_bias_dist(),
                )


# =============================================================================
# Diagnostics
# =============================================================================

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


@torch.no_grad()
def estimate_terms_for_logging(
    model: BBB,
    data: torch.Tensor,
    target: torch.Tensor,
    num_samples: int = 1,
):
    """
    Diagnostic-only estimate of raw KL and NLL.
    Training itself uses Pyro Trace_ELBO.
    """
    raw_kl = 0.0
    nll = 0.0

    for _ in range(num_samples):
        weights = []
        log_q = 0.0
        log_p = 0.0

        for layer in model.layers:
            weight, bias = layer.sample_posterior()
            weights.append((weight, bias))

            log_q = log_q + layer.posterior_weight_dist().log_prob(weight)
            log_q = log_q + layer.posterior_bias_dist().log_prob(bias)

            log_p = log_p + log_mixture_prior(
                weight,
                layer.pi,
                layer.s1,
                layer.s2,
            ).sum()

            log_p = log_p + log_mixture_prior(
                bias,
                layer.pi,
                layer.s1,
                layer.s2,
            ).sum()

        logits = model.forward_given_weights(data, weights)

        raw_kl = raw_kl + (log_q - log_p).item() / num_samples
        nll = nll + F.cross_entropy(logits, target, reduction="sum").item() / num_samples

    return raw_kl, nll


# =============================================================================
# Training and evaluation
# =============================================================================

def train_epoch(
    model: BBB,
    optimizer,
    elbo,
    loader,
    hyper: BBB_Hyper,
    train_device: torch.device,
):
    model.train()

    loss_sum = 0.0
    kl_diag = 0.0
    nll_diag = 0.0

    num_batches = len(loader)

    # Equal KL weighting only.
    beta = hyper.kl_scale / float(num_batches)

    for batch_id, (data, target) in enumerate(loader):
        data = data.to(train_device, non_blocking=True)
        target = target.to(train_device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        loss = elbo.differentiable_loss(
            model.pyro_model,
            model.pyro_guide,
            data,
            target,
            beta,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at batch {batch_id}: {loss.item()}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
            error_if_nonfinite=True,
        )

        optimizer.step()

        # Avoid numerical extremes in rho.
        with torch.no_grad():
            for layer in model.layers:
                layer.weight_rho.clamp_(-12.0, 2.0)
                layer.bias_rho.clamp_(-12.0, 2.0)

        loss_sum += loss.item() / num_batches

        # Logging diagnostics from first batch only.
        if batch_id == 0:
            raw_kl, nll = estimate_terms_for_logging(
                model,
                data,
                target,
                num_samples=1,
            )
            kl_diag = raw_kl
            nll_diag = nll

    return loss_sum, kl_diag, nll_diag


@torch.no_grad()
def evaluate(
    model: BBB,
    loader,
    eval_device: torch.device,
    infer: bool = True,
    samples: int = 1,
) -> float:
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
                probs = F.softmax(logits, dim=1)

                probs_sum = probs if probs_sum is None else probs_sum + probs

            pred = (probs_sum / samples).argmax(dim=1)

        correct += pred.eq(target).sum().item()
        total += target.numel()

    return correct / total


def clone_state_dict_to_cpu(model: nn.Module):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def BBB_run(
    hyper: BBB_Hyper,
    train_loader,
    valid_loader,
    test_loader,
    n_input: int,
    n_output: int,
    run_id: int = 0,
):
    print("Using device:", device)
    print("Training configuration:")

    for key, value in sorted(hyper.__dict__.items()):
        print(f"  {key} = {value}")

    pyro.clear_param_store()

    model = BBB(n_input, n_output, hyper).to(device)

    print("Initial posterior diagnostic:")
    print(" ", format_snr_summary(posterior_snr_summary(model)))

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=hyper.lr,
        momentum=hyper.momentum,
    )

    elbo = Trace_ELBO(num_particles=hyper.n_samples)

    train_losses = np.zeros(hyper.max_epoch)
    train_kl_diag = np.zeros(hyper.max_epoch)
    train_nll_diag = np.zeros(hyper.max_epoch)

    valid_mean_accs = np.zeros(hyper.max_epoch)
    test_mean_accs = np.zeros(hyper.max_epoch)
    valid_mc_accs = np.zeros(hyper.max_epoch)
    test_mc_accs = np.zeros(hyper.max_epoch)

    snr_q50_db = np.zeros(hyper.max_epoch)
    snr_q75_db = np.zeros(hyper.max_epoch)

    best_score = -1.0
    best_state = None

    for epoch in range(hyper.max_epoch):
        train_loss, raw_kl, nll = train_epoch(
            model=model,
            optimizer=optimizer,
            elbo=elbo,
            loader=train_loader,
            hyper=hyper,
            train_device=device,
        )

        valid_mean_acc = evaluate(
            model=model,
            loader=valid_loader,
            eval_device=device,
            infer=True,
            samples=1,
        )

        test_mean_acc = evaluate(
            model=model,
            loader=test_loader,
            eval_device=device,
            infer=True,
            samples=1,
        )

        valid_mc_acc = evaluate(
            model=model,
            loader=valid_loader,
            eval_device=device,
            infer=False,
            samples=hyper.n_test_samples,
        )

        test_mc_acc = evaluate(
            model=model,
            loader=test_loader,
            eval_device=device,
            infer=False,
            samples=hyper.n_test_samples,
        )

        score = valid_mean_acc if hyper.best_by == "mean" else valid_mc_acc

        if score > best_score:
            best_score = score
            best_state = clone_state_dict_to_cpu(model)

        summary = posterior_snr_summary(model)

        snr_q50_db[epoch] = summary["snr_q50_db"]
        snr_q75_db[epoch] = summary["snr_q75_db"]

        message = (
            f"Epoch {epoch + 1:03d}/{hyper.max_epoch} | "
            f"Loss {train_loss:.4f} | "
            f"KLraw {raw_kl:.2f} | "
            f"NLL {nll:.2f} | "
            f"ValidMeanErr {100.0 * (1.0 - valid_mean_acc):.3f}% | "
            f"TestMeanErr {100.0 * (1.0 - test_mean_acc):.3f}% | "
            f"ValidMCErr {100.0 * (1.0 - valid_mc_acc):.3f}% | "
            f"TestMCErr {100.0 * (1.0 - test_mc_acc):.3f}%"
        )

        if hyper.snr_log_every and (
            (epoch + 1) % hyper.snr_log_every == 0 or epoch == 0
        ):
            message += " | " + format_snr_summary(summary)

        print(message)

        train_losses[epoch] = train_loss
        train_kl_diag[epoch] = raw_kl
        train_nll_diag[epoch] = nll

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

    with open(path + ".csv", "w", newline="") as file:
        writer = csv.writer(file, delimiter=",", lineterminator="\n")

        writer.writerow(
            [
                "epoch",
                "valid_mean_error",
                "test_mean_error",
                "valid_mc_error",
                "test_mc_error",
                "train_loss",
                "train_kl_raw_first_batch",
                "train_nll_first_batch",
                "snr_q50_db",
                "snr_q75_db",
            ]
        )

        for i in range(hyper.max_epoch):
            writer.writerow(
                [
                    i + 1,
                    1.0 - valid_mean_accs[i],
                    1.0 - test_mean_accs[i],
                    1.0 - valid_mc_accs[i],
                    1.0 - test_mc_accs[i],
                    train_losses[i],
                    train_kl_diag[i],
                    train_nll_diag[i],
                    snr_q50_db[i],
                    snr_q75_db[i],
                ]
            )

    torch.save(model.state_dict(), path + ".pth")

    if best_state is not None:
        torch.save(best_state, path + "_best.pth")

    print("Saved final model to:", path + ".pth")
    print("Saved best-valid model to:", path + "_best.pth")
    print("Saved log to:", path + ".csv")

    return model


# =============================================================================
# Data
# =============================================================================

def make_data_loaders(hyper: BBB_Hyper):
    if hyper.dataset in ["mnist", "fmnist"]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255.0 / 126.0),
            ]
        )

        if hyper.dataset == "mnist":
            train_data = datasets.MNIST(
                root="data",
                train=True,
                download=True,
                transform=transform,
            )

            test_data = datasets.MNIST(
                root="data",
                train=False,
                download=True,
                transform=transform,
            )

        else:
            train_data = datasets.FashionMNIST(
                root="data2",
                train=True,
                download=True,
                transform=transform,
            )

            test_data = datasets.FashionMNIST(
                root="data2",
                train=False,
                download=True,
                transform=transform,
            )

        n_input = 28 * 28
        n_output = 10

    elif hyper.dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                ),
            ]
        )

        train_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform,
        )

        test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transform,
        )

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


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pyro Bayes by Backprop training script."
    )

    # Backwards-compatible positional arguments:
    #     python BBB.py 1200 mnist
    parser.add_argument("pos_hidden", nargs="?", type=int)
    parser.add_argument("pos_dataset", nargs="?", type=str)

    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["mnist", "fmnist", "cifar10"],
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)

    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-test-samples", type=int, default=None)

    # No --kl-weighting argument.
    # Equal KL weighting is always used:
    #     beta = kl_scale / num_batches
    parser.add_argument(
        "--kl-scale",
        type=float,
        default=None,
        help="Multiplier on equal-weighted KL term. Paper-faithful value is 1.0.",
    )

    parser.add_argument(
        "--mu-init",
        type=str,
        default=None,
        choices=["small", "kaiming", "uniform"],
    )

    parser.add_argument("--mu-init-scale", type=float, default=None)

    parser.add_argument("--rho-init", type=float, default=None)
    parser.add_argument("--rho-init-std", type=float, default=None)

    parser.add_argument(
        "--best-by",
        type=str,
        default=None,
        choices=["mean", "mc"],
    )

    parser.add_argument("--snr-log-every", type=int, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument("--id", type=int, default=0)

    return parser.parse_args()


def main():
    global device

    args = parse_args()

    hyper = BBB_Hyper()

    # Positional compatibility.
    if args.pos_hidden is not None:
        hyper.hidden_units = args.pos_hidden

    if args.pos_dataset is not None:
        hyper.dataset = args.pos_dataset

    # Named overrides.
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
            raise RuntimeError("--device cuda requested, but CUDA is not available.")
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(hyper.seed)

    # Apply hidden-unit-specific default prior/rho settings.
    apply_default_prior_for_hidden_units(hyper)

    # rho override must happen after apply_default_prior_for_hidden_units().
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