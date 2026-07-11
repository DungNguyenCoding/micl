"""Pyro-based mean-field variational local learning for Bayesian FL.

The implementation is deliberately small and explicit: it models an MLP's
weights and biases as latent variables and uses ``AutoDiagonalNormal`` to learn
a diagonal Gaussian posterior. This makes the arrays sent through Flower directly
comparable to the diagonal Gaussian arrays used by the online Laplace method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import RunConfig


@dataclass
class BayesianMlpSpec:
    input_dim: int
    hidden_dims: List[int]
    num_classes: int
    names: List[str]
    shapes: List[Tuple[int, ...]]

    @property
    def latent_dim(self) -> int:
        return int(sum(np.prod(shape) for shape in self.shapes))


def build_mlp_spec(input_shape: Sequence[int], num_classes: int, hidden_dims: Sequence[int]) -> BayesianMlpSpec:
    """Return latent-site names/shapes matching ``model.MLP`` parameter order."""
    input_dim = int(np.prod(tuple(input_shape)))
    dims = [input_dim] + [int(x) for x in hidden_dims] + [int(num_classes)]
    names: List[str] = []
    shapes: List[Tuple[int, ...]] = []
    for layer_idx in range(len(dims) - 1):
        names.extend([f"w{layer_idx}", f"b{layer_idx}"])
        shapes.extend([(dims[layer_idx + 1], dims[layer_idx]), (dims[layer_idx + 1],)])
    return BayesianMlpSpec(input_dim=input_dim, hidden_dims=list(hidden_dims), num_classes=int(num_classes), names=names, shapes=shapes)


def split_flat(flat: np.ndarray | torch.Tensor, spec: BayesianMlpSpec, device: torch.device) -> Dict[str, torch.Tensor]:
    """Split a flat latent vector into a dictionary keyed by Pyro site names."""
    flat_t = torch.as_tensor(flat, dtype=torch.float32, device=device).flatten()
    if flat_t.numel() != spec.latent_dim:
        raise ValueError(f"Expected latent dim {spec.latent_dim}, got {flat_t.numel()}")
    out: Dict[str, torch.Tensor] = {}
    cursor = 0
    for name, shape in zip(spec.names, spec.shapes):
        n = int(np.prod(shape))
        out[name] = flat_t[cursor : cursor + n].view(shape)
        cursor += n
    return out




def logits_from_flat(flat: np.ndarray | torch.Tensor, x: torch.Tensor, spec: BayesianMlpSpec, device: torch.device) -> torch.Tensor:
    """Deterministic forward pass through the posterior mean parameters."""
    state = split_flat(flat, spec, device)
    z = torch.flatten(x, start_dim=1)
    num_layers = len(spec.shapes) // 2
    for layer_idx in range(num_layers):
        z = F.linear(z, state[f"w{layer_idx}"], state[f"b{layer_idx}"])
        if layer_idx < num_layers - 1:
            z = F.relu(z)
    return z


@torch.no_grad()
def mean_posterior_nll(trainloader: DataLoader, flat_loc: np.ndarray, spec: BayesianMlpSpec, device: torch.device) -> float:
    """Cross-entropy/NLL of the posterior mean on the local train data."""
    total_loss = 0.0
    total_examples = 0
    for x, y in trainloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = logits_from_flat(flat_loc, x, spec, device)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.detach().cpu())
        total_examples += int(y.numel())
    return float(total_loss / max(total_examples, 1))


def _softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of softplus for positive tensors."""
    x = torch.clamp(x, min=1.0e-8)
    return x + torch.log(-torch.expm1(-x))


def _make_pyro_model(spec: BayesianMlpSpec, prior_loc: Dict[str, torch.Tensor], prior_scale: Dict[str, torch.Tensor]):
    """Create a Pyro model closure."""
    import pyro
    import pyro.distributions as dist

    def bnn_model(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        num_layers = len(spec.shapes) // 2
        for layer_idx in range(num_layers):
            w_name = f"w{layer_idx}"
            b_name = f"b{layer_idx}"
            w = pyro.sample(w_name, dist.Normal(prior_loc[w_name], prior_scale[w_name]).to_event(2))
            b = pyro.sample(b_name, dist.Normal(prior_loc[b_name], prior_scale[b_name]).to_event(1))
            x = F.linear(x, w, b)
            if layer_idx < num_layers - 1:
                x = F.relu(x)
        logits = x
        with pyro.plate("data", logits.shape[0]):
            pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits

    return bnn_model


def train_vi_local(
    trainloader: DataLoader,
    input_shape: Sequence[int],
    num_classes: int,
    hidden_dims: Sequence[int],
    global_loc: np.ndarray,
    global_scale: np.ndarray,
    device: torch.device,
    cfg: RunConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """Run local Pyro SVI and return posterior loc/scale and metrics.

    The returned metrics include an ELBO/free-energy proxy from Pyro SVI and a
    closed-form diagonal-Gaussian KL between the learned local posterior and the
    incoming global posterior/prior. This gives the log files enough information
    to plot VI complexity-vs-likelihood behavior later.
    """
    import pyro
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoDiagonalNormal
    from pyro.infer.autoguide.initialization import init_to_value
    from pyro.optim import Adam

    pyro.clear_param_store()
    pyro.set_rng_seed(int(seed))

    spec = build_mlp_spec(input_shape=input_shape, num_classes=num_classes, hidden_dims=hidden_dims)
    loc_state = split_flat(global_loc, spec, device)
    scale_np = np.maximum(np.asarray(global_scale, dtype=np.float32), float(cfg.vi_min_scale))
    scale_state = split_flat(scale_np, spec, device)
    bnn_model = _make_pyro_model(spec, loc_state, scale_state)
    guide = AutoDiagonalNormal(
        bnn_model,
        init_loc_fn=init_to_value(values=loc_state),
        init_scale=float(cfg.vi_init_scale),
    )

    # Initialize guide parameters before trying to seed its scale vector.
    first_batch = next(iter(trainloader), None)
    if first_batch is None:
        return np.asarray(global_loc, dtype=np.float32), scale_np, 0.0, {"vi_elbo_loss": 0.0}
    x0, y0 = first_batch
    x0 = x0.to(device)
    y0 = y0.to(device)
    guide(x0, y0)

    # AutoDiagonalNormal stores a single flat mean and scale. We initialize both
    # from the global posterior so prior iteration is warm-started.
    try:
        pyro.param("AutoDiagonalNormal.loc").data.copy_(torch.as_tensor(global_loc, dtype=torch.float32, device=device))
        constrained_scale = pyro.param("AutoDiagonalNormal.scale")
        constrained_scale.data.copy_(torch.as_tensor(scale_np, dtype=torch.float32, device=device))
        if hasattr(constrained_scale, "unconstrained"):
            constrained_scale.unconstrained().data.copy_(_softplus_inverse(torch.as_tensor(scale_np, dtype=torch.float32, device=device)))
    except Exception:
        # Pyro versions differ slightly in how constrained parameters are exposed.
        # The guide remains valid because prior_loc/prior_scale are still the
        # global posterior parameters.
        pass

    svi = SVI(
        bnn_model,
        guide,
        Adam({"lr": float(cfg.vi_lr)}),
        Trace_ELBO(num_particles=int(cfg.vi_particles)),
    )

    loss_sum = 0.0
    steps = 0
    for _epoch in range(int(cfg.local_epochs)):
        for x, y in trainloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss_sum += float(svi.step(x, y))
            steps += 1

    posterior = guide.get_posterior()

    base_dist = getattr(posterior, "base_dist", posterior)

    if hasattr(base_dist, "loc") and hasattr(base_dist, "scale"):
        loc_t = base_dist.loc
        scale_t = base_dist.scale
    else:
        loc_t = posterior.mean
        scale_t = getattr(posterior, "stddev", torch.sqrt(posterior.variance))

    loc = loc_t.detach().cpu().numpy().astype(np.float32, copy=True)
    scale = scale_t.detach().cpu().numpy().astype(np.float32, copy=True)
    scale = np.maximum(scale, float(cfg.vi_min_scale)).astype(np.float32, copy=False)
    if float(getattr(cfg, "vi_max_scale", 0.0)) > 0.0:
        scale = np.minimum(scale, float(cfg.vi_max_scale)).astype(np.float32, copy=False)

    prior_loc = np.asarray(global_loc, dtype=np.float64).reshape(-1)
    prior_scale = np.maximum(np.asarray(global_scale, dtype=np.float64).reshape(-1), float(cfg.vi_min_scale))
    loc64 = loc.astype(np.float64, copy=False).reshape(-1)
    scale64 = np.maximum(scale.astype(np.float64, copy=False).reshape(-1), float(cfg.vi_min_scale))
    kl_terms = (scale64 ** 2 + (loc64 - prior_loc) ** 2) / (prior_scale ** 2) - 1.0 + 2.0 * (np.log(prior_scale) - np.log(scale64))
    vi_kl = float(0.5 * np.sum(kl_terms))
    n_examples = int(len(trainloader.dataset)) if hasattr(trainloader, "dataset") else 0
    vi_kl_per_example = float(vi_kl / max(n_examples, 1))
    vi_kl_per_param = float(vi_kl / max(loc64.size, 1))
    vi_likelihood = mean_posterior_nll(trainloader, loc, spec, device)
    snr = np.abs(loc64) / (scale64 + 1.0e-12)
    stats = {
        "vi_elbo_loss": float(loss_sum / max(steps, 1)),
        "vi_kl_loss": vi_kl,
        "vi_kl_loss_per_example": vi_kl_per_example,
        "vi_kl_per_param": vi_kl_per_param,
        "vi_likelihood_loss": float(vi_likelihood),
        "vi_complexity_cost": vi_kl_per_example,
        "vi_effective_lr": float(cfg.vi_lr),
        "vi_loc_l2": float(np.linalg.norm(loc64)),
        "vi_scale_mean": float(scale64.mean()),
        "vi_scale_std": float(scale64.std()),
        "vi_scale_p50": float(np.percentile(scale64, 50)),
        "vi_scale_p90": float(np.percentile(scale64, 90)),
        "vi_scale_max": float(scale64.max()),
        "vi_snr_raw_mean": float(snr.mean()),
        "vi_snr_raw_p50": float(np.percentile(snr, 50)),
        "vi_snr_raw_p90": float(np.percentile(snr, 90)),
    }
    return loc, scale, loss_sum / max(steps, 1), stats
