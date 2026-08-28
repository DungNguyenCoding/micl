"""Bayesian-Torch variational trainer for the two-phase AirComp protocol.

This backend intentionally preserves the external protocol used by the
existing Pyro implementation:

Phase 1
-------
Prior:
    N(mu_t, diag(rho_t)^-1)

Posterior:
    N(mu_t, diag(rho_local)^-1)

Bayesian-Torch native rho/softplus scale parameters are optimized while
posterior means remain fixed.

Phase 2
-------
Prior:
    N(mu_t, diag(rho_t)^-1)

Posterior:
    N(mu_local, diag(rho_{t+1})^-1)

Bayesian-Torch posterior means are optimized while posterior scales are
fixed.  The learned posterior mean is converted to the paper's natural
coordinate before communication:

    nu = (rho_local / rho_{t+1}) * mu_local

Important
---------
This is not coordinate-for-coordinate identical to the Pyro optimizer.

Pyro optimizes:
    direct precision rho
    natural coordinate nu

Bayesian-Torch optimizes:
    unconstrained scale rho_BT
    posterior mean mu

The server-side Gaussian state and AirComp communication coordinates,
however, remain identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bayesian_torch_adapter import (
    BayesianTorchParameterAdapter,
)
from config import ModelConfig, TrainingConfig
from serialization import ParameterLayout


@dataclass
class PrecisionPhaseResult:
    precision: np.ndarray
    average_loss: float
    local_steps: int
    precision_delta_l2: float
    precision_delta_max_abs: float
    precision_changed_fraction: float
    applied_gradient_l2_mean: float
    applied_gradient_max_abs: float


@dataclass
class NaturalMeanPhaseResult:
    nu: np.ndarray
    implied_mean: np.ndarray
    average_loss: float
    local_steps: int


class BayesianTorchVITrainer:
    """Local Bayesian-Torch implementation of the Proposed two-phase VI."""

    def __init__(
        self,
        model: torch.nn.Module,
        layout: ParameterLayout,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        device: torch.device,
        learning_rate: float | None = None,
    ) -> None:

        optimizer_name = str(
            train_cfg.optimizer
        ).strip().lower()

        if optimizer_name != "sgd":
            raise ValueError(
                "BayesianTorchVITrainer currently supports "
                "optimizer='sgd' only"
            )

        self.layout = layout
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device

        self.learning_rate = (
            float(train_cfg.learning_rate)
            if learning_rate is None
            else float(learning_rate)
        )

        # The deterministic model remains the definition of the AirComp
        # coordinate layout.  The adapter creates a private Bayesian copy.
        self.adapter = BayesianTorchParameterAdapter(
            model,
            layout,
        )

        self.adapter.model.to(
            device
        )

        # Match the current Pyro path, where the deterministic CNN is put
        # into evaluation mode during local Bayesian optimization.
        self.adapter.model.eval()

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def train_precision_phase(
        self,
        *,
        global_mean: np.ndarray,
        global_precision: np.ndarray,
        loader: DataLoader,
        seed: int,
    ) -> PrecisionPhaseResult:
        """Optimize Bayesian-Torch scale with posterior mean fixed."""

        mean = self._validated_mean(
            global_mean,
            "global_mean",
        )

        prior_precision = (
            self._validated_precision(
                global_precision,
                "global_precision",
            )
        )

        self._seed(
            seed
        )

        self.adapter.set_prior(
            mean,
            prior_precision,
        )

        self.adapter.set_posterior_mean(
            mean
        )

        self.adapter.set_posterior_precision(
            prior_precision
        )

        # Critical reference for delta anchoring.
        #
        # Bayesian-Torch float32 rho/softplus represents an exact incoming
        # precision of 10000 as approximately 9999.997.  This representational
        # offset must not become a fake transmitted Delta-rho.
        bt_initial_precision = (
            self.adapter
            .posterior_precision_vector()
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64)
        )

        self.adapter.set_trainable(
            mean=False,
            scale=True,
        )

        parameters = (
            self.adapter
            .scale_parameters()
        )

        def project_precision() -> None:
            """Project native BT precision only if it leaves allowed range."""

            with torch.no_grad():

                precision = (
                    self.adapter
                    .posterior_precision_vector()
                    .detach()
                )

                clipped = precision.clamp(
                    min=float(
                        self.model_cfg.min_precision
                    ),
                    max=float(
                        self.model_cfg.max_precision
                    ),
                )

                # Avoid unnecessary rho -> precision -> rho round-trips.
                if not torch.equal(
                    precision,
                    clipped,
                ):
                    self.adapter.set_posterior_precision(
                        clipped
                    )

        (
            average_loss,
            local_steps,
            applied_gradient_l2_mean,
            applied_gradient_max_abs,
        ) = self._run_sgd(
            loader=loader,
            parameters=parameters,
            after_step=project_precision,
        )

        bt_final_precision = (
            self.adapter
            .posterior_precision_vector()
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64)
        )

        # Native Bayesian-Torch precision change.
        native_delta = (
            bt_final_precision
            - bt_initial_precision
        )

        # Anchor that change to the exact float64 AirComp coordinate.
        precision = (
            prior_precision
            + native_delta
        )

        precision = np.clip(
            precision,
            float(
                self.model_cfg.min_precision
            ),
            float(
                self.model_cfg.max_precision
            ),
        ).astype(np.float64)

        delta = (
            precision
            - prior_precision
        )

        return PrecisionPhaseResult(
            precision=precision,
            average_loss=float(
                average_loss
            ),
            local_steps=int(
                local_steps
            ),
            precision_delta_l2=float(
                np.linalg.norm(
                    delta
                )
            ),
            precision_delta_max_abs=float(
                np.max(
                    np.abs(
                        delta
                    )
                )
            ),
            precision_changed_fraction=float(
                np.count_nonzero(
                    delta
                )
                / max(
                    1,
                    delta.size,
                )
            ),
            applied_gradient_l2_mean=float(
                applied_gradient_l2_mean
            ),
            applied_gradient_max_abs=float(
                applied_gradient_max_abs
            ),
        )

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def train_natural_mean_phase(
        self,
        *,
        global_mean: np.ndarray,
        prior_global_precision: np.ndarray,
        next_global_precision: np.ndarray,
        local_precision: np.ndarray,
        loader: DataLoader,
        seed: int,
    ) -> NaturalMeanPhaseResult:
        """Optimize native posterior mean and convert it to nu."""

        mean = self._validated_mean(
            global_mean,
            "global_mean",
        )

        prior_precision = (
            self._validated_precision(
                prior_global_precision,
                "prior_global_precision",
            )
        )

        next_precision = (
            self._validated_precision(
                next_global_precision,
                "next_global_precision",
            )
        )

        local_precision = (
            self._validated_precision(
                local_precision,
                "local_precision",
            )
        )

        self._seed(
            seed
        )

        # Eq. (15): regularize against round-start global posterior.
        self.adapter.set_prior(
            mean,
            prior_precision,
        )

        # Pyro's nu initialization implies exactly mu_local = mu_t.
        # Since this backend optimizes native mu, initialize it directly.
        self.adapter.set_posterior_mean(
            mean
        )

        # Phase-2 covariance is server's newly aggregated covariance.
        self.adapter.set_posterior_precision(
            next_precision
        )

        self.adapter.set_trainable(
            mean=True,
            scale=False,
        )

        parameters = (
            self.adapter
            .mean_parameters()
        )

        (
            average_loss,
            local_steps,
            _gradient_l2_mean,
            _gradient_max_abs,
        ) = self._run_sgd(
            loader=loader,
            parameters=parameters,
            after_step=None,
        )

        implied_mean = (
            self.adapter
            .posterior_mean_vector()
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float32)
        )

        # Convert Bayesian-Torch native posterior mean into exactly the
        # natural coordinate expected by the existing AirComp server.
        nu = (
            (
                local_precision
                / next_precision
            )
            * implied_mean.astype(
                np.float64
            )
        ).astype(np.float32)

        return NaturalMeanPhaseResult(
            nu=nu,
            implied_mean=implied_mean,
            average_loss=float(
                average_loss
            ),
            local_steps=int(
                local_steps
            ),
        )

    # ------------------------------------------------------------------
    # Shared SGD / ELBO approximation
    # ------------------------------------------------------------------

    def _run_sgd(
        self,
        *,
        loader: DataLoader,
        parameters: Sequence[
            torch.nn.Parameter
        ],
        after_step: Callable[
            [],
            None,
        ]
        | None,
    ) -> tuple[
        float,
        int,
        float,
        float,
    ]:

        parameters = list(
            parameters
        )

        if not parameters:
            raise ValueError(
                "No trainable Bayesian-Torch parameters"
            )

        optimizer = torch.optim.SGD(
            parameters,
            lr=float(
                self.learning_rate
            ),
            momentum=float(
                self.train_cfg.momentum
            ),
            weight_decay=float(
                self.train_cfg.weight_decay
            ),
        )

        dataset_size = max(
            1,
            int(
                len(
                    loader.dataset
                )
            ),
        )

        total_normalized_loss = 0.0
        local_steps = 0

        gradient_l2_values: list[
            float
        ] = []

        gradient_max_values: list[
            float
        ] = []

        for _ in range(
            int(
                self.train_cfg.local_epochs
            )
        ):

            for batch in loader:

                if (
                    not isinstance(
                        batch,
                        (tuple, list),
                    )
                    or len(batch) < 2
                ):
                    raise ValueError(
                        "Bayesian local loader must return "
                        "(inputs, targets)"
                    )

                inputs = batch[0].to(
                    self.device,
                    non_blocking=True,
                )

                targets = batch[1].to(
                    self.device,
                    non_blocking=True,
                )

                batch_size = max(
                    1,
                    int(
                        targets.numel()
                    ),
                )

                # Same stochastic full-dataset likelihood estimator as Pyro.
                likelihood_scale = (
                    float(
                        dataset_size
                    )
                    / float(
                        batch_size
                    )
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                loss = self._negative_elbo(
                    inputs=inputs,
                    targets=targets,
                    likelihood_scale=likelihood_scale,
                )

                if not bool(
                    torch.isfinite(
                        loss
                    ).item()
                ):
                    raise FloatingPointError(
                        "Non-finite Bayesian-Torch "
                        f"training loss: {loss}"
                    )

                loss.backward()

                clip_norm = float(
                    self.train_cfg.gradient_clip_norm
                )

                if clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        parameters,
                        max_norm=clip_norm,
                    )

                # Record APPLIED gradients after clipping, matching the
                # diagnostics produced by the Pyro implementation.
                (
                    gradient_l2,
                    gradient_max_abs,
                ) = self._gradient_stats(
                    parameters
                )

                gradient_l2_values.append(
                    gradient_l2
                )

                gradient_max_values.append(
                    gradient_max_abs
                )

                optimizer.step()

                if after_step is not None:
                    after_step()

                total_normalized_loss += (
                    float(
                        loss
                        .detach()
                        .cpu()
                    )
                    / float(
                        dataset_size
                    )
                )

                local_steps += 1

        if local_steps <= 0:
            raise RuntimeError(
                "Bayesian local training completed "
                "zero optimizer steps"
            )

        average_loss = (
            total_normalized_loss
            / float(
                local_steps
            )
        )

        gradient_l2_mean = (
            float(
                np.mean(
                    gradient_l2_values
                )
            )
            if gradient_l2_values
            else 0.0
        )

        gradient_max_abs = (
            float(
                np.max(
                    gradient_max_values
                )
            )
            if gradient_max_values
            else 0.0
        )

        return (
            float(
                average_loss
            ),
            int(
                local_steps
            ),
            gradient_l2_mean,
            gradient_max_abs,
        )

    def _negative_elbo(
        self,
        *,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        likelihood_scale: float,
    ) -> torch.Tensor:
        """MC estimate of data NLL plus analytic diagonal-Gaussian KL."""

        mc_samples = max(
            1,
            int(
                self.train_cfg.mc_train_samples
            ),
        )

        data_nll = torch.zeros(
            (),
            dtype=torch.float32,
            device=self.device,
        )

        for _ in range(
            mc_samples
        ):

            logits = (
                self.adapter
                .model(
                    inputs
                )
            )

            # Defensive compatibility if a future Bayesian-Torch layer
            # configuration exposes (output, KL).
            if isinstance(
                logits,
                tuple,
            ):
                logits = logits[0]

            data_nll = (
                data_nll
                + F.cross_entropy(
                    logits,
                    targets,
                    reduction="sum",
                )
            )

        data_nll = (
            data_nll
            / float(
                mc_samples
            )
        )

        # The adapter returns the FULL coordinate-sum KL.
        # Pyro's KL is likewise over the complete event vector.
        kl = (
            self.adapter
            .total_kl_sum()
        )

        return (
            float(
                likelihood_scale
            )
            * data_nll
            + float(
                self.train_cfg.kl_weight
            )
            * kl
        )

    # ------------------------------------------------------------------
    # Validation / diagnostics
    # ------------------------------------------------------------------

    def _validated_mean(
        self,
        value: np.ndarray,
        name: str,
    ) -> np.ndarray:

        result = np.asarray(
            value,
            dtype=np.float32,
        ).reshape(-1)

        if (
            result.size
            != self.layout.total_numel
        ):
            raise ValueError(
                f"{name} contains {result.size} values, "
                f"expected {self.layout.total_numel}"
            )

        if not np.all(
            np.isfinite(
                result
            )
        ):
            raise ValueError(
                f"{name} contains non-finite values"
            )

        return result

    def _validated_precision(
        self,
        value: np.ndarray,
        name: str,
    ) -> np.ndarray:

        result = np.asarray(
            value,
            dtype=np.float64,
        ).reshape(-1)

        if (
            result.size
            != self.layout.total_numel
        ):
            raise ValueError(
                f"{name} contains {result.size} values, "
                f"expected {self.layout.total_numel}"
            )

        if not np.all(
            np.isfinite(
                result
            )
        ):
            raise ValueError(
                f"{name} contains non-finite values"
            )

        if np.any(
            result <= 0.0
        ):
            raise ValueError(
                f"{name} must be strictly positive"
            )

        return np.clip(
            result,
            float(
                self.model_cfg.min_precision
            ),
            float(
                self.model_cfg.max_precision
            ),
        ).astype(np.float64)

    @staticmethod
    def _gradient_stats(
        parameters: Sequence[
            torch.nn.Parameter
        ],
    ) -> tuple[
        float,
        float,
    ]:

        squared_l2 = 0.0
        maximum = 0.0

        for parameter in parameters:

            if parameter.grad is None:
                continue

            gradient = (
                parameter.grad
                .detach()
            )

            squared_l2 += float(
                torch.sum(
                    gradient
                    .to(torch.float64)
                    .square()
                )
                .cpu()
            )

            if gradient.numel() > 0:
                maximum = max(
                    maximum,
                    float(
                        torch.max(
                            torch.abs(
                                gradient
                            )
                        )
                        .cpu()
                    ),
                )

        return (
            float(
                np.sqrt(
                    squared_l2
                )
            ),
            float(
                maximum
            ),
        )

    def _seed(
        self,
        seed: int,
    ) -> None:

        seed = int(
            seed
        )

        np.random.seed(
            seed % (2**32)
        )

        torch.manual_seed(
            seed
        )

        if (
            self.device.type
            == "cuda"
        ):
            torch.cuda.manual_seed_all(
                seed
            )
