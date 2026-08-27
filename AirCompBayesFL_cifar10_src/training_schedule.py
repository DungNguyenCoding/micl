"""Round-level learning-rate schedules shared by all FL methods.

The schedule is keyed by *logical* round so methods with different numbers of
physical Flower rounds (e.g. Proposed has rho/nu phases) receive the same
learning rate at the same FL round.
"""

from __future__ import annotations

import math

from config import TrainingConfig


def learning_rate_for_round(
    train_cfg: TrainingConfig,
    logical_round: int,
    total_logical_rounds: int,
) -> float:
    """Return the learning rate for one logical FL round.

    ``constant`` reproduces the legacy behavior. ``cosine`` starts exactly at
    ``learning_rate`` on logical round 1 and reaches ``min_learning_rate`` on
    the final logical round. Round 0 (initial evaluation) is treated as round 1.
    """
    base_lr = float(train_cfg.learning_rate)
    scheduler = str(train_cfg.lr_scheduler).strip().lower()

    if scheduler == "constant":
        return base_lr
    if scheduler != "cosine":
        raise ValueError(f"Unsupported lr_scheduler: {train_cfg.lr_scheduler!r}")

    min_lr = float(train_cfg.min_learning_rate)
    total = max(1, int(total_logical_rounds))
    if total <= 1:
        return base_lr

    round_index = min(max(1, int(logical_round)), total)
    progress = float(round_index - 1) / float(total - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine
