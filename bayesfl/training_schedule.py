"""Round-level learning-rate schedule with fixed cosine horizon."""

from __future__ import annotations

import math

from config import TrainingConfig


def learning_rate_for_round(train_cfg: TrainingConfig, server_round: int) -> float:
    """Return LR for 1-based Flower server round.

    Cosine uses 0-based r=server_round-1 and the fixed horizon H from
    training.lr_decay_rounds.  The run length does not alter the schedule.
    """
    base = float(train_cfg.learning_rate)
    mode = str(train_cfg.lr_scheduler).strip().lower()
    if mode == "constant":
        return base
    if mode != "cosine":
        raise ValueError(f"Unsupported lr_scheduler: {train_cfg.lr_scheduler!r}")
    minimum = float(train_cfg.min_learning_rate)
    horizon = max(1, int(train_cfg.lr_decay_rounds))
    if horizon == 1:
        return minimum
    r = min(max(int(server_round) - 1, 0), horizon - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * float(r) / float(horizon - 1)))
    return minimum + (base - minimum) * cosine
