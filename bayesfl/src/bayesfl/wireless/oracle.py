"""Client-local KL tail oracle. Server-facing responses contain scalars only."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from bayesfl.wireless.payload import integer, payload_bits, positive


@dataclass(frozen=True)
class Response:
    count: int
    discarded_kl: float


class ScoreState:
    def __init__(self, scores: np.ndarray):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError("Scores must be a finite nonnegative vector")
        self.scores = scores.copy()
        # Stable sort preserves ascending coordinate index on equal scores.
        self.order = np.argsort(-scores, kind="stable")
        self.sorted_scores = scores[self.order]
        self.tails = np.empty(scores.size + 1, dtype=np.float64)
        self.tails[-1] = 0.0
        with np.errstate(over="raise", invalid="raise"):
            self.tails[:-1] = np.cumsum(self.sorted_scores[::-1], dtype=np.float64)[::-1]
        if not np.isfinite(self.tails).all():
            raise FloatingPointError("KL tails overflowed float64")
        self.negative_sorted = -self.sorted_scores

    @property
    def d(self) -> int:
        return self.scores.size

    def at_count(self, m: int) -> Response:
        m = integer(m, "keep_count")
        if m > self.d:
            raise ValueError("Count exceeds d")
        return Response(m, float(self.tails[m]))

    def at_price(self, price: float, weight: float, window_s: float, spectral_efficiency: float) -> Response:
        price = float(price)
        if not math.isfinite(price) or price < 0:
            raise ValueError("Price must be finite and nonnegative")
        weight = positive(weight, "weight")
        U, c = positive(window_s, "window_s"), positive(spectral_efficiency, "spectral_efficiency")
        if self.d == 1:
            return self.at_count(1)
        # Dividing sequentially avoids overflowing the denominator unnecessarily.
        theta = price / weight / U / c * 64.0
        if not math.isfinite(theta):
            if math.isinf(theta) and theta > 0:
                n = 0
            else:
                raise FloatingPointError("Invalid price threshold")
        else:
            n = int(np.searchsorted(self.negative_sorted, -theta, side="left"))
        sparse = min(self.d - 1, max(1, n))
        sparse_bw = payload_bits(sparse, self.d) / U / c
        dense_bw = payload_bits(self.d, self.d) / U / c
        a = weight * float(self.tails[sparse]) + price * sparse_bw
        b = price * dense_bw
        if not math.isfinite(a) or not math.isfinite(b):
            raise FloatingPointError("Priced objective overflow; lower initial price or rescale units")
        # Strict comparison; no numerical epsilon changes the local optimizer.
        choose_dense = b < a or (b == a and (dense_bw, 0.0, self.d) < (sparse_bw, self.tails[sparse], sparse))
        return self.at_count(self.d if choose_dense else sparse)

    def mask(self, m: int) -> np.ndarray:
        self.at_count(m)
        out = np.zeros(self.d, dtype=bool)
        out[self.order[:m]] = True
        return out


class LocalOracles:
    """Frozen-state/testing adapter. Production uses scalar RPC responses instead."""
    def __init__(self, states):
        self.states = list(states)

    def counts(self, counts):
        if len(counts) != len(self.states):
            raise ValueError("Client count mismatch")
        return [s.at_count(m) for s, m in zip(self.states, counts)]

    def price(self, price, weights, window_s, efficiencies):
        return [s.at_price(price, w, window_s, c) for s, w, c in zip(self.states, weights, efficiencies)]
