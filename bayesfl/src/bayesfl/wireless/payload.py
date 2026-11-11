"""Exact array payload sizes. Bits, bytes and integer counts are not ratios."""
from __future__ import annotations
import math
from numbers import Integral


def integer(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def positive(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is required and must be finite and positive") from exc
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return out


def payload_bytes(m: int, d: int, *, posterior: bool = True) -> int:
    d = integer(d, "d")
    m = integer(m, "m")
    if m > d:
        raise ValueError("m exceeds d")
    if not posterior:
        if m != d:
            raise ValueError("Wireless FedAvg supports dense uploads only")
        return 4 * d
    return 8 * d if m == d else 8 * m + (d + 7) // 8


def payload_bits(m: int, d: int, *, posterior: bool = True) -> int:
    return 8 * payload_bytes(m, d, posterior=posterior)


def minimum_count(d: int) -> int:
    """Minimum-payload endpoint, breaking an exact payload tie toward dense."""
    return d if payload_bits(d, d) <= payload_bits(1, d) else 1


def largest_feasible_count(capacity_bits: float, d: int) -> int | None:
    """Largest feasible count, checking the separate bitmap-free dense format."""
    d = integer(d, "d")
    cap = float(capacity_bits)
    if not math.isfinite(cap) or cap < 0:
        raise ValueError("Capacity must be finite and nonnegative")
    if payload_bits(d, d) <= cap:
        return d
    if d == 1 or payload_bits(1, d) > cap:
        return None
    m = min(d - 1, max(1, math.floor((cap - 8 * ((d + 7) // 8)) / 64)))
    while m > 1 and payload_bits(m, d) > cap:
        m -= 1
    while m < d - 1 and payload_bits(m + 1, d) <= cap:
        m += 1
    return m if payload_bits(m, d) <= cap else None
