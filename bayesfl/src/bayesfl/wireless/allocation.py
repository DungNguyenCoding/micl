"""Method 1: price coordination, separate dense option and feasible incumbent.

Float64 dual values are diagnostics, NOT certified mathematical lower bounds.
The deployed solver never enumerates the Cartesian count space.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from typing import Protocol, Sequence
from bayesfl.wireless.config import SolverConfig
from bayesfl.wireless.oracle import Response
from bayesfl.wireless.payload import integer, positive, payload_bits, minimum_count, largest_feasible_count


class InfeasibleAllocation(ValueError):
    pass


class Oracles(Protocol):
    def counts(self, counts: Sequence[int]) -> list[Response]: ...
    def price(self, price: float, weights, window_s: float, efficiencies) -> list[Response]: ...


@dataclass
class Allocation:
    counts: list[int]
    bandwidth_hz: list[float]
    discarded_kl: list[float]
    objective: float
    lower_bound_diagnostic: float
    gap_diagnostic: float
    price_queries: int
    status: str
    certified_optimal: bool = False

    def to_dict(self):
        return asdict(self)


def _inputs(d, weights, U, efficiencies, B):
    d = integer(d, "d")
    weights = [positive(w, "weight") for w in weights]
    cs = [positive(c, "spectral_efficiency") for c in efficiencies]
    if not weights or len(weights) != len(cs):
        raise ValueError("One positive weight/channel efficiency per client is required")
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Weights must be normalized once across all clients")
    return d, weights, positive(U, "window_s"), cs, positive(B, "bandwidth_hz")


def _responses(responses, k, d):
    if len(responses) != k:
        raise ValueError("Missing oracle response")
    for r in responses:
        integer(r.count, "response count")
        if r.count > d or not math.isfinite(r.discarded_kl) or r.discarded_kl < 0:
            raise ValueError("Invalid count/KL response")
        if r.count == d and r.discarded_kl != 0:
            raise ValueError("Dense KL tail must be zero")


def allocate(oracles: Oracles, *, d: int, weights, window_s: float,
             efficiencies, bandwidth_hz: float, policy: str = "joint",
             settings: SolverConfig | None = None, posterior: bool = True) -> Allocation:
    d, weights, U, cs, B = _inputs(d, weights, window_s, efficiencies, bandwidth_hz)
    k, settings = len(weights), settings or SolverConfig()

    def bw(counts):
        return [payload_bits(m, d, posterior=posterior) / U / c for m, c in zip(counts, cs)]

    def result(rs, bands, status, queries=0, lb=0.0):
        _responses(rs, k, d)
        f = math.fsum(w * r.discarded_kl for w, r in zip(weights, rs))
        # A small negative floating gap is not promoted to a certificate.
        return Allocation([r.count for r in rs], bands, [r.discarded_kl for r in rs],
                          f, lb, f - lb, queries, status, False)

    if policy == "dense":
        counts = [d] * k
        bands = bw(counts)
        if math.fsum(bands) > B:
            raise InfeasibleAllocation("Dense uploads cannot meet this deadline/bandwidth")
        return result([Response(d, 0.0) for _ in range(k)], bands, "all_dense")
    if not posterior:
        raise ValueError("Adaptive mean-only FedAvg is not implemented")
    if policy not in {"joint", "equal_bandwidth", "uniform"}:
        raise ValueError("Invalid allocation policy")
    minimum = [minimum_count(d)] * k
    min_bw = bw(minimum)
    if math.fsum(min_bw) > B:
        raise InfeasibleAllocation("Even minimum-payload all-client uploads exceed bandwidth")
    # Make the equal-share allocation conservative under binary64 summation.
    share = B / k
    while math.fsum([share] * k) > B:
        share = math.nextafter(share, 0.0)
    equal = [largest_feasible_count(share * U * c, d) for c in cs]
    # Recheck with the exact computation used for required Hz.
    for j, m in enumerate(equal):
        while m is not None and payload_bits(m, d) / U / cs[j] > share:
            if m == d:
                m = largest_feasible_count(math.nextafter(payload_bits(d, d), 0.0), d)
            elif m > 1:
                m -= 1
            else:
                m = None
        equal[j] = m
    if policy in {"equal_bandwidth", "uniform"}:
        if any(m is None for m in equal):
            raise InfeasibleAllocation("At least one equal-bandwidth client cannot upload even one coordinate")
        counts = equal if policy == "equal_bandwidth" else [min(equal)] * k
        rs = oracles.counts(counts)
        if [r.count for r in rs] != counts:
            raise ValueError("Oracle did not honor requested counts")
        # Version A really assigns equal bandwidth; early finish waits for slot end.
        return result(rs, [share] * k, policy)

    dense_bw = bw([d] * k)
    if math.fsum(dense_bw) <= B:
        return result([Response(d, 0.0) for _ in range(k)], dense_bw, "all_dense")
    rs = oracles.counts(minimum)
    if [r.count for r in rs] != minimum:
        raise ValueError("Minimum-count oracle mismatch")
    incumbent = result(rs, min_bw, "minimum_fallback")
    if all(m is not None for m in equal):
        rs = oracles.counts(equal)
        if [r.count for r in rs] != equal:
            raise ValueError("Equal-count oracle mismatch")
        candidate = result(rs, bw(equal), "equal_incumbent")
        if candidate.objective < incumbent.objective:
            incumbent = candidate

    lb, queries = 0.0, 0

    def query(price):
        nonlocal incumbent, lb, queries
        rs = oracles.price(price, weights, U, cs)
        _responses(rs, k, d)
        bands = bw([r.count for r in rs])
        demand = math.fsum(bands)
        f = math.fsum(w * r.discarded_kl for w, r in zip(weights, rs))
        dual = f + price * (demand - B)
        if not math.isfinite(dual) or not math.isfinite(f):
            raise FloatingPointError("Nonfinite allocator objective")
        lb = max(lb, dual)
        queries += 1
        if demand <= B and (f, demand) < (incumbent.objective, math.fsum(incumbent.bandwidth_hz)):
            incumbent = result(rs, bands, "feasible_price")
        return demand

    demand = query(0.0)
    status = "zero_distortion" if demand <= B else "bracket_limit"
    if demand > B:
        lo, hi = 0.0, positive(settings.initial_price, "initial_price")
        found = False
        for _ in range(integer(settings.max_bracket_queries, "max_bracket_queries")):
            demand = query(hi)
            if demand <= B:
                found = True
                break
            lo = hi
            if hi > float.fromhex("0x1.fffffffffffffp+1023") / 2:
                status = "price_overflow_limit"
                break
            hi *= 2
        if found:
            status = "iteration_limit"
            for _ in range(integer(settings.max_bisection_queries, "max_bisection_queries", 0)):
                gap = incumbent.objective - lb
                if 0 <= gap <= settings.gap_atol + settings.gap_rtol * max(1.0, incumbent.objective):
                    status = "numerical_gap_target"
                    break
                if hi - lo <= settings.price_atol + settings.price_rtol * max(abs(lo), abs(hi)):
                    status = "price_tolerance"
                    break
                mid = lo + (hi - lo) / 2
                if mid == lo or mid == hi:
                    status = "price_resolution"
                    break
                if query(mid) > B:
                    lo = mid
                else:
                    hi = mid
    incumbent.price_queries = queries
    incumbent.lower_bound_diagnostic = lb
    incumbent.gap_diagnostic = incumbent.objective - lb
    incumbent.status = status
    if incumbent.gap_diagnostic < -1e-9 * max(1.0, incumbent.objective):
        raise FloatingPointError("Dual diagnostic exceeds incumbent; numerical oracle inconsistency")
    if math.fsum(incumbent.bandwidth_hz) > B:
        raise RuntimeError("Allocator attempted to return an infeasible incumbent")
    return incumbent
