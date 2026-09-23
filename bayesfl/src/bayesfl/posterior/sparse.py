"""Post-training coordinate selection and validated packed-bitmap FOLA packets.

The baseline stores raw precision/omega. It is NEVER overwritten with the
score-only precision floor. A floor_for_score run uses surrogate Gaussians
N(mu, 1/max(omega, floor)) for selection only; it is not raw zero-omega KL.
No Flower imports: the actual client and server use these same functions.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from bayesfl.config import ExperimentConfig
from bayesfl.posterior.packing import ParameterLayout, pack_fola, unpack_fola

PROTOCOL = "sparse_fola_bitmap_precision_v1"
BITORDER = "little"
RULES = {"kl_global_local", "kl_local_global", "random"}


def dimension(layout: ParameterLayout) -> int:
    return sum(math.prod(s) for s in layout.shapes)


def layout_manifest(layout: ParameterLayout, *, representation: str = "mean_and_raw_precision") -> dict:
    offset, entries = 0, []
    for name, shape in zip(layout.names, layout.shapes):
        count = math.prod(shape)
        entries.append(dict(name=name, shape=list(shape), offset=offset, count=count,
                            representation=representation, dtype="float32"))
        offset += count
    if offset <= 0 or len(set(layout.names)) != len(layout.names):
        raise ValueError("Layout must have nonempty, uniquely named coordinates")
    manifest = dict(version=1, d=offset, entries=entries, order="named_parameters/C",
                    auxiliary_policy="baseline_named_parameters_only", bitmap_bitorder=BITORDER)
    manifest["layout_id"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest


def flatten(arrays: Sequence[np.ndarray], layout: ParameterLayout) -> np.ndarray:
    layout.validate(arrays)
    if any(a.dtype != np.float32 for a in arrays):
        raise ValueError("FOLA transport requires float32; implicit wire casts are forbidden")
    return np.ascontiguousarray(np.concatenate([a.reshape(-1) for a in arrays]))


def unflatten(vector: np.ndarray, layout: ParameterLayout) -> list[np.ndarray]:
    if vector.ndim != 1 or vector.size != dimension(layout):
        raise ValueError("Vector/layout dimension mismatch")
    out, offset = [], 0
    for shape in layout.shapes:
        count = math.prod(shape)
        out.append(vector[offset:offset + count].reshape(shape).copy())
        offset += count
    return out


def posterior_vectors(arrays, layout):
    means, precs = unpack_fola(arrays, layout)
    mu, precision = flatten(means, layout), flatten(precs, layout)
    validate_raw(mu, precision)
    return mu, precision


def validate_raw(mu: np.ndarray, precision: np.ndarray) -> None:
    if mu.ndim != 1 or precision.shape != mu.shape or mu.dtype != np.float32 or precision.dtype != np.float32:
        raise ValueError("Means and raw precisions must be aligned one-dimensional float32 arrays")
    if not np.isfinite(mu).all() or not np.isfinite(precision).all() or np.any(precision < 0):
        raise ValueError("Raw FOLA state contains nonfinite means or negative/nonfinite precision")


def snapshot_id(arrays: Sequence[np.ndarray], layout: ParameterLayout, round_id: int,
                *, representation: str = "mean_and_raw_precision") -> str:
    h = hashlib.sha256(f"{layout_manifest(layout, representation=representation)['layout_id']}:{round_id}".encode())
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def keep_count(d: int, ratio: float) -> int:
    if isinstance(d, bool) or not isinstance(d, Integral) or d <= 0:
        raise ValueError("d must be a positive integer")
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("keep ratio must be in (0,1]")
    # Python binary64 product followed by ceil, shared by codec and budgeting.
    return min(int(d), math.ceil(float(ratio) * int(d)))


def diagonal_kl(mu_p, var_p, mu_q, var_q) -> np.ndarray:
    """Per-coordinate KL(P || Q), evaluated in float64, never a training loss."""
    mp, vp, mq, vq = (np.asarray(a, dtype=np.float64) for a in (mu_p, var_p, mu_q, var_q))
    if mp.ndim != 1 or mp.size == 0 or any(a.shape != mp.shape for a in (vp, mq, vq)):
        raise ValueError("KL requires nonempty aligned vectors")
    if any(not np.isfinite(a).all() for a in (mp, vp, mq, vq)) or np.any(vp <= 0) or np.any(vq <= 0):
        raise ValueError("KL requires finite means and finite strictly positive variances")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        s = np.log(vp) - np.log(vq)
        score = 0.5 * (np.expm1(s) - s + (mp - mq) ** 2 / vq)
    if not np.isfinite(score).all() or np.any(score < -1e-10):
        raise FloatingPointError("Invalid Gaussian KL score")
    return np.maximum(score, 0.0)


def score_variance(precision: np.ndarray, policy: str, floor: float) -> tuple[np.ndarray, float]:
    p = np.asarray(precision, dtype=np.float64)
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Invalid raw precision")
    if policy == "strict":
        if np.any(p <= 0):
            raise ValueError("A finite Gaussian KL cannot be computed from zero raw precision")
        changed = 0.0
    elif policy == "floor_for_score":
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError("Invalid score-only floor")
        changed = float(np.mean(p < floor))
        p = np.maximum(p, floor)
    else:
        raise ValueError("Unknown scoring precision policy")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        var = 1.0 / p
    if not np.isfinite(var).all() or np.any(var <= 0):
        raise ValueError("Scoring variances cannot be represented in float64")
    return var, changed


def dedicated_rng(seed: int, client_id: int, round_id: int, *, domain: str = "mask") -> np.random.Generator:
    # No Python hash(), no consumption of NumPy/Python/Torch global RNG streams.
    digest = hashlib.sha256(f"bayesfl-v1:{domain}:{int(seed)}:{int(client_id)}:{int(round_id)}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def select_mask(scores: np.ndarray | None, d: int, m: int, *, rng=None) -> np.ndarray:
    if not isinstance(m, Integral) or isinstance(m, bool) or not 0 <= m <= d:
        raise ValueError("m must be an integer in [0,d]")
    mask = np.zeros(d, dtype=bool)
    if scores is None:
        if rng is None:
            raise ValueError("Random selection needs a dedicated generator")
        mask[rng.choice(d, size=m, replace=False)] = True
    else:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (d,) or not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError("Invalid selection scores")
        # Linear-time partition; ties match sorting by (-score, global index).
        if m == d:
            mask[:] = True
        elif m > 0:
            threshold = np.partition(scores, d - m)[d - m]
            mask[scores > threshold] = True
            ties = np.flatnonzero(scores == threshold)
            mask[ties[:m - int(mask.sum())]] = True
    return mask


def encode_packet(local_mu: np.ndarray, local_precision: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    validate_raw(local_mu, local_precision)
    if mask.dtype != np.bool_ or mask.shape != local_mu.shape:
        raise ValueError("Mask must be an aligned boolean vector")
    # Boolean indexing is ascending coordinate order, never score order.
    arrays = [np.packbits(mask, bitorder=BITORDER),
              np.ascontiguousarray(local_mu[mask]), np.ascontiguousarray(local_precision[mask])]
    assert sum(a.nbytes for a in arrays) == 8 * int(mask.sum()) + (mask.size + 7) // 8
    return arrays


def compress_update(global_arrays, local_arrays, layout, cfg: ExperimentConfig, *, round_id: int,
                    client_id: int, base_snapshot_id: str) -> tuple[list[np.ndarray], dict]:
    start = time.perf_counter()
    if cfg.compression.selection_rule not in RULES:
        raise ValueError("Sparse upload requires an explicit selection rule")
    if snapshot_id(global_arrays, layout, round_id) != base_snapshot_id:
        raise ValueError("Broadcast snapshot does not match the fit instruction")
    global_mu, global_p = posterior_vectors(global_arrays, layout)
    local_mu, local_p = posterior_vectors(local_arrays, layout)
    if cfg.fola.mode != "paper_reference" and (np.any(global_p <= 0) or np.any(local_p <= 0)):
        raise ValueError("online_recurrence requires positive precision")
    gv, gfrac = score_variance(global_p, cfg.compression.precision_policy, cfg.score_precision_floor)
    lv, lfrac = score_variance(local_p, cfg.compression.precision_policy, cfg.score_precision_floor)
    rule = cfg.compression.selection_rule
    scores = None
    if rule == "kl_global_local":
        scores = diagonal_kl(global_mu, gv, local_mu, lv)
    elif rule == "kl_local_global":
        scores = diagonal_kl(local_mu, lv, global_mu, gv)
    d = global_mu.size
    m = keep_count(d, cfg.compression.keep_ratio)
    mask = select_mask(scores, d, m, rng=dedicated_rng(cfg.runtime.seed, client_id, round_id))
    packet = encode_packet(local_mu, local_p, mask)
    metadata = dict(protocol=PROTOCOL, round_id=int(round_id), client_id=int(client_id),
                    base_snapshot_id=base_snapshot_id, layout_id=layout_manifest(layout)["layout_id"],
                    covariance_representation="precision", d=int(d), m=int(m),
                    selection_rule=rule, score_dtype="float64", bitmap_policy="always",
                    precision_policy=cfg.compression.precision_policy,
                    score_precision_floor=cfg.score_precision_floor,
                    keep_ratio_requested=cfg.compression.keep_ratio, keep_ratio_actual=m / d,
                    score_global_floored_fraction=gfrac, score_local_floored_fraction=lfrac,
                    selection_time_seconds=time.perf_counter() - start)
    return packet, metadata


def decode_packet(arrays: Sequence[np.ndarray], metadata: Mapping, *, global_arrays,
                  layout: ParameterLayout, cfg: ExperimentConfig, round_id: int,
                  client_id: int, base_snapshot_id: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Reconstruct exclusively from the packet and immutable broadcast snapshot."""
    d = dimension(layout)
    expected = dict(protocol=PROTOCOL, round_id=int(round_id), client_id=int(client_id),
                    base_snapshot_id=base_snapshot_id, layout_id=layout_manifest(layout)["layout_id"],
                    covariance_representation="precision", d=d,
                    m=keep_count(d, cfg.compression.keep_ratio),
                    selection_rule=cfg.compression.selection_rule, score_dtype="float64",
                    bitmap_policy="always", precision_policy=cfg.compression.precision_policy,
                    score_precision_floor=cfg.score_precision_floor)
    for key, value in expected.items():
        if key not in metadata or metadata[key] != value:
            raise ValueError(f"Invalid sparse metadata {key}: expected {value!r}")
    for key in ("round_id", "client_id", "d", "m"):
        if isinstance(metadata[key], bool) or not isinstance(metadata[key], Integral):
            raise ValueError(f"{key} must be an integer")
    if len(arrays) != 3:
        raise ValueError("Sparse packets contain exactly bitmap/means/raw precisions")
    bitmap, values, precs = arrays
    if bitmap.dtype != np.uint8 or bitmap.ndim != 1 or bitmap.size != (d + 7) // 8:
        raise ValueError("Invalid packed bitmap storage")
    bits = np.unpackbits(bitmap, bitorder=BITORDER)
    if np.any(bits[d:]):
        raise ValueError("Nonzero bitmap padding")
    mask = bits[:d].astype(bool)
    m = int(mask.sum())
    if m != expected["m"] or values.shape != (m,) or precs.shape != (m,):
        raise ValueError("Bitmap cardinality/value lengths do not match the round")
    validate_raw(values, precs)
    if cfg.fola.mode != "paper_reference" and np.any(precs <= 0):
        raise ValueError("online_recurrence packet has nonpositive precision")
    mu, precision = posterior_vectors(global_arrays, layout)  # copies, never local untransmitted state
    mu[mask], precision[mask] = values, precs
    return pack_fola(unflatten(mu, layout), unflatten(precision, layout)), mask
