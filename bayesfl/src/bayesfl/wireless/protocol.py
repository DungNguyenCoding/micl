"""Version 2 mixed dense/sparse packets bound to exact integer grants."""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np
from bayesfl.wireless import PROTOCOL
from bayesfl.wireless.payload import integer, positive, payload_bytes
from bayesfl.posterior.packing import pack_fola
from bayesfl.posterior.sparse import (dimension, flatten, unflatten, posterior_vectors,
    validate_raw, snapshot_id, layout_manifest, encode_packet)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def binding(global_arrays, layout, *, round_id: int, client_id: int, run_token: str, posterior=True) -> dict:
    rep = "mean_and_raw_precision" if posterior else "mean_only"
    return dict(protocol=PROTOCOL, round_id=integer(round_id, "round_id"),
                client_id=integer(client_id, "client_id", 0), run_token=str(run_token),
                base_snapshot_id=snapshot_id(global_arrays, layout, round_id, representation=rep),
                layout_id=layout_manifest(layout, representation=rep)["layout_id"])


def make_grant(base: dict, *, count: int, d: int, bandwidth_hz: float,
               window_s: float, efficiency: float, posterior: bool = True) -> dict:
    count = integer(count, "count")
    size = payload_bytes(count, d, posterior=posterior)
    bw = positive(bandwidth_hz, "bandwidth_hz")
    U, c = positive(window_s, "window_s"), positive(efficiency, "efficiency")
    duration = 8 * size / bw / c
    if duration > U and not math.isclose(duration, U, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Granted bandwidth cannot carry the integer payload within the upload interval")
    grant = dict(base, count=count, d=integer(d, "d"), bandwidth_hz=bw,
                 window_s=U, efficiency=c, posterior=bool(posterior), payload_bytes=size)
    grant["grant_id"] = hashlib.sha256(canonical(grant)).hexdigest()
    return grant


def validate_grant(grant: dict) -> None:
    copied = dict(grant)
    digest = copied.pop("grant_id", None)
    if digest != hashlib.sha256(canonical(copied)).hexdigest():
        raise ValueError("Grant digest mismatch")
    if copied.get("protocol") != PROTOCOL or type(copied.get("posterior")) is not bool:
        raise ValueError("Invalid grant protocol/representation")
    expected = make_grant({k: copied[k] for k in ("protocol", "round_id", "client_id", "run_token", "base_snapshot_id", "layout_id")},
                          count=copied["count"], d=copied["d"], bandwidth_hz=copied["bandwidth_hz"],
                          window_s=copied["window_s"], efficiency=copied["efficiency"], posterior=copied["posterior"])
    if expected != grant:
        raise ValueError("Malformed or extra grant fields")


def encode(mu: np.ndarray, precision: np.ndarray | None, mask: np.ndarray, grant: dict):
    validate_grant(grant)
    d, m = grant["d"], grant["count"]
    if mu.dtype != np.float32 or mu.shape != (d,) or not np.isfinite(mu).all():
        raise ValueError("Invalid local means")
    if mask.dtype != bool or mask.shape != (d,) or int(mask.sum()) != m:
        raise ValueError("Mask differs from integer grant")
    if grant["posterior"]:
        validate_raw(mu, precision)
        arrays = [mu.copy(), precision.copy()] if m == d else encode_packet(mu, precision, mask)
    else:
        if m != d or precision is not None:
            raise ValueError("FedAvg requires a dense mean-only packet")
        arrays = [mu.copy()]
    if sum(a.nbytes for a in arrays) != grant["payload_bytes"]:
        raise RuntimeError("Encoder/payload contract mismatch")
    metadata = {k: grant[k] for k in ("protocol", "round_id", "client_id", "run_token", "base_snapshot_id", "layout_id", "count", "d", "grant_id")}
    metadata["packet_format"] = "dense" if m == d else "bitmap"
    metadata["covariance_representation"] = "precision" if grant["posterior"] else "none"
    return arrays, metadata


def decode(arrays, metadata, *, grant, global_arrays, layout):
    validate_grant(grant)
    d, m = grant["d"], grant["count"]
    if d != dimension(layout):
        raise ValueError("Grant/layout mismatch")
    expected_binding = binding(global_arrays, layout, round_id=grant["round_id"], client_id=grant["client_id"],
                               run_token=grant["run_token"], posterior=grant["posterior"])
    for key, value in expected_binding.items():
        if grant.get(key) != value or metadata.get(key) != value:
            raise ValueError(f"Packet/grant binding mismatch: {key}")
    for key in ("count", "d", "grant_id"):
        if metadata.get(key) != grant[key]:
            raise ValueError(f"Packet differs from grant: {key}")
    if type(metadata.get("count")) is not int or type(metadata.get("d")) is not int:
        raise ValueError("Count metadata must be integer")
    if metadata.get("packet_format") != ("dense" if m == d else "bitmap"):
        raise ValueError("Packet format mismatch")
    if metadata.get("covariance_representation") != ("precision" if grant["posterior"] else "none"):
        raise ValueError("Representation mismatch")
    if any(not isinstance(a, np.ndarray) or a.ndim != 1 for a in arrays):
        raise ValueError("Packet arrays must be one-dimensional ndarrays")
    if sum(a.nbytes for a in arrays) != grant["payload_bytes"]:
        raise ValueError("Observed packet bytes differ from the grant")
    if not grant["posterior"]:
        if len(arrays) != 1 or arrays[0].shape != (d,) or arrays[0].dtype != np.float32 or not np.isfinite(arrays[0]).all():
            raise ValueError("Malformed dense FedAvg packet")
        return unflatten(arrays[0], layout), np.ones(d, dtype=bool)
    if m == d:
        if len(arrays) != 2 or any(a.shape != (d,) for a in arrays):
            raise ValueError("Malformed dense posterior packet")
        validate_raw(*arrays)
        mu, prec = (a.copy() for a in arrays)
        mask = np.ones(d, dtype=bool)
    else:
        if len(arrays) != 3:
            raise ValueError("Sparse posterior needs bitmap/mean/precision arrays")
        bitmap, values, precisions = arrays
        if bitmap.dtype != np.uint8 or bitmap.shape != ((d + 7) // 8,):
            raise ValueError("Malformed bitmap")
        bits = np.unpackbits(bitmap, bitorder="little")
        if np.any(bits[d:]) or int(bits[:d].sum()) != m:
            raise ValueError("Bitmap padding or cardinality mismatch")
        if values.shape != (m,) or precisions.shape != (m,):
            raise ValueError("Selected value shape mismatch")
        validate_raw(values, precisions)
        mask = bits[:d].astype(bool)
        mu, prec = (a.copy() for a in posterior_vectors(global_arrays, layout))
        mu[mask], prec[mask] = values, precisions
    return pack_fola(unflatten(mu, layout), unflatten(prec, layout)), mask


def aggregate(decoded, masks, samples, *, global_arrays, layout, cfg):
    from bayesfl.strategies.common import normalized_example_weights, weighted_average_arrays
    from bayesfl.posterior.aggregation import aggregate_fola
    if len(decoded) != cfg.federation.num_clients or len(samples) != len(decoded) or len(masks) != len(decoded):
        raise ValueError("All-client aggregation is required")
    weights = normalized_example_weights(samples)
    if cfg.method == "fedavg":
        return weighted_average_arrays(decoded, weights)
    out = aggregate_fola(decoded, weights, layout=layout, cfg=cfg)
    omitted = ~np.logical_or.reduce(masks)
    if omitted.any():
        mu, p = posterior_vectors(out, layout)
        old_mu, old_p = posterior_vectors(global_arrays, layout)
        mu[omitted], p[omitted] = old_mu[omitted], old_p[omitted]
        out = pack_fola(unflatten(mu, layout), unflatten(p, layout))
    posterior_vectors(out, layout)
    return out
