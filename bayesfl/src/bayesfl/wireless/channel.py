"""Deterministic fixed-PSD digital links; modeled time is not emulator time."""
from __future__ import annotations
import csv
import hashlib
import math
from pathlib import Path
import numpy as np
from bayesfl.posterior.sparse import dedicated_rng
from bayesfl.wireless.payload import integer, positive
from bayesfl.wireless.allocation import InfeasibleAllocation


class Channel:
    def __init__(self, wireless, num_clients: int, *, trace_path: Path | None = None):
        self.cfg = wireless
        self.k = num_clients
        self.trace = {}
        self.trace_sha256 = None
        if wireless.channel.mode == "trace":
            path = trace_path or Path(wireless.channel.trace_csv).expanduser()
            data = path.read_bytes()
            self.trace_sha256 = hashlib.sha256(data).hexdigest()
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    r, cid = int(row["round"]), int(row["client_id"])
                    if str(r) != row["round"].strip() or str(cid) != row["client_id"].strip():
                        raise ValueError("Trace round and client_id must be decimal integers")
                    if r < 1 or not 0 <= cid < num_clients or (r, cid) in self.trace:
                        raise ValueError("Invalid or duplicate channel-trace entry")
                    self.trace[r, cid] = (positive(row["uplink_gain"], "uplink_gain"),
                                          positive(row["downlink_gain"], "downlink_gain"))
            if not self.trace:
                raise ValueError("Channel trace is empty")

    def round(self, round_id: int) -> dict:
        round_id = integer(round_id, "round_id")
        cfg = self.cfg.channel
        gains_ul, gains_dl = [], []
        for cid in range(self.k):
            if cfg.mode == "trace":
                if (round_id, cid) not in self.trace:
                    raise ValueError(f"Missing channel trace for round={round_id}, client={cid}")
                ul, dl = self.trace[round_id, cid]
            else:
                ul = dl = float(cfg.path_gains[cid])
                if cfg.mode == "block_rayleigh":
                    fade = float(dedicated_rng(cfg.seed, cid, round_id, domain="wireless-ul-v2").exponential())
                    ul *= fade
                    dl *= fade if cfg.reciprocal_fading else float(
                        dedicated_rng(cfg.seed, cid, round_id, domain="wireless-dl-v2").exponential())
            gains_ul.append(positive(ul, "uplink gain"))
            gains_dl.append(positive(dl, "downlink gain"))
        snr_ul = [p * g / cfg.noise_psd_w_hz for p, g in zip(cfg.uplink_psd_w_hz, gains_ul)]
        snr_dl = [cfg.downlink_psd_w_hz * g / cfg.noise_psd_w_hz for g in gains_dl]
        cu = [math.log1p(positive(x, "uplink SNR")) / math.log(2) for x in snr_ul]
        cd = [math.log1p(positive(x, "downlink SNR")) / math.log(2) for x in snr_dl]
        for c in cu + cd:
            positive(c, "spectral efficiency")
        return dict(uplink_gain=gains_ul, downlink_gain=gains_dl, uplink_snr=snr_ul,
                    downlink_snr=snr_dl, efficiencies=cu, downlink_efficiency=min(cd))


def round_timing(cfg, *, samples, downlink_bytes: int, channel: dict) -> dict:
    w = cfg.wireless
    prep = [n * cfg.training.local_epochs * per_sample + fixed + score
            for n, per_sample, fixed, score in zip(samples, w.compute.seconds_per_sample_epoch,
                                                   w.compute.fixed_seconds, w.compute.score_seconds)]
    if len(prep) != cfg.federation.num_clients or not all(math.isfinite(x) and x >= 0 for x in prep):
        raise ValueError("Invalid preparation durations")
    dl = 8 * downlink_bytes / w.downlink_bandwidth_hz / channel["downlink_efficiency"]
    window = w.round_target_s - dl - max(prep) - w.control_reserve_s
    if not math.isfinite(window) or window <= 0:
        raise InfeasibleAllocation("No upload interval remains after multicast, preparation and control reservation")
    return dict(downlink_seconds=dl, preparation_seconds=prep, upload_window_seconds=window)
