"""PREPARE -> scalar queries -> exact grants -> packets -> release -> evaluate.

Only communicated packets enter aggregation. All methods use fixed-length slots;
early equal-bandwidth completion is idle time, not an unreported shorter round.
"""
from __future__ import annotations
import math
import time
import numpy as np
from bayesfl.communication import atomic_json
from bayesfl.logging_utils import CsvRecorder
from bayesfl.posterior.sparse import dimension
from bayesfl.wireless.allocation import allocate, InfeasibleAllocation
from bayesfl.wireless.oracle import Response
from bayesfl.wireless.payload import payload_bytes, payload_bits, minimum_count
from bayesfl.wireless.protocol import binding, canonical, make_grant, decode, aggregate
from bayesfl.wireless.channel import round_timing
from bayesfl.wireless.transport import ExchangeError


class RemoteOracles:
    def __init__(self, controller):
        self.controller = controller

    def counts(self, counts):
        responses = self.controller.exchange("tail", [{"count": int(m)} for m in counts])
        return [Response(r.metrics["count"], r.metrics["discarded_kl"]) for r in responses]

    def price(self, price, weights, window_s, efficiencies):
        responses = self.controller.exchange("price", [dict(price=price, weight=w, window_s=window_s, efficiency=c)
                                    for w, c in zip(weights, efficiencies)])
        return [Response(r.metrics["count"], r.metrics["discarded_kl"]) for r in responses]


class Controller:
    def __init__(self, cfg, state, channel, transport, samples, logger):
        self.cfg, self.state, self.channel, self.transport = cfg, state, channel, transport
        self.samples, self.logger = list(samples), logger
        if len(samples) != cfg.federation.num_clients or any(type(n) is not int or n <= 0 for n in samples):
            raise ValueError("One positive realized sample count per participating client is required")
        self.k = len(samples)
        self.weights = [n / sum(samples) for n in samples]
        self.d = dimension(state.layout)
        self.rounds = CsvRecorder(state.run_dir / "metrics/wireless_rounds.csv")
        self.clients = CsvRecorder(state.run_dir / "metrics/wireless_clients.csv")
        self.bindings, self.control_s, self.grants = [], 0.0, None

    def exchange(self, phase, additions=None, parameters=None):
        state, w = self.state, self.cfg.wireless
        additions = additions or [{} for _ in range(self.k)]
        ins = [dict(base, phase=phase, **extra) for base, extra in zip(self.bindings, additions)]
        request_bytes = len(canonical(ins))
        request_time = 8 * request_bytes / w.control_rate_bps + w.control_exchange_latency_s
        if self.control_s + request_time > w.control_reserve_s:
            raise InfeasibleAllocation(f"Control reservation exhausted before {phase}; no hidden deadline extension")
        state.event("control_request", control_bytes=request_bytes, modeled_seconds=request_time, operation=phase)
        self.control_s += request_time
        failure = None
        try:
            replies = self.transport.exchange(parameters or [], ins, state.pending_round)
        except ExchangeError as exc:
            replies, failure = exc.replies, exc
            state.incomplete_accounting = True
        # Charge received arrays before checking their contents, even if invalid.
        durations, energies = [], []
        for cid in sorted(replies):
            r = replies[cid]
            size = sum(a.nbytes for a in r.arrays)
            control = len(canonical(dict(client_id=cid, num_examples=r.num_examples, metrics=r.metrics)))
            seconds = 8 * control / w.control_rate_bps
            state.event("reply", uplink_bytes=size, control_bytes=control,
                        modeled_seconds=seconds, client_id=cid, operation=phase)
            self.control_s += seconds
            if phase == "upload" and self.grants is not None:
                g = self.grants[cid]
                t = 8 * size / g["bandwidth_hz"] / g["efficiency"]
                durations.append(t)
                energies.append(w.channel.uplink_psd_w_hz[cid] * g["bandwidth_hz"] * t)
        if phase == "upload" and durations:
            state.event("upload_airtime", modeled_seconds=max(durations), energy_j=math.fsum(energies),
                        completed_replies=len(durations))
        if failure is not None:
            raise failure
        if set(replies) != set(range(self.k)):
            state.incomplete_accounting = True
            raise RuntimeError("Not all clients replied")
        if self.control_s > w.control_reserve_s:
            raise InfeasibleAllocation("Observed canonical control cost exceeded reserved time")
        out = [replies[cid] for cid in range(self.k)]
        for cid, r in enumerate(out):
            if type(r.num_examples) is not int or r.num_examples != self.samples[cid]:
                raise ValueError("Client sample count differs from the pinned partition")
            for key, value in self.bindings[cid].items():
                if r.metrics.get(key) != value:
                    raise ValueError(f"Mismatched {phase} response binding: {key}")
            if phase != "upload" and r.arrays:
                raise ValueError("Control response leaked ungranted model/score arrays")
        return out

    def step(self, round_id):
        started = time.perf_counter()
        s, w = self.state, self.cfg.wireless
        if s.pending_round is not None or round_id != s.round_id + 1:
            raise ValueError("Wireless rounds must execute once in order")
        ch = self.channel.round(round_id)
        dl_bytes = sum(a.nbytes for a in s.current)
        timing = round_timing(self.cfg, samples=self.samples, downlink_bytes=dl_bytes, channel=ch)
        U, cs = timing["upload_window_seconds"], ch["efficiencies"]
        required = self.d if w.policy == "dense" else minimum_count(self.d)
        if math.fsum(payload_bits(required, self.d, posterior=self.cfg.method == "fola") / U / c for c in cs) > w.uplink_bandwidth_hz:
            # Known from channel/sizes before spending modeled data-plane time.
            raise InfeasibleAllocation("Known minimum/dense all-client payload is infeasible for this slot")
        s.pending_round = round_id
        before_bytes, before_time, before_control = s.total_bytes, s.elapsed_s, s.control_bytes
        self.control_s, self.grants = 0.0, None
        self.bindings = [binding(s.current, s.layout, round_id=round_id, client_id=i,
                         run_token=s.run_token, posterior=self.cfg.method == "fola") for i in range(self.k)]
        s.event("multicast", downlink_bytes=dl_bytes, modeled_seconds=timing["downlink_seconds"],
                energy_j=w.channel.downlink_psd_w_hz * w.downlink_bandwidth_hz * timing["downlink_seconds"],
                recipients=self.k)
        # Explicit modeled parallel full preparation. Failure is conservatively
        # charged the reserved barrier; actual partial-device time is not measured.
        s.event("preparation_barrier", modeled_seconds=max(timing["preparation_seconds"]),
                timing_kind="model_not_wall_clock")
        prepared = self.exchange("prepare", parameters=s.current)
        for r in prepared:
            if r.metrics.get("d") != self.d:
                raise ValueError("Client prepared an incompatible dimension")
        allocation_start = time.perf_counter()
        allocation = allocate(RemoteOracles(self), d=self.d, weights=self.weights, window_s=U,
                              efficiencies=cs, bandwidth_hz=w.uplink_bandwidth_hz, policy=w.policy,
                              settings=w.solver, posterior=self.cfg.method == "fola")
        allocation_wall = time.perf_counter() - allocation_start
        self.grants = [make_grant(base, count=m, d=self.d, bandwidth_hz=b, window_s=U, efficiency=c,
                                 posterior=self.cfg.method == "fola")
                       for base, m, b, c in zip(self.bindings, allocation.counts, allocation.bandwidth_hz, cs)]
        uploads = self.exchange("upload", [dict(grant_json=canonical(g).decode()) for g in self.grants])
        decoded, masks = [], []
        for i, reply in enumerate(uploads):
            arrays, mask = decode(reply.arrays, reply.metrics, grant=self.grants[i], global_arrays=s.current, layout=s.layout)
            if self.cfg.method == "fola" and self.cfg.fola.mode != "paper_reference":
                from bayesfl.posterior.sparse import posterior_vectors
                if np.any(posterior_vectors(arrays, s.layout)[1] <= 0):
                    raise ValueError("Online recurrence requires positive precision")
            decoded.append(arrays)
            masks.append(mask)
        output = aggregate(decoded, masks, self.samples, global_arrays=s.current, layout=s.layout, cfg=self.cfg)
        self.exchange("release")
        elapsed = s.elapsed_s - before_time
        idle = w.round_target_s - elapsed
        if idle < -1e-10 * max(1.0, w.round_target_s):
            raise RuntimeError("Modeled round missed target despite allocation/control checks")
        s.event("slot_idle", modeled_seconds=max(0.0, idle))
        observed = s.total_bytes - before_bytes
        expected = dl_bytes + sum(g["payload_bytes"] for g in self.grants)
        if observed != expected:
            raise RuntimeError(f"Observed arrays {observed} differ from granted {expected}")
        actual_tails = [float(r.metrics["actual_discarded_kl"]) for r in uploads]
        if any(not math.isfinite(x) or x < 0 for x in actual_tails):
            raise ValueError("Invalid actual discarded KL report")
        for i, (reply, grant) in enumerate(zip(uploads, self.grants)):
            duration = payload_bits(grant["count"], self.d, posterior=self.cfg.method == "fola") / grant["bandwidth_hz"] / cs[i]
            self.clients.append(dict(round=round_id, client_id=i, num_examples=self.samples[i], weight=self.weights[i],
                keep_count=grant["count"], d=self.d, keep_ratio=grant["count"]/self.d,
                bandwidth_hz=grant["bandwidth_hz"], uplink_gain=ch["uplink_gain"][i], downlink_gain=ch["downlink_gain"][i],
                uplink_snr=ch["uplink_snr"][i], spectral_efficiency=cs[i],
                modeled_preparation_s=timing["preparation_seconds"][i], modeled_upload_s=duration,
                upload_array_bytes=grant["payload_bytes"], implied_tx_power_w=w.channel.uplink_psd_w_hz[i]*grant["bandwidth_hz"],
                modeled_tx_energy_j=w.channel.uplink_psd_w_hz[i]*grant["bandwidth_hz"]*duration,
                topm_tail_kl=allocation.discarded_kl[i], actual_discarded_kl=actual_tails[i],
                global_floor_fraction=reply.metrics["score_global_floored_fraction"],
                local_floor_fraction=reply.metrics["score_local_floored_fraction"],
                local_steps=reply.metrics["local_steps"],
                preparation_wall_seconds=prepared[i].metrics["preparation_wall_seconds"]))
        s.current, s.round_id, s.pending_round = output, round_id, None
        s.elapsed_s = round_id * w.round_target_s  # canonical fixed-slot clock, avoids sum drift
        s.local_steps += sum(int(r.metrics["local_steps"]) for r in uploads)
        s.examples_seen += sum(self.samples) * self.cfg.training.local_epochs
        s.last_round_stats = dict(round_array_bytes=observed, round_control_bytes=s.control_bytes-before_control,
            modeled_round_seconds=w.round_target_s, modeled_downlink_seconds=timing["downlink_seconds"],
            modeled_preparation_seconds=max(timing["preparation_seconds"]),
            modeled_upload_window_seconds=U, modeled_control_seconds=self.control_s,
            modeled_idle_seconds=max(0.0, idle), allocated_bandwidth_hz=math.fsum(allocation.bandwidth_hz),
            mean_keep_ratio=math.fsum(allocation.counts)/self.k/self.d,
            topm_reconstruction_objective=allocation.objective,
            actual_reconstruction_objective=math.fsum(a*b for a,b in zip(self.weights, actual_tails)),
            dual_lower_bound_diagnostic=allocation.lower_bound_diagnostic,
            objective_gap_diagnostic=allocation.gap_diagnostic, certified_optimal=False,
            price_queries=allocation.price_queries, allocation_status=allocation.status,
            allocation_wall_seconds=allocation_wall, round_wall_time_seconds=time.perf_counter()-started)
        self.rounds.append(dict(round=round_id, **s.evaluation_fields()))
        root = s.run_dir / "wireless/allocations"
        root.mkdir(exist_ok=True)
        atomic_json(root / f"round_{round_id:06d}.json", dict(round=round_id, allocation=allocation.to_dict(),
                     channel=ch, timing=timing, actual_discarded_kl=actual_tails))
        self.logger.info("Wireless round %d: arrays=%d modeled_s=%.6f keep=%.5f policy=%s status=%s queries=%d",
            round_id, observed, w.round_target_s, s.last_round_stats["mean_keep_ratio"], w.policy, allocation.status, allocation.price_queries)

    def run(self, evaluate):
        try:
            if self.state.round_id == 0 and not (self.state.run_dir/"wireless_resume/latest.json").exists():
                evaluate(0, self.state.current)
            while self.state.stop_reason() == "running":
                self.step(self.state.round_id + 1)
                evaluate(self.state.round_id, self.state.current)
            self.state.finalize()
        except Exception as exc:
            self.state.finalize(failed=str(exc))
            raise
