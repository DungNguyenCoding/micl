# Validation: observed checks versus required implementation tests

## Checks actually performed on the supplied baseline

All checks used an extracted working copy of `baseline_src.zip`, not the live
research server. No training source was modified. No dataset benchmark or GPU
wireless experiment was launched.

| Check | Observed result | Evidence |
|---|---|---|
| Existing available pytest suite | 128 passed, 5 skipped, 9.82 seconds in captured run | `evidence/baseline_pytest.txt` |
| Primary archived CSV endpoint/byte checks | 24 endpoints passed; both datasets R150/R300 reference | `evidence/baseline_primary_endpoints.txt` |
| Resolved configurations | Read all 24 primary selected runs | `evidence/primary_configuration_audit.json` |
| Exact-rational small-instance mathematical checks | 600 cases; 34,076 local responses checked against all counts; 16,200 monotonic-demand checks; 16,800 dual/incumbent bound checks | `evidence/review_checks.json` |
| Tiny joint comparison spaces | 24,653 count combinations enumerated across cases | Same |
| Integer count -> floating ratio -> ceil probe | m25 becomes26 for d80202; m111 becomes112 for d878538 | Same |
| Full-retention v1 packet | d17 packet has139 bytes vs136 dense bytes (3-byte bitmap) | Same |
| Unknown root wireless config probe | Loads without error and loses wireless section | Same |

Mathematical checks included d2..6, one to three clients, ideal and byte-padded
bitmaps, b in {1,2,32}, zeros/ties and rational costs/prices. Tiny enumeration is a
**test oracle**, not a proposed deployed optimization method. This review's code
is not the production allocator and makes no universal/global-optimality claim.

The environment had CPU Torch2.10.0, torchvision0.25.0, NumPy2.3.5,
PyYAML6.0.3 and pytest9.0.2; exact interpreter/package metadata is in
`evidence/source_manifest.json`. Flower, Ray and Bayesian-Torch were unavailable.
The five skip reports cover Flower integration and Bayesian-Torch checks. These
results do not validate the pinned Python3.10/CUDA/Flower1.29 server stack. Do not
modify its dependencies to match this review environment.

## Repeat the read-only review probes

From the repository containing this additive documentation directory:

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PYTHONDONTWRITEBYTECODE=1 \
/home/micl/anaconda3/envs/dungndh/bin/python \
  docs/wireless/evidence/review_checks.py --root "$PWD"
```

The probe uses NumPy/PyYAML, existing baseline helpers and a disposable system
TEMP directory for one YAML. It prints JSON to the terminal and does not alter
training data, source, configs or metrics. A future implemented wireless section
will change the config probe outcome; that is not a backward-compatibility test
failure by itself.

The captured baseline test invocation was `python -m pytest -q -p no:cacheprovider`
with `PYTHONPATH=src` and `PYTHONDONTWRITEBYTECODE=1`, in the copied repository.
A live full suite can create its documented test artifacts; do not call it
strictly filesystem-read-only.

## Required before a production wireless release

### Numerical layer

Check stable KL and identical surrogate policy; reverse-cumulative tails; zero and
tied scores; integer counts; d2; m1 and md; sparse messages larger than dense;
exact byte padding for both real dimensions; minimum/all-dense/zero-distortion/
infeasible cases; finite input rejection; bracketing limits; overflow; immutable
feasible incumbent on every exit; declared floating tolerance; equality recovery.
Compare local responses with direct all-count evaluation on tiny problems. Check
bounds and feasible objective against tiny joint enumeration. Verify nonincreasing
actual demand, not assumed monotonic counts. A negative numerical UB-LB must be
reported/investigated, not silently clamped into a false certificate.

### Packet/protocol layer

Actual ndarray bytes equal allocator prediction. m remains integer across client,
grant, decoder and ledger. Exercise mixed dense/sparse senders in one round. Reject
wrong format/version, stale snapshot, altered m, duplicate replies, invalid padding,
wrong dtype, malformed shape, negative/nonfinite precision and incorrect client id.
The server reconstructs exclusively from received payload plus its broadcast.
No extra posterior or full score arrays may appear in scalar report transport.

### Learner preservation

At identical local states/masks, the new reconstruction/reducer equals the old
result, including epsilon and all-client omission restoration. Full retention gives
dense baseline arrays. Wireless-disabled initialization, training, seeded masks,
metrics and fingerprints stay unchanged. Test original paper_reference and any
other mode explicitly claimed as supported. Do not claim support for untested modes.

### Lifecycle on real pinned Flower/Ray

Run PREPARE, several PRICE queries and UPLOAD with a newly instantiated client
between messages. Assert exactly one trainer invocation, unchanged cached state,
no second model broadcast, no consumed training RNG during allocation, and correct
logical client identities. Test early query termination, stale grants, missing
cache and cleanup. Test concurrent clients without shared-cache leakage.
A mock client or an in-process dictionary does not certify Flower persistence.

### Clock/accounting layer

Separate emulator wall time, modeled max local preparation, multicast, control,
upload and aggregation assumptions. Verify bits/bytes and Hz/kHz/MHz conversions,
dB/linear and PSD/power conversions. Reject nonfinite/zero spectral efficiency,
negative U, bandwidth overuse and power-limit violations under the declared model.
Do not rescale infeasible grants to fit: that changes upload durations.
Use real partition sizes, not their mean for each client.

Test infeasibility before and after preparation. Preserve consumed cost of a failed
attempt. If fixed slots are selected, charge consistent slot time to Version A,
Version B and uniform controls; otherwise declare a different clock policy.

### Resume/reproducibility

Uninterrupted versus clean-resumed runs must agree on global arrays, masks/counts,
channel realizations/trace indices, true round/slot, time, model bytes, control
queries and objective metrics within stated numerical tolerances. Changes to
physical parameters, channel seed, packet version, deadline, solver policy and
local learner must invalidate a wireless continuation. Check disabled-wireless
legacy hashes and legacy resumes separately. Mid-round recovery is unsupported
until a transactional design is implemented and tested.

### Output, plotting and handoff

Preserve all legacy primary endpoints and filenames. New wireless plots must read
explicit modeled-time fields and exact recorded per-client allocations. Test missing
rounds, missing traces, mixed protocols, malformed counters and incomplete runs.
Update the exporter to capture required wireless metrics, trace/config hashes and
schemas. Its old 24-run check is not a wireless validation path.

Every implementation gate requires accompanying docs showing what passed, what
skipped, exact environment and untested scope. Do not copy these review test counts
as if they were validation of subsequently changed code.
