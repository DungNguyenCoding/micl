# Communication accounting and budget checkpoints

Current model: float32 means and raw precision (4+4 bytes/coordinate), a little-bit-order packed bitmap for every sparse upload, one dense shared downlink per round, and one distinct uplink per participating client. This is **modeled payload accounting**, not a measured physical multicast or wireless/AirComp implementation.

```text
m = ceil(r*d)                         # same binary64 multiplication as existing encoder
bitmap_bytes = ceil(d/8)
C_dense_fola = 8*d*(K+1)
C_fedavg    = 4*d*(K+1)
C_sparse    = 8*d + K*(8*m + bitmap_bytes)
B           = R_dense*C_dense_fola
R_sparse    = floor(B/C_sparse)
```

K=20 in both primary profiles. Extra evaluation/initialization model-array traffic is zero under the recorded central-evaluation/server-local initialization convention. If participation, payload dtype, metadata accounting or auxiliary arrays change, recompute from the actual ledger instead of using these constants. GB = 10^9 bytes; disk GiB is a different unit.

Dense FOLA R150 and FedAvg R300 have equal modeled bytes. Sparse endpoints are the **last whole affordable round**, so their final bytes can be slightly below B. Verify both `R*C <= B` and `(R+1)*C > B` plus the actual CSV counter. Never interpolate over missing rounds or select the best test accuracy instead of the required endpoint.



## MNIST s10 (also s5)


Budget: **2,021,090,400 bytes = 2.021090400 GB**. d=80,202.

| Keep r | m | Bytes/round | Target round | Endpoint GB | Per-round saving |
| --- | --- | --- | --- | --- | --- |
| 0.75 | 60,152 | 10,466,456 | 193 | 2.020026008 | 22.3207% |
| 0.50 | 40,101 | 7,258,296 | 278 | 2.017806288 | 46.1308% |
| 0.25 | 20,051 | 4,050,296 | 498 | 2.017047408 | 69.9398% |
| 0.10 | 8,021 | 2,125,496 | 950 | 2.019221200 | 84.2251% |
| 0.05 | 4,011 | 1,483,896 | 1362 | 2.021066352 | 88.9869% |


## CIFAR-10 alpha=.01, f=1.0


Budget: **22,139,157,600 bytes = 22.139157600 GB**. d=878,538.

| Keep r | m | Bytes/round | Target round | Endpoint GB | Per-round saving |
| --- | --- | --- | --- | --- | --- |
| 0.75 | 658,904 | 114,649,304 | 193 | 22.127315672 | 22.3214% |
| 0.50 | 439,269 | 79,507,704 | 278 | 22.103141712 | 46.1309% |
| 0.25 | 219,635 | 44,366,264 | 499 | 22.138765736 | 69.9404% |
| 0.10 | 87,854 | 23,281,304 | 950 | 22.117238800 | 84.2262% |
| 0.05 | 43,927 | 16,252,984 | 1362 | 22.136564208 | 88.9881% |


## Do not confuse these cases

The r=.25 target is **MNIST R498** but **CIFAR R499**. Their bitmap rounding and model dimensions differ.

The old R120 alternative uses B=1,616,872,320 (MNIST) or 17,711,326,080 (CIFAR), FedAvg R240, and sparse targets 154, 222, 399, 760, 1089. It is not the primary current figure. Stage-22 output directory names still contain their original R150-budget targets; an R120 plot must separate the source directory target from the evaluation target.

Historical unicast equations `C_dense_fola=16*K*d`, `C_sparse=K*(8*d+8*m+bitmap_bytes)` are preserved only in historical documents. Do not reuse their old 197-round keep=.5 affordability for current multicast plots.

At r=1, dense FOLA sends no bitmap. A sparse-packet r=1 experiment pays K bitmap payloads and is not the same byte-cost curve, even when numerical aggregation agrees. The **Full-posterior BayesFL** legend denotes the dense baseline.

Per-round savings do not mean that equal-budget endpoints spent that much less total communication. Sparse runs receive more full-model training rounds; no local FLOP, wall-clock, energy, overfitting or uncertainty-calibration benefit is established by these byte calculations alone.

The file `metrics/global_metrics.csv` column `cumulative_all_array_bytes` is the endpoint record used by this project. Saved checkpoints, compressed NPZ size and JSON file size are not communication traffic. Keep any multicast migration backups and metadata; never change old byte counters by hand to make a figure fit.
