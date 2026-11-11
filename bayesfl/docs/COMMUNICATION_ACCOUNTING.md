# Communication accounting: legacy bytes and wireless resources

## Legacy paper benchmark (unchanged)

Float32 mean/precision pairs, dimension d, K=20, one dense modeled multicast:

```
C_dense_FOLA = 8*d*(K+1)
C_dense_FedAvg = 4*d*(K+1)
m = ceil(keep_ratio*d)
C_sparse_v1 = 8*d + K*(8*m + ceil(d/8))
```

Legacy sparse v1 always carries a bitmap, even at full retention. The plotted
Full-posterior BayesFL is the dense format, not a v1 bitmap packet at r=1.
Initialization/central evaluation add no model transmissions in these studies.
Budgets: MNIST R150 = 2,021,090,400 bytes; CIFAR R150 = 22,139,157,600.
Corresponding FedAvg R300 has equal modeled array bytes, not automatically equal time.
At .75/.5/.25/.1/.05, sparse rounds are MNIST193/278/498/950/1362 and
CIFAR193/278/499/950/1362. GB is decimal 1e9 bytes.

## Wireless v2 actual model-array cost

For each posterior upload:

```
bytes_v2(m) = 8*m + ceil(d/8),  1 <= m < d
bytes_v2(d) = 8*d
FedAvg dense upload = 4*d
round bytes = dense multicast bytes + sum(actual per-client packet bytes)
```

Counts are granted as integers, never recovered by ceil((m/d)*d). Different clients and
rounds can have different counts/formats. Do not verify wireless cumulative bytes using
a fixed legacy keep ratio or round*constant formula. Padded bitmap cost is operative.
Arrived rejected packets are charged; unknown failed/undecodable replies flag incomplete
accounting. This is a logical array boundary, not whole-message measured radio traffic.

## Additional wireless ledgers

Control bytes are canonical UTF-8 JSON instructions/replies, outside model-array counts.
A declared shared control rate and per-exchange latency convert these to modeled time.
Control reservation is subtracted from the round's upload window. Exceeding it fails;
unused reservation becomes idle. Startup identity discovery, CSI estimation, Flower/
network framing and retransmissions are excluded, not physically free.

Time = dense multicast + max(modeled full local preparation) + modeled control + maximum
upload duration + idle to the fixed slot boundary. Emulator wall time stays separate.
There is no implicit clock-speedup from sparse uploads in a fixed-slot experiment;
the main comparison is posterior/accuracy preservation within a common time resource.

Fixed PSD implies data-plane transmit energy = PSD * allocated bandwidth * airtime.
The cumulative energy field includes UL+DL only, not compute/control/circuit/idle energy.
No noise is added to successfully decoded values. This is digital OFDMA-style modeling,
not analog/AirComp aggregation or a physical-channel implementation.

The old byte-budget controller remains unchanged. Wireless uses separate round/time
limits and a separate fingerprint/checkpoint contract. No simultaneous strict byte cap
is silently imposed on the dynamic allocator.
