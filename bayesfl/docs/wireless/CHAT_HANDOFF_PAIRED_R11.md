# Chat handoff — CIFAR wireless paired R11 + prelong gate

## Proven state before this package

- Protected CIFAR paper learner is `paper_basiccnn` d878538, 20 full clients, seed0 Dirichlet alpha0.01/full fraction1.0, E10 B32, SGD .02 cosine horizon400, lambda .01, clipping10, FOLA paper_reference, precision `floor_for_score` only for scoring. Older experiments and 24 primary endpoints already verified.
- Wireless extension installed on server and validated: 59 wireless tests initially passed, 197 full-suite tests passed, real Flower/Ray synthetic 2-round smoke passed, CIFAR equal R2 smoke passed.
- Equal-bandwidth CIFAR20 R20 38.68%; modeled 3600s; array bytes 1.658473 GB, control 1.612769 MB.
- Joint CIFAR20 R20 37.58%; modeled 3600s; array bytes 1.531769 GB, control 14.487163 MB. These are separate learning trajectories.
- Earlier **allocation-only** R10->R11 same-posterior probe completed, 20 local FOLA training calls exactly once, equal KL51.41562517 and joint KL1.425165575 (~97.2281% lower). This did NOT send packets, aggregate or evaluate.
- Source checkpoint location: `outputs/run_wireless_cifar_eqb_pilot20_v1_cifar10_a0p01_f1_equal_bandwidth_seed0_20261109_182157/checkpoints/global_round_0010.npz`; SHA256 `1bba3cd62b72f4da6e7e0b7c3fbfe687804e26653e64c389ae2b9c8807b85e4a`.
- Original partition: `outputs/partitions/cifar10_paper_dirichlet_a0.01_n20_seed0.npz` and content SHA256 `914fe035f7b504b0059ca85b5a6a194bd41027fb2cfe2765ee777142d68280f2`.
- Fixed radio pilot: path-loss topology 500m, UL 1 MHz, 180s slot, ~58.973s upload, 1 MHz assumed control rate/30s reserve, fixed PSD conservative vs 24dBm cap. It is `EXAMPLE_NOT_LOCKED`, not a measured deployment.

## New additive files

- `scripts/probe_cifar_paired_r11.py`: one local R11 preparation, two policy allocations, actual production upload/decoder, original FOLA aggregate, shared test evaluation and nonfeasible full FOLA oracle; verifies source hashes.
- `scripts/report_cifar_prelong_gate.py`: fails closed on arithmetic/training/evaluation checks and creates minimal shareable evidence.
- `scripts/run_cifar_prelong_gate.sh`: runs existing tests, full regression, 24 endpoints, radio preflight, paired R11 test and report **once**, no long training.
- `tests/wireless/test_paired_r11_probe.py`, `test_prelong_report.py`: synthetic tests with no datasets or network.
- `docs/wireless/CIFAR_PAIRED_PRELONG_GATE.md`: protocol, limitations, 150-round config-generation example.

## Server execution

```
cd ~/DungNDH/micl/bayesfl || exit 1
pgrep -af 'bayesfl.main|_run_gpu_queue.sh|_run_wireless_queue.sh' || true
nvidia-smi
bash scripts/run_cifar_prelong_gate.sh 0
```

Uploads one `~/DungNDH/micl/bayesfl_cifar_prelong_gate_*.zip` with R11 accuracy/NLL and per-client packet/solver/control diagnostics. Do **not** assume predicted R11 accuracy; wait for actual terminal evidence. A PASS authorizes *considering* new controlled fixed-scenario pilots, not automatic full training or paper-level wireless claims.

## Key scientific caveats

- Source global R10 is from the **equal-bandwidth trajectory**, so repeat from joint R10 for symmetric evidence before asserting generalized benefit.
- Dense FOLA oracle R11 is only a reference; check its infeasibility at the fixed 180s slot.
- Joint minimizes discarded KL, not CIFAR test loss, bytes or energy. Neither the 97.2% scalar KL gain nor any one-step outcome proves long-run accuracy improvement.
- A single-source, single-seed test cannot certify global integer optimality, statistical significance or measured RF throughput.
- Original configs, outputs, datasets, queues and plots are not modified by this package. Pair probe runs offline local transport (actual codec and aggregate), not a Ray multi-client network transfer; historical Flower/Ray smoke already exists.
