# CIFAR-10 AirCompBayesFL Experiment Journal

This document records the experimental decisions used to determine the
final CIFAR-10 configuration. Failed or rejected configurations are
retained because they provide evidence for subsequent design decisions.

## General protocol

Unless explicitly stated otherwise:

- Dataset: CIFAR-10
- Clients: 40
- Labels per client: 1
- Mean samples per client: 50
- Model: CIFAR10ResidualCNN
- Trainable parameters: 78,042
- Wireless power: 23 dBm
- Primary tuning seed: 12025

---

## EXP-004 — Effect of large local epoch count

### Question

Can increasing the local epoch count to E=10 improve the baseline under
the highly non-IID one-label-per-client setting?

### Setup

Batch size: 256  
Optimizer: SGD  
Momentum: 0.9  
Scheduler: cosine  
Minimum learning rate: 1e-4  
Local epochs: 10

### Results

| E | Initial LR | Best FedAvg accuracy | Final accuracy | Max update L2 first 10 |
|---:|---:|---:|---:|---:|
| 10 | 0.050 | 18.90% | 18.56% | unstable |
| 10 | 0.002 | 16.90% | 15.22% | 0.1443 |
| 10 | 0.001 | 16.24% | 14.07% | 0.0856 |

### Observation

E=10 with LR=0.05 produced large and unstable updates.

Reducing the initial learning rate to 0.001–0.002 substantially reduced
the update magnitude, but did not recover good test accuracy.

With approximately 50 samples/client and batch size 256, each local
epoch contains approximately one batch. E=10 therefore results in
roughly ten consecutive local optimization steps using data belonging
to the same class before aggregation.

### Interpretation

The results suggest that the poor E=10 behavior is not caused only by
the initial learning rate. Excessive local optimization under the
one-label-per-client partition likely increases client drift, while
momentum can reinforce consecutive same-class update directions.

This conclusion applies to the current experimental setup and should
not be interpreted as a general statement that E=10 is unsuitable for
federated learning.

### Decision

Reject E=10 as the final baseline configuration.

Proceed to E=5 while keeping SGD momentum 0.9 and cosine learning-rate
scheduling.

### Status

Rejected as final configuration; retained as an informative tuning
result.

---

## EXP-005 — E=5 local-epoch screen

### Question

Does reducing the local epoch count from E=10 to E=5 improve FedAvg
under the one-label-per-client CIFAR-10 partition while retaining
SGD momentum 0.9 and cosine learning-rate scheduling?

### Fixed setting

- Dataset: CIFAR-10
- Clients: 40
- Labels/client: 1
- Mean samples/client: 50
- Model: CIFAR10ResidualCNN
- Trainable parameters: 78,042
- Batch size: 256
- Optimizer: SGD
- Momentum: 0.9
- Weight decay: 0
- LR scheduler: cosine
- Minimum LR: 1e-4
- Local epochs: 5
- Seed: 12025
- Wireless power: 23 dBm
- Logical rounds: 80

### Results

| E | Initial LR | Best accuracy | Best round | Final accuracy | Final NLL | Max update L2 first 10 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.003 | 17.09% | 47 | 15.51% | 2.3761 | 0.0816 |
| 5 | 0.005 | 17.53% | 62 | 14.62% | 2.3750 | 0.1222 |

### Observation

Reducing the local epoch count from E=10 to E=5 controlled the global
update magnitude. The maximum global model update L2 norm during the
first ten rounds remained approximately 0.08-0.12.

However, neither E=5 configuration produced a clear improvement in test
accuracy.

The LR=0.005 configuration achieved the higher peak accuracy of 17.53%,
but accuracy subsequently decreased to 14.62% at the final round.

The LR=0.003 configuration was somewhat less volatile but achieved a
lower peak accuracy of 17.09%.

### Comparison with previous experiments

The E=5 configurations are not clearly better than the earlier E=1
FedAvg experiment, which reached approximately 17.62% best accuracy
and 17.28% final accuracy.

They therefore do not justify selecting E=5 as the final FedAvg
baseline.

### Interpretation

The E=5 results suggest that the poor behavior observed with large local
epoch counts is not solely caused by excessively large update magnitude.

After reducing E and controlling the update norm, performance remains
limited. Under the one-label-per-client partition, repeated local
optimization on a single class may still introduce client drift.

This remains an empirical hypothesis based on the current partition and
seed rather than a general conclusion about E=5 in federated learning.

### Decision

Do not select E=5 as the final baseline yet.

Proceed to an intermediate E=3 experiment.

Test two learning rates:

- E=3, momentum=0.9, initial LR=0.01
- E=3, momentum=0.9, initial LR=0.02

Both retain cosine scheduling to 1e-4.

If E=3 also fails to improve over the earlier E=1 reference, return to
E=1 and tune the optimizer around that regime rather than continuing to
force larger local epoch counts.

### Status

Rejected as current final configuration; retained as evidence that
reducing E from 10 to 5 alone is insufficient.

---

## EXP-006 — E=3 local-epoch screen

### Question

Does reducing local optimization to E=3 provide a better compromise
between local learning and client drift under the one-label-per-client
CIFAR-10 partition?

### Fixed setting

- Dataset: CIFAR-10
- Clients: 40
- Labels/client: 1
- Mean samples/client: 50
- Model: CIFAR10ResidualCNN
- Trainable parameters: 78,042
- Batch size: 256
- Optimizer: SGD
- Momentum: 0.9
- Weight decay: 0
- LR scheduler: cosine
- Minimum LR: 1e-4
- Local epochs: 3
- Seed: 12025
- Wireless power: 23 dBm
- Logical rounds: 80

### Results

| E | Initial LR | Best accuracy | Best round | Final accuracy | Final NLL |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.010 | 19.13% | 58 | 17.19% | 2.3514 |
| 3 | 0.020 | 19.01% | 68 | 18.71% | 2.3193 |

### Observation

E=3 produced a clear improvement over the E=5 and low-learning-rate
E=10 screens.

The LR=0.010 run achieved the highest instantaneous accuracy, 19.13%,
but did not retain it. Accuracy eventually stabilized around 17.2%.

The LR=0.020 run achieved a similar peak of 19.01%, but its late-round
behavior was substantially more stable. From approximately rounds
63-80, accuracy remained near 18.7-19.0%, with a final accuracy of
18.71%.

The LR=0.020 run also achieved a lower final NLL than LR=0.010.

AirComp NMSE remained approximately 0.05 throughout both runs, so the
large accuracy oscillations do not coincide with a major change in
wireless reconstruction error.

### Interpretation

The results suggest that E=3 is a better local-optimization regime than
E=5 or E=10 for the current extremely non-IID partition.

The LR=0.020 trajectory remains unstable during much of the early and
middle training period, but becomes substantially more stable after the
cosine schedule reduces the effective learning rate into approximately
the 0.001-0.003 range.

The LR=0.010 run reaches that low-learning-rate regime earlier but loses
more accuracy near the end.

This suggests that an initial learning rate between 0.010 and 0.020 may
provide a better balance.

This remains an empirical observation for seed 12025 and the current
partition.

### Decision

E=3 is retained as the current leading local-epoch configuration.

Current leading FedAvg candidate:

- E=3
- SGD momentum=0.9
- initial LR=0.020
- cosine schedule
- minimum LR=1e-4

Before freezing it, perform two additional informative tests:

1. E=3, initial LR=0.015, to interpolate between the two E=3 results.
2. E=2, initial LR=0.020, to test whether one fewer same-class local
   step further reduces client drift while holding LR fixed.

### Status

Promising. Not yet frozen as the final baseline.

---

## EXP-007 — E=3 LR refinement and E=2 comparison

### Question

Can the E=3 configuration be improved by interpolating the initial
learning rate between 0.010 and 0.020, and does reducing the local
epoch count further to E=2 reduce client drift?

### Fixed setting

- Dataset: CIFAR-10
- Clients: 40
- Labels/client: 1
- Mean samples/client: 50
- Model: CIFAR10ResidualCNN
- Trainable parameters: 78,042
- Batch size: 256
- Optimizer: SGD
- Momentum: 0.9
- Weight decay: 0
- LR scheduler: cosine
- Minimum LR: 1e-4
- Seed: 12025
- Wireless power: 23 dBm
- Logical rounds: 80

### Results

| Setting | Best acc. | Best round | Final acc. | Final NLL | Late mean R60-80 | Late std |
|---|---:|---:|---:|---:|---:|---:|
| E=3, LR=0.015 | 18.10% | 66 | 17.29% | 2.3302 | 17.40% | 0.56 pp |
| E=2, LR=0.020 | 19.11% | 58 | 19.04% | 2.3014 | 18.94% | 0.10 pp |

### Observation

Interpolating the E=3 initial learning rate to 0.015 did not improve
performance. It produced a lower best accuracy and lower final accuracy
than the previous E=3, LR=0.020 configuration.

Reducing the local epoch count to E=2 while retaining LR=0.020 produced
the strongest and most stable FedAvg result observed so far.

The E=2 run achieved:

- best accuracy: 19.11%
- final accuracy: 19.04%
- mean accuracy over rounds 60-80: 18.94%
- standard deviation over rounds 60-80: approximately 0.10 percentage
  points

The narrow late-round accuracy range, 18.71%-19.04%, indicates that the
result is a stable plateau rather than an isolated peak.

### Interpretation

The comparison provides additional evidence that reducing the number of
consecutive local updates is beneficial in the current one-label-per-
client setting.

E=3 with a moderate learning rate did not outperform E=2, while E=2
improved both final accuracy and late-round stability.

This is consistent with the client-drift hypothesis developed during
the E=10 and E=5 experiments: when each client's data belongs to only
one class, repeated local optimization can move local models toward
class-specific objectives before global aggregation.

This remains an empirical conclusion for the current partition and
seed.

### Decision

E=2, LR=0.020 is the current leading FedAvg configuration.

Before freezing it, perform one final E=1 control experiment under the
same cosine-scheduler framework.

Because batch size 256 exceeds the typical local dataset size, E=1
corresponds to approximately one optimizer step per client per round.
With the optimizer state recreated each round, momentum does not
materially affect this one-step E=1 case.

Screen:

- E=1, initial LR=0.040
- E=1, initial LR=0.060

If neither clearly improves over the E=2 result, freeze E=2,
LR=0.020 as the baseline training regime.

### Status

E=2, LR=0.020 is the current leading candidate. Final E=1 control
pending.

---

## EXP-008 — Final E=1 FedAvg control and baseline selection

### Question

Does reducing local optimization to E=1 improve convergence relative
to the current E=2 candidate under the same CIFAR-10 one-label-per-
client setting?

### Fixed setting

- Dataset: CIFAR-10
- Clients: 40
- Labels/client: 1
- Mean samples/client: 50
- Model: CIFAR10ResidualCNN
- Trainable parameters: 78,042
- Batch size: 256
- Optimizer: SGD
- Momentum parameter: 0.9
- Weight decay: 0
- LR scheduler: cosine
- Minimum LR: 1e-4
- Seed: 12025
- Wireless power: 23 dBm
- Logical rounds: 80

### Results

| Setting | Best acc. | Best round | Final acc. | Final NLL | Late mean R60-80 | Late std |
|---|---:|---:|---:|---:|---:|---:|
| E=2, LR=0.020 | 19.11% | 58 | 19.04% | 2.3014 | 18.94% | 0.103 pp |
| E=1, LR=0.040 | 19.39% | 61 | 19.39% | 2.2281 | 19.38% | 0.007 pp |
| E=1, LR=0.060 | 20.22% | 78 | 20.22% | 2.2258 | 20.20% | 0.021 pp |

### Observation

E=1 substantially improved late-round behavior relative to E=2.

The LR=0.060 configuration achieved the best accuracy observed during
the FedAvg tuning campaign:

- best accuracy: 20.22%
- final accuracy: 20.22%
- mean accuracy over rounds 60-80: 20.20%
- late-round range: 20.15%-20.22%
- final NLL: 2.2258

Unlike several earlier configurations, the final result is not an
isolated accuracy peak. Accuracy converges to a narrow and stable
late-round plateau.

### Momentum note

Because the typical client contains approximately 50 samples while the
batch size is 256, E=1 normally corresponds to one optimizer step per
client per communication round.

The local optimizer is recreated for each round. Consequently, although
the configuration retains momentum=0.9, momentum does not materially
affect a one-step E=1 local update because no subsequent local optimizer
step uses the accumulated velocity.

This should be documented when reporting the selected configuration.

### Interpretation

Across the tuning campaign, configurations with many consecutive local
updates performed poorly under the one-label-per-client partition.

The E=3 versus E=2 comparison at the same initial LR=0.020 provides the
clearest controlled evidence that reducing the number of consecutive
same-class local updates improves stability.

The E=1 LR=0.060 configuration further improves the observed accuracy
and late-round stability, although its different learning rate means
the improvement cannot be attributed exclusively to the epoch count.

The results are consistent with the hypothesis that client drift is a
major optimization constraint under the current highly non-IID
partition.

### Decision

Freeze the FedAvg baseline training regime as:

- local epochs: 1
- batch size: 256
- optimizer: SGD
- momentum configuration: 0.9
- initial learning rate: 0.060
- scheduler: cosine
- minimum learning rate: 1e-4
- logical rounds: 80

Use exactly the same training schedule for the next Proposed/Pyro
seed-12025 experiment.

Do not continue FedAvg hyperparameter tuning before evaluating
Proposed/Pyro under this selected regime.

### Status

Selected FedAvg baseline configuration.

---

## EXP-009 — Selected training regime: Proposed/Pyro vs FedAvg

### Question

Does the selected FedAvg optimization regime transfer successfully to
the Proposed Bayesian AirComp method implemented with Pyro?

### Fixed setting

- Dataset: CIFAR-10
- Clients: 40
- Labels/client: 1
- Mean samples/client: 50
- Model: CIFAR10ResidualCNN
- Trainable parameters: 78,042
- Local epochs: 1
- Batch size: 256
- Optimizer: SGD
- Momentum configuration: 0.9
- Weight decay: 0
- Initial LR: 0.060
- LR scheduler: cosine
- Minimum LR: 1e-4
- Logical rounds: 80
- Seed: 12025
- Wireless power: 23 dBm
- Proposed initial prior std: 0.01
- Pyro MC samples: 5

### Results

| Method | Best predictive accuracy | Best logical round | Final predictive accuracy | Final NLL |
|---|---:|---:|---:|---:|
| FedAvg | 20.22% | 78 | 20.22% | 2.2258 |
| Proposed / Pyro | 20.10% | 74 | 19.93% | 2.2078 |

Late-round predictive accuracy:

| Method | Mean R60-80 | Std R60-80 |
|---|---:|---:|
| FedAvg | 20.20% | 0.021 pp |
| Proposed / Pyro | 19.69% | 0.226 pp |

Proposed final ECE: 0.0837.

Posterior-mean accuracy and the corresponding calibration comparison
will be extracted separately from metrics.csv.

### Observation

The selected E=1, LR=0.060 regime successfully transfers to the
Proposed/Pyro method.

The Proposed method reaches essentially the same accuracy region as
FedAvg. Its best predictive accuracy is only 0.12 percentage points
below FedAvg, while its final accuracy is 0.29 percentage points below
FedAvg.

The Proposed final NLL is lower than FedAvg by approximately 0.018.

The Proposed predictive accuracy exhibits more late-round variation
than FedAvg. However, Proposed evaluation uses Monte-Carlo posterior
sampling, so part of this variation may arise from predictive sampling
rather than parameter instability.

### AirComp observation

The two Proposed physical phases exhibit substantially different NMSE.

Precision-phase NMSE remains relatively small, typically around 0.02
late in training.

Natural-mean phase NMSE becomes substantially larger and can reach
approximately 0.3-0.6 during late training.

This should not yet be interpreted as evidence of catastrophic
communication failure. NMSE is normalized by update energy, and late
training updates are small. Therefore normalized error can increase
even when absolute reconstruction error remains limited.

The phase-specific update norm and absolute aggregation error should be
considered together before making a stronger conclusion.

### Interpretation

The selected optimization regime is adequate for both deterministic
FedAvg and Proposed/Pyro.

There is no evidence from seed 12025 that further optimizer tuning is
necessary before comparing Bayesian software implementations.

The Proposed result also indicates that the previous poor Bayesian runs
were primarily associated with unsuitable optimization settings rather
than an inability of the Bayesian protocol to learn CIFAR-10.

A direct claim that Pyro is better or worse than FedAvg is not justified
from one seed.

### Decision

Freeze the common optimization regime:

- E=1
- batch size=256
- initial LR=0.060
- cosine schedule to 1e-4
- SGD
- 80 logical rounds

Do not tune Pyro independently.

Proceed to the Bayesian-Torch implementation using the same data,
partition, model family, seed, wireless configuration, optimization
schedule, and communication protocol as closely as the library
parameterization permits.

The Bayesian-Torch implementation must be kept separate from the
validated Pyro implementation so that both backends remain reproducible.

### Status

FedAvg baseline selected.

Proposed/Pyro baseline successfully completed.

Ready for Bayesian-library comparison stage.

---

## EXP-010 — Final Pyro reference metrics and posterior-precision observation

### Purpose

Freeze the exact seed-12025 Pyro reference metrics before implementing
the Bayesian-Torch backend.

### Selected common training regime

- Local epochs: 1
- Batch size: 256
- Optimizer: SGD
- Momentum configuration: 0.9
- Initial LR: 0.060
- Scheduler: cosine
- Minimum LR: 1e-4
- Logical rounds: 80
- Seed: 12025
- Initial prior std: 0.01

### Reference results

| Metric | FedAvg | Proposed / Pyro |
|---|---:|---:|
| Best accuracy | 20.22% | 20.10% predictive |
| Final accuracy | 20.22% | 19.93% predictive |
| Final posterior-mean accuracy | 20.22% | 19.78% |
| Late predictive mean R60-80 | 20.20% | 19.69% |
| Final NLL | 2.2258 | 2.2078 |
| Final ECE | 0.10085 | 0.08374 |

For Proposed/Pyro:

- predictive late std R60-80: 0.226 percentage points
- posterior-mean late std R60-80: 0.042 percentage points

The much lower posterior-mean variability indicates that part of the
predictive-accuracy variation comes from Monte-Carlo posterior sampling.

### Posterior-precision observation

At the final logical round:

- posterior precision mean = 10000
- posterior variance = 0.0001
- initial precision = 1 / 0.01^2 = 10000
- global precision update L2 = approximately 2.12e-9

Thus the final posterior precision is effectively unchanged from its
initial value.

This suggests that the current direct-precision Pyro optimization has
very weak adaptation of the posterior uncertainty parameters.

The posterior mean clearly learns, but posterior variance adaptation
appears numerically limited.

A full-trajectory precision diagnostic should be checked before making
a stronger statement that precision is frozen throughout all rounds.

### Interpretation

The Pyro implementation achieves accuracy close to FedAvg and improves
NLL and ECE on seed 12025, but its learned uncertainty parameters appear
almost unchanged from initialization.

Therefore the upcoming Bayesian-Torch comparison must distinguish
between:

1. differences caused by the software/inference parameterization, and
2. differences caused by changing the Bayesian model or FL protocol.

A naive replacement of Pyro with a different BNN algorithm would not
constitute a controlled library comparison.

### Decision

Preserve the current Pyro implementation and results unchanged.

Implement Bayesian-Torch as a separate backend/variant and keep the
dataset, partition, network topology, optimization schedule, AirComp
configuration, logical-round count, and seed fixed.

### Status

Pyro reference frozen. Bayesian-Torch implementation stage begins.

---

## EXP-011 — 100-round full-cosine convergence check

### Motivation

Before implementing the Bayesian-Torch backend, evaluate whether
FedAvg and Proposed/Pyro obtain higher or more stable final accuracy
when trained for 100 logical rounds.

This experiment is treated as a new 100-round training-budget
configuration rather than an extension of the previous 80-round run.

### Common configuration

- Dataset: CIFAR-10
- Clients: 40
- Labels/client: 1
- Mean samples/client: 50
- Model: CIFAR10ResidualCNN
- Trainable parameters: 78,042
- Local epochs: 1
- Batch size: 256
- Optimizer: SGD
- Momentum configuration: 0.9
- Weight decay: 0
- Initial LR: 0.060
- LR scheduler: cosine
- Minimum LR: 0.0001
- Seed: 12025
- Wireless power: 23 dBm
- Logical rounds: 100

### LR schedule

The existing scheduler is used without source-code modification.

For this experiment, cosine decay spans all 100 logical rounds:

- round 1: LR = 0.060
- round 100: LR = 0.0001

Therefore this experiment has a different LR trajectory from the
previous 80-round experiment.

### Methods

FedAvg:

- 100 logical rounds
- 100 physical Flower rounds

Proposed/Pyro:

- 100 logical rounds
- 200 physical Flower rounds

### Purpose

Compare final accuracy, NLL, calibration, and late-round stability over
a longer common training budget before beginning the Bayesian-Torch
implementation.

### Status

Pending execution.



### Results

The 100-round full-cosine experiments completed successfully.

| Method | Best predictive acc. | Final predictive acc. | Mean R80-100 | Std R80-100 | Final NLL |
|---|---:|---:|---:|---:|---:|
| FedAvg | 20.94% | 20.94% | 20.936% | 0.008 pp | 2.2107 |
| Proposed/Pyro | 20.64% | 19.61% | 20.274% | 0.276 pp | 2.2042 |

FedAvg reaches a clear practical convergence plateau. From rounds
90-100 its test accuracy remains exactly 20.94%.

The Proposed/Pyro posterior-predictive accuracy remains more variable,
with late values approximately between 19.6% and 20.6%. However, its
late-window mean is stable: rounds 80-89 average approximately 20.29%,
while rounds 90-100 average approximately 20.26%.

Therefore the low final predictive sample of 19.61% should not by
itself be interpreted as degradation of the learned global model.
Posterior-predictive evaluation uses Monte-Carlo weight samples and is
expected to contain sampling variation.

The Proposed run also obtains a slightly lower final NLL than FedAvg.

### Interpretation

Extending the common training budget from 80 to 100 logical rounds
improves the deterministic FedAvg baseline from approximately 20.22%
to 20.94%.

Proposed/Pyro reaches a new best predictive accuracy of 20.64%.
Its late predictive mean is approximately 20.27%, although individual
Monte-Carlo evaluations remain noisy.

There is no strong evidence from the final 20 rounds that either method
would obtain a materially different accuracy level simply by extending
the same schedule to substantially more rounds.

FedAvg can be considered converged at 100 rounds.

For Proposed/Pyro, posterior-mean accuracy will be used as the final
low-noise convergence diagnostic before freezing the 100-round horizon.

### Decision

Do not launch a 120-round experiment at this stage.

Use 100 logical rounds as the candidate final comparison horizon.
Confirm the Proposed/Pyro posterior-mean trajectory, then proceed to
the Bayesian-Torch implementation if it is also practically flat.

### Status

100-round experiments complete.
FedAvg convergence confirmed.
Proposed/Pyro posterior-mean convergence check pending.


---


### Posterior-mean convergence confirmation

The low-noise Proposed/Pyro posterior-mean trajectory was examined over
logical rounds 80-100.

Results:

- best posterior-mean accuracy: 20.33%
- best logical round: 88
- final posterior-mean accuracy: 20.31%
- mean accuracy over rounds 80-100: 20.30%
- standard deviation over rounds 80-100: 0.0164 percentage points
- minimum over rounds 80-100: 20.26%
- maximum over rounds 80-100: 20.33%
- fitted accuracy slope: approximately +0.0010 percentage points/round

The fitted slope corresponds to only approximately +0.02 percentage
points across the entire 20-round late-training window. The
posterior-mean trajectory can therefore be considered practically
converged.

### Posterior-precision diagnostic

Across the complete 100-round Proposed/Pyro trajectory:

- posterior precision mean minimum: 10000
- posterior precision mean maximum: 10000
- posterior precision mean final: 10000
- posterior variance minimum: 0.0001
- posterior variance maximum: 0.0001
- posterior variance final: 0.0001
- maximum global precision-update L2: approximately 1.45e-6
- final global precision-update L2: approximately 2.12e-9

The initial precision is also 10000 because initial_prior_std=0.01.

Therefore the posterior precision remains effectively unchanged from
initialization throughout training. The Proposed/Pyro implementation
learns a useful global posterior mean, but uncertainty adaptation is
numerically negligible under the current direct-precision
parameterization and optimization regime.

### Final convergence decision

The 100-logical-round horizon is accepted as the final baseline
comparison horizon.

Final seed-12025 reference:

- FedAvg converged accuracy: approximately 20.94%
- Proposed/Pyro best predictive accuracy: approximately 20.64%
- Proposed/Pyro late predictive mean: approximately 20.27%
- Proposed/Pyro final posterior-mean accuracy: approximately 20.31%
- Proposed/Pyro late posterior-mean average: approximately 20.30%

No 120-round extension is required before the Bayesian-Torch
comparison.

The Bayesian-Torch implementation should use the same 100-logical-round
training budget and cosine schedule.

### Status

Baseline optimization and convergence study complete.

Ready for Bayesian-Torch implementation.


