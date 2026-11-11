# Algorithm notes — preserved learner, new transmission controller

The original FOLA and FedAvg trainers, model definitions, data partitioning and FOLA
reducer remain unchanged by wireless v1. FOLA paper_reference retains incoming raw
precision, accumulates squared minibatch-mean task gradients and optimizes its original
CE-plus-prior objective with the chosen clipping policy. It is not a newly implemented
variational inference method or per-example Fisher estimator.

Legacy FedKLS performs full local training then fixed-ratio model-wide KL(global||local)
or equal-cardinality random upload. Prior-fill reconstruction and the epsilon reducer
are shared with dense FOLA; all-client-omitted coordinates are restored to the broadcast.
No layer-wise masks, permanent model pruning or sparse local training is implied.

Wireless v1 uses the same learning/scoring/reducer functions but separates preparation
from selection/transmission. Current-round scores stay at clients and scalar price
responses determine exact per-client counts. Raw precision remains on the wire;
CIFAR's floor_for_score is a scoring-only surrogate, not a modified uploaded posterior.

The new packet format distinguishes dense m=d (no bitmap) and sparse m<d (padded bitmap).
It does not change legacy v1 packets. The server never uses a hidden full local model
from a sparse client. Actual discarded-score totals are reported separately for random
uploads and optimal top-m prefixes. Frozen-state KL comparisons do not prove higher
future prediction accuracy, less overfitting, or reduced local computation.

Read [wireless/IMPLEMENTATION.md](wireless/IMPLEMENTATION.md) for thresholds, lifecycle,
physical/timing assumptions, numerical diagnostic limits and proposal adaptations.
The price multiplier is not the local FOLA regularization coefficient. Integer bisection
is not a universal globally optimal integer allocation algorithm.
