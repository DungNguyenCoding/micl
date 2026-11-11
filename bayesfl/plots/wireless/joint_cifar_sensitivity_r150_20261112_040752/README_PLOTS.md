# Joint CIFAR-10 R150 sensitivity plots

Two-column panels: LEFT: 3 upload-window cases at Bmax=1 MHz; RIGHT: 3 bandwidth cases at U=58.973 seconds. Each includes the original joint R150 reference as a dashed line.

Outputs: side-by-side accuracy, NLL, mean retention, and cumulative array+control bytes; accuracy versus modeled time; weighted discarded KL (from R2, log axis); endpoint versus Bmax x U.

**Cautions:** these are single-seed joint-only digital-link simulations; training runtime differs from modeled time; KL curves compare different trajectories; control bytes are modeled serialized payload counts, not physical RF bytes.
The original scenario is tagged EXAMPLE_NOT_LOCKED.
