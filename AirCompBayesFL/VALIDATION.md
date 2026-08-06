# Validation status

Validation performed for version 1.3.1 in the packaging environment:

```text
python -m compileall -q .
```

passed for all Python files.

The dependency-light test set passed:

```text
14 passed
```

It covers:

- the disclosed 62,346-parameter CNN size;
- ideal AirComp weighted summation;
- reference-distance channel scaling;
- backend/device helpers;
- physical precision/natural-mean phase scheduling;
- Eq. (33)/(34) coordinate initialization and round trip;
- exact ideal two-phase Gaussian conflation;
- phase-aware CSV schemas;
- direct-rho source contract and phase-2 round-start-prior contract.

The packaging environment does not contain Pyro, Flower, or Ray and has no
NVIDIA GPU, so a full end-to-end Pyro/Flower/CUDA run was not executed there.
The user's previously validated Windows environment uses PyTorch 2.5.1+cu121,
Pyro 1.9.1, Flower 1.32.1, Ray 2.55.1, and an RTX 3060 Laptop GPU.

## v1.3.2 precision-numerics validation

The v1.3.1 user run showed non-zero phase losses while every local and global
precision value remained exactly 400.0.  v1.3.2 adds a regression test proving
that a 1e-6 direct-rho update at rho=400 survives client/server aggregation.
The precision path is float64; CNN inference remains float32.

Dependency-light validation performed during packaging:

```text
python -m compileall .
PYTHONPATH=. pytest -q
16 passed
```

An end-to-end Pyro/CUDA run still requires the user's NVIDIA environment.
