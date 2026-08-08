# Validation status — v1.4.2

Packaging-environment validation:

```text
python -m compileall -q .
PYTHONPATH=. pytest -q
27 passed
```

The tests cover:

- the disclosed 62,346-parameter CNN size;
- ideal and wireless AirComp aggregation;
- reference-distance channel scaling;
- runtime/backend helpers;
- Algorithm-1 precision/natural-mean phase scheduling;
- Eq. (33)/(34) coordinate transforms;
- float64 precision-state preservation;
- unified KKT power-control configuration;
- posterior-mean diagnostics;
- v1.4.2 deterministic `Delta-w` payload semantics;
- additive server application of the received FedAvg/FedProx update;
- deterministic ideal/received/global update-norm logging fields.

The packaging environment does not expose an NVIDIA GPU and does not execute
the user's complete Windows CUDA/Flower/Pyro workload. End-to-end GPU validation
should therefore be run in the existing Windows environment that has already
validated PyTorch 2.5.1+cu121, Pyro 1.9.1, Flower 1.32.1, Ray 2.55.1, and the
RTX 3060 Laptop GPU.
