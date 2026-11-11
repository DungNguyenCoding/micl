# Selected CIFAR-10 legacy profile

BasicCNN (`paper_basiccnn`), d878538, N20/full participation, seed0,
paper_dirichlet alpha.01, local_data_fraction1.0, E10/B32, SGD momentum0/WD0.
FOLA paper_reference lambda.01, lr.02 cosine horizon400/min.0001, clip10, zero initial
omega, score-only floor_for_score1e-12. FedAvg lr.01 constant, clip10.

The selected legacy figure uses dense FOLA R150 (53.10%) and FedAvg R300 (47.28%) at
22,139,157,600 bytes. Sparse .75/.5/.25/.1/.05 targets193/278/499/950/1362.
All results/paths are in RESULTS_AND_PLOTTING.md and EXPERIMENT_REGISTRY.json.
Older R120, alpha.1, half-data, ResNet and unclipped studies are separate history.

Wireless generation clones the saved learning configs but requires an explicit physical
scenario and separate resource/controller identity. These legacy endpoints are not
wireless measurements, and their rounds ratio does not establish equal modeled time.
