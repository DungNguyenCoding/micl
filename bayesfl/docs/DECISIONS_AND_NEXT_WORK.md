# Decisions and next work — wireless v1

The completed legacy paper study is retained: MNIST CNN s10/N20/seed0 with unclipped
Stage18/22 FOLA and recovered clipped Stage23 FedAvg; CIFAR BasicCNN alpha.01/f1/N20/seed0;
both dense R150-byte references. MNIST s5/seed3, N100, older architectures and R120 cuts
remain separate historical/alternative studies. Do not retune or rerun them by default.
Cleanup Stage2/3B remains complete as previously recorded; no further cleanup is included.

The user authorized full wireless source implementation and associated docs. That work
is now supplied as **digital reliable orthogonal transmission**, based on the attached
FedKLS_Wireless_Develop proposal, not the formerly discussed prospective AirComp idea.

Implemented: exact integer grant allocator, local current-round oracles, packet v2,
persistent multi-phase clients, local/Flower adapters, modeled resource accounting,
strict clean resume, study/queue/results/export tooling and tests. Preserve the opt-in
boundary and disabled legacy config fingerprint.

Next gates: target-environment Flower/Ray test; short CUDA dataset smoke; approve physical
scenario and comparable deadlines; then long controlled wireless experiments. An example
scenario is not a final paper lock. No wireless performance superiority is claimed from
a CPU synthetic test. No general integer optimum or finite-precision certificate is claimed.

After every approved change, update source/tests, CHAT_HANDOFF, wireless status/validation,
scenario identity, exact new run names and recorded results. Default handoff export still
selects legacy24; list new wireless runs explicitly. Never relabel legacy byte-budget
curves as wireless time curves or change historical records to conceal differing baselines.
