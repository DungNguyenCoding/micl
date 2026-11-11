# Wireless v1 verification evidence

`pytest_final.txt` is the final available suite result (186 passed,6 skipped).
`pytest_first_full.txt` is an earlier intermediate run, not the final test count.
`environment.json` records the actual authoring environment and unexecuted runtime gates.
`legacy_config_compatibility.json` compares original891 YAML configurations/fingerprints.
`baseline_preservation.json` records unchanged original source/output invariants.
`legacy_endpoints.txt` rechecks the actual24 selected raw CSV endpoints and byte totals.
`synthetic_six_policies.txt` and `synthetic_summary.txt` are synthetic CPU protocol/trainer
checks, not MNIST/CIFAR learning evidence. Installer verification is recorded separately
when the release overlay is assembled. None of these logs certify target CUDA/Flower/Ray.
