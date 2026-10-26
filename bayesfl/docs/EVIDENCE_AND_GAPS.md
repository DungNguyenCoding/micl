# Evidence, corrections, and outstanding checks

Documentation date: **2026-10-26 (Asia/Seoul)**. This is a documentation refresh, not a new training-code release.

## Evidence precedence

For observed results, use the exact run's raw metrics and saved configuration. For intended future work, use the user's latest explicit decision. If an artifact disagrees with that decision, flag the disagreement; never silently rewrite an old experiment to match a new intention.

| Evidence | What it supports | What it does not support |
| --- | --- | --- |
| Current user decisions and pasted plot preflight | Final profiles, chosen baseline, exact reported paths/accuracies | Recomputed metrics or exact current source implementation |
| [Stage-3 registry](evidence/STAGE3_REGISTRY_20261026_211438.txt) | Presence of scripts/configs; literal run-name matches; 42 protected and 121 archived folders at audit time | Full dependency graph or semantic equivalence of configs |
| [Stage-2 terminal log](evidence/STAGE2_MOVE_AND_VERIFY_TERMINAL.txt) | Actual 121-folder move and follow-up verification | An independent external backup |
| User statement “done, stage 3B” | Reported completion of the 13-file organization | A newly uploaded post-Stage-3B verification transcript |
| [Source inventory excerpt](evidence/SOURCE_INVENTORY_EXCERPT_20261026.txt) | Current module filenames and recorded metadata | Current code bodies or current hashes |
| Supplied root docs and docs.zip | Historical protocol explanation and earlier implementation/validation | Current locks without the corrections below |
| Earlier mounted bayesfl.zip | Inspection of an older source snapshot, including the queue format, FOLA equations and CIFAR config templates | Later MNIST CNN implementation or strict-FedAvg patches |
| [CIFAR R120 table](evidence/CIFAR_R120_RESULTS_HISTORICAL.txt) | Earlier 17.711326080 GB results | R150-budget endpoint accuracies |
| Integer arithmetic in this refresh | Budget and affordable-round tables for the stated payload model | Physical radio traffic, speed or measured accuracy |

The older source ZIP inspected for structure has SHA-256
`d6358dd692df9480b01b603460c148e6b6c7e21717020f4f526761cdb8b17983`.
Its latest ZIP-member timestamp is 2026-10-06; it lacks `models/mnist_cnn.py` and later strict-FedAvg code. **Do not install it over the current repository.** This refresh does not contain a replacement `src/`.

## Important corrections

| Old statement | Current treatment |
| --- | --- |
| MNIST MLP, N=100, E=10, B=32 is the official lock | Historical. Main paper: CNN 16–32, N=20, fixed 10/client, seed 0, E=5, B=10, LR=0.10. |
| CIFAR paper profile is ResNet-56 / alpha=.1 / half data | Historical profiles. Main paper: BasicCNN, alpha=.01, local-data fraction 1.0, 20/20 clients. |
| Communication is `16*K*d` per dense FOLA round | Historical unicast count. Current plots use `8*d*(K+1)` under modeled single multicast downlink. |
| CIFAR R120 is the current plot reference | Historical table retained; current reference is R150 / FedAvg R300. |
| MNIST clipped FedAvg R300 is 88.12% | That is the old deleted Stage-12 run's reported value. The selected recovered Stage-23 run reports **88.23%**. Do not merge them. |
| Dense FOLA should use clipped Stage-24 | User reverted to **unclipped Stage-18**. Stage-24 remains an ablation. |
| Every trainer is unchanged since the original sparse release | Only a historical preservation claim. Later clipping controls and strict-FedAvg work happened. |
| No resume support / sparse study not yet implemented | Historical. Sparse runs and clean completed-round resume are in the current workflow. |
| 398 or 408 configs / 110 tests describe the live repository | Historical snapshot counts/results. Stage-3 recorded 898 files under configs; no fresh full suite was run for this refresh. |
| Sparse full retention and dense FOLA have identical bytes | Same mathematical inputs can aggregate identically, but the sparse packet still has a bitmap. The dense curve is not a sparse bitmap run at r=1. |
| `refs=0`, `NO_STATIC_RUN_MATCH` or a `debug_` name means unused | Not valid evidence for deletion. |
| Identical seed/name guarantees bitwise replay | False as a guarantee; read actual partition, configuration, source and runtime records. |

## Outstanding evidence (do not invent)

1. **CIFAR R150-budget accuracies:** completion through targets is recorded, but the supplied numeric table is R120-based. Extract the raw R150/R300 and sparse endpoints before quoting final paper numbers.
2. **Current full source audit:** current `mnist_cnn.py`, `factory.py`, `config.py`, `training/deterministic.py`, `experiment_state.py` and other recently changed files need live inspection before an implementation patch. Paths are known; their current bodies were not supplied in this documentation task.
3. **Current resolved configs:** core locks are decision/terminal-supported. Additional MNIST precision-score policy, exact CNN layer details, preprocessing, evaluation and runtime controls must be read from the live YAML/source rather than inferred from the older MLP template.
4. **Latest full test results:** absent. Old CPU test counts and archive checks are not current training certification.
5. **Git status:** .gitignore was discussed but no applied file/status was supplied. Do not claim Git portability or ignored-file tracking has already been verified.
6. **Recovery difference:** the recovered clipped FedAvg endpoint differs from the old reported endpoint. Preserve both provenance records; the cause was not established.

`tools/export_handoff.py --check` reads current primary metrics and validates exact byte endpoints. `--export` additionally copies a lightweight current-code/metrics snapshot outside the repository. Neither operation runs training or edits experiment data. This closes the most useful handoff gaps without pretending the documentation alone contains the code.
