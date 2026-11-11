# BayesFL / FedKLS continuation instructions

Read `docs/CHAT_HANDOFF.md`, `docs/COMMAND_RULES.md`, `docs/LOCKED_EXPERIMENTS.md`,
`docs/wireless/README.md` and `docs/wireless/STATUS.md` before changing this project.
These are workflow preferences; the user's latest explicit request remains decisive.

The current code includes an **opt-in reliable digital orthogonal wireless extension**,
not AirComp. Preserve full local trainers, raw precision, original reducer, partitions,
legacy packet v1 and old byte-budget resume. Wireless v2 has a separate controller,
integer grants and checkpoints. Do not convert integer counts to ratios and back.
Do not upload unselected state/scores to the server, retrain during price queries,
drop clients, redraw infeasible channels, or silently enlarge deadlines.

Keep the selected legacy MNIST s10/N20/seed0 and CIFAR alpha .01/f1/N20/seed0
experiments unchanged. Their FOLA R150/FedAvg R300 relation is equal **bytes**, not
wireless time. Wireless numerical scenario settings are not yet paper locks.

Commands use `~/DungNDH/micl/bayesfl`, explicit Python
`/home/micl/anaconda3/envs/dungndh/bin/python`, and `PYTHONPATH=$PWD/src`.
E<=10 unless the user explicitly changes it. Clone saved locked learning configs.
New ordinary/recovery outputs use `run_`; smoke tests use clearly named `smoke_`.
No overwritten outputs, concurrent writers or automatic long queues. One serial queue
per GPU; separate wireless runner/grammar from historical byte-budget queues.

When giving a simulation queue, include generation, configuration validation, shared
partition preparation, serial-per-GPU nohup launch and terminal-only result summaries.
Do not create result .txt/CSV/tee unless requested; normal execution logs and requested
figures are exceptions. Inspect completion, not merely a queue FINISHED message.

Update source, tests, relevant docs and the new-chat handoff together. Preserve
historical evidence; never promote synthetic tests to dataset/GPU validation or
numerical gap diagnostics to certified global optimality. Target Flower/Ray/CUDA
execution is unverified in this source release. No dependency migration or new Git
ignore policy is authorized. Existing archive managers/manifests and protected outputs
remain untouched.
