# Add the CIFAR-10 scenario pilot (no production source replacement)

Save `FedKLS_CIFAR10_Wireless_Pilot_Setup_v1.zip` beside `bayesfl/` and run:

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
unzip -n ../FedKLS_CIFAR10_Wireless_Pilot_Setup_v1.zip
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" scripts/check_cifar_wireless_scenario_v1.py
```

Read `docs/wireless/CIFAR10_STAGED_PLAN.md` for all values, source attribution,
explicit assumptions, commands and limits. The scenario is deliberately marked
`EXAMPLE_NOT_LOCKED`, so the study generator requires `--allow-example`.

The installation adds new files only. Existing data, outputs, source, and documentation
are not overwritten. No simulation starts during extraction or preflight.

Next: run the separately named two-round CIFAR-10 equal-bandwidth smoke described in
the plan. Then inspect equal-bandwidth training before testing joint allocation.
Do not launch all six policies; the dense FOLA policy is infeasible at this pilot's
180-second slot. Do not enable Rayleigh fading blindly for a long queue.

The preflight expects the original seed-0 CIFAR partition NPZ and saved protected run
metadata to exist. A lightweight handoff without that NPZ cannot run the partition
integrity check. Use the actual research repository rather than inventing a replacement.
