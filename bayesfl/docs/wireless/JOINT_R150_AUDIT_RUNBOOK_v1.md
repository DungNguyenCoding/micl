# CIFAR-10 Wireless FedKLS — Joint-only R150 audit

This is an **additive, read-only** diagnostic for the 150-round experiment `cifar_eqb_joint150_fixed_v1`. It intentionally ignores the equal-bandwidth run, retaining joint KL (`kl_global_local`) as the primary development workflow.

## Objective

Confirm the **recorded** learning trajectory and wireless process: 151 global evaluations including R0, 150 wireless rounds, 3,000 client records, 150 price/allocation JSONs, per-client local step counts, integer sparse payload sizes, spectrum/bandwidth budget, power caps, modeled airtime and control reserve, exact KL-tail arithmetic, price request/reply counts, event ledger bytes and energy, committed resume-file integrity, and study scenario hash.

Use at the repository root:

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" scripts/audit_joint_wireless_run_v1.py --root "$PWD" --tag cifar_eqb_joint150_fixed_v1 --rounds 150
```

On PASS, upload the printed `bayesfl_joint_R150_audit_*.zip` from the parent directory. The ZIP contains only resolved/source configs, scalar JSON/CSV, allocation JSON, event JSONL, and derived summaries; **no model tensors, checkpoints, datasets or client posterior arrays**. The source files are not modified. The script creates new evidence *outside* the repository.

A **pass** supports internal engineering consistency under the specified fixed-PSD, reliable-digital simulation. It does not establish true physical energy/latency, FedKLS accuracy superiority, or global integer optimality. `EXAMPLE_NOT_LOCKED` means the radio compute and control timing is still an explicitly assumed pilot scenario. Output `JOINT_AUDIT_REPORT.md` is an analysis report, not automatically copied to canonical handoff docs. After reviewing evidence, update `docs/CHAT_HANDOFF.md`, `docs/wireless/STATUS.md`, `docs/wireless/VALIDATION.md`, and registries in a separate previewed documentation-only action.

If a check fails, do not edit historical results to make it pass. Send the full error and inspect the source path it names.
