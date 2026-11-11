#!/usr/bin/env python3
"""Audited joint-only CIFAR10 bandwidth/deadline sensitivity plots.

Input: seven COMPLETED runs under outputs/ or seven audited ZIPs.
Output: four PNG/PDF figures, endpoint/trajectory CSVs, and provenance JSON.
Read-only with respect to source runs, checkpoints, configs, and audit ZIPs.

Example (server):
  python scripts/plot_joint_cifar_sensitivity_v1.py --root "$PWD" \
      --output-dir "plots/wireless/joint_sensitivity_$(date +%Y%m%d_%H%M%S)"
Example (audited ZIP collection):
  python scripts/plot_joint_cifar_sensitivity_v1.py --audit-dir /path/to/audits \
      --output-dir /tmp/sensitivity-figure-output
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CASES = (
    ("reference", "cifar_eqb_joint150_fixed_v1", "Reference (1 MHz, U=58.97 s)"),
    ("u170", "cifar_joint_u170_r150_v1", "U = 48.97 s"),
    ("u160", "cifar_joint_u160_r150_v1", "U = 38.97 s"),
    ("u150", "cifar_joint_u150_r150_v1", "U = 28.97 s"),
    ("b0550", "cifar_joint_b0550_r150_v1", "B = 0.55 MHz"),
    ("b0750", "cifar_joint_b0750_r150_v1", "B = 0.75 MHz"),
    ("b0950", "cifar_joint_b0950_r150_v1", "B = 0.95 MHz"),
)
U_CASES = ("reference", "u170", "u160", "u150")
B_CASES = ("reference", "b0550", "b0750", "b0950")
COLORS = {
    "reference": "#252933",
    "u170": "#2563a5", "u160": "#d97706", "u150": "#b42318",
    "b0550": "#b42318", "b0750": "#d97706", "b0950": "#2563a5",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def round_no(value: str | int | float) -> int:
    f = float(value)
    n = int(f)
    require(f == n, f"Noninteger round identifier: {value}")
    return n


def numeric(value: Any, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Missing or invalid {label}: {value!r}") from None
    require(math.isfinite(out), f"Nonfinite {label}: {value!r}")
    return out


def close(a: float, b: float, label: str, *, atol: float = 1e-6, rtol: float = 1e-9) -> None:
    require(math.isclose(a, b, abs_tol=atol, rel_tol=rtol), f"Inconsistent {label}: {a} vs {b}")


def csv_rows(content: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))


class Archive:
    def __init__(self, path: Path):
        self.path = path
        self.zip = zipfile.ZipFile(path)
        names = [n for n in self.zip.namelist() if n.startswith("scripts/configs/wireless/") and n.endswith("/study_manifest.json")]
        require(len(names) == 1, f"Ambiguous archived study in {path.name}")
        self.tag = Path(names[0]).parent.name
        self._members = set(self.zip.namelist())

    def read(self, rel: str) -> bytes:
        require(rel in self._members, f"Missing {rel} in {self.path.name}")
        return self.zip.read(rel)

    def validate_hashes(self) -> None:
        hashes = json.loads(self.read("audit/SOURCE_HASHES.json"))
        for rel, digest in hashes.items():
            require(sha256(self.read(rel)) == digest, f"Audit HASH MISMATCH: {self.path.name} : {rel}")
        audit = json.loads(self.read("audit/joint_audit.json"))
        require(audit.get("status") == "PASS_ENGINEERING_RECORD_INTEGRITY", f"Audit did not pass: {self.path.name}")
        print(f"PASS archive: {self.path.name}, {len(hashes)} hashes")


class Live:
    def __init__(self, root: Path, tag: str):
        self.root = root
        self.tag = tag
        self.path = root

    def read(self, rel: str) -> bytes:
        p = self.root / rel
        require(p.is_file(), f"Missing live source file: {p}")
        return p.read_bytes()


@dataclass
class Case:
    key: str
    tag: str
    label: str
    source: Any
    run_rel: str
    config: dict[str, Any]
    summary: dict[str, Any]
    rounds: list[dict[str, str]]
    evaluations: list[dict[str, str]]
    bandwidth_hz: float
    upload_window_s: float
    round_target_s: float
    hashes: dict[str, str]

    def values(self, field: str, include_zero: bool = False) -> list[float]:
        data = self.evaluations if field in ("accuracy", "nll") else self.rounds
        if not include_zero and data is self.evaluations:
            data = [r for r in data if round_no(r["round"]) >= 1]
        return [numeric(r[field], f"{self.key}:{field}") for r in data]


def find_run_rel(source: Archive | Live, tag: str) -> str:
    manifest = json.loads(source.read(f"scripts/configs/wireless/{tag}/study_manifest.json"))
    jobs = manifest.get("jobs", [])
    selected = [j for j in jobs if "_joint_seed0" in j.get("run_name", "")]
    require(len(selected) == 1, f"Expected exactly one joint seed0 job in {tag}, found {len(selected)}")
    name = selected[0]["run_name"]
    if isinstance(source, Archive):
        matches = {"/".join(n.split("/")[:2]) for n in source._members if n.startswith("outputs/" + name + "_") and "/run_summary.json" in n}
    else:
        matches = {str(p.relative_to(source.root)) for p in (source.root / "outputs").glob(name + "_*") if p.is_dir() and (p / "run_summary.json").is_file()}
    require(len(matches) == 1, f"Expected one completed output for {tag}; found {len(matches)}: {sorted(matches)}")
    return next(iter(matches))


def load_case(key: str, tag: str, label: str, source: Archive | Live) -> Case:
    rel = find_run_rel(source, tag)
    raw = {}
    for sub in ("run_summary.json", "resolved_config.yaml", "metrics/global_metrics.csv", "metrics/wireless_rounds.csv"):
        raw[sub] = source.read(f"{rel}/{sub}")
    summary = json.loads(raw["run_summary.json"])
    cfg = yaml.safe_load(raw["resolved_config.yaml"])
    evaluations = csv_rows(raw["metrics/global_metrics.csv"])
    rounds = csv_rows(raw["metrics/wireless_rounds.csv"])
    require(summary.get("completed_rounds") == 150 and summary.get("failure") is None and summary.get("stop_reason") == "max_rounds", f"Incomplete R150 {key}")
    require(len(evaluations) == 151 and len(rounds) == 150, f"Bad global/round row count {key}: {len(evaluations)}, {len(rounds)}")
    require([round_no(row["round"]) for row in evaluations] == list(range(151)), f"Noncontiguous evaluation rounds {key}")
    require([round_no(row["round"]) for row in rounds] == list(range(1, 151)), f"Noncontiguous wireless rounds {key}")
    wireless = cfg["wireless"]
    require(wireless["policy"] == "joint" and wireless["backend"] == "flower" and cfg["runtime"]["seed"] == 0, f"Unexpected policy/backend/seed in {key}")
    require(cfg["training"]["rounds"] == 150 and cfg["training"]["local_epochs"] == 10 and cfg["training"]["batch_size"] == 32, f"Unexpected training settings {key}")
    require(wireless["channel"]["mode"] == "fixed", f"Not a fixed-channel run: {key}")
    for r, (e, w) in enumerate(zip(evaluations[1:], rounds), start=1):
        close(numeric(e["mean_keep_ratio"], f"{key} keep {r}"), numeric(w["mean_keep_ratio"], f"{key} keep {r}"), f"{key} mean keep R{r}")
        for k in ("cumulative_all_array_bytes", "cumulative_control_bytes", "cumulative_modeled_time_seconds", "actual_reconstruction_objective"):
            close(numeric(e[k], f"{key}:{k}"), numeric(w[k], f"{key}:{k}"), f"{key} {k} R{r}", atol=1e-5)
        require(e["allocation_status"] in ("numerical_gap_target", "price_tolerance"), f"Unexpected allocation status R{r} {key}: {e['allocation_status']}")
        require(e["array_accounting_complete"].lower() == "true", f"Incomplete array accounting {key} R{r}")
        require(numeric(w["allocated_bandwidth_hz"], f"BW {key}") <= numeric(wireless["uplink_bandwidth_hz"], f"limit {key}") + 1e-4, f"Bandwidth exceeds budget {key} R{r}")
        require(numeric(w["modeled_control_seconds"], f"ctrl {key}") <= numeric(wireless["control_reserve_s"], f"ctrl reserve {key}") + 1e-6, f"Control exceeds reserve {key} R{r}")
    for key0, summary_key in (("cumulative_all_array_bytes", "cumulative_all_array_bytes"), ("cumulative_control_bytes", "cumulative_control_bytes"), ("cumulative_modeled_time_seconds", "cumulative_modeled_time_seconds")):
        close(numeric(evaluations[-1][key0], key0), numeric(summary[summary_key], summary_key), f"{key} final {key0}", atol=1e-5)
    close(numeric(evaluations[-1]["accuracy"], "acc"), numeric(summary["last_evaluation"]["accuracy"], "summary acc"), f"{key} final accuracy")
    upload = numeric(rounds[0]["modeled_upload_window_seconds"], f"{key} upload")
    require(all(math.isclose(numeric(r["modeled_upload_window_seconds"], "U"), upload, abs_tol=1e-7) for r in rounds), f"Changing upload time {key}")
    hashes = {x: sha256(data) for x, data in raw.items()}
    return Case(key, tag, label, source, rel, cfg, summary, rounds, evaluations,
                float(wireless["uplink_bandwidth_hz"]), upload, float(wireless["round_target_s"]), hashes)


def normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    c.pop("run_name", None)
    c["wireless"].pop("uplink_bandwidth_hz", None)
    c["wireless"].pop("round_target_s", None)
    return c


def validate_comparability(cases: dict[str, Case]) -> None:
    ref = cases["reference"]
    for case in cases.values():
        require(normalize_config(case.config) == normalize_config(ref.config), f"Unexpected configuration differences: {case.key}")
        b = case.bandwidth_hz / 1e6
        if case.key.startswith("u"):
            close(b, 1.0, f"{case.key} fixed B")
        elif case.key.startswith("b"):
            close(case.upload_window_s, ref.upload_window_s, f"{case.key} fixed U")
    require(ref.round_target_s == 180 and ref.bandwidth_hz == 1e6, "Unexpected reference scenario")
    close(cases["u170"].round_target_s, 170, "u170 slot")
    close(cases["u160"].round_target_s, 160, "u160 slot")
    close(cases["u150"].round_target_s, 150, "u150 slot")
    for key, expected_b in (("b0550", 550000), ("b0750", 750000), ("b0950", 950000)):
        close(cases[key].bandwidth_hz, expected_b, f"{key} B")
    print("PASS: 7 matched joint runs, differing only in uplink bandwidth or round deadline")


def make_endpoint_rows(cases: dict[str, Case]) -> list[dict[str, Any]]:
    baseline_resource = cases["reference"].bandwidth_hz * cases["reference"].upload_window_s
    out = []
    for key, tag, label in CASES:
        case = cases[key]
        ev = case.evaluations
        rr = case.rounds
        counts = {}
        for row in rr:
            counts[row["allocation_status"]] = counts.get(row["allocation_status"], 0) + 1
        out.append({
            "case": key, "study_tag": tag, "run_dir": case.run_rel,
            "Bmax_MHz": case.bandwidth_hz / 1e6, "U_seconds": case.upload_window_s,
            "round_target_seconds": case.round_target_s,
            "BU_relative_percent": 100 * (case.bandwidth_hz * case.upload_window_s) / baseline_resource,
            "accuracy_R50_pct": 100 * float(ev[50]["accuracy"]),
            "accuracy_R100_pct": 100 * float(ev[100]["accuracy"]),
            "accuracy_R150_pct": 100 * float(ev[150]["accuracy"]),
            "nll_R150": float(ev[150]["nll"]),
            "mean_retention_R150_percent": 100 * float(rr[149]["mean_keep_ratio"]),
            "mean_retention_R131_R150_percent": 100 * sum(float(r["mean_keep_ratio"]) for r in rr[130:]) / 20,
            "weighted_discarded_KL_R150": float(rr[149]["actual_reconstruction_objective"]),
            "cumulative_model_array_GB": float(ev[150]["cumulative_all_array_bytes"]) / 1e9,
            "cumulative_control_MB": float(ev[150]["cumulative_control_bytes"]) / 1e6,
            "cumulative_array_plus_control_GB": (float(ev[150]["cumulative_all_array_bytes"]) + float(ev[150]["cumulative_control_bytes"])) / 1e9,
            "modeled_time_hours": float(ev[150]["cumulative_modeled_time_seconds"]) / 3600,
            "price_queries_total": sum(int(float(r["price_queries"])) for r in rr),
            "price_tolerance_rounds": counts.get("price_tolerance", 0),
            "numerical_gap_target_rounds": counts.get("numerical_gap_target", 0),
            "max_control_time_s": max(float(r["modeled_control_seconds"]) for r in rr),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_trajectory_rows(cases: dict[str, Case]) -> list[dict[str, Any]]:
    out = []
    for key, _, _ in CASES:
        case = cases[key]
        for r in range(151):
            ev = case.evaluations[r]
            wire = None if r == 0 else case.rounds[r-1]
            out.append({
                "case": key, "round": r,
                "accuracy_pct": float(ev["accuracy"]) * 100,
                "nll": float(ev["nll"]),
                "modeled_time_hours": float(ev["cumulative_modeled_time_seconds"]) / 3600,
                "mean_retention_pct": "" if wire is None else float(wire["mean_keep_ratio"]) * 100,
                "weighted_discarded_kl": "" if wire is None else float(wire["actual_reconstruction_objective"]),
                "cumulative_model_array_GB": float(ev["cumulative_all_array_bytes"]) / 1e9,
                "cumulative_control_MB": float(ev["cumulative_control_bytes"]) / 1e6,
                "cumulative_total_recorded_GB": (float(ev["cumulative_all_array_bytes"]) + float(ev["cumulative_control_bytes"])) / 1e9,
                "price_queries": "" if wire is None else int(wire["price_queries"]),
                "allocator_status": "" if wire is None else wire["allocation_status"],
            })
    return out


def figures(cases: dict[str, Case], endpoints: list[dict[str, Any]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42,
                         "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    def plot_group(ax, keyset, field, *, x_time=False, from_round=0):
        for key in keyset:
            case = cases[key]
            if field == "accuracy":
                ys = [float(r["accuracy"]) * 100 for r in case.evaluations]
                rounds = list(range(151))
            elif field == "nll":
                ys = [float(r["nll"]) for r in case.evaluations]
                rounds = list(range(151))
            elif field == "keep":
                ys = [100 * float(r["mean_keep_ratio"]) for r in case.rounds]
                rounds = list(range(1,151))
            elif field == "payload":
                ys = [(float(r["cumulative_all_array_bytes"]) + float(r["cumulative_control_bytes"])) / 1e9 for r in case.evaluations]
                rounds = list(range(151))
            elif field == "kl":
                ys = [float(r["actual_reconstruction_objective"]) for r in case.rounds]
                rounds = list(range(1,151))
            else:
                raise ValueError(field)
            pts = [(r,y) for r,y in zip(rounds,ys) if r >= from_round and math.isfinite(y) and (field!='kl' or y>0)]
            xs = [r * case.round_target_s / 3600 if x_time else r for r,_ in pts]
            vals = [y for _,y in pts]
            ax.plot(xs, vals, color=COLORS[key], lw=2.2 if key=="reference" else 1.7,
                    ls="--" if key=="reference" else "-", label=case.label,
                    alpha=0.96)
        ax.grid(alpha=0.20, linewidth=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    fig, axs = plt.subplots(4, 2, figsize=(14.3, 15.4), sharex='col')
    rowfields = [("accuracy", "Test accuracy (%)"), ("nll", "Test NLL (lower is better)"),
                 ("keep", "Mean retention (%)"), ("payload", "Cumulative array + control (GB)")]
    for col,(names,title) in enumerate(((U_CASES,"A. Upload-window sweep | Bmax = 1 MHz"),
                                        (B_CASES,"B. Bandwidth sweep | U = 58.973 s"))):
        axs[0,col].set_title(title, weight='bold', pad=10)
        for row,(fld,ylabel) in enumerate(rowfields):
            plot_group(axs[row,col], names, fld)
            axs[row,col].set_ylabel(ylabel)
        axs[0,col].legend(loc='lower right' if col==0 else 'lower right', framealpha=0.95)
        axs[-1,col].set_xlabel("FL communication round")
        axs[-1,col].set_xlim(0,150)
    fig.suptitle("Joint FedKLS / CIFAR-10: six R150 wireless-resource scenarios + validated reference", fontsize=14, weight='bold', y=0.995)
    fig.text(0.5,0.01,"Same seed, partition, learner, channel gains, PSD and solver. Modeled times differ in the U sweep. Single-seed pilot.", ha='center', fontsize=9, color='#475569')
    fig.tight_layout(rect=(0,0.03,1,0.97), h_pad=2.0, w_pad=2.0)
    save(fig,"joint_sensitivity_side_by_side")

    fig,axs=plt.subplots(1,2,figsize=(14.2,5.0))
    for ax,names,title in zip(axs,(U_CASES,B_CASES), ("A. Upload-window sweep | Bmax = 1 MHz", "B. Bandwidth sweep | U = 58.973 s")):
        plot_group(ax,names,'accuracy',x_time=True)
        ax.set_title(title,weight='bold')
        ax.set_ylabel("Test accuracy (%)")
        ax.set_xlabel("Cumulative modeled time (hours)")
        ax.legend(loc='lower right')
    fig.suptitle("Joint FedKLS / CIFAR-10: accuracy versus modeled elapsed time", fontsize=13, weight='bold')
    fig.tight_layout(rect=(0,0,1,0.93))
    save(fig,"joint_accuracy_vs_modeled_time")

    fig,axs=plt.subplots(1,2,figsize=(14.2,5.0))
    for ax,names,title in zip(axs,(U_CASES,B_CASES), ("A. Upload-window sweep | Bmax = 1 MHz", "B. Bandwidth sweep | U = 58.973 s")):
        plot_group(ax,names,'kl',from_round=2)
        ax.set_title(title,weight='bold')
        ax.set_ylabel("Weighted discarded KL (log scale)")
        ax.set_xlabel("FL communication round (R2–R150)")
        ax.set_yscale('log')
        ax.legend(loc='upper right')
    fig.suptitle("Joint FedKLS: discarded KL on each run's own evolving posteriors",fontsize=13,weight='bold')
    fig.text(0.5,0.015,"R1 excluded: initial precision scoring floor creates a very large outlier. Curves are NOT same-posterior comparisons.",ha='center',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,0.045,1,0.93))
    save(fig,"joint_discarded_kl_by_round")

    fig,axs=plt.subplots(1,2,figsize=(14.2,5.0))
    for ax,field,ylabel in zip(axs,("accuracy_R150_pct","mean_retention_R150_percent"),("R150 test accuracy (%)","R150 mean retention (%)")):
        for row in endpoints:
            key=row['case']
            marker='D' if key=='reference' else ('o' if key.startswith('u') else 's')
            ax.scatter(row['BU_relative_percent'],row[field],s=90 if key=='reference' else 65,
                       color=COLORS[key],marker=marker,edgecolors='white',linewidths=.7,zorder=3)
            ax.annotate('Reference' if key=='reference' else key,
                        (row['BU_relative_percent'],row[field]),xytext=(5,6),textcoords='offset points',fontsize=8)
        ax.set_xlabel("Bmax x U (% of reference)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        ax.set_xlim(40,108)
    fig.suptitle("Joint FedKLS: endpoint resource sensitivity (one seed; no monotonicity assumption)", fontsize=13,weight='bold')
    fig.tight_layout(rect=(0,0,1,0.93))
    save(fig,"joint_resource_endpoints")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--audit-dir", type=Path, help="Directory containing 7 verified bayesfl_joint_R150_audit_*.zip files")
    grp.add_argument("--root", type=Path, default=Path.cwd(), help="BayesFL repository root (default: current directory)")
    parser.add_argument("--output-dir", type=Path, help="NEW output directory (never overwrite existing files)")
    parser.add_argument("--check-only",action='store_true',help="Validate all seven runs, print endpoints, create no files")
    args=parser.parse_args(argv)
    require(args.check_only or args.output_dir is not None, "--output-dir required without --check-only")
    if args.audit_dir:
        available = {}
        archives = []
        for p in sorted(args.audit_dir.glob("bayesfl_joint_R150_audit_*.zip")):
            archive=Archive(p)
            archives.append(archive)
            if archive.tag in [tag for _,tag,_ in CASES]:
                require(archive.tag not in available, f"Duplicate audit archives for {archive.tag}")
                archive.validate_hashes()
                available[archive.tag]=archive
        require(len(available)==7, f"Expected seven audit ZIPs, found {sorted(available)}")
        sources=available
    else:
        root=args.root.expanduser().resolve()
        require((root / "scripts/configs/wireless").is_dir(), f"Not a BayesFL repo: {root}")
        sources={tag:Live(root,tag) for _,tag,_ in CASES}
    cases={key:load_case(key,tag,label,sources[tag]) for key,tag,label in CASES}
    validate_comparability(cases)
    endpoints=make_endpoint_rows(cases)
    print('\n{:<10s} {:>10s} {:>9s} {:>10s} {:>10s} {:>12s} {:>10s}'.format('Scenario','B (MHz)','U (s)','Acc R150','Keep R150','Payload GB','PriceTol'))
    for r in endpoints:
        print('{:<10s} {:>10.2f} {:>9.3f} {:>9.2f}% {:>9.2f}% {:>12.3f} {:>10d}'.format(r['case'],r['Bmax_MHz'],r['U_seconds'],r['accuracy_R150_pct'],r['mean_retention_R150_percent'],r['cumulative_array_plus_control_GB'],r['price_tolerance_rounds']))
    if args.check_only:
        print('PASS: CHECK ONLY; no files written')
        return 0
    out=args.output_dir.expanduser().resolve()
    require(not out.exists(), f"Output directory already exists; refusing overwrite: {out}")
    out.mkdir(parents=True)
    write_csv(out/'joint_sensitivity_endpoints.csv',endpoints)
    write_csv(out/'joint_sensitivity_trajectories.csv',make_trajectory_rows(cases))
    provenance={
        'study':'CIFAR-10 joint-only sensitivity R150; seed0; simulated fixed channels; EXAMPLE_NOT_LOCKED',
        'audited_zip_mode':bool(args.audit_dir),
        'cases':{c.key:{'study_tag':c.tag,'run_rel':c.run_rel,'run_summary_sha256':c.hashes['run_summary.json'],
                        'resolved_config_sha256':c.hashes['resolved_config.yaml'],
                        'global_metrics_sha256':c.hashes['metrics/global_metrics.csv'],
                        'wireless_rounds_sha256':c.hashes['metrics/wireless_rounds.csv']} for c in cases.values()},
        'methodology':'All seven runs match in resolved config after removing run_name, wireless.uplink_bandwidth_hz and wireless.round_target_s.',
        'limitations':['Single seed, fixed channels, uncalibrated compute/control times',
                       'Different modeled round duration in U-sweep; compare rounds AND elapsed hours',
                       'KL curves are evaluated on independently evolving posteriors, not shared posteriors',
                       'Solver numerical diagnostics are not proof of integer global optimality'],
    }
    (out/'joint_sensitivity_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    figures(cases,endpoints,out)
    (out/'README_PLOTS.md').write_text(
        '# Joint CIFAR-10 R150 sensitivity plots\n\n'
        'Two-column panels: LEFT: 3 upload-window cases at Bmax=1 MHz; RIGHT: 3 bandwidth cases at U=58.973 seconds. '
        'Each includes the original joint R150 reference as a dashed line.\n\n'
        'Outputs: side-by-side accuracy, NLL, mean retention, and cumulative array+control bytes; '
        'accuracy versus modeled time; weighted discarded KL (from R2, log axis); endpoint versus Bmax x U.\n\n'
        '**Cautions:** these are single-seed joint-only digital-link simulations; training runtime differs from modeled time; '
        'KL curves compare different trajectories; control bytes are modeled serialized payload counts, not physical RF bytes.\n'
        'The original scenario is tagged EXAMPLE_NOT_LOCKED.\n', encoding='utf-8')
    print(f'\nPASS: Generated 4 two-column figures (PNG + PDF), 2 CSVs and provenance in {out}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, FileNotFoundError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
