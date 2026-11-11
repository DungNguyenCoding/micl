"""Standard-library-only inspection of variable-cost wireless trajectories."""
from __future__ import annotations
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def whole(value):
    n=Decimal(str(value))
    if not n.is_finite() or n != n.to_integral_value():
        raise ValueError(f"Expected integer metric, got {value!r}")
    return int(n)


def number(value):
    value=float(value)
    if not math.isfinite(value):raise ValueError("Nonfinite metric")
    return value


def inspect_run(run_dir: Path, *, time_budget_s: float | None = None) -> dict:
    run_dir=Path(run_dir).resolve()
    protocol=json.loads((run_dir/'wireless/protocol.json').read_text())
    if protocol.get('protocol')!='fedkls_wireless_integer_precision_v2':
        raise ValueError(f"Not a wireless-v2 run: {run_dir}")
    trace=run_dir/'wireless/channel_trace.csv'
    if protocol.get('trace_sha256') is not None and (not trace.is_file() or sha(trace)!=protocol['trace_sha256']):
        raise ValueError('Pinned channel trace is missing or changed')
    metadata=json.loads((run_dir/'wireless/layout_manifest.json').read_text())
    d=whole(metadata['d']);k=whole(protocol['num_clients'])
    method=protocol['method'];target=number(protocol['round_target_s'])
    global_rows=rows(run_dir/'metrics/global_metrics.csv')
    rounds=rows(run_dir/'metrics/wireless_rounds.csv') if (run_dir/'metrics/wireless_rounds.csv').is_file() else []
    clients=rows(run_dir/'metrics/wireless_clients.csv') if (run_dir/'metrics/wireless_clients.csv').is_file() else []
    rids=[whole(r['round']) for r in global_rows]
    if rids!=list(range(len(global_rows))) or not rids:raise ValueError("Missing or duplicate global rounds")
    end=rids[-1]
    if [whole(r['round']) for r in rounds]!=list(range(1,end+1)):
        raise ValueError("Wireless round logs do not align with global evaluations")
    grouped={r:[] for r in range(1,end+1)}
    for row in clients:
        r=whole(row['round'])
        if r not in grouped:raise ValueError("Uncommitted client row")
        grouped[r].append(row)
    total=0
    for r, rr in enumerate(rounds,1):
        cr=grouped[r]
        if sorted(whole(x['client_id']) for x in cr)!=list(range(k)):
            raise ValueError(f"Missing/duplicate client reports at R{r}")
        ul=0
        for row in cr:
            m=whole(row['keep_count'])
            if not 1<=m<=d or whole(row['d'])!=d:raise ValueError("Invalid recorded count")
            b=(4*d if method=='fedavg' else (8*d if m==d else 8*m+(d+7)//8))
            if method=='fedavg' and m!=d:raise ValueError("Unsupported sparse FedAvg record")
            if whole(row['upload_array_bytes'])!=b:raise ValueError("Count/packet byte mismatch")
            ul+=b
        expected=ul+(4*d if method=='fedavg' else 8*d)
        total+=expected
        if whole(rr['round_array_bytes'])!=expected or whole(global_rows[r]['cumulative_all_array_bytes'])!=total:
            raise ValueError("Wireless actual cumulative bytes do not match client packets")
        if not math.isclose(number(global_rows[r]['cumulative_modeled_time_seconds']),r*target,rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError("Modeled fixed-slot clock mismatch")
    if whole(global_rows[0]['cumulative_all_array_bytes'])!=0:raise ValueError("Unexpected model-array cost at initialization")
    summary=json.loads((run_dir/'run_summary.json').read_text())
    if summary.get('failure') or summary.get('pending_round') is not None:
        raise ValueError("Failed/partial wireless runs are not complete comparison inputs")
    if whole(summary['completed_rounds'])!=end:raise ValueError("Summary/evaluation mismatch")
    if time_budget_s is not None:
        budget=number(time_budget_s)
        if budget<0 or budget>number(global_rows[-1]['cumulative_modeled_time_seconds']):
            raise ValueError("Requested modeled-time budget is outside the observed trajectory; no extrapolation")
        chosen=[r for r in global_rows if number(r['cumulative_modeled_time_seconds'])<=budget][-1]
    else:chosen=global_rows[-1]
    accuracy=number(chosen['fola_mean_accuracy'] if method=='fola' else chosen['accuracy'])
    if not 0<=accuracy<=1:raise ValueError("Expected stored fractional accuracy")
    latest=run_dir/'wireless_resume/latest.json'
    if latest.is_file():
        p=json.loads(latest.read_text());name=p['state_file']
        if Path(name).name!=name:raise ValueError("Invalid checkpoint metadata path")
        state_path=latest.parent/name
        if sha(state_path)!=p['state_sha256']:raise ValueError("Checkpoint metadata hash mismatch")
        state=json.loads(state_path.read_text())
        if whole(state['round_id'])!=end:raise ValueError("Metrics have no matching evaluated-round commit")
        for name,digest in state['tracked_sha256'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError("Unsafe tracked metric path")
            if sha(run_dir/name)!=digest:raise ValueError(f"Committed metric changed: {name}")
    return dict(run_dir=str(run_dir), method=method, policy=protocol['policy'], selection_rule=protocol['selection_rule'],
        completed_rounds=end, selected_round=whole(chosen['round']), accuracy_percent=100*accuracy,
        cumulative_array_bytes=whole(chosen['cumulative_all_array_bytes']),
        cumulative_modeled_time_seconds=number(chosen['cumulative_modeled_time_seconds']),
        cumulative_control_bytes=whole(chosen['cumulative_control_bytes']),
        cumulative_modeled_tx_energy_j=number(chosen['cumulative_modeled_tx_energy_j']),
        metrics_sha256=sha(run_dir/'metrics/global_metrics.csv'),
        input_hashes={str(p.relative_to(run_dir)):sha(p) for p in (run_dir/'metrics/global_metrics.csv',
             run_dir/'metrics/wireless_rounds.csv',run_dir/'metrics/wireless_clients.csv',
             run_dir/'resolved_config.yaml',run_dir/'wireless/protocol.json',
             run_dir/'wireless/layout_manifest.json',run_dir/'run_summary.json',run_dir/'wireless/channel_trace.csv') if p.is_file()},
        scope='actual recorded arrays and modeled wireless resources; not measured radio latency')


def print_results(results):
    print('\nWIRELESS RESULTS — exact recorded endpoints; times are modeled, not GPU wall time')
    print(f"{'Method / policy / selector':<48} {'Round':>6} {'Acc %':>9} {'Seconds':>12} {'Array GB':>12} {'Ctrl bytes':>12} {'Tx J':>12}")
    for r in results:
        label=f"{r['method']} / {r['policy']} / {r['selection_rule']}"
        print(f"{label:<48} {r['selected_round']:>6} {r['accuracy_percent']:>9.3f} {r['cumulative_modeled_time_seconds']:>12.5f} {r['cumulative_array_bytes']/1e9:>12.6f} {r['cumulative_control_bytes']:>12} {r['cumulative_modeled_tx_energy_j']:>12.5f}")
    if len(results)>1:
        times=[r['cumulative_modeled_time_seconds'] for r in results]
        if all(math.isclose(x,times[0],rel_tol=1e-10,abs_tol=1e-10) for x in times):
            print('Endpoints have the same modeled time; verify matched scenario/partition/training before causal claims.')
        else:print('WARNING: endpoint modeled times differ. Specify a common covered time budget for comparison.')
    print('PASS: variable per-client packet sizes and completed evaluations verified.')
