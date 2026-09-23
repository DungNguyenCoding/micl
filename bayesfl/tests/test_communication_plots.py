import csv
import json
from pathlib import Path
import pytest
from bayesfl.communication_plots import load_curve, plot_comparison, print_summary


def make_run(path, *, costs=(0,100,200), acc=(.1,.9,.3), seed=0, method='fola_dense', stop='max_rounds', budget=250):
    path.mkdir(parents=True,exist_ok=True)
    (path/'metrics').mkdir(exist_ok=True)
    with (path/'metrics/global_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['round','cumulative_all_array_bytes','accuracy','seed','method'])
        w.writeheader()
        for r,(c,a) in enumerate(zip(costs,acc)):
            w.writerow(dict(round=r,cumulative_all_array_bytes=c,accuracy=a,seed=seed,method=method))
    (path/'run_summary.json').write_text(json.dumps(dict(stop_reason=stop,budget_bytes=budget,
                                                        budget_metric='cumulative_all_array_bytes')))
    (path/'resolved_config.yaml').write_text('method: fola\nmodel: {name: test}\n')
    return path


def test_latest_at_budget_not_best_or_overshoot(tmp_path,capsys):
    c=load_curve(make_run(tmp_path))
    assert c.at_budget(150)['round']==1
    assert c.at_budget(200)['accuracy']==.3
    with pytest.raises(ValueError,match='covered'): c.at_budget(201)
    print_summary([c],budget=200)
    assert '30.000%' in capsys.readouterr().out


def test_exhausted_whole_round_remainder_is_a_covered_budget(tmp_path):
    c=load_curve(make_run(tmp_path,stop='communication_budget'))
    assert c.coverage_end==250 and c.at_budget(250)['round']==2
    with pytest.raises(ValueError): c.at_budget(251)


def test_plot_observed_points_and_export_not_fabricated_round_cost(tmp_path):
    curves=[load_curve(make_run(tmp_path/'dense')),load_curve(make_run(tmp_path/'sparse',costs=(0,70,140),method='fola_sparse_random'))]
    paths=plot_comparison(curves,tmp_path/'plots')
    assert len(paths)==3 and all(p.exists() and p.stat().st_size>100 for p in paths)
    with paths[-1].open() as f: rows=list(csv.DictReader(f))
    assert [int(r['array_bytes']) for r in rows]==[0,100,200,0,70,140]


def test_missing_byte_history_rejected(tmp_path):
    path=make_run(tmp_path)
    p=path/'metrics/global_metrics.csv'
    p.write_text(p.read_text().replace('cumulative_all_array_bytes','old_round_estimate'))
    with pytest.raises(ValueError,match='Historical'):load_curve(path)


def test_average_seeds_checks_distinct_matched_ids_and_settings(tmp_path):
    curves=[load_curve(make_run(tmp_path/'s0',seed=0),label='FOLA'),
            load_curve(make_run(tmp_path/'s1',seed=1),label='FOLA')]
    assert plot_comparison(curves,tmp_path/'plots',average_seeds=True)[0].exists()
    curves[1].seed=0
    with pytest.raises(ValueError,match='distinct'):plot_comparison(curves,tmp_path/'bad',average_seeds=True)


def test_integer_byte_counts_are_not_rounded_through_float(tmp_path):
    big=2**54+1
    c=load_curve(make_run(tmp_path,costs=(0,big,big+2)))
    assert c.rows[1][c.metric]==big


def test_undecodable_packet_cannot_claim_complete_accounting(tmp_path):
    path=make_run(tmp_path)
    (path/'run_summary.json').write_text(json.dumps({'array_accounting_complete':False}))
    with pytest.raises(ValueError,match='unknown array payload'):load_curve(path)
