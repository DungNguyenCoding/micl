import copy
import numpy as np
import pytest
from bayesfl.offline_smoke import smoke_config, npy_roundtrip
from bayesfl.posterior.packing import ParameterLayout
from bayesfl.posterior.sparse import (compress_update, decode_packet, encode_packet, layout_manifest,
                                      snapshot_id, select_mask)


def packet_case(d=9, ratio=.5, rule='kl_global_local', initial_precision=1.0):
    layout = ParameterLayout(('weight',), ((d,),))
    cfg = smoke_config('fola_sparse_' + rule, keep_ratio=ratio, initial_precision=initial_precision)
    before = [np.linspace(-1, 1, d, dtype=np.float32), np.full(d, initial_precision, dtype=np.float32)]
    local = [before[0] + .25, before[1] + np.linspace(.1, 1, d, dtype=np.float32)]
    sid = snapshot_id(before, layout, 1)
    arrays, metadata = compress_update(before, local, layout, cfg, round_id=1, client_id=0, base_snapshot_id=sid)
    kw = dict(global_arrays=before, layout=layout, cfg=cfg, round_id=1, client_id=0, base_snapshot_id=sid)
    return before, local, arrays, metadata, kw


def test_tiny_bitmap_cost_and_order():
    arrays = encode_packet(np.array([.7, 10, .7, 20], np.float32), np.array([2, 3, 4, 5], np.float32),
                           np.array([True, False, True, False]))
    assert arrays[0].tolist() == [5]
    assert sum(a.nbytes for a in arrays) == 17
    np.testing.assert_array_equal(arrays[2], [2, 4])


@pytest.mark.parametrize('d', [1, 4, 8, 9, 17])
@pytest.mark.parametrize('rule', ['kl_global_local', 'kl_local_global', 'random'])
def test_packet_roundtrip_and_no_hidden_state(d, rule):
    before, local, arrays, metadata, kw = packet_case(d=d, rule=rule)
    arrays, _ = npy_roundtrip(arrays)
    effective, mask = decode_packet(arrays, metadata, **kw)
    assert arrays[0].dtype == np.uint8
    assert arrays[1].dtype == arrays[2].dtype == np.float32
    assert len(arrays) == 3 and arrays[0].size == (d + 7)//8
    for j in (0, 1):
        np.testing.assert_array_equal(effective[j][mask], local[j][mask])
        np.testing.assert_array_equal(effective[j][~mask], before[j][~mask])
    assert int(mask.sum()) == metadata['m']


@pytest.mark.parametrize('key,value', [('round_id',2), ('base_snapshot_id','stale'), ('client_id',1),
                                      ('layout_id','wrong'), ('covariance_representation','variance'),
                                      ('m',1), ('selection_rule','random'), ('precision_policy','unexpected')])
def test_metadata_rejection(key, value):
    _, _, arrays, metadata, kw = packet_case()
    metadata[key] = value
    with pytest.raises(ValueError): decode_packet(arrays, metadata, **kw)


@pytest.mark.parametrize('damage', ['padding','short','bitmapdtype','meandtype','prec_nan','prec_negative','mean_inf','cardinality','extra'])
def test_packet_corruption(damage):
    _, _, arrays, metadata, kw = packet_case()
    if damage == 'padding': arrays[0][-1] |= np.uint8(128)
    if damage == 'short': arrays[0] = arrays[0][:-1]
    if damage == 'bitmapdtype': arrays[0] = arrays[0].astype(np.float32)
    if damage == 'meandtype': arrays[1] = arrays[1].astype(np.float64)
    if damage == 'prec_nan': arrays[2][0] = np.nan
    if damage == 'prec_negative': arrays[2][0] = -1
    if damage == 'mean_inf': arrays[1][0] = np.inf
    if damage == 'cardinality': arrays[1] = arrays[1][:-1]
    if damage == 'extra': arrays.append(np.zeros(9, np.float32))
    with pytest.raises(ValueError): decode_packet(arrays, metadata, **kw)


def test_zero_raw_precision_is_explicitly_supported_not_transformed():
    before, local, arrays, metadata, kw = packet_case(initial_precision=0)
    effective, mask = decode_packet(arrays, metadata, **kw)
    assert metadata['precision_policy'] == 'floor_for_score'
    assert metadata['score_global_floored_fraction'] == 1.0
    np.testing.assert_array_equal(effective[1][~mask], 0)
    np.testing.assert_array_equal(effective[1][mask], local[1][mask])


def test_snapshot_and_layout_deterministic():
    b, _, _, _, kw = packet_case()
    layout = kw['layout']
    assert snapshot_id(b, layout, 1) == snapshot_id([a.copy() for a in b], layout, 1)
    assert snapshot_id(b, layout, 1) != snapshot_id(b, layout, 2)
    assert layout_manifest(layout)['entries'][0]['count'] == 9
