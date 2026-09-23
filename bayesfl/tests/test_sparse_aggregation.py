from pathlib import Path
import numpy as np
import pytest
from bayesfl.offline_smoke import smoke_config
from bayesfl.experiment_state import RunState
from bayesfl.posterior.aggregation import aggregate_fola
from bayesfl.posterior.packing import ParameterLayout
from bayesfl.posterior.sparse import compress_update, encode_packet


@pytest.mark.parametrize('mode', ['paper_reference', 'online_recurrence'])
def test_original_reducer_golden(mode):
    fixture=Path(__file__).parent/'fixtures'/'uploaded_fola_golden.npz'
    layout=ParameterLayout(('a','b'),((2,3),(2,)))
    cfg=smoke_config('fola_dense'); cfg.fola.mode=mode
    with np.load(fixture, allow_pickle=False) as z:
        clients=[[z[f'client_{k}_{j}'].copy() for j in range(4)] for k in range(3)]
        output=aggregate_fola(clients,[.1,.3,.6],layout=layout,cfg=cfg)
        for j,a in enumerate(output): np.testing.assert_array_equal(a,z[f'{mode}_{j}'])


@pytest.mark.parametrize('mode', ['paper_reference', 'online_recurrence'])
def test_mixed_masks_and_all_omitted_are_unchanged(tmp_path, mode):
    cfg=smoke_config('fola_sparse_kl_global_local', keep_ratio=.5)
    cfg.fola.mode=mode
    layout=ParameterLayout(('w',),((4,),))
    before=[np.array([1,2,3,4],np.float32),np.array([1,2,4,8],np.float32)]
    state=RunState(cfg,layout,before,tmp_path)
    instruction=state.prepare_round(1,[0,1],before)
    locals_=[[np.array([5,6,7,8],np.float32), np.array([3,4,5,6],np.float32)],
             [np.array([9,10,11,12],np.float32),np.array([7,8,9,10],np.float32)]]
    selections=[np.array([1,1,0,0],bool),np.array([1,0,1,0],bool)]
    effective=[]; masks=[]
    for cid in [0,1]:
        _, meta=compress_update(before,locals_[cid],layout,cfg,round_id=1,client_id=cid,
                                base_snapshot_id=instruction['base_snapshot_id'])
        wire=encode_packet(*locals_[cid],selections[cid])
        reconstructed, mask=state.receive_packet(round_id=1,recipient_id=cid,arrays=wire,
                                                   metrics=meta,num_examples=[1,3][cid],expected_client_id=cid)
        effective.append(reconstructed); masks.append(mask)
    out=aggregate_fola(effective,[.25,.75],layout=layout,cfg=cfg)
    out=state.finish_round(out,masks=masks,counts=[1,3],client_metrics=[{},{}])
    expected_p=.25*effective[0][1]+.75*effective[1][1]
    numerator=.25*effective[0][1]*effective[0][0]+.75*effective[1][1]*effective[1][0]
    epsilon=cfg.fola.aggregation_epsilon if mode=='paper_reference' else 0
    expected_mu=numerator/(expected_p+epsilon)
    np.testing.assert_allclose(out[0][:3], expected_mu[:3], rtol=2e-7)
    np.testing.assert_allclose(out[1][:3], expected_p[:3], rtol=2e-7)
    assert out[0][3] == before[0][3] and out[1][3] == before[1][3]
    assert np.all(out[1]>0)
