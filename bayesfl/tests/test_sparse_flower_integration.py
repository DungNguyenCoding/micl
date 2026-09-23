"""Real Flower legacy API tests on synthetic data; no Ray or downloads.

Skipped, not emulated, when Flower is absent. Run these in the pinned 1.29.0
research environment before the Ray/dataset smoke experiment.
"""
import copy
import logging
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

flwr=pytest.importorskip('flwr',reason='Real Flower 1.29 API unavailable; do not substitute a mocked flwr package')
from flwr.common import DisconnectRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from bayesfl.budget_server import BudgetServer
from bayesfl.client import BayesFLNumPyClient
from bayesfl.evaluation import CentralEvaluator
from bayesfl.experiment_state import RunState
from bayesfl.offline_smoke import tiny_model, synthetic_dataset, smoke_config
from bayesfl.posterior.packing import ParameterLayout, initial_fola_state, model_to_ndarrays
from bayesfl.strategies.research_strategy import ResearchStrategy


class LocalProxy(ClientProxy):
    """Only transport is local. NumPyClient adapters and serializers are real."""
    def __init__(self,cid,client,sparse):
        super().__init__(cid)
        self.client=client.to_client()
        self.sparse=sparse
        self.fit_rounds=[]
    def get_properties(self,ins,timeout,group_id):return self.client.get_properties(ins)
    def get_parameters(self,ins,timeout,group_id):return self.client.get_parameters(ins)
    def fit(self,ins,timeout,group_id):
        self.fit_rounds.append(int(ins.config['server_round']))
        result=self.client.fit(ins)
        arrays=parameters_to_ndarrays(result.parameters)
        if self.sparse:
            assert len(arrays)==3
            assert [a.dtype for a in arrays]==[np.dtype('uint8'),np.dtype('float32'),np.dtype('float32')]
        return result
    def evaluate(self,ins,timeout,group_id):raise AssertionError('Central evaluation must not download to clients')
    def reconnect(self,ins,timeout,group_id):return DisconnectRes(reason='test complete')


def test_actual_flower_serializer_preserves_bitmap_dtype():
    arrays=[np.array([5],np.uint8),np.array([.7,.7],np.float32),np.array([2.,3.],np.float32)]
    wire=ndarrays_to_parameters(arrays)
    restored=parameters_to_ndarrays(wire)
    assert sum(a.nbytes for a in restored)==17
    assert sum(len(t) for t in wire.tensors)>17
    for a,b in zip(arrays,restored):np.testing.assert_array_equal(a,b);assert a.dtype==b.dtype


@pytest.mark.parametrize('method',['fedavg_dense','fola_dense','fola_sparse_kl_global_local',
                                  'fola_sparse_kl_local_global','fola_sparse_random'])
def test_flower_client_strategy_server_budget_loop(tmp_path,monkeypatch,method):
    import bayesfl.client as client_module
    import bayesfl.evaluation as evaluation_module
    torch.set_num_threads(1)
    torch.manual_seed(7)
    blueprint=tiny_model()
    cfg=smoke_config(method,rounds=8)
    def loader(_cfg,_path,cid,shuffle_seed):
        data=synthetic_dataset(cid)
        return DataLoader(data,batch_size=4,shuffle=True,
                          generator=torch.Generator().manual_seed(shuffle_seed)),len(data)
    monkeypatch.setattr(client_module,'load_client_loader',loader)
    monkeypatch.setattr(client_module,'build_model',lambda cfg:copy.deepcopy(blueprint))
    monkeypatch.setattr(evaluation_module,'build_model',lambda cfg:copy.deepcopy(blueprint))
    monkeypatch.setattr(client_module,'resolve_device',lambda _:torch.device('cpu'))
    layout=ParameterLayout.from_model(blueprint)
    arrays=initial_fola_state(blueprint,1) if cfg.method=='fola' else model_to_ndarrays(blueprint)
    per_round=906 if cfg.sparse_enabled else 1184 if cfg.method=='fola' else 592
    cfg.communication.max_communication_bytes=2*per_round+1
    state=RunState(cfg,layout,arrays,tmp_path,partition_sha256='synthetic')
    for name in ('metrics','posterior','reliability','checkpoints'):(tmp_path/name).mkdir(exist_ok=True)
    evaluator=CentralEvaluator(cfg,DataLoader(synthetic_dataset(20),batch_size=32),tmp_path,
                               logger=logging.getLogger('test'),state=state)
    strategy=ResearchStrategy(cfg=cfg,layout=layout,initial_arrays=arrays,run_dir=tmp_path,
                              logger=logging.getLogger('test'),state=state)
    strategy.evaluate_fn=lambda rnd,arrays,_:evaluator.evaluate(rnd,arrays)
    manager=SimpleClientManager()
    proxies=[]
    for cid in range(3):
        client=BayesFLNumPyClient(client_id=cid,cfg=cfg,partition_path='not-read',average_client_size=7)
        proxy=LocalProxy(f'opaque_proxy_{cid}',client,cfg.sparse_enabled)
        manager.register(proxy)
        proxies.append(proxy)
    server=BudgetServer(client_manager=manager,strategy=strategy)
    server.set_max_workers(1)
    components=ServerAppComponents(server=server,config=ServerConfig(num_rounds=8))
    assert components.server is server
    history,_=server.fit(num_rounds=8,timeout=60)
    assert state.round_id==2 and state.budget_used==2*per_round
    assert state.stop_reason()=='communication_budget'
    assert [r for r,_ in history.losses_centralized]==[0,1,2]
    assert sum(len(p.fit_rounds) for p in proxies)==4
    assert state.ledger.totals['initialization_array_bytes']==0
    assert state.ledger.totals['evaluation_array_bytes']==0
    assert (tmp_path/'checkpoints/global_round_0002.npz').exists()
