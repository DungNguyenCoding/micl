"""Real Flower 1.29 adapters, ephemeral clients and Context persistence; no Ray.
Skipped rather than replaced with a fake flwr module when Flower is absent.
"""
import threading
import numpy as np
import pytest
flwr=pytest.importorskip('flwr',reason='Real Flower required for wireless multi-phase integration')
from flwr.app import Context,RecordDict
from flwr.common import Code,GetPropertiesIns,FitIns,ndarrays_to_parameters,parameters_to_ndarrays,DisconnectRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import SimpleClientManager
from bayesfl.wireless.smoke import smoke_config,synthetic_trainer,run_smoke
from bayesfl.wireless.transport import flower_client,FlowerTransport


class EphemeralProxy(ClientProxy):
    def __init__(self,cid,cfg,lock):
        super().__init__(f'opaque-{cid}')
        self.context=Context(run_id=1,node_id=100+cid,node_config={'partition-id':cid},state=RecordDict(),run_config={})
        self.cfg=cfg;self.operations=[];self.trains=0;self.lock=lock
    def client(self):
        def trainer(*a):
            with self.lock:
                self.trains+=1
                return synthetic_trainer(*a)
        # A fresh Python client object on EVERY invocation.
        return flower_client(self.cfg,'synthetic',7,self.context,trainer=trainer)
    def get_properties(self,ins,timeout,group_id):return self.client().get_properties(ins)
    def get_parameters(self,ins,timeout,group_id):raise AssertionError('No client initialization upload')
    def fit(self,ins,timeout,group_id):
        self.operations.append(ins.config['phase'])
        r=self.client().fit(ins)
        assert r.status.code==Code.OK
        if ins.config['phase']!='upload':assert parameters_to_ndarrays(r.parameters)==[]
        return r
    def evaluate(self,ins,timeout,group_id):raise AssertionError('No distributed evaluation')
    def reconnect(self,ins,timeout,group_id):return DisconnectRes(reason='test complete')


def test_real_flower_multiphase_persistent_state(tmp_path):
    cfg=smoke_config(rounds=2)
    manager=SimpleClientManager();lock=threading.Lock()
    proxies=[EphemeralProxy(i,cfg,lock) for i in range(3)]
    for x in proxies:manager.register(x)
    transport=FlowerTransport(manager,cfg)
    state=run_smoke(cfg,tmp_path/'flower',transport=transport)
    assert state.round_id==2
    for x in proxies:
        assert x.trains==2
        assert x.operations.count('prepare')==2 and x.operations.count('upload')==2
        assert x.operations.count('release')==2
        assert 'wireless_meta' not in x.context.state
