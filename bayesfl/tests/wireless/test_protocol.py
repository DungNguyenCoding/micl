from __future__ import annotations
import copy
import json
import math
import numpy as np
import pytest
from bayesfl.posterior.packing import ParameterLayout,pack_fola
from bayesfl.wireless.protocol import binding,make_grant,encode,decode,aggregate,canonical
from bayesfl.wireless.client import ClientEngine,MemoryStore
from bayesfl.wireless.smoke import smoke_config
from bayesfl.wireless.oracle import ScoreState
from bayesfl.wireless.reporting import inspect_run
from bayesfl.wireless.payload import payload_bits
from bayesfl.wireless.state import file_sha


@pytest.fixture
def context():
    d=65;layout=ParameterLayout(('x',),((d,),))
    prior=pack_fola([np.zeros(d,np.float32)],[np.ones(d,np.float32)])
    mu=np.arange(d,dtype=np.float32)/d;prec=np.full(d,2,np.float32)
    b=binding(prior,layout,round_id=1,client_id=0,run_token='test')
    return d,layout,prior,mu,prec,b


@pytest.mark.parametrize('m',[1,2,25,64,65])
def test_packet_roundtrip(context,m):
    d,layout,prior,mu,prec,b=context
    mask=np.zeros(d,bool);mask[:m]=True
    g=make_grant(b,count=m,d=d,bandwidth_hz=payload_bits(m,d)/2,window_s=2,efficiency=1)
    arrays,meta=encode(mu,prec,mask,g)
    restored,actual=decode(arrays,meta,grant=g,global_arrays=prior,layout=layout)
    assert np.array_equal(actual,mask)
    assert np.array_equal(restored[0][mask],mu[mask])
    assert np.array_equal(restored[1][mask],prec[mask])
    assert np.array_equal(restored[0][~mask],prior[0][~mask])
    assert len(arrays)==(2 if m==d else 3)


@pytest.mark.parametrize('corruption',['padding','count','snapshot','dtype','grant','negative','format'])
def test_packet_rejects_corruption(context,corruption):
    d,layout,prior,mu,prec,b=context;mask=np.zeros(d,bool);mask[:4]=True
    g=make_grant(b,count=4,d=d,bandwidth_hz=10000,window_s=2,efficiency=1)
    arrays,meta=encode(mu,prec,mask,g)
    if corruption=='padding':arrays[0][-1]|=128
    elif corruption=='count':meta['count']=5
    elif corruption=='snapshot':meta['base_snapshot_id']='wrong'
    elif corruption=='dtype':arrays[1]=arrays[1].astype(np.float64)
    elif corruption=='grant':g['count']=5
    elif corruption=='negative':arrays[2][0]=-1
    elif corruption=='format':meta['packet_format']='dense'
    with pytest.raises(ValueError):decode(arrays,meta,grant=g,global_arrays=prior,layout=layout)


def test_all_omitted_and_dense_matches_reducer(context):
    from bayesfl.posterior.aggregation import aggregate_fola
    d,layout,prior,mu,prec,b=context
    prior[0][:]=.6
    cfg=smoke_config()
    clients=[pack_fola([mu],[prec])]*3
    full=[np.ones(d,bool)]*3
    actual=aggregate(clients,full,[5,7,9],global_arrays=prior,layout=layout,cfg=cfg)
    expected=aggregate_fola(clients,[5/21,7/21,9/21],layout=layout,cfg=cfg)
    for a,e in zip(actual,expected):assert np.array_equal(a,e)
    empty=[np.zeros(d,bool)]*3
    actual=aggregate([prior]*3,empty,[5,7,9],global_arrays=prior,layout=layout,cfg=cfg)
    for a,e in zip(actual,prior):assert np.array_equal(a,e)


def test_prepare_once_scalar_queries_and_replay_rejection(context):
    d,layout,prior,mu,prec,b=context;cfg=smoke_config();store=MemoryStore();calls=[]
    def trainer(*args):calls.append(1);return pack_fola([mu],[prec]),5,{'local_steps':2},layout
    def engine():return ClientEngine(cfg,0,'unused',7,store,trainer=trainer)
    arrays,n,meta=engine().handle(prior,dict(b,phase='prepare'))
    assert arrays==[] and calls==[1]
    for price in [0,1,10]:
        arrays,n,meta=engine().handle([],dict(b,phase='price',price=price,weight=.3,window_s=1,efficiency=2))
        assert arrays==[] and type(meta['count']) is int
    assert calls==[1]
    with pytest.raises(ValueError):engine().handle(prior,dict(b,phase='prepare'))
    with pytest.raises(ValueError):engine().handle([],dict(b,phase='tail',round_id=2,count=1))
    g=make_grant(b,count=4,d=d,bandwidth_hz=10000,window_s=1,efficiency=2)
    arrays,n,meta=engine().handle([],dict(b,phase='upload',grant_json=canonical(g).decode()))
    assert len(arrays)==3 and calls==[1]
    with pytest.raises(ValueError):engine().handle([],dict(b,phase='upload',grant_json=canonical(g).decode()))
    engine().handle([],dict(b,phase='release'));assert store.load() is None


def test_zero_precision_policy(context):
    d,layout,prior,mu,prec,b=context;prior[1][:]=0
    b=binding(prior,layout,round_id=1,client_id=0,run_token='test')
    def trainer(*a):return pack_fola([mu],[prec]),5,{},layout
    cfg=smoke_config()
    with pytest.raises(ValueError,match='zero raw precision'):
        ClientEngine(cfg,0,'unused',7,MemoryStore(),trainer=trainer).handle(prior,dict(b,phase='prepare'))
    cfg.compression.precision_policy='floor_for_score';store=MemoryStore()
    ClientEngine(cfg,0,'unused',7,store,trainer=trainer).handle(prior,dict(b,phase='prepare'))
    meta,arrays=store.load();assert meta['global_floor_fraction']==1
    assert np.array_equal(arrays['precision'],prec) and np.all(prior[1]==0)
