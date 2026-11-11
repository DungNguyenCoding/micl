"""Independent pre-implementation checks; NOT an installed wireless allocator."""
from fractions import Fraction as F
from itertools import product
import random, json, math
from pathlib import Path
import numpy as np
import yaml
import argparse, sys, tempfile
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path.cwd(), help='BayesFL root; checks do not edit it.')
args=parser.parse_args()
root=args.root.expanduser().resolve()
if not (root/'src/bayesfl/config.py').is_file():
    parser.error('Use --root pointing to the BayesFL repository.')
sys.dont_write_bytecode=True
sys.path.insert(0,str(root/'src'))
from bayesfl.config import load_config
from bayesfl.posterior.sparse import keep_count, encode_packet

rng=random.Random(9271)

def payload(m,d,b,padded):
    return 2*b*d if m==d else (8*((d+7)//8) if padded else d)+2*b*m

def response(s,pi,U,c,b,lam,padded):
    d=len(s); theta=2*b*lam/(pi*U*c)
    m=min(d-1,max(1,sum(v>theta for v in s)))
    return min([m,d],key=lambda j:(pi*sum(s[j:])+lam*F(payload(j,d,b,padded))/(U*c), F(payload(j,d,b,padded))/(U*c), pi*sum(s[j:]),j))

summary={}
local_checks=0; bound_checks=0; combo_count=0; monotonic_pairs=0
for padded in [False,True]:
 for case in range(300):
    K=rng.randint(1,3); d=rng.randint(2,6); b=rng.choice([1,2,32]); U=F(rng.randint(1,4))
    ns=[rng.randint(1,10) for _ in range(K)]; pis=[F(n,sum(ns)) for n in ns]
    cs=[F(rng.randint(1,7),rng.randint(1,3)) for _ in range(K)]
    ss=[sorted([F(rng.randint(0,20),rng.randint(1,5)) for _ in range(d)],reverse=True) for _ in range(K)]
    costs=[[F(payload(m,d,b,padded))/(U*cs[k]) for m in range(1,d+1)] for k in range(K)]
    objectives=[[pis[k]*sum(ss[k][m:]) for m in range(1,d+1)] for k in range(K)]
    vmin=[min(range(1,d+1),key=lambda j:(costs[k][j-1],objectives[k][j-1])) for k in range(K)]
    wmin=sum(costs[k][vmin[k]-1] for k in range(K))
    wdense=sum(v[-1] for v in costs)
    B=wmin+F(rng.randint(0,10),10)*(wdense-wmin)
    opt=None
    for vec in product(range(1,d+1),repeat=K):
        combo_count+=1
        w=sum(costs[k][vec[k]-1] for k in range(K))
        if w<=B:
            obj=sum(objectives[k][vec[k]-1] for k in range(K))
            opt=obj if opt is None else min(opt,obj)
    assert opt is not None
    incumbent=sum(objectives[k][vmin[k]-1] for k in range(K))
    prev=None
    for lam in [F(0)]+[F(2)**i for i in range(-12,15)]:
        vec=[]
        for k in range(K):
            m=response(ss[k],pis[k],U,cs[k],b,lam,padded)
            score=objectives[k][m-1]+lam*costs[k][m-1]
            exhaustive=min(objectives[k][j]+lam*costs[k][j] for j in range(d))
            assert score==exhaustive
            local_checks+=1;vec.append(m)
        demand=sum(costs[k][vec[k]-1] for k in range(K))
        obj=sum(objectives[k][vec[k]-1] for k in range(K))
        if prev is not None:
            assert demand<=prev;monotonic_pairs+=1
        prev=demand
        lower=obj+lam*(demand-B)
        assert lower<=opt
        if demand<=B:incumbent=min(incumbent,obj)
        assert opt<=incumbent
        bound_checks+=1
summary['exact_rational_oracle']={'cases':600,'local_responses_checked':local_checks,'demand_monotonicity_pairs':monotonic_pairs,'bounds_checked':bound_checks,'joint_count_combinations':combo_count,'status':'PASS','scope':'proposal threshold and weak-duality identities, not a production solver proof'}

summary['integer_roundtrip']={}
for d in [80202,878538]:
    ids=np.arange(1,d+1,dtype=np.float64)
    back=np.ceil((ids/d)*d).astype(np.int64)
    bad=np.flatnonzero(back!=ids)
    example=int(bad[0]+1) if len(bad) else None
    summary['integer_roundtrip'][str(d)]={'m_values_changing_via_ratio':len(bad),'first_m':example,'recomputed_m':keep_count(d,example/d) if example else None}

m=3;d=17
mu=np.arange(d,dtype=np.float32);prec=np.ones(d,dtype=np.float32)
for m in [1,d]:
    mask=np.arange(d)<m
    p=encode_packet(mu,prec,mask)
    actual=sum(a.nbytes for a in p)
    assert actual==8*m+math.ceil(d/8)
summary['legacy_full_retention']={'d':17,'actual_sparse_v1_bytes':actual,'dense_bytes':8*d,'bitmap_bytes':math.ceil(d/8),'status':'PASS: bitmap retained, so v2 format required'}

cfg_path=root/'scripts/configs/fola_mnist.yaml'
raw=yaml.safe_load(cfg_path.read_text());raw['wireless']={'enabled':True,'round_target_s':1.0}
with tempfile.TemporaryDirectory(prefix='bayesfl_wireless_review_') as tmp:
    p=Path(tmp)/'unknown_wireless_probe.yaml'
    p.write_text(yaml.safe_dump(raw))
    cfg=load_config(p)
    seen='wireless' in cfg.to_dict()
summary['config_probe']={'wireless_section_survives_to_dict':seen,'status':('WIRELESS SECTION PRESENT: compare this newer code to audited baseline' if seen else 'CONFIRMED: root wireless section currently silently ignored')}
print(json.dumps(summary,indent=2))
