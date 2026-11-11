from __future__ import annotations
import itertools
import math
import numpy as np
import pytest
from bayesfl.wireless.payload import payload_bits, payload_bytes, largest_feasible_count
from bayesfl.wireless.oracle import ScoreState, LocalOracles
from bayesfl.wireless.allocation import allocate, InfeasibleAllocation
from bayesfl.wireless.config import SolverConfig


@pytest.mark.parametrize('d', [1,2,7,8,9,37,80202,878538])
def test_payload(d):
    for m in {1, max(1,d//2),d}:
        assert payload_bits(m,d) == 8*payload_bytes(m,d)
        assert payload_bytes(m,d) == (8*d if m == d else 8*m+(d+7)//8)
    assert largest_feasible_count(payload_bits(d,d),d) == d


@pytest.mark.parametrize('bad', [True,0,-1,1.5,float('nan')])
def test_bad_counts(bad):
    with pytest.raises(ValueError):payload_bytes(bad,10)


def test_roundtrip_bug_not_used():
    assert math.ceil((25/80202)*80202)==26
    assert payload_bytes(25,80202)==8*25+(80202+7)//8
    assert math.ceil((111/878538)*878538)==112


@pytest.mark.parametrize('d',[1,2,3,8,17,65])
def test_oracle_matches_all_counts(d):
    rng=np.random.default_rng(123+d)
    for _ in range(40):
        scores=rng.integers(0,8,d).astype(float)/4
        state=ScoreState(scores)
        last=math.inf
        for price in [0,1e-6,.001,.01,.1,1,100]:
            response=state.at_price(price,.3,2.3,1.7)
            f=lambda m:.3*state.tails[m]+price*payload_bits(m,d)/2.3/1.7
            best=min(range(1,d+1),key=lambda m:(f(m),payload_bits(m,d),state.tails[m]))
            assert f(response.count)==pytest.approx(f(best),rel=1e-12,abs=1e-12)
            demand=payload_bits(response.count,d)
            assert demand<=last
            last=demand


def test_zero_scores_and_ties():
    s=ScoreState(np.zeros(64))
    assert s.at_price(0,1,1,1).count==1
    s=ScoreState(np.array([3.,2.,2.,0.]))
    assert np.flatnonzero(s.mask(2)).tolist()==[0,1]
    assert s.at_count(4).discarded_kl==0


def test_tiny_exhaustive_primal_bounds_and_equal_incumbent():
    rng=np.random.default_rng(91)
    for d in [2,3,5,9]:
        for trial in range(30):
            states=[ScoreState(rng.integers(0,8,d).astype(float)) for _ in range(3)]
            oracles=LocalOracles(states)
            weights=[.2,.3,.5];cs=[.5,1.,2.];U=2.
            minb=sum(min(payload_bits(1,d),payload_bits(d,d))/U/c for c in cs)
            dense=sum(payload_bits(d,d)/U/c for c in cs)
            B=minb+(dense-minb)*rng.random()
            out=allocate(oracles,d=d,weights=weights,window_s=U,efficiencies=cs,bandwidth_hz=B,
                         settings=SolverConfig(max_bisection_queries=70))
            feasible=[(ms,sum(w*s.tails[m] for w,s,m in zip(weights,states,ms)))
                      for ms in itertools.product(range(1,d+1),repeat=3)
                      if math.fsum(payload_bits(m,d)/U/c for m,c in zip(ms,cs))<=B]
            optimum=min(f for _,f in feasible)
            assert math.fsum(out.bandwidth_hz)<=B
            assert out.objective>=optimum-1e-10
            assert out.lower_bound_diagnostic<=optimum+1e-9
            assert out.certified_optimal is False
            try:
                a=allocate(oracles,d=d,weights=weights,window_s=U,efficiencies=cs,bandwidth_hz=B,policy='equal_bandwidth')
            except InfeasibleAllocation:pass
            else:assert out.objective<=a.objective+1e-10


def test_infeasible_and_dense():
    oracle=LocalOracles([ScoreState(np.ones(17))]*2)
    with pytest.raises(InfeasibleAllocation):
        allocate(oracle,d=17,weights=[.5,.5],window_s=1,efficiencies=[1,1],bandwidth_hz=1)
    out=allocate(oracle,d=17,weights=[.5,.5],window_s=1,efficiencies=[1,1],bandwidth_hz=10000)
    assert out.counts==[17,17] and out.objective==0


def test_limited_search_returns_feasible_incumbent():
    out=allocate(LocalOracles([ScoreState(np.full(100,1e12))]*2),d=100,weights=[.5,.5],
                 window_s=1,efficiencies=[1,1],bandwidth_hz=2000,
                 settings=SolverConfig(initial_price=1e-20,max_bracket_queries=1,max_bisection_queries=0))
    assert math.fsum(out.bandwidth_hz)<=2000
    assert out.status=='bracket_limit'
