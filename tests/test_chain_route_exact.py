import itertools
import math
from dataclasses import replace

import pytest

from planning.chain_route import ChainSource, solve_chain_route
from planning.chain_route_exact import solve_chain_route_exact, refine_truncated_route


def exhaustive(covers, sources, start, background, dynamic, speed):
    """Enumerate complete interleavings; do not use either planner evaluator."""
    n, k = len(sources), len(covers)
    best = math.inf
    for source_order in itertools.permutations(range(n)):
        for slots in itertools.combinations(range(n+k), n):
            chosen = set(slots); si = ci = 0; total = 0.; current = start
            for j in range(n+k):
                if j in chosen:
                    s = sources[source_order[si]]
                    target, fee = (s.position.x,s.position.y), s.service_s
                    si += 1
                else:
                    target, fee = covers[ci], background+dynamic*(n-si)
                    ci += 1
                total += math.dist(current,target)/speed+fee
                current = target
            best = min(best,total)
    return best


@pytest.mark.parametrize('n,k', list(itertools.product(range(4), range(4))))
@pytest.mark.parametrize('background,dynamic,speed', [(0.,0.,5.),(3.,6.,2.5)])
def test_complete_enumeration_matches(n,k,background,dynamic,speed):
    covers = [(9.*j,-7.*j+2.) for j in range(k)]
    sources = [ChainSource((i*5.-8.,i*i*3.),service_s=2.+i) for i in range(n)]
    result = solve_chain_route_exact(covers,sources,(-5.,6.),background_scan_s=background,scan_source_s=dynamic,speed_mps=speed)
    assert result.cost_s == pytest.approx(exhaustive(covers,sources,(-5.,6.),background,dynamic,speed),abs=1e-10)
    assert result.exact and result.cost_s == result.lower_bound_s
    assert [i for kind,i in result.order if kind=='cover'] == list(range(k))
    assert sorted(i for kind,i in result.order if kind=='source') == list(range(n))
    old=solve_chain_route(covers,sources,(-5.,6.),max_expansions=10000,background_scan_s=background,scan_source_s=dynamic,speed_mps=speed)
    assert old.cost_s == pytest.approx(result.cost_s,abs=1e-9)


def test_no_return_leg_and_dynamic_scan_timing():
    assert solve_chain_route_exact([(10.,0.)],[],scan_source_s=6.).cost_s == 2.
    r=solve_chain_route_exact([(0.,0.)],[ChainSource((0.,0.),5.)],background_scan_s=2.,scan_source_s=6.)
    assert r.order == (('source',0),('cover',0)) and r.cost_s==7.


def test_exact_and_equal_incumbent_objects_are_preserved():
    covers=[(0.,0.)];sources=[ChainSource((3.,4.))]
    old=solve_chain_route(covers,sources,max_expansions=0)
    preserved, log=refine_truncated_route(replace(old,exact=True),covers,sources)
    assert not log['called'] and log['reason']=='incumbent_exact'
    tied=replace(solve_chain_route_exact(covers,sources),exact=False)
    result,log=refine_truncated_route(tied,covers,sources)
    assert result is tied and log['called'] and not log['applied']


def test_improvement_returns_complete_route_and_separate_work():
    sources=[ChainSource((1.,0.))];covers=[(10.,0.)]
    exact=solve_chain_route_exact(covers,sources)
    # Feasible but inferior cover-first route: 10/5+6 + 9/5+5 = 14.8.
    old=replace(exact,cost_s=14.8,lower_bound_s=0.,exact=False,order=(('cover',0),('source',0)))
    result, log=refine_truncated_route(old,covers,sources)
    assert result.exact and log['applied'] and log['proxy_saved_s']==pytest.approx(7.8) and log['first_action_changed']


def test_duplicate_coordinates_preserve_task_identity_and_subtolerance_tie():
    sources=[ChainSource((0.,0.),2.),ChainSource((0.,0.),3.)]
    covers=[(0.,0.),(0.,0.)]
    r=solve_chain_route_exact(covers,sources,scan_source_s=0.)
    assert r.cost_s==5. and len(r.order)==4 and len(set(r.order))==4
    old=replace(r,cost_s=r.cost_s+5e-10,exact=False)
    selected,log=refine_truncated_route(old,covers,sources,scan_source_s=0.)
    assert selected is old and not log['applied']


@pytest.mark.parametrize('kwargs',[{'speed_mps':0},{'background_scan_s':-1},{'scan_source_s':float('nan')},{'speed_mps':True}])
def test_invalid_costs(kwargs):
    with pytest.raises(ValueError):solve_chain_route_exact([],[],**kwargs)


def test_fixed_size_guard_and_maximum_state_case():
    covers=[(float(i),0.) for i in range(22)];sources=[ChainSource((0.,float(i))) for i in range(8)]
    result=solve_chain_route_exact(covers,sources)
    assert result.expanded<=23*256*9 and result.generated<=9*result.expanded
    with pytest.raises(ValueError):solve_chain_route_exact(covers+[(-1.,0.)],sources)
    with pytest.raises(ValueError):solve_chain_route_exact(covers,sources+[ChainSource((1.,1.))])
    old=replace(result,exact=False)
    kept,log=refine_truncated_route(old,covers,sources+[ChainSource((1.,1.))])
    assert kept is old and not log['called'] and log['reason']=='fixed_work_limit'
