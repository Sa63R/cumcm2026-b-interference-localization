"""Independent enumeration of the finite matrix-cost proxy, no scenarios."""
import itertools
import math
import random

import pytest

from planning.chain_route import ChainSource, solve_chain_route
from planning.matrix_chain_route import solve_matrix_chain_route


def orders(n, k, source_order=None):
    for permutation in ([source_order] if source_order is not None else itertools.permutations(range(n))):
        for slots in itertools.combinations(range(n+k), k):
            chosen = set(slots)
            ci, si = iter(range(k)), iter(permutation)
            yield tuple(("cover", next(ci)) if j in chosen else ("source", next(si)) for j in range(n+k))


def cost(order, covers, sources, start, b, w, speed):
    location, serviced, total = start, set(), 0.
    for kind, i in order:
        if kind == "cover":
            target = covers[i]
            fee = b[i] + sum(w[i][j] for j in range(len(sources)) if j not in serviced)
        else:
            target = (sources[i].position.x, sources[i].position.y)
            fee = sources[i].service_s
            serviced.add(i)
        total += math.hypot(location[0]-target[0], location[1]-target[1])/speed + fee
        location = target
    return total


def check(result, covers, sources, start, b, w, speed=5):
    assert sorted(i for kind, i in result.order if kind == "source") == list(range(len(sources)))
    assert [i for kind, i in result.order if kind == "cover"] == list(range(len(covers)))
    assert result.cost_s == pytest.approx(cost(result.order,covers,sources,start,b,w,speed),abs=1e-8)
    assert 0 <= result.lower_bound_s <= result.cost_s+1e-8
    assert math.isfinite(result.runtime_s) and result.runtime_s >= 0


@pytest.mark.parametrize("seed", range(20))
def test_exhaustive_source_permutations_and_cover_interleavings(seed):
    rng = random.Random(seed)
    n, k = seed % 5, (seed//5) % 4
    covers = [(rng.uniform(-40,40),rng.uniform(-40,40)) for _ in range(k)]
    sources = [ChainSource((rng.uniform(-40,40),rng.uniform(-40,40)),rng.uniform(0,12)) for _ in range(n)]
    start = (12.,-8.); speed = rng.uniform(1,7)
    b = [rng.uniform(0,16) for _ in covers]
    w = [[rng.choice((0.,rng.uniform(0,20))) for _ in sources] for _ in covers]
    optimum = min(cost(o,covers,sources,start,b,w,speed) for o in orders(n,k))
    previous = math.inf
    for budget in (0,1,4,100000):
        r = solve_matrix_chain_route(covers,sources,start,background_scan_s=b,source_scan_s=w,
                                    speed_mps=speed,max_expansions=budget)
        check(r,covers,sources,start,b,w,speed)
        assert r.lower_bound_s <= optimum+1e-8 <= r.cost_s+2e-8
        assert r.cost_s <= previous+1e-8 and r.expanded <= budget
        previous = r.cost_s
    assert r.exact and r.cost_s == pytest.approx(optimum,abs=1e-8)


@pytest.mark.parametrize("source_order", list(itertools.permutations(range(3))))
def test_fixed_source_dp_uses_identity_not_only_visited_count(source_order):
    covers=[(10.,10.),(-4.,30.),(17.,-8.)]
    sources=[ChainSource((8,30),2),ChainSource((-10,-2),4),ChainSource((3,10),6)]
    b=[3,1,7];w=[[0,20,1],[50,0,3],[0,1,40]]
    best=min(cost(o,covers,sources,(0,0),b,w,5) for o in orders(3,3,source_order))
    r=solve_matrix_chain_route(covers,sources,background_scan_s=b,source_scan_s=w,
                              initial_source_order=source_order,max_expansions=0)
    check(r,covers,sources,(0,0),b,w)
    assert r.cost_s <= best+1e-8


@pytest.mark.parametrize("budget", [0,1,200])
@pytest.mark.parametrize("scan", [0.,6.])
def test_constant_or_zero_matrix_degenerates_to_original_solver(budget,scan):
    covers=[(10,0),(20,12),(-4,8)]
    sources=[ChainSource((12,4),0),ChainSource((-2,10),0),ChainSource((8,-3),0)]
    old=solve_chain_route(covers,sources,max_expansions=budget,background_scan_s=12,scan_source_s=scan)
    new=solve_matrix_chain_route(covers,sources,max_expansions=budget,
                                background_scan_s=[12]*3,source_scan_s=[[scan]*3 for _ in covers])
    for key in ("order","cost_s","lower_bound_s","expanded","generated","dominance_pruned","bound_pruned","exact"):
        assert getattr(new,key)==getattr(old,key)


def test_overlapping_positions_still_have_distinct_source_identity_and_matrix_credit():
    covers=[(0,0),(0,0)];sources=[ChainSource((0,0),0),ChainSource((0,0),0)]
    b=[7,9];w=[[0,100],[40,0]]
    r=solve_matrix_chain_route(covers,sources,background_scan_s=b,source_scan_s=w)
    check(r,covers,sources,(0,0),b,w)
    assert r.exact and r.cost_s==16
    assert r.order.index(('source',1)) < r.order.index(('cover',0))
    assert r.order.index(('source',0)) < r.order.index(('cover',1))


def test_large_supported_task_shape_is_bounded_and_nonmutating():
    covers=[(i*10,20) for i in range(22)]
    sources=[ChainSource((i*13,-4),i%3) for i in range(16)]
    b=[6]*22;w=[[0 if (i+k)%3 else 6 for i in range(16)] for k in range(22)]
    before=[row[:] for row in w]
    r=solve_matrix_chain_route(covers,sources,background_scan_s=b,source_scan_s=w,max_expansions=3)
    check(r,covers,sources,(0,0),b,w)
    assert r.expanded<=3 and r.generated<=17*3 and w==before


@pytest.mark.parametrize("b,w", [
    (None,[[0,0]]),([0],None),([],[[0,0]]),([0,0],[[0,0]]),
    ([0],[]),([0],[[0]]),([0],[[0,0,0]]),([0],[0]),
    ([True],[[0,0]]),([math.inf],[[0,0]]),([-1],[[0,0]]),
    ([0],[[math.nan,0]]),([0],[[0,math.inf]]),([0],[[False,0]]),
    ([0],[[-1,0]]),([0],[["6",0]]),([0],[[10**400,0]])])
def test_invalid_matrix_rejected(b,w):
    with pytest.raises(ValueError):
        solve_matrix_chain_route([(0,0)],[ChainSource((0,0)),ChainSource((1,0))],
                                 background_scan_s=b,source_scan_s=w)


def test_empty_matrices_are_validated_even_for_no_tasks():
    with pytest.raises(ValueError):
        solve_matrix_chain_route([],[],background_scan_s=[0],source_scan_s=[])
    r=solve_matrix_chain_route([],[],background_scan_s=[],source_scan_s=[])
    assert r.exact and r.order==() and r.cost_s==0


def test_aggregate_matrix_overflow_rejected():
    with pytest.raises(ValueError,match='cost scale'):
        solve_matrix_chain_route([(0,0)],[ChainSource((0,0)),ChainSource((0,0))],
                                 background_scan_s=[0],source_scan_s=[[1e308,1e308]])
