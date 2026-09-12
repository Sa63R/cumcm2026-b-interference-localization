"""Independent permutation/interleaving reference for release-constrained routing."""
import itertools
import math
import random

import pytest

from planning.chain_route import ChainSource
from planning.matrix_chain_route import solve_matrix_chain_route
from planning.release_chain_route import solve_release_chain_route
from tests.test_matrix_chain_route import orders, cost, check


def feasible(order, releases):
    k = 0
    for kind, i in order:
        if kind == 'cover':
            if i != k: return False
            k += 1
        elif k < releases[i]:
            return False
    return True


def solve(covers, sources, b, w, releases, **kwargs):
    return solve_release_chain_route(covers, sources, background_scan_s=b,
        source_scan_s=w, release_indices=releases, **kwargs)


@pytest.mark.parametrize('seed', range(25))
def test_all_orders_and_interleavings_n_k_at_most_four(seed):
    rng = random.Random(seed)
    n, k = seed % 5, seed // 5
    covers = [(rng.uniform(-50, 50), rng.uniform(-50, 50)) for _ in range(k)]
    sources = [ChainSource((rng.uniform(-50, 50), rng.uniform(-50, 50)), rng.uniform(0, 8)) for _ in range(n)]
    b = [rng.uniform(0, 12) for _ in covers]
    w = [[rng.choice((0., rng.uniform(0, 18))) for _ in sources] for _ in covers]
    releases = [rng.randrange(k+1) for _ in sources]
    optimum = min(cost(o, covers, sources, (0,0), b, w, 5)
                  for o in orders(n, k) if feasible(o, releases))
    previous = math.inf
    for budget in (0, 1, 4, 100000):
        r = solve(covers, sources, b, w, releases, max_expansions=budget)
        assert feasible(r.order, releases)
        check(r, covers, sources, (0,0), b, w)
        assert r.lower_bound_s <= optimum+1e-8 <= r.cost_s+2e-8
        assert r.cost_s <= previous+1e-8 and r.expanded <= budget
        previous = r.cost_s
    assert r.exact and r.cost_s == pytest.approx(optimum, abs=1e-8)


@pytest.mark.parametrize('budget', [0, 1, 200])
@pytest.mark.parametrize('seed', range(5))
def test_zero_release_exact_degeneracy_to_original_matrix_solver(seed, budget):
    rng = random.Random(seed)
    covers = [(rng.uniform(-40,40), rng.uniform(-40,40)) for _ in range(4)]
    sources = [ChainSource((rng.uniform(-40,40), rng.uniform(-40,40)), 5.) for _ in range(4)]
    b = [3., 9., 0., 18.]; w = [[rng.choice((0.,6.)) for _ in sources] for _ in covers]
    old = solve_matrix_chain_route(covers, sources, background_scan_s=b, source_scan_s=w,
                                  max_expansions=budget)
    new = solve(covers, sources, b, w, [0]*4, max_expansions=budget)
    for key in ('order','cost_s','lower_bound_s','expanded','generated','dominance_pruned','bound_pruned','exact'):
        assert getattr(new,key) == getattr(old,key)


@pytest.mark.parametrize('source_order', list(itertools.permutations(range(3))))
def test_fixed_source_interleaving_dp_waits_for_releases(source_order):
    covers=[(10.,10.),(-4.,30.),(17.,-8.),(0.,0.)]
    sources=[ChainSource((8,30),2),ChainSource((-10,-2),4),ChainSource((3,10),6)]
    b=[3,1,7,0]; w=[[0,20,1],[50,0,3],[0,1,40],[6,6,6]]; releases=[0,3,1]
    best=min(cost(o,covers,sources,(0,0),b,w,5) for o in orders(3,4,source_order) if feasible(o,releases))
    r=solve(covers,sources,b,w,releases,initial_source_order=source_order,max_expansions=0)
    assert feasible(r.order,releases) and r.cost_s <= best+1e-8


@pytest.mark.parametrize('budget', [0,1,200])
def test_all_sources_released_at_tail_cannot_use_source_first_incumbent(budget):
    covers=[(100.,0.),(200.,0.)]
    sources=[ChainSource((0.,0.),0.),ChainSource((1.,0.),0.)]
    b=[0.,0.]; w=[[100.,100.],[100.,100.]]
    r=solve(covers,sources,b,w,[2,2],max_expansions=budget)
    assert r.order[:2] == (('cover',0),('cover',1))
    optimum=min(cost(o,covers,sources,(0,0),b,w,5) for o in orders(2,2) if feasible(o,[2,2]))
    assert r.lower_bound_s <= optimum+1e-8 and r.cost_s == pytest.approx(optimum)
    assert r.cost_s > solve(covers,sources,b,w,[0,0],max_expansions=200).cost_s


def test_same_coordinate_tasks_retain_distinct_release_and_matrix_credit():
    covers=[(0,0),(0,0)]; sources=[ChainSource((0,0),0),ChainSource((0,0),0)]
    r=solve(covers,sources,[1,2],[[10,20],[30,40]],[1,2])
    assert feasible(r.order,[1,2]) and r.cost_s == 73.
    assert r.order.index(('source',0)) < r.order.index(('cover',1))
    assert r.order.index(('source',1)) > r.order.index(('cover',1))


def test_maximum_standard_shape_is_bounded_without_mutating_inputs():
    covers=[(i*10.,20.) for i in range(22)]
    sources=[ChainSource((i*13.,-4.),5.) for i in range(16)]
    b=[6.]*22; w=[[float((i+k)%3==0)*6 for i in range(16)] for k in range(22)]
    releases=[i%23 for i in range(16)]
    before=repr((covers,sources,b,w,releases))
    r=solve(covers,sources,b,w,releases,max_expansions=3)
    assert feasible(r.order,releases) and r.expanded <= 3 and r.generated <= 51
    check(r,covers,sources,(0,0),b,w)
    assert repr((covers,sources,b,w,releases)) == before


@pytest.mark.parametrize('releases',[None,[],[0],[0,0,0],[-1,0],[0,3],[0,True],[0,1.],[0,'1'],[0,math.inf]])
def test_invalid_releases_are_rejected(releases):
    with pytest.raises(ValueError): solve([(0,0),(1,0)],[ChainSource((0,0)),ChainSource((1,0))],
                                          [0,0],[[0,0],[0,0]],releases)


def test_empty_tasks_and_no_covers_validate_release():
    r=solve([],[],[],[],[])
    assert r.order == () and r.exact and r.cost_s == 0.
    with pytest.raises(ValueError): solve([],[],[],[],[0])
    with pytest.raises(ValueError): solve([],[ChainSource((0,0))],[],[],[1])
    r=solve([],[ChainSource((0,0),5)],[],[],[0])
    assert r.order == (('source',0),) and r.cost_s == 5.


@pytest.mark.parametrize('b,w', [([0],[[0,0]]),([math.inf,0],[[0],[0]]),
    ([0,0],[[True],[0]]),([0,0],[[-1],[0]]),([0,0],[[1e308],[1e308]])])
def test_existing_matrix_validation_and_overflow_remain(b,w):
    with pytest.raises(ValueError): solve([(0,0),(1,0)],[ChainSource((0,0))],b,w,[0])
