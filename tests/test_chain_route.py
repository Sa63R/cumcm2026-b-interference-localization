"""Independent finite-graph checks; no simulator or scene truth is used."""

import itertools
import math
import random

import pytest

from planning.chain_route import ChainSource, solve_chain_route
from simulator_client.state import Position


def path_cost(path, covers, sources, start=(0, 0), *, background=0.0, scan=6.0, speed=5.0):
    current, cost, cleared = Position.coerce(start), 0.0, 0
    for kind, index in path:
        if kind == "source":
            destination, fee = sources[index].position, sources[index].service_s
            cleared += 1
        else:
            assert kind == "cover"
            destination = Position.coerce(covers[index])
            fee = background + scan * (len(sources) - cleared)
        cost += math.hypot(destination.x - current.x, destination.y - current.y) / speed + fee
        current = destination
    return cost


def all_orders(source_count, cover_count, *, source_order=None):
    # Enumerate independently: choose positions of the covers, then a source
    # permutation. Cover indices are inserted in increasing order by definition.
    count = source_count + cover_count
    permutations = [source_order] if source_order is not None else itertools.permutations(range(source_count))
    for permutation in permutations:
        for cover_slots in itertools.combinations(range(count), cover_count):
            source_iter, cover_iter = iter(permutation), iter(range(cover_count))
            slots = set(cover_slots)
            yield tuple(("cover", next(cover_iter)) if i in slots
                        else ("source", next(source_iter)) for i in range(count))


def brute(covers, sources, start=(0, 0), *, background=0.0, scan=6.0, speed=5.0, source_order=None):
    return min((path_cost(order, covers, sources, start, background=background, scan=scan, speed=speed), order)
               for order in all_orders(len(sources), len(covers), source_order=source_order))


def assert_result(result, covers, sources, start=(0, 0), *, background=0.0, scan=6.0, speed=5.0):
    assert len(result.order) == len(covers) + len(sources)
    assert [i for kind, i in result.order if kind == "cover"] == list(range(len(covers)))
    assert sorted(i for kind, i in result.order if kind == "source") == list(range(len(sources)))
    assert result.next_action == (result.order[0] if result.order else None)
    assert result.cost_s == pytest.approx(path_cost(result.order, covers, sources, start,
                                                   background=background, scan=scan, speed=speed))
    assert math.isfinite(result.lower_bound_s)
    assert 0 <= result.lower_bound_s <= result.cost_s + 1e-8
    assert 0 <= result.runtime_s < math.inf
    assert all(type(value) is int and value >= 0 for value in
               (result.expanded, result.generated, result.dominance_pruned, result.bound_pruned))


@pytest.mark.parametrize("seed", range(16))
def test_exact_against_all_source_permutations_and_cover_interleavings(seed):
    rng = random.Random(seed)
    n, k = seed % 4 + 1, seed % 3 + 1
    start = (rng.uniform(-50, 50), rng.uniform(-50, 50))
    covers = [(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(k)]
    sources = [ChainSource((rng.uniform(-100, 100), rng.uniform(-100, 100)), rng.uniform(0, 12))
               for _ in range(n)]
    background, scan, speed = rng.uniform(0, 10), rng.uniform(0, 15), rng.uniform(1, 9)
    optimum, _ = brute(covers, sources, start, background=background, scan=scan, speed=speed)
    result = solve_chain_route(covers, sources, start, max_expansions=100_000,
                               background_scan_s=background, scan_source_s=scan, speed_mps=speed)
    assert_result(result, covers, sources, start, background=background, scan=scan, speed=speed)
    assert result.exact
    assert result.cost_s == pytest.approx(optimum, abs=1e-8)
    assert result.lower_bound_s == pytest.approx(optimum, abs=1e-8)


@pytest.mark.parametrize("seed", range(8))
def test_budgeted_bounds_and_feasibility_against_exhaustive_optimum(seed):
    rng = random.Random(seed + 128)
    covers = [(rng.uniform(-200, 200), rng.uniform(-200, 200)) for _ in range(3)]
    sources = [ChainSource((rng.uniform(-200, 200), rng.uniform(-200, 200)), i + 0.75) for i in range(4)]
    optimum, _ = brute(covers, sources, background=7.25, scan=8.5)
    old_upper = math.inf
    for budget in (0, 1, 3, 10, 1000):
        result = solve_chain_route(covers, sources, max_expansions=budget,
                                   background_scan_s=7.25, scan_source_s=8.5)
        assert_result(result, covers, sources, background=7.25, scan=8.5)
        assert result.expanded <= budget
        assert result.lower_bound_s <= optimum + 1e-8
        assert optimum <= result.cost_s + 1e-8
        assert result.cost_s <= old_upper + 1e-8
        old_upper = result.cost_s


def test_dynamic_scanning_credit_prefers_sources_before_colocated_covers():
    covers = [(0, 0)] * 4
    sources = [ChainSource((0, 0), 5.0), ChainSource((0, 0), 7.0)]
    result = solve_chain_route(covers, sources, background_scan_s=3, scan_source_s=6)
    assert_result(result, covers, sources, background=3)
    assert result.order[:2] == (("source", 0), ("source", 1))
    assert result.cost_s == pytest.approx(24)
    cover_first = tuple(("cover", k) for k in range(4)) + (("source", 0), ("source", 1))
    assert path_cost(cover_first, covers, sources, background=3) - result.cost_s == 48


def test_interleaving_can_beat_both_end_block_orders_with_zero_search_budget():
    covers = [(10, 0), (20, 0), (30, 0)]
    sources = [ChainSource((15, 0), 2)]
    result = solve_chain_route(covers, sources, max_expansions=0, scan_source_s=0.1, speed_mps=1)
    assert_result(result, covers, sources, scan=0.1, speed=1)
    assert result.order == (("cover", 0), ("source", 0), ("cover", 1), ("cover", 2))
    assert result.cost_s == pytest.approx(32.1)
    cover_actions = tuple(("cover", i) for i in range(3))
    for order in (cover_actions + (("source", 0),), (("source", 0),) + cover_actions):
        assert result.cost_s < path_cost(order, covers, sources, scan=0.1, speed=1)


@pytest.mark.parametrize("source_order", tuple(itertools.permutations(range(3))))
def test_supplied_source_order_gets_optimal_chain_interleaving_even_at_budget_zero(source_order):
    covers = [(10, 10), (-4, 30), (17, -8)]
    sources = [ChainSource((8, 30), 2), ChainSource((-10, -2), 4), ChainSource((3, 10), 6)]
    optimum_for_order, _ = brute(covers, sources, source_order=source_order, background=3, scan=2)
    result = solve_chain_route(covers, sources, max_expansions=0, initial_source_order=source_order,
                               background_scan_s=3, scan_source_s=2)
    assert_result(result, covers, sources, background=3, scan=2)
    assert result.cost_s <= optimum_for_order + 1e-8


def test_empty_source_only_and_cover_only():
    empty = solve_chain_route([], [], max_expansions=0)
    assert_result(empty, [], [])
    assert empty.exact and empty.cost_s == empty.lower_bound_s == 0
    assert empty.expanded == empty.generated == 0
    covers = [(3, 4), (3, 4), (0, 0)]
    cover_only = solve_chain_route(covers, [], max_expansions=0, background_scan_s=9, speed_mps=2)
    assert_result(cover_only, covers, [], background=9, speed=2)
    assert cover_only.exact and cover_only.cost_s == pytest.approx(32)
    sources = [ChainSource((1, 2), 1), ChainSource((-3, 2), 8), ChainSource((12, 12), 0)]
    source_only = solve_chain_route([], sources, max_expansions=1000)
    assert_result(source_only, [], sources)
    assert source_only.exact and source_only.cost_s == pytest.approx(brute([], sources)[0])


@pytest.mark.parametrize("budget", (0, 1, 100))
def test_thirty_six_covers_sixteen_sources_remain_finite_with_limited_expansions(budget):
    rng = random.Random(431)
    covers = [(1500 * math.cos(i * 0.28), 1500 * math.sin(i * 0.28)) for i in range(36)]
    sources = [ChainSource((rng.uniform(-1200, 1200), rng.uniform(-1200, 1200)), 5 + i % 3)
               for i in range(16)]
    result = solve_chain_route(covers, sources, max_expansions=budget, background_scan_s=24,
                               initial_source_order=tuple(range(16)))
    assert_result(result, covers, sources, background=24)
    assert result.expanded <= budget
    assert result.generated <= 17 * budget
    cover_actions = tuple(("cover", i) for i in range(36))
    source_actions = tuple(("source", i) for i in range(16))
    assert result.cost_s <= path_cost(cover_actions + source_actions, covers, sources, background=24)
    assert result.cost_s <= path_cost(source_actions + cover_actions, covers, sources, background=24)


def test_joint_search_improves_heuristic_source_orders_and_closes_the_bound():
    rng = random.Random(0)
    covers = [(rng.uniform(-200, 200), rng.uniform(-200, 200)) for _ in range(4)]
    sources = [ChainSource((rng.uniform(-200, 200), rng.uniform(-200, 200)), 5) for _ in range(5)]
    initial = solve_chain_route(covers, sources, max_expansions=0, background_scan_s=12)
    searched = solve_chain_route(covers, sources, max_expansions=100_000, background_scan_s=12)
    optimum, _ = brute(covers, sources, background=12)
    assert_result(searched, covers, sources, background=12)
    assert searched.exact and searched.expanded > 0
    assert searched.dominance_pruned > 0 and searched.bound_pruned > 0
    assert searched.cost_s < initial.cost_s - 30
    assert searched.cost_s == pytest.approx(optimum, abs=1e-8)
    assert searched.lower_bound_s == pytest.approx(optimum, abs=1e-8)


@pytest.mark.parametrize("kwargs", [
    {"max_expansions": -1}, {"max_expansions": True}, {"max_expansions": 1.5},
    {"speed_mps": 0}, {"speed_mps": -1}, {"speed_mps": True}, {"speed_mps": math.inf},
    {"scan_source_s": -1}, {"scan_source_s": math.nan}, {"scan_source_s": "6"},
    {"background_scan_s": -2}, {"background_scan_s": math.inf}, {"background_scan_s": True},
    {"initial_source_order": [0, 0]}, {"initial_source_order": [1]},
    {"initial_source_order": [False]}, {"initial_source_order": 1},
])
def test_invalid_solver_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        solve_chain_route([(0, 0)], [ChainSource((0, 0))], **kwargs)


@pytest.mark.parametrize("value", (-1, math.nan, math.inf, True, "5", 10**400))
def test_invalid_source_service_rejected(value):
    with pytest.raises(ValueError):
        ChainSource((0, 0), value)


@pytest.mark.parametrize("covers,sources,start", [
    (None, [], (0, 0)), ([], None, (0, 0)),
    ([], [(0, 0)], (0, 0)), ([(math.nan, 0)], [], (0, 0)),
    ([], [], (True, 0)), ([], [], (math.inf, 0)),
    ([], [ChainSource((0, 0))] * 17, (0, 0)),
])
def test_invalid_tasks_rejected(covers, sources, start):
    with pytest.raises(ValueError):
        solve_chain_route(covers, sources, start)


def test_cost_scale_overflow_is_rejected():
    with pytest.raises(ValueError, match="cost scale"):
        solve_chain_route([(1, 0)], [], speed_mps=1e-320)


def test_source_dataclass_normalizes_position_and_service():
    source = ChainSource([1, 2], 3)
    assert source.position == Position(1, 2)
    assert type(source.service_s) is float
