"""Independent exhaustive checks for directed, multiple-choice task routing."""

from itertools import permutations, product
import math
import random

import pytest

from planning.mode_route import RouteMode, solve_mode_route
from planning.state_route import RouteTask, solve_state_route
from simulator_client.state import Position


def cost_of(groups, order, start=(0.0, 0.0), speed=5.0):
    """Evaluate the declared objective directly, independently of solver DP."""
    point = Position.coerce(start)
    cost = 0.0
    for group, index in order:
        mode = groups[group][index]
        cost += math.hypot(point.x - mode.entry.x, point.y - mode.entry.y) / speed + mode.service_s
        point = mode.exit
    return cost


def exhaustive(groups, start=(0.0, 0.0), speed=5.0, group_order=None):
    orders = permutations(range(len(groups))) if group_order is None else [group_order]
    choices = [range(len(group)) for group in groups]
    best = math.inf
    for order in orders:
        for modes in product(*choices):
            route = tuple((group, modes[group]) for group in order)
            best = min(best, cost_of(groups, route, start, speed))
    return 0.0 if not groups else best


def random_groups(seed, count=5):
    rng = random.Random(seed)
    return tuple(tuple(RouteMode((rng.uniform(-500, 500), rng.uniform(-500, 500)),
                                  (rng.uniform(-500, 500), rng.uniform(-500, 500)),
                                  rng.uniform(0, 30))
                       for _ in range(rng.randint(1, 3))) for _ in range(count))


def check_feasible(groups, result, start=(0.0, 0.0), speed=5.0):
    assert sorted(group for group, _ in result.order) == list(range(len(groups)))
    assert all(0 <= mode < len(groups[group]) for group, mode in result.order)
    assert result.cost_s == pytest.approx(cost_of(groups, result.order, start, speed), abs=1e-8)
    assert result.runtime_s >= 0


@pytest.mark.parametrize("seed", range(10))
def test_complete_search_matches_permutations_times_modes(seed):
    groups = random_groups(seed)
    start = Position(57.0, -19.0)
    optimum = exhaustive(groups, start, speed=3.5)
    result = solve_mode_route(groups, start, max_expansions=100000, speed_mps=3.5)
    check_feasible(groups, result, start, speed=3.5)
    assert result.exact
    assert result.cost_s == pytest.approx(optimum, abs=1e-8)
    assert result.lower_bound_s == pytest.approx(optimum, abs=1e-8)


@pytest.mark.parametrize("seed", range(4))
def test_anytime_bounds_with_directed_modes_and_zero_budget(seed):
    groups = random_groups(100 + seed)
    optimum = exhaustive(groups)
    previous_cost = math.inf
    for budget in (0, 1, 3, 20, 100000):
        result = solve_mode_route(groups, max_expansions=budget)
        check_feasible(groups, result)
        assert result.expanded <= budget
        assert result.lower_bound_s <= optimum + 1e-8
        assert optimum <= result.cost_s + 1e-8
        assert result.cost_s <= previous_cost + 1e-8
        if result.exact:
            assert result.cost_s == pytest.approx(optimum, abs=1e-8)
        previous_cost = result.cost_s


def test_nonminimal_local_service_can_have_best_exit():
    groups = ((RouteMode((0, 0), (0, 0), 0), RouteMode((0, 0), (100, 0), 1)),
              (RouteMode((100, 0), (100, 0), 0),))
    result = solve_mode_route(groups, initial_order=(0, 1), max_expansions=0)
    # A fixed-order mode DP must retain the more expensive local mode because
    # its exit saves the subsequent 20 s transition. Greedy mode choice fails.
    assert result.order == ((0, 1), (1, 0))
    assert result.cost_s == 1.0
    assert result.lower_bound_s <= 1.0
    assert result.expanded == 0


def test_astar_improves_a_nonoptimal_incumbent_and_exercises_pruning():
    groups = random_groups(0)
    initial = solve_mode_route(groups, max_expansions=0)
    complete = solve_mode_route(groups, max_expansions=100000)
    assert initial.cost_s - complete.cost_s > 30.0
    assert complete.cost_s == pytest.approx(exhaustive(groups), abs=1e-8)
    assert complete.exact and complete.expanded > 0
    assert complete.dominance_pruned > 0 and complete.bound_pruned > 0


def test_entry_exit_transitions_are_directed_and_do_not_charge_internal_distance():
    groups = ((RouteMode((0, 0), (100, 0), 2),),
              (RouteMode((100, 0), (100, 0), 3),))
    assert cost_of(groups, ((0, 0), (1, 0))) == 5.0
    assert cost_of(groups, ((1, 0), (0, 0))) == 45.0
    result = solve_mode_route(groups, max_expansions=100)
    assert result.cost_s == result.lower_bound_s == 5.0
    assert result.order == ((0, 0), (1, 0))
    assert result.exact


@pytest.mark.parametrize("seed", range(3))
def test_given_order_is_preserved_as_an_incumbent_after_mode_dp(seed):
    groups = random_groups(200 + seed)
    initial_order = (3, 0, 4, 2, 1)
    fixed_optimum = exhaustive(groups, group_order=initial_order)
    result = solve_mode_route(groups, initial_order=initial_order, max_expansions=0)
    check_feasible(groups, result)
    assert result.cost_s <= fixed_optimum + 1e-8


@pytest.mark.parametrize("seed", range(5))
def test_singleton_point_modes_match_old_zero_scan_solver_optimum(seed):
    rng = random.Random(300 + seed)
    points = [(rng.uniform(-400, 400), rng.uniform(-400, 400)) for _ in range(6)]
    services = [rng.uniform(0, 100) for _ in points]
    tasks = [RouteTask(point, bool(i % 2), services[i]) for i, point in enumerate(points)]
    groups = [(RouteMode(point, point, services[i]),) for i, point in enumerate(points)]
    old = solve_state_route(tasks, scan_source_s=0, max_expansions=100000)
    new = solve_mode_route(groups, max_expansions=100000)
    assert old.exact and new.exact
    assert new.cost_s == pytest.approx(old.cost_s, abs=1e-8)
    assert new.lower_bound_s == pytest.approx(old.lower_bound_s, abs=1e-8)
    check_feasible(groups, new)


def test_one_group_selects_entry_plus_service_not_exit_or_service_alone():
    groups = [(RouteMode((100, 0), (0, 0), 0), RouteMode((0, 0), (200, 0), 7),
               RouteMode((10, 0), (0, 0), 6), RouteMode((0, 0), (0, 0), 9))]
    result = solve_mode_route(groups, max_expansions=0)
    assert result.order == ((0, 1),)
    assert result.cost_s == 7.0
    assert result.lower_bound_s <= result.cost_s


def test_empty_input_and_supported_maximum_with_deterministic_ties():
    empty = solve_mode_route([], initial_order=(), max_expansions=0)
    assert empty.order == ()
    assert empty.cost_s == empty.lower_bound_s == 0.0
    assert empty.exact and empty.expanded == 0
    groups = [(RouteMode((0, 0), (0, 0)),) * 4 for _ in range(22)]
    first = solve_mode_route(groups, max_expansions=0)
    second = solve_mode_route(groups, max_expansions=0)
    assert first.order == second.order == tuple((i, 0) for i in range(22))
    assert first.cost_s == first.lower_bound_s == 0.0
    check_feasible(groups, first)


@pytest.mark.parametrize("service", [-1, math.nan, math.inf, -math.inf, True, "1", 10**1000])
def test_invalid_mode_service_rejected(service):
    with pytest.raises(ValueError, match="service_s"):
        RouteMode((0, 0), (0, 0), service)


@pytest.mark.parametrize("coordinate", [math.nan, math.inf, 2000001, True, "bad", 10**1000])
def test_invalid_mode_coordinates_rejected(coordinate):
    with pytest.raises(ValueError):
        RouteMode((coordinate, 0), (0, 0))
    with pytest.raises(ValueError):
        RouteMode((0, 0), (0, coordinate))


@pytest.mark.parametrize("speed", [0, -1, math.nan, math.inf, True, "5", 10**1000])
def test_invalid_speed_rejected(speed):
    with pytest.raises(ValueError, match="speed_mps"):
        solve_mode_route([], speed_mps=speed)


@pytest.mark.parametrize("budget", [-1, 0.5, math.inf, True, "100"])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError, match="max_expansions"):
        solve_mode_route([], max_expansions=budget)


@pytest.mark.parametrize("groups", [None, [None], [[]], [[object()]],
    [(RouteMode((0, 0), (0, 0)),) * 5],
    [(RouteMode((0, 0), (0, 0)),)] * 23])
def test_invalid_groups_rejected(groups):
    with pytest.raises(ValueError):
        solve_mode_route(groups)


@pytest.mark.parametrize("order", [(0,), (0, 0), (0, 2), (True, 0), (0.0, 1),
                                  ((0, 0), (1, 0)), 1, "01"])
def test_initial_order_must_be_a_complete_group_permutation(order):
    mode = RouteMode((0, 0), (0, 0))
    with pytest.raises(ValueError, match="initial_order"):
        solve_mode_route([(mode,), (mode,)], initial_order=order)


def test_overflowing_costs_rejected_instead_of_false_numeric_certificates():
    with pytest.raises(ValueError, match="cost scale"):
        solve_mode_route([(RouteMode((0, 0), (0, 0), 1e308),)] * 2)
    with pytest.raises(ValueError, match="cost scale"):
        solve_mode_route([(RouteMode((100, 0), (100, 0)),)], speed_mps=1e-320)
