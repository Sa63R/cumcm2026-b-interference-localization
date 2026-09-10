"""Anytime subset search for a *frozen* Q3 task ordering model.

This is not an optimizer over hidden source positions or future observations.
For fixed point tasks, a cover visit pays a supplied base cost and an extra
``scan_source_s`` for each source task not yet executed. A source visit pays
its fixed service cost. The objective is open-path travel time plus service.

The compressed state is (visited_mask, last_task); order histories reaching
the same state have identical continuations, so the cheaper one dominates.
An MST on unvisited points plus a cheapest connection and unavoidable service
costs is an admissible remaining-cost bound. A* with an incumbent returns an
optimal order when exhausted, or a feasible order and a model-only gap when
its deterministic expansion budget is exhausted.
"""

from dataclasses import dataclass
from functools import lru_cache
import heapq
import math
import time

from simulator_client.state import Position


@dataclass(frozen=True)
class RouteTask:
    position: Position
    is_source: bool
    service_s: float = 0.0


@dataclass(frozen=True)
class StateRouteResult:
    order: tuple[int, ...]
    cost_s: float
    lower_bound_s: float
    expanded: int
    generated: int
    dominance_pruned: int
    bound_pruned: int
    exact: bool
    runtime_s: float


def solve_state_route(tasks, start=(0.0, 0.0), *, speed_mps=5.0,
                      scan_source_s=6.0, max_expansions=5000,
                      travel_times_s=None, initial_times_s=None):
    """Optimize the declared finite task model, with at most 22 tasks.

    Numerical comparisons use conservative 1e-9 s tolerances. The certificate
    concerns the floating-point instance, not exact real Euclidean distances.
    Service costs and positions must remain frozen throughout this call.
    Optional nonnegative symmetric travel times define a separate finite
    matrix objective. They need not satisfy a triangle inequality: an MST
    still lower-bounds every Hamiltonian continuation on the same matrix.
    """
    began = time.perf_counter()
    tasks = tuple(tasks)
    n = len(tasks)
    if n > 22:
        raise ValueError("state route supports at most 22 fixed tasks")
    if (not math.isfinite(speed_mps) or speed_mps <= 0
            or not math.isfinite(scan_source_s) or scan_source_s < 0):
        raise ValueError("speed must be positive and scan cost nonnegative")
    if (isinstance(max_expansions, bool) or not isinstance(max_expansions, int)
            or max_expansions < 0):
        raise ValueError("max_expansions must be a nonnegative integer")
    for task in tasks:
        if not isinstance(task.is_source, bool):
            raise ValueError("is_source must be boolean")
        if not math.isfinite(task.service_s) or task.service_s < 0:
            raise ValueError("service costs must be finite and nonnegative")
    if (travel_times_s is None) != (initial_times_s is None):
        raise ValueError("travel and initial times must be supplied together")
    if travel_times_s is not None:
        if len(travel_times_s)!=n or len(initial_times_s)!=n or any(len(row)!=n for row in travel_times_s):
            raise ValueError("travel matrix dimensions must match tasks")
        values=list(initial_times_s)+[v for row in travel_times_s for v in row]
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values):
            raise ValueError("travel times must be finite and nonnegative")
        if any(travel_times_s[i][i]!=0 for i in range(n)):
            raise ValueError("travel matrix diagonal must be zero")
        if any(travel_times_s[i][j]!=travel_times_s[j][i] for i in range(n) for j in range(i)):
            raise ValueError("travel matrix must be symmetric")
    if not n:
        return StateRouteResult((), 0.0, 0.0, 0, 0, 0, 0, True,
                                time.perf_counter() - began)
    points = [Position.coerce(task.position) for task in tasks]
    start = Position.coerce(start)
    distances = ([[a.distance_to(b) / speed_mps for b in points] for a in points]
                 if travel_times_s is None else [list(row) for row in travel_times_s])
    initial = ([start.distance_to(p) / speed_mps for p in points]
               if initial_times_s is None else list(initial_times_s))
    all_mask = (1 << n) - 1
    source_mask = sum(1 << i for i, task in enumerate(tasks) if task.is_source)
    services = [task.service_s for task in tasks]

    @lru_cache(maxsize=None)
    def indices(mask):
        result = []
        while mask:
            bit = mask & -mask
            result.append(bit.bit_length() - 1)
            mask ^= bit
        return tuple(result)

    @lru_cache(maxsize=None)
    def base_service(mask):
        return sum(services[i] for i in indices(mask))

    @lru_cache(maxsize=None)
    def mst(mask):
        vertices = indices(mask)
        if len(vertices) < 2:
            return 0.0
        first, *rest = vertices
        cheapest = {i: distances[first][i] for i in rest}
        value = 0.0
        while cheapest:
            vertex = min(cheapest, key=cheapest.get)
            value += cheapest.pop(vertex)
            for other in cheapest:
                cheapest[other] = min(cheapest[other], distances[vertex][other])
        return value

    def estimate(mask, last):
        remaining = all_mask ^ mask
        if not remaining:
            return 0.0
        row = initial if last < 0 else distances[last]
        # Dropping every source/cover ordering penalty can only lower cost.
        return (min(row[i] for i in indices(remaining)) + mst(remaining)
                + base_service(remaining))

    def increment(mask, last, i):
        movement = initial[i] if last < 0 else distances[last][i]
        scan = 0.0 if tasks[i].is_source else scan_source_s * (
            source_mask & ~mask).bit_count()
        return movement + services[i] + scan

    def route_cost(order):
        cost, mask, last = 0.0, 0, -1
        for i in order:
            cost += increment(mask, last, i)
            mask |= 1 << i
            last = i
        return cost

    # Several deterministic seeds followed by true-objective 2-opt improve
    # the upper bound. Unlike Euclidean 2-opt, reversal changes scan costs too.
    candidates = []
    for source_bonus in (0.0, scan_source_s * (n - source_mask.bit_count())):
        order, mask, last = [], 0, -1
        while len(order) < n:
            i = min(indices(all_mask ^ mask), key=lambda j: (
                increment(mask, last, j) - source_bonus * tasks[j].is_source, j))
            order.append(i)
            mask |= 1 << i
            last = i
        candidates.append(order)
    best_order = min(candidates, key=route_cost)
    upper = route_cost(best_order)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                alternate = best_order[:i] + best_order[i:j + 1][::-1] + best_order[j + 1:]
                score = route_cost(alternate)
                if score < upper - 1e-9:
                    best_order, upper, improved = alternate, score, True
                    break
            if improved:
                break

    lower = estimate(0, -1)
    queue = [(lower, 0.0, 0, -1, ())]
    best_g = {(0, -1): 0.0}
    expanded = generated = dominance_pruned = bound_pruned = 0
    while queue and expanded < max_expansions:
        bound, cost, mask, last, order = heapq.heappop(queue)
        if cost > best_g.get((mask, last), math.inf) + 1e-9:
            dominance_pruned += 1
            continue
        if bound > upper - 1e-9:
            bound_pruned += 1
            continue
        expanded += 1
        for i in indices(all_mask ^ mask):
            generated += 1
            new_mask = mask | (1 << i)
            new_cost = cost + increment(mask, last, i)
            if new_cost >= best_g.get((new_mask, i), math.inf) - 1e-9:
                dominance_pruned += 1
                continue
            best_g[new_mask, i] = new_cost
            new_order = order + (i,)
            if new_mask == all_mask:
                if new_cost < upper:
                    upper, best_order = new_cost, new_order
                continue
            new_bound = new_cost + estimate(new_mask, i)
            if new_bound > upper - 1e-9:
                bound_pruned += 1
                continue
            heapq.heappush(queue, (new_bound, new_cost, new_mask, i, new_order))
    # Stale queue entries can weaken this bound, never overstate it. Incumbent
    # descendants need not remain on the queue, so cap the bound at its cost.
    lower = min(upper, queue[0][0]) if queue else upper
    return StateRouteResult(tuple(best_order), upper, lower, expanded, generated,
                            dominance_pruned, bound_pruned, not queue,
                            time.perf_counter() - began)
