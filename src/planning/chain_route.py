"""Bounded routing of source tasks interleaved with an ordered coverage chain.

Only the <=16 sources use a bitmask. A prefix length represents the coverage
chain, whose order is preserved even when it has more than 22 stations. The
compressed state (source_mask, cover_prefix, last_source_or_cover) completely
determines the frozen model's remaining costs.

The movement lower bound is the maximum of two relaxations: delete all sources
and follow the remaining cover chain; or delete all covers and lower-bound the
remaining source path by an entry edge plus a source MST. These bounds must NOT
be added, because they may charge the same movement. Fixed services/base scans
are added once; nonnegative dynamic scan charges are omitted from the bound.

The route and numerical gap concern this finite point-task proxy only. Future
observations, source discoveries and actual localization paths are not modeled.
"""

from dataclasses import dataclass
from functools import lru_cache
import heapq
import math
import time

from simulator_client.state import Position


_TOLERANCE_S = 1e-9


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (positive and value == 0):
        sign = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {sign}")
    return float(value)


@dataclass(frozen=True)
class ChainSource:
    position: Position
    service_s: float = 5.0

    def __post_init__(self):
        object.__setattr__(self, "position", Position.coerce(self.position))
        object.__setattr__(self, "service_s", _number(self.service_s, "service_s"))


@dataclass(frozen=True)
class ChainRouteResult:
    order: tuple[tuple[str, int], ...]
    cost_s: float
    lower_bound_s: float
    expanded: int
    generated: int
    dominance_pruned: int
    bound_pruned: int
    exact: bool
    runtime_s: float

    @property
    def next_action(self):
        return self.order[0] if self.order else None


def solve_chain_route(cover_points, sources, start=(0.0, 0.0), *, max_expansions=100,
                      background_scan_s=0.0, scan_source_s=6.0, speed_mps=5.0,
                      initial_source_order=None):
    """Visit each source once and every cover in its supplied order.

    Visiting a source pays its frozen ``service_s``. Cover k pays
    ``background_scan_s + scan_source_s * remaining_source_count`` at the time
    it is visited. Costs are seconds, with straight movement at ``speed_mps``.
    There is no return leg.

    Incumbents include all covers before source-greedy/2-opt routes, sources
    before covers, and greedy interleavings. For every retained source order,
    an O(N*K) label DP additionally optimizes the entire interleaving exactly.
    ``initial_source_order`` may supply another complete source permutation.
    A* then searches other source permutations and interleavings together.

    ``max_expansions`` limits A* expansions, not incumbent construction. The
    sparse state space is O(2**N * (K+1) * (N+1)), with <=N+1 transitions/state.
    Source MSTs are cached by mask, independent of cover prefix. ``exact`` means
    frontier closure with 1e-9 s comparisons, not an exact-arithmetic proof.
    """
    began = time.perf_counter()
    speed_mps = _number(speed_mps, "speed_mps", positive=True)
    background_scan_s = _number(background_scan_s, "background_scan_s")
    scan_source_s = _number(scan_source_s, "scan_source_s")
    if (isinstance(max_expansions, bool) or not isinstance(max_expansions, int)
            or max_expansions < 0):
        raise ValueError("max_expansions must be a nonnegative integer")
    start = Position.coerce(start)
    try:
        covers = tuple(Position.coerce(p) for p in cover_points)
        sources = tuple(sources)
    except TypeError as exc:
        raise ValueError("cover_points and sources must be sequences") from exc
    n, cover_count = len(sources), len(covers)
    if n > 16:
        raise ValueError("chain route supports at most 16 sources")
    if any(not isinstance(source, ChainSource) for source in sources):
        raise ValueError("sources must contain ChainSource objects")
    if initial_source_order is not None:
        try:
            initial_source_order = tuple(initial_source_order)
        except TypeError as exc:
            raise ValueError("initial_source_order must be a permutation of source indices") from exc
        if (len(initial_source_order) != n
                or any(type(i) is not int for i in initial_source_order)
                or set(initial_source_order) != set(range(n))):
            raise ValueError("initial_source_order must be a permutation of source indices")
    if not n and not cover_count:
        return ChainRouteResult((), 0.0, 0.0, 0, 0, 0, 0, True,
                                time.perf_counter() - began)

    points = tuple(source.position for source in sources) + covers + (start,)
    origin = len(points) - 1
    travel = [[a.distance_to(b) / speed_mps for b in points] for a in points]
    services = tuple(source.service_s for source in sources)
    scan_fees = tuple(background_scan_s + scan_source_s * (n - i) for i in range(n + 1))
    maximum_cost = ((n + cover_count) * max(max(row) for row in travel)
                    + sum(services) + cover_count * (background_scan_s + scan_source_s * n))
    if not math.isfinite(maximum_cost):
        raise ValueError("chain route cost scale must remain finite")
    all_mask = (1 << n) - 1
    cover_suffix = [0.0] * (cover_count + 1)
    for k in range(cover_count - 2, -1, -1):
        cover_suffix[k] = travel[n + k][n + k + 1] + cover_suffix[k + 1]

    @lru_cache(maxsize=None)
    def indices(mask):
        result = []
        while mask:
            bit = mask & -mask
            result.append(bit.bit_length() - 1)
            mask ^= bit
        return tuple(result)

    @lru_cache(maxsize=None)
    def service_bound(mask):
        return sum(services[i] for i in indices(mask))

    @lru_cache(maxsize=None)
    def source_mst(mask):
        vertices = indices(mask)
        if len(vertices) < 2:
            return 0.0
        first, *rest = vertices
        cheapest = {i: travel[first][i] for i in rest}
        result = 0.0
        while cheapest:
            vertex = min(cheapest, key=lambda i: (cheapest[i], i))
            result += cheapest.pop(vertex)
            for other in cheapest:
                cheapest[other] = min(cheapest[other], travel[vertex][other])
        return result

    def location(k, last):
        # last=-1 means the latest cover, or the external start if k=0.
        return last if last >= 0 else n + k - 1 if k else origin

    def estimate(mask, k, last):
        remaining = all_mask ^ mask
        current = location(k, last)
        cover_move = travel[current][n + k] + cover_suffix[k] if k < cover_count else 0.0
        source_move = (min(travel[current][i] for i in indices(remaining)) + source_mst(remaining)
                       if remaining else 0.0)
        return (max(cover_move, source_move) + service_bound(remaining)
                + (cover_count - k) * background_scan_s)

    def route_cost(order):
        cost, current, cleared = 0.0, origin, 0
        for kind, index in order:
            if kind == "source":
                vertex, service = index, services[index]
                cleared += 1
            else:
                vertex = n + index
                service = scan_fees[cleared]
            cost += travel[current][vertex] + service
            current = vertex
        return cost

    def source_greedy(anchor):
        order, remaining, current = [], set(range(n)), anchor
        while remaining:
            chosen = min(remaining, key=lambda i: (travel[current][i], i))
            order.append(chosen)
            remaining.remove(chosen)
            current = chosen
        return tuple(order)

    def source_path_cost(order, anchor):
        if not order:
            return 0.0
        return travel[anchor][order[0]] + sum(travel[a][b] for a, b in zip(order, order[1:]))

    def source_two_opt(order, anchor):
        cost = source_path_cost(order, anchor)
        for _ in range(n):
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    alternate = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                    alternate_cost = source_path_cost(alternate, anchor)
                    if alternate_cost < cost - _TOLERANCE_S:
                        order, cost, improved = alternate, alternate_cost, True
                        break
                if improved:
                    break
            if not improved:
                break
        return order

    def greedy_interleave(source_bonus):
        order, mask, k, current = [], 0, 0, origin
        while mask != all_mask or k < cover_count:
            choices = [(travel[current][i] + services[i]
                        - source_bonus * scan_source_s * (cover_count - k), "source", i)
                       for i in indices(all_mask ^ mask)]
            if k < cover_count:
                choices.append((travel[current][n + k] + scan_fees[mask.bit_count()], "cover", k))
            _, kind, index = min(choices)
            order.append((kind, index))
            if kind == "source":
                mask |= 1 << index
                current = index
            else:
                k += 1
                current = n + index
        return tuple(order)

    def interleave_fixed_sources(source_order):
        # State (i,k,last_kind) merges prefixes from the two ordered lists.
        # Counts i,k determine dynamic scans; last_kind determines the endpoint.
        labels = {(0, 0, -1): 0.0}
        parents = {}
        for i in range(n + 1):
            for k in range(cover_count + 1):
                for last_kind in (-1, 0, 1):
                    previous = (i, k, last_kind)
                    cost = labels.get(previous)
                    if cost is None:
                        continue
                    current = (source_order[i - 1] if last_kind == 0
                               else n + k - 1 if last_kind == 1 else origin)
                    if i < n:
                        source = source_order[i]
                        new_cost = cost + (travel[current][source] + services[source])
                        key = (i + 1, k, 0)
                        if new_cost < labels.get(key, math.inf):
                            labels[key] = new_cost
                            parents[key] = previous, ("source", source)
                    if k < cover_count:
                        new_cost = cost + (travel[current][n + k] + scan_fees[i])
                        key = (i, k + 1, 1)
                        if new_cost < labels.get(key, math.inf):
                            labels[key] = new_cost
                            parents[key] = previous, ("cover", k)
        cost, key = min((value, key) for key, value in labels.items()
                        if key[0] == n and key[1] == cover_count)
        path = []
        while key in parents:
            key, action = parents[key]
            path.append(action)
        return cost, tuple(reversed(path))

    source_orders = set()
    if initial_source_order is not None:
        source_orders.add(initial_source_order)
    for anchor in (origin, n + cover_count - 1 if cover_count else origin):
        greedy = source_greedy(anchor)
        source_orders.update((greedy, source_two_opt(greedy, anchor)))
    candidates = []
    for bonus in (0, 1):
        order = greedy_interleave(bonus)
        candidates.append((route_cost(order), order))
        source_orders.add(tuple(index for kind, index in order if kind == "source"))
    cover_order = tuple(("cover", k) for k in range(cover_count))
    for source_order in sorted(source_orders):
        source_actions = tuple(("source", i) for i in source_order)
        for order in (cover_order + source_actions, source_actions + cover_order):
            candidates.append((route_cost(order), order))
        candidates.append(interleave_fixed_sources(source_order))
    _, best_order = min(candidates)
    # Use exactly the transition grouping used by A* and the public evaluator.
    upper = route_cost(best_order)
    queue = [(estimate(0, 0, -1), 0.0, 0, 0, -1, ())]
    best_g = {(0, 0, -1): 0.0}
    expanded = generated = dominance_pruned = bound_pruned = 0
    while queue:
        bound, cost, mask, k, last, order = heapq.heappop(queue)
        if cost > best_g.get((mask, k, last), math.inf) + _TOLERANCE_S:
            dominance_pruned += 1
            continue
        if bound > upper - _TOLERANCE_S:
            bound_pruned += 1
            continue
        if expanded >= max_expansions:
            heapq.heappush(queue, (bound, cost, mask, k, last, order))
            break
        expanded += 1
        current = location(k, last)
        transitions = [("source", i, mask | (1 << i), k, i,
                        travel[current][i] + services[i]) for i in indices(all_mask ^ mask)]
        if k < cover_count:
            transitions.append(("cover", k, mask, k + 1, -1,
                                travel[current][n + k] + scan_fees[mask.bit_count()]))
        for kind, index, new_mask, new_k, new_last, increment in transitions:
            generated += 1
            new_cost = cost + increment
            key = (new_mask, new_k, new_last)
            if new_cost >= best_g.get(key, math.inf) - _TOLERANCE_S:
                dominance_pruned += 1
                continue
            best_g[key] = new_cost
            new_order = order + ((kind, index),)
            if new_mask == all_mask and new_k == cover_count:
                if new_cost < upper:
                    upper, best_order = new_cost, new_order
                continue
            new_bound = new_cost + estimate(new_mask, new_k, new_last)
            if new_bound > upper - _TOLERANCE_S:
                bound_pruned += 1
                continue
            heapq.heappush(queue, (new_bound, new_cost, new_mask, new_k, new_last, new_order))
    lower = min(upper, queue[0][0]) if queue else upper
    return ChainRouteResult(tuple(best_order), upper, lower, expanded, generated,
                            dominance_pruned, bound_pruned, not queue,
                            time.perf_counter() - began)
