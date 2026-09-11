"""Anytime routing through one frozen entry/service/exit mode per task group.

The state is (visited groups, last group, last mode). Different modes may have
different exits, so transitions are directed. Only histories with the same full
state have identical continuations and may dominate each other.

For a remaining group pair, minimize travel over both directions and every
mode pair. An MST of this undirected relaxation, a cheapest entry from the
current exit, and each remaining group's cheapest service lower-bound every
continuation. The minima need not share compatible modes: relaxing that
compatibility makes the bound weaker, never larger.

All guarantees concern this finite frozen proxy, within 1e-9 s comparison
tolerance. They do not imply optimality of a feedback policy or of Q3 itself.
"""

from dataclasses import dataclass
from functools import lru_cache
import heapq
import math
import time

from simulator_client.state import Position


_TOLERANCE_S = 1e-9


def _position(value):
    try:
        point = Position.coerce(value)
    except (TypeError, OverflowError) as exc:
        raise ValueError("mode coordinates must be finite numbers") from exc
    if any(isinstance(v, bool) for v in (point.x, point.y)):
        raise ValueError("mode coordinates must be finite numbers, not booleans")
    return point


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class RouteMode:
    entry: Position
    exit: Position
    service_s: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "entry", _position(self.entry))
        object.__setattr__(self, "exit", _position(self.exit))
        if not _finite_number(self.service_s) or self.service_s < 0:
            raise ValueError("service_s must be finite and nonnegative")
        object.__setattr__(self, "service_s", float(self.service_s))


@dataclass(frozen=True)
class ModeRouteResult:
    order: tuple[tuple[int, int], ...]
    cost_s: float
    lower_bound_s: float
    expanded: int
    generated: int
    dominance_pruned: int
    bound_pruned: int
    exact: bool
    runtime_s: float


def solve_mode_route(groups, start=(0.0, 0.0), *, max_expansions=100,
                     speed_mps=5.0, initial_order=None):
    """Return one mode per group and a lower bound for this frozen model.

    ``initial_order`` is an optional complete permutation of group indices.
    Dynamic programming chooses its best compatible modes, preserving this
    feasible incumbent even with a zero expansion budget. Two deterministic
    nearest-neighbor seeds and at most n improving group-order reversals also
    build incumbents. Every reversed order is rescored by mode DP; an ordinary
    symmetric 2-opt edge-difference formula would be invalid here.

    ``max_expansions`` bounds A* state expansions, not incumbent construction.
    ``exact`` means the search frontier is closed within comparison tolerance,
    not a certificate about real-arithmetic distances or unfrozen predictions.
    """
    began = time.perf_counter()
    if (isinstance(max_expansions, bool) or not isinstance(max_expansions, int)
            or max_expansions < 0):
        raise ValueError("max_expansions must be a nonnegative integer")
    if not _finite_number(speed_mps) or speed_mps <= 0:
        raise ValueError("speed_mps must be finite and positive")
    start = _position(start)
    try:
        groups = tuple(tuple(group) for group in groups)
    except TypeError as exc:
        raise ValueError("groups must be a sequence of nonempty mode sequences") from exc
    n = len(groups)
    if n > 22:
        raise ValueError("mode route supports at most 22 groups")
    if any(not 1 <= len(group) <= 4 for group in groups):
        raise ValueError("every group must contain 1 to 4 modes")
    if any(not isinstance(mode, RouteMode) for group in groups for mode in group):
        raise ValueError("every mode must be a RouteMode")
    if initial_order is not None:
        try:
            initial_order = tuple(initial_order)
        except TypeError as exc:
            raise ValueError("initial_order must be a permutation of group indices") from exc
        if (len(initial_order) != n
                or any(type(i) is not int for i in initial_order)
                or set(initial_order) != set(range(n))):
            raise ValueError("initial_order must be a permutation of group indices")
    if not n:
        return ModeRouteResult((), 0.0, 0.0, 0, 0, 0, 0, True,
                               time.perf_counter() - began)

    modes, labels, members = [], [], []
    for group_index, group in enumerate(groups):
        members.append(tuple(range(len(modes), len(modes) + len(group))))
        modes.extend(group)
        labels.extend((group_index, k) for k in range(len(group)))
    services = [mode.service_s for mode in modes]
    initial = [start.distance_to(mode.entry) / speed_mps for mode in modes]
    travel = [[a.exit.distance_to(b.entry) / speed_mps for b in modes] for a in modes]
    # Reject overflowing cost scales rather than emitting a spurious inf/NaN
    # certificate. Physical coordinates remain governed by Position's limits.
    maximum_travel = max(initial + [max(row) for row in travel])
    if not math.isfinite(n * maximum_travel + sum(max(m.service_s for m in g) for g in groups)):
        raise ValueError("mode route cost scale must remain finite")
    increments = [[distance + services[j] for j, distance in enumerate(row)] for row in travel]
    initial_increments = [distance + service for distance, service in zip(initial, services)]
    group_min_service = [min(services[v] for v in group) for group in members]
    relaxed = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            relaxed[i][j] = relaxed[j][i] = min(
                min(travel[a][b], travel[b][a]) for a in members[i] for b in members[j])
    entry_minima = [[min(row[v] for v in group) for group in members]
                    for row in travel + [initial]]
    all_mask = (1 << n) - 1

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
        return sum(group_min_service[i] for i in indices(mask))

    @lru_cache(maxsize=None)
    def mst(mask):
        vertices = indices(mask)
        if len(vertices) < 2:
            return 0.0
        first, *rest = vertices
        cheapest = {i: relaxed[first][i] for i in rest}
        value = 0.0
        while cheapest:
            vertex = min(cheapest, key=lambda i: (cheapest[i], i))
            value += cheapest.pop(vertex)
            for other in cheapest:
                cheapest[other] = min(cheapest[other], relaxed[vertex][other])
        return value

    def estimate(mask, last_group, last_mode):
        remaining = all_mask ^ mask
        if not remaining:
            return 0.0
        row = entry_minima[-1] if last_group < 0 else entry_minima[members[last_group][last_mode]]
        return (min(row[i] for i in indices(remaining)) + mst(remaining)
                + service_bound(remaining))

    @lru_cache(maxsize=4096)
    def order_modes(order):
        # Label includes the mode sequence to make equal-cost ties deterministic.
        first = order[0]
        values = {v: (initial_increments[v], (labels[v],)) for v in members[first]}
        for group in order[1:]:
            next_values = {}
            for v in members[group]:
                cost, path = min((cost + increments[u][v], path)
                                 for u, (cost, path) in values.items())
                next_values[v] = cost, path + (labels[v],)
            values = next_values
        return min(values.values())

    seeds = []
    if initial_order is not None:
        seeds.append(initial_order)
    for include_service in (False, True):
        order, mask, last = [], 0, -1
        while mask != all_mask:
            row = initial if last < 0 else travel[last]
            chosen = min((v for i in indices(all_mask ^ mask) for v in members[i]),
                         key=lambda v: (row[v] + (services[v] if include_service else 0.0), labels[v]))
            group, _ = labels[chosen]
            order.append(group)
            mask |= 1 << group
            last = chosen
        seeds.append(tuple(order))
    upper, best_order = min(order_modes(order) for order in seeds)
    group_order = tuple(i for i, _ in best_order)
    for _ in range(n):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                alternate = group_order[:i] + group_order[i:j + 1][::-1] + group_order[j + 1:]
                cost, path = order_modes(alternate)
                if cost < upper - _TOLERANCE_S:
                    upper, best_order, group_order = cost, path, alternate
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    lower = estimate(0, -1, -1)
    queue = [(lower, 0.0, 0, -1, -1, ())]
    best_g = {(0, -1, -1): 0.0}
    expanded = generated = dominance_pruned = bound_pruned = 0
    while queue:
        bound, cost, mask, last_group, last_mode, order = heapq.heappop(queue)
        if cost > best_g.get((mask, last_group, last_mode), math.inf) + _TOLERANCE_S:
            dominance_pruned += 1
            continue
        if bound > upper - _TOLERANCE_S:
            bound_pruned += 1
            continue
        if expanded >= max_expansions:
            heapq.heappush(queue, (bound, cost, mask, last_group, last_mode, order))
            break
        expanded += 1
        row = (initial_increments if last_group < 0
               else increments[members[last_group][last_mode]])
        for group in indices(all_mask ^ mask):
            for mode_index, vertex in enumerate(members[group]):
                generated += 1
                new_mask = mask | (1 << group)
                new_cost = cost + row[vertex]
                key = (new_mask, group, mode_index)
                if new_cost >= best_g.get(key, math.inf) - _TOLERANCE_S:
                    dominance_pruned += 1
                    continue
                best_g[key] = new_cost
                new_order = order + ((group, mode_index),)
                if new_mask == all_mask:
                    if new_cost < upper:
                        upper, best_order = new_cost, new_order
                    continue
                new_bound = new_cost + estimate(new_mask, group, mode_index)
                if new_bound > upper - _TOLERANCE_S:
                    bound_pruned += 1
                    continue
                heapq.heappush(queue, (new_bound, new_cost, new_mask, group, mode_index, new_order))
    # Stale frontier entries can only weaken the reported bound. Cap at the
    # incumbent because its already generated descendants need not be queued.
    lower = min(upper, queue[0][0]) if queue else upper
    return ModeRouteResult(tuple(best_order), upper, lower, expanded, generated,
                           dominance_pruned, bound_pruned, not queue,
                           time.perf_counter() - began)
