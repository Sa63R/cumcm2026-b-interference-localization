"""Exact open-route ordering for the few remaining omnidirectional stations."""

import math

from simulator_client.state import Position


def exact_open_route(points, start=(0.0, 0.0)):
    """Visit at most six supplied points along a shortest Euclidean open path.

    The external start is fixed and need not be one of the points. No return
    leg is added. Subset dynamic programming enumerates every visitation order
    in O(n**2 * 2**n) time and O(n * 2**n) space. "Exact" refers to the
    combinatorial optimization over floating-point Euclidean edge lengths.

    Coordinates are sorted before optimization, so ties are deterministic and
    independent of the caller's input order. Duplicate positions remain
    separate visits. The six-point cap keeps this routine bounded for online
    Q3 reordering; this is not intended for Q4's larger station sets.
    """
    start = Position.coerce(start)
    locations = []
    for item in points:
        locations.append(Position.coerce(item))
        if len(locations) > 6:
            raise ValueError("exact_open_route accepts at most six remaining points")
    locations.sort(key=lambda p: (p.x, p.y))
    n = len(locations)
    if n == 0:
        return ()
    edges = [[a.distance_to(b) for b in locations] for a in locations]
    costs = [[math.inf] * n for _ in range(1 << n)]
    parents = [[-1] * n for _ in range(1 << n)]
    for last, position in enumerate(locations):
        costs[1 << last][last] = start.distance_to(position)
    for mask in range(1, 1 << n):
        for last in range(n):
            if not mask & (1 << last):
                continue
            previous_mask = mask ^ (1 << last)
            if not previous_mask:
                continue
            best, previous = min(
                (costs[previous_mask][other] + edges[other][last], other)
                for other in range(n) if previous_mask & (1 << other)
            )
            costs[mask][last], parents[mask][last] = best, previous
    mask = (1 << n) - 1
    last = min(range(n), key=lambda index: (costs[mask][index], index))
    route = []
    while last >= 0:
        route.append(locations[last])
        previous = parents[mask][last]
        mask ^= 1 << last
        last = previous
    return tuple(reversed(route))
