"""Finite deterministic route refinement inside certified clearance disks."""

from dataclasses import dataclass
from itertools import permutations
import math

from simulator_client.state import Position
from .state_route import RouteTask, solve_state_route


@dataclass(frozen=True)
class CertifiedDisk:
    channel: int
    center: Position
    radius: float
    near: bool = False

    def __post_init__(self):
        if type(self.channel) is not int or not 1 <= self.channel <= 20:
            raise ValueError("certified channel must be an integer in 1..20")
        if not math.isfinite(self.radius) or not 0 <= self.radius <= 19.9:
            raise ValueError("certified disk radius must be in [0,19.9]")
        object.__setattr__(self, "center", Position.coerce(self.center))


@dataclass(frozen=True)
class ClearVisit:
    channel: int
    position: Position


@dataclass(frozen=True)
class BaselineTail:
    visits: tuple[ClearVisit, ...]
    cost_us: int
    expanded_after_each: tuple[int, ...]


def path_cost_us(start, visits):
    previous, result = Position.coerce(start), 0
    for visit in visits:
        # Match the simulator's rounding separately for every actual movement.
        result += round(previous.distance_to(visit.position) / 5 * 1_000_000) + 5_000_000
        previous = visit.position
    return result


def original_clear_point(disk, current):
    if disk.near:
        return disk.center
    # Intentionally match EfficientSearch._clear's arithmetic, even when the
    # current point is already inside the safe disk.
    distance = disk.center.distance_to(current)
    position = disk.center
    if distance > 0:
        fraction = min(1.0, disk.radius / distance)
        position = Position(disk.center.x + fraction * (current.x - disk.center.x),
                            disk.center.y + fraction * (current.y - disk.center.y))
    return position


def predict_baseline_tail(disks, start, *, max_expansions, max_total_expansions,
                          total_expansions, scan_source_s=0.0):
    """Reproduce v1's repeated frozen-point solve then actual safe-edge clear.

    The region, near point and source set are fixed; clearing does not change
    channel. This function is applicable only after discovery and while every
    unresolved source already has a certificate (there are no new measures).
    """
    remaining = sorted(disks, key=lambda d: d.channel)
    if len({d.channel for d in remaining}) != len(remaining):
        raise ValueError("duplicate source channel")
    current, expanded = Position.coerce(start), total_expansions
    visits, budgets = [], []
    while remaining:
        tasks = [RouteTask(d.center, True, 5.0) for d in remaining]
        budget = max(0, min(max_expansions, max_total_expansions - expanded))
        route = solve_state_route(tasks, current, scan_source_s=scan_source_s,
                                  max_expansions=budget)
        expanded += route.expanded
        disk = remaining.pop(route.order[0])
        current = original_clear_point(disk, current)
        visits.append(ClearVisit(disk.channel, current))
        budgets.append(expanded)
    visits = tuple(visits)
    return BaselineTail(visits, path_cost_us(start, visits), tuple(budgets))


def project_disk(point, disk):
    point = Position.coerce(point)
    distance = disk.center.distance_to(point)
    if distance <= disk.radius:
        return point
    if distance == 0 or disk.radius == 0:
        return disk.center
    fraction = disk.radius / distance
    return Position(disk.center.x + fraction * (point.x - disk.center.x),
                    disk.center.y + fraction * (point.y - disk.center.y))


def _local_cost(point, previous, following):
    return point.distance_to(previous) + (point.distance_to(following) if following is not None else 0.0)


def improve_certified_tail(disks, start, incumbent):
    """Enumerate at most 24 permutations, perform bounded feasible descent.

    No convergence or global optimum certificate is asserted. The exact
    microsecond cost of the supplied incumbent remains the acceptance test.
    """
    disks = tuple(sorted(disks, key=lambda d: d.channel))
    if not 2 <= len(disks) <= 4:
        raise ValueError("tail refinement requires 2..4 certified sources")
    if sorted(d.channel for d in disks) != sorted(v.channel for v in incumbent):
        raise ValueError("incumbent must visit every source exactly once")
    start = Position.coerce(start)
    best, best_cost = tuple(incumbent), path_cost_us(start, incumbent)
    for order in permutations(disks):
        points, previous = [], start
        for disk in order:
            previous = project_disk(previous, disk)
            points.append(previous)
        # Alternating coordinate sweeps; each accepted step decreases the
        # continuous full path, while final selection uses rounded actual cost.
        for sweep in range(32):
            changed = False
            indices = range(len(order)) if sweep % 2 == 0 else range(len(order) - 1, -1, -1)
            for index in indices:
                disk, point = order[index], points[index]
                previous = start if index == 0 else points[index - 1]
                following = points[index + 1] if index + 1 < len(points) else None
                if following is None:
                    trial = project_disk(previous, disk)
                    if _local_cost(trial, previous, None) < _local_cost(point, previous, None) - 1e-9:
                        points[index], changed = trial, True
                    continue
                gradient_x = gradient_y = 0.0
                for neighbor in (previous, following):
                    d = max(point.distance_to(neighbor), 1e-8)
                    gradient_x += (point.x - neighbor.x) / d
                    gradient_y += (point.y - neighbor.y) / d
                current_cost = _local_cost(point, previous, following)
                step = max(1.0, 8 * disk.radius)
                for _ in range(12):
                    trial = project_disk(Position(point.x - step * gradient_x,
                                                  point.y - step * gradient_y), disk)
                    if _local_cost(trial, previous, following) < current_cost - 1e-9:
                        points[index], changed = trial, True
                        break
                    step *= 0.5
            if not changed:
                break
        candidate = tuple(ClearVisit(d.channel, p) for d, p in zip(order, points))
        cost = path_cost_us(start, candidate)
        if cost < best_cost:
            best, best_cost = candidate, cost
    return best, best_cost
