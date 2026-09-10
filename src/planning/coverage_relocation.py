"""Move one future coverage site inside its convex full-coverage feasible set."""

from dataclasses import dataclass
import math

from .disk_cover import disk_cover_radius
from simulator_client.state import Position


class CoverageOracleBudget(Exception):
    pass


class CoverageOracle:
    def __init__(self, maximum=12000):
        self.maximum = maximum
        self.calls = 0
        self.cache = {}

    def __call__(self, sites):
        key = tuple(sorted(set((Position.coerce(p).x, Position.coerce(p).y) for p in sites)))
        if key not in self.cache:
            if self.calls >= self.maximum:
                raise CoverageOracleBudget()
            self.cache[key] = disk_cover_radius(key)
            self.calls += 1
        return self.cache[key]


@dataclass(frozen=True)
class Relocation:
    point: Position
    fraction: float
    coverage_radius_m: float


def feasible_ray_point(fixed, old, target, *, oracle=disk_cover_radius, steps=20, margin=1e-5):
    """Last certified point on one ray segment; not a 2D optimum certificate.

    With all other sites fixed, Omega is the part of the arena they do not
    cover. Feasible replacement sites form intersection_{x in Omega} B(x,r),
    hence a convex set. Starting from a feasible old point, ray feasibility is
    an interval. Every retained bisection point is nevertheless re-certified
    by the complete disk-cover oracle, including interior holes.
    """
    if type(steps) is not int or not 1 <= steps <= 30:
        raise ValueError("steps must be an integer in [1,30]")
    if not math.isfinite(margin) or not 0 < margin < 1:
        raise ValueError("margin must be positive and less than one metre")
    old, target = Position.coerce(old), Position.coerce(target)
    fixed = tuple(fixed)
    maximum = 1000. - margin
    old_radius = oracle(fixed + (old,))
    if old_radius > maximum:
        raise ValueError("Existing future cover plan is not certified")
    target_radius = oracle(fixed + (target,))
    if target_radius <= maximum:
        return Relocation(target, 1., target_radius)
    lo, hi, chosen, radius = 0., 1., old, old_radius
    for _ in range(steps):
        middle = (lo + hi) / 2
        point = Position(old.x + middle * (target.x-old.x), old.y + middle * (target.y-old.y))
        measured_radius = oracle(fixed + (point,))
        if measured_radius <= maximum:
            lo, chosen, radius = middle, point, measured_radius
        else:
            hi = middle
    return Relocation(chosen, lo, radius)


def project_to_segment(point, start, end):
    point, start, end = map(Position.coerce, (point, start, end))
    dx, dy = end.x-start.x, end.y-start.y
    length2 = dx*dx+dy*dy
    fraction = max(0., min(1., ((point.x-start.x)*dx+(point.y-start.y)*dy)/length2)) if length2 else 0.
    return Position(start.x+fraction*dx, start.y+fraction*dy)
