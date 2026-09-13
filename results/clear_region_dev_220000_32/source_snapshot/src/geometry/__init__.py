"""Planar geometry with explicit finite, empty, and unbounded set semantics.

Coordinates are metres. No dependency on simulator truth or scientific packages.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable

Point = tuple[float, float]
EPS = 1e-8


def point(value) -> Point:
    """Accept coordinate pairs and the client's Position class."""
    if hasattr(value, "x") and hasattr(value, "y"):
        value = (value.x, value.y)
    if len(value) != 2:
        raise ValueError("a point needs two coordinates")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(v) for v in result):
        raise ValueError("coordinates must be finite")
    return result


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def cross(a: Point, b: Point, c: Point) -> float:
    """Twice the signed area of triangle abc."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


@dataclass(frozen=True)
class HalfPlane:
    """Closed half-plane a*x+b*y <= c; normals are normalized internally."""

    a: float
    b: float
    c: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.a, self.b, self.c)):
            raise ValueError("half-plane coefficients must be finite")
        scale = math.hypot(self.a, self.b)
        if scale == 0:
            raise ValueError("half-plane normal cannot be zero")
        object.__setattr__(self, "a", self.a / scale)
        object.__setattr__(self, "b", self.b / scale)
        object.__setattr__(self, "c", self.c / scale)

    def residual(self, p: Point) -> float:
        return self.a * p[0] + self.b * p[1] - self.c

    def contains(self, p: Point, tolerance: float = EPS) -> bool:
        return self.residual(point(p)) <= tolerance


def _deduplicate(vertices: Iterable[Point]) -> tuple[Point, ...]:
    result = []
    for p in vertices:
        if not result or distance(result[-1], p) > EPS:
            result.append(p)
    if len(result) > 1 and distance(result[0], result[-1]) <= EPS:
        result.pop()
    return tuple(result)


def clip_polygon(vertices: Iterable[Point], halfplane: HalfPlane) -> tuple[Point, ...]:
    """Clip a convex polygon, segment, or point by one closed half-plane."""
    vertices = tuple(point(p) for p in vertices)
    if not vertices:
        return ()
    if len(vertices) == 1:
        return vertices if halfplane.contains(vertices[0]) else ()
    output = []
    for start, end in zip(vertices, vertices[1:] + vertices[:1]):
        rs, re = halfplane.residual(start), halfplane.residual(end)
        sin, ein = rs <= EPS, re <= EPS
        if sin != ein:
            t = max(0.0, min(1.0, rs / (rs - re)))
            output.append((start[0] + t * (end[0] - start[0]),
                           start[1] + t * (end[1] - start[1])))
        if ein:
            output.append(end)
    return _deduplicate(output)


def convex_hull(points: Iterable[Point]) -> tuple[Point, ...]:
    """Monotone-chain hull; collinear sets return their endpoints."""
    points = sorted(set(point(p) for p in points))
    if len(points) <= 1:
        return tuple(points)
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return tuple(lower[:-1] + upper[:-1])


def polygon_area(vertices: Iterable[Point]) -> float:
    vertices = tuple(vertices)
    if len(vertices) < 3:
        return 0.0
    # Translate to reduce cancellation when the region is small but far away.
    origin = vertices[0]
    return abs(sum(cross(origin, vertices[i], vertices[i + 1])
                   for i in range(1, len(vertices) - 1))) / 2.0


def polygon_diameter(vertices: Iterable[Point]) -> float:
    """Maximum vertex-pair distance, using hull + rotating calipers."""
    hull = convex_hull(vertices)
    n = len(hull)
    if n == 0:
        raise ValueError("empty sets have no localization diameter")
    if n <= 2:
        return distance(hull[0], hull[-1])
    result, j = 0.0, 1
    for i in range(n):
        ni = (i + 1) % n
        while cross(hull[i], hull[ni], hull[(j + 1) % n]) > cross(hull[i], hull[ni], hull[j]) + 1e-12:
            j = (j + 1) % n
        result = max(result, distance(hull[i], hull[j]), distance(hull[ni], hull[j]))
        nj = (j + 1) % n
        if abs(cross(hull[i], hull[ni], hull[nj]) - cross(hull[i], hull[ni], hull[j])) <= 1e-12:
            result = max(result, distance(hull[i], hull[nj]), distance(hull[ni], hull[nj]))
    return result


@dataclass(frozen=True)
class Circle:
    center: Point
    radius: float

    def __post_init__(self):
        object.__setattr__(self, "center", point(self.center))
        if not math.isfinite(self.radius) or self.radius < 0:
            raise ValueError("circle radius must be finite and nonnegative")

    def contains(self, p: Point, tolerance: float = EPS) -> bool:
        return distance(self.center, point(p)) <= self.radius + tolerance


def _diameter_circle(a: Point, b: Point) -> Circle:
    center = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return Circle(center, max(distance(center, a), distance(center, b)))


def _circumcircle(a: Point, b: Point, c: Point) -> Circle | None:
    bx, by, cx, cy = b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1]
    det = 2 * (bx * cy - by * cx)
    if abs(det) <= 1e-14 * max(1.0, math.hypot(bx, by) * math.hypot(cx, cy)):
        return None
    b2, c2 = bx * bx + by * by, cx * cx + cy * cy
    center = (a[0] + (cy * b2 - by * c2) / det,
              a[1] + (bx * c2 - cx * b2) / det)
    return Circle(center, max(distance(center, p) for p in (a, b, c)))


def _circle_two_boundary(points: list[Point], p: Point, q: Point) -> Circle:
    """Smallest circle containing points with p,q on its boundary."""
    diameter = _diameter_circle(p, q)
    left, right = None, None
    for r in points:
        if diameter.contains(r):
            continue
        side = cross(p, q, r)
        circle = _circumcircle(p, q, r)
        if circle is None:
            continue
        center_side = cross(p, q, circle.center)
        if side > 0 and (left is None or center_side > cross(p, q, left.center)):
            left = circle
        elif side < 0 and (right is None or center_side < cross(p, q, right.center)):
            right = circle
    if left is None and right is None:
        # Covers the collinear case, including endpoints beyond p or q.
        return max((_diameter_circle(a, b) for a in (p, q) for b in points + [p, q]),
                   key=lambda item: item.radius)
    if left is None:
        return right
    if right is None:
        return left
    return min((left, right), key=lambda item: item.radius)


def minimum_enclosing_circle(points: Iterable[Point]) -> Circle:
    """Hull preprocessing O(n log n), then expected O(h) incremental MEC.

    The final radius is expanded to the actual maximum vertex distance to make
    the returned circle an enclosure even in the presence of rounding error.
    """
    points = list(convex_hull(points))
    if not points:
        raise ValueError("empty sets do not have a usable enclosing circle")
    random.Random(2026).shuffle(points)
    circle = None
    for i, p in enumerate(points):
        if circle is not None and circle.contains(p):
            continue
        circle = Circle(p, 0.0)
        for j, q in enumerate(points[:i]):
            if circle.contains(q):
                continue
            if circle.radius == 0.0:
                circle = _diameter_circle(p, q)
            else:
                circle = _circle_two_boundary(points[:j + 1], p, q)
    return Circle(circle.center, max(distance(circle.center, p) for p in points))


def disk_halfplanes(center: Point, radius: float, sides: int = 128, *, outer: bool = True) -> tuple[HalfPlane, ...]:
    """Tangent (outer) or chord (inner) polygon constraints for a disk."""
    center = point(center)
    if (not math.isfinite(radius) or radius <= 0 or isinstance(sides, bool)
            or not isinstance(sides, int) or sides < 8):
        raise ValueError("radius must be positive and sides must be an integer >= 8")
    apothem = radius if outer else radius * math.cos(math.pi / sides)
    return tuple(HalfPlane(math.cos(2 * math.pi * k / sides), math.sin(2 * math.pi * k / sides),
                           math.cos(2 * math.pi * k / sides) * center[0]
                           + math.sin(2 * math.pi * k / sides) * center[1] + apothem)
                 for k in range(sides))


def disk_polygon(center: Point, radius: float, sides: int = 128, *, outer: bool = True) -> tuple[Point, ...]:
    disk_halfplanes(center, radius, sides, outer=outer)  # validation
    center = point(center)
    circumradius = radius / math.cos(math.pi / sides) if outer else radius
    return tuple((center[0] + circumradius * math.cos(2 * math.pi * (k + 0.5) / sides),
                  center[1] + circumradius * math.sin(2 * math.pi * (k + 0.5) / sides))
                 for k in range(sides))


def bearing_halfplanes(position: Point, bearing_deg: float, error_deg: float = 1.005) -> tuple[HalfPlane, HalfPlane]:
    """Forward angular wedge, including wraparound through 0 degrees."""
    position = point(position)
    if not math.isfinite(bearing_deg) or not 0 < error_deg < 90:
        raise ValueError("bearing must be finite and error must be in (0,90)")
    lo, hi = math.radians(bearing_deg - error_deg), math.radians(bearing_deg + error_deg)
    a, b = math.sin(lo), -math.cos(lo)
    c, d = -math.sin(hi), math.cos(hi)
    return (HalfPlane(a, b, a * position[0] + b * position[1]),
            HalfPlane(c, d, c * position[0] + d * position[1]))


@dataclass(frozen=True)
class HalfPlaneIntersection:
    status: str
    vertices: tuple[Point, ...]
    feasible_point: Point | None
    dimension: int | None

    @property
    def diameter(self) -> float | None:
        if self.status == "empty":
            return None
        if self.status == "unbounded":
            return math.inf
        return polygon_diameter(self.vertices)


def halfplane_intersection(halfplanes: Iterable[HalfPlane]) -> HalfPlaneIntersection:
    """Analyze a general intersection without silently imposing a bounding box.

    Pairwise boundary intersections plus boundary projections find a feasible
    point (the closest point to the origin has zero, one, or two active faces).
    Feasible recession directions classify unbounded sets. Complexity O(m^3),
    suitable for the small number of bearing constraints in Question 1; use
    clip_polygon for repeated bounded localization updates.
    """
    hps = tuple(halfplanes)
    candidates = [(0.0, 0.0)] + [(hp.a * hp.c, hp.b * hp.c) for hp in hps]
    intersections = []
    for i, first in enumerate(hps):
        for second in hps[i + 1:]:
            det = first.a * second.b - first.b * second.a
            if abs(det) < 1e-15:
                continue
            intersections.append(((first.c * second.b - first.b * second.c) / det,
                                  (first.a * second.c - first.c * second.a) / det))
    feasible = [p for p in candidates + intersections if all(hp.contains(p) for hp in hps)]
    if not feasible:
        return HalfPlaneIntersection("empty", (), None, None)
    directions = [(1.0, 0.0)] if not hps else [(sign * hp.b, -sign * hp.a) for hp in hps for sign in (-1, 1)]
    unbounded = any(all(hp.a * d[0] + hp.b * d[1] <= 1e-14 for hp in hps) for d in directions)
    vertices = convex_hull(p for p in intersections if all(hp.contains(p) for hp in hps))
    if unbounded:
        return HalfPlaneIntersection("unbounded", vertices, feasible[0], None)
    if not vertices:
        vertices = (feasible[0],)
    # The hull already reduces exactly collinear vertices to their endpoints.
    # An absolute area threshold would misclassify small 2-D triangles as lines.
    dimension = min(2, len(vertices) - 1)
    return HalfPlaneIntersection("bounded", vertices, feasible[0], dimension)


__all__ = ["Point", "HalfPlane", "HalfPlaneIntersection", "Circle", "point", "distance", "cross",
           "clip_polygon", "convex_hull", "polygon_area", "polygon_diameter", "minimum_enclosing_circle",
           "disk_polygon", "disk_halfplanes", "bearing_halfplanes", "halfplane_intersection"]
