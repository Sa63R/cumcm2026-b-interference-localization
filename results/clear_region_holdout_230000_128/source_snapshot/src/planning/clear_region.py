"""Closest safe clearance point for a convex outer source region.

The safe set is the intersection of radius-r disks centred on every polygon
vertex. A closest point is either the query itself, a one-circle radial
projection, or an intersection of two active circle boundaries. This is a
local movement optimum, not an optimum of the remaining search trajectory.
"""

import math

from simulator_client.state import Position


def nearest_clear_point(vertices, current, *, radius=19.9, fallback=None):
    """Return a certified candidate, or None if none can be verified.

    Candidate circles have a tiny inward numerical margin. All returned
    candidates, including the caller's old feasible point, are checked against
    the full requested radius with no positive feasibility tolerance.
    """
    if isinstance(radius, bool) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    current = Position.coerce(current)
    centers = tuple(dict.fromkeys(Position.coerce(v) for v in vertices))
    points = (*centers, current)
    if any(not math.isfinite(v) for p in points for v in (p.x, p.y)):
        raise ValueError("coordinates must be finite")
    if not centers:
        return None

    def feasible(p):
        return all(p.distance_to(c) <= radius for c in centers)

    if feasible(current):
        return current
    candidates = []
    if fallback is not None:
        old = Position.coerce(fallback)
        if math.isfinite(old.x) and math.isfinite(old.y) and feasible(old):
            candidates.append(old)
    # Keep original-radius intersections too, so exactly tangent feasible
    # sets are not lost. Strict final checks reject outward rounding.
    for r in (radius - min(1e-7, radius * 1e-9), radius):
        for center in centers:
            d = center.distance_to(current)
            if d:
                scale = r / d
                p = Position(center.x + scale * (current.x - center.x),
                             center.y + scale * (current.y - center.y))
                if feasible(p):
                    candidates.append(p)
        for i, a in enumerate(centers):
            for b in centers[i + 1:]:
                dx, dy = b.x - a.x, b.y - a.y
                d = math.hypot(dx, dy)
                if not d or d > 2 * r:
                    continue
                height = math.sqrt(max(0.0, (r - d / 2) * (r + d / 2)))
                mid_x, mid_y = a.x + dx / 2, a.y + dy / 2
                for sign in (-1.0, 1.0):
                    p = Position(mid_x - sign * height * dy / d,
                                 mid_y + sign * height * dx / d)
                    if feasible(p):
                        candidates.append(p)
    return min(candidates, key=lambda p: (current.distance_to(p), p.x, p.y), default=None)


def nearest_near_clear_point(near_point, current, *, radius=19.9):
    """Project onto B(near_point, radius-5), using only the public near bound."""
    if radius <= 5:
        raise ValueError("near clearance radius must exceed five metres")
    return nearest_clear_point([near_point], current, radius=radius - 5,
                               fallback=near_point)
