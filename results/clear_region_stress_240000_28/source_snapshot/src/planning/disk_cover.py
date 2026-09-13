"""Finite Voronoi witnesses certify coverage of the entire arena disk.

For sites P, max_{x in arena} min_p ||x-p|| occurs at a clipped Voronoi
vertex, a Voronoi-edge / arena-circle intersection, or the farthest point on
an arena-circle arc from its nearest site. We enumerate a superset of these
witnesses: all site circumcentres, all pair-bisector / boundary intersections,
and all radial antipodes. Interior holes are checked, not just the boundary.
"""

import math

from simulator_client.state import Position


def disk_cover_radius(stations, arena_radius=1800.0):
    """Return the maximum nearest-station distance over a centred disk.

    Finite floating-point geometry; callers keep a positive certification
    margin. Duplicate stations are discarded. No source truth is used.
    """
    points = sorted(set((Position.coerce(p).x, Position.coerce(p).y) for p in stations))
    if not points:
        return math.inf
    if not math.isfinite(arena_radius) or arena_radius <= 0:
        raise ValueError("arena_radius must be finite and positive")
    candidates = [(0.0, 0.0), (arena_radius, 0.0), (-arena_radius, 0.0),
                  (0.0, arena_radius), (0.0, -arena_radius)]
    radius2 = arena_radius * arena_radius
    for ax, ay in points:
        norm = math.hypot(ax, ay)
        if norm > 1e-12:
            candidates.append((-arena_radius * ax / norm, -arena_radius * ay / norm))
    for i, (ax, ay) in enumerate(points):
        for j in range(i + 1, len(points)):
            bx, by = points[j]
            ux, uy = bx - ax, by - ay
            norm2 = ux * ux + uy * uy
            if norm2 <= 1e-20:
                continue
            rhs = ux * (ax + bx) / 2 + uy * (ay + by) / 2
            footx, footy = rhs * ux / norm2, rhs * uy / norm2
            rest = radius2 - footx * footx - footy * footy
            if rest >= -1e-7:
                factor = math.sqrt(max(0.0, rest) / norm2)
                candidates.extend(((footx - uy * factor, footy + ux * factor),
                                   (footx + uy * factor, footy - ux * factor)))
            for k in range(j + 1, len(points)):
                cx, cy = points[k]
                vx, vy = cx - ax, cy - ay
                determinant = ux * vy - uy * vx
                if determinant == 0.0:
                    continue
                # Translate by a before solving to avoid cancellation in
                # differences of squared absolute coordinates.
                local_rhs = norm2 / 2
                local_rhs2 = (vx * vx + vy * vy) / 2
                x = ax + (local_rhs * vy - uy * local_rhs2) / determinant
                y = ay + (ux * local_rhs2 - local_rhs * vx) / determinant
                if x * x + y * y <= radius2 + 1e-6:
                    candidates.append((x, y))
    return math.sqrt(max(min((x - ax)**2 + (y - ay)**2 for ax, ay in points)
                         for x, y in candidates))


def certifies_disk_cover(stations, reception_radius=1000.0, *, margin=1e-5):
    return disk_cover_radius(stations) <= reception_radius - margin
