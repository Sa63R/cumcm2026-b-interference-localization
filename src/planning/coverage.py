"""Finite spatial covers for discovery and optical clearance."""

import math

from simulator_client.state import Position


def _triangle_intersects_disk(vertices, radius):
    """Exact convex/edge test, with tolerance solely for contact roundoff."""
    edges = tuple(zip(vertices, vertices[1:] + vertices[:1]))
    crosses = [a[0] * b[1] - a[1] * b[0] for a, b in edges]
    if all(value >= -1e-9 for value in crosses) or all(value <= 1e-9 for value in crosses):
        return True
    for a, b in edges:
        dx, dy = b[0] - a[0], b[1] - a[1]
        fraction = max(0.0, min(1.0, -(a[0] * dx + a[1] * dy) / (dx * dx + dy * dy)))
        if math.hypot(a[0] + fraction * dx, a[1] + fraction * dy) <= radius + 1e-9:
            return True
    return False


def _triangular_cover(spacing=990.0, radius=1800.0):
    """All vertices of equilateral triangles intersecting the source disk."""
    height = spacing * math.sqrt(3) / 2
    extent = math.ceil(2 * radius / (math.sqrt(3) * spacing)) + 1
    indices = set()

    def coordinates(index):
        i, j = index
        return (spacing * (i + j / 2), height * j)

    for i in range(-extent, extent):
        for j in range(-extent, extent):
            triangles = [((i, j), (i + 1, j), (i, j + 1)),
                         ((i + 1, j + 1), (i, j + 1), (i + 1, j))]
            for triangle in triangles:
                vertices = tuple(coordinates(index) for index in triangle)
                if _triangle_intersects_disk(vertices, radius):
                    indices.update(triangle)
    return [Position(*coordinates(index)) for index in sorted(indices)]


def nearest_order(points, start=(0.0, 0.0)):
    """Deterministic nearest-neighbour ordering; no target data are consulted."""
    remaining = [Position.coerce(point) for point in points]
    current = Position.coerce(start)
    ordered = []
    while remaining:
        index = min(range(len(remaining)), key=lambda i: (
            current.distance_to(remaining[i]), remaining[i].x, remaining[i].y,
        ))
        current = remaining.pop(index)
        ordered.append(current)
    return tuple(ordered)


def improve_open_route(points, start=(0.0, 0.0)):
    """Deterministic first-improvement 2-opt with a fixed external start.

    The path does not return to the start. Reversals may include its last point,
    which removes unnecessary long terminal jumps from nearest-neighbour tours.
    Every accepted reversal strictly reduces geometric path length; the finite
    set of permutations therefore guarantees termination.
    """
    route = [Position.coerce(point) for point in points]
    start = Position.coerce(start)
    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 1):
            previous = start if i == 0 else route[i - 1]
            for j in range(i + 1, len(route)):
                before = previous.distance_to(route[i])
                after = previous.distance_to(route[j])
                if j + 1 < len(route):
                    before += route[j].distance_to(route[j + 1])
                    after += route[i].distance_to(route[j + 1])
                if after < before - 1e-8:
                    route[i:j + 1] = reversed(route[i:j + 1])
                    improved = True
                    break
            if improved:
                break
    return tuple(route)


def coverage_points(problem=3, *, variant="adaptive"):
    """Return a cover valid for the 1800 m source disk and R >= 1000 m.

    Q3: origin plus six points on a 1500 m ring.
    Q4: every vertex of every 650 m square intersecting the source disk.
    Q4 triangular: vertices of intersecting 990 m equilateral triangles.
    Both Q4 covers necessarily include points outside the disk.
    """
    if variant == "triangular" and problem not in (4, "q4"):
        raise ValueError("triangular coverage is only defined for problem 4")
    if problem in (3, "q3"):
        points = [Position(0.0, 0.0)] + [
            Position(1500.0 * math.cos(i * math.pi / 3),
                     1500.0 * math.sin(i * math.pi / 3))
            for i in range(6)
        ]
    elif problem in (4, "q4"):
        if variant == "triangular":
            return improve_open_route(nearest_order(_triangular_cover()))
        spacing = 650.0
        radius = 1800.0
        indices = set()
        lower = math.floor(-radius / spacing)
        upper = math.ceil(radius / spacing)
        for i in range(lower, upper):
            for j in range(lower, upper):
                x0, x1 = i * spacing, (i + 1) * spacing
                y0, y1 = j * spacing, (j + 1) * spacing
                nearest_x = max(x0, min(0.0, x1))
                nearest_y = max(y0, min(0.0, y1))
                if math.hypot(nearest_x, nearest_y) <= radius:
                    indices.update(((i, j), (i + 1, j),
                                    (i, j + 1), (i + 1, j + 1)))
        points = [Position(i * spacing, j * spacing) for i, j in sorted(
            indices, key=lambda p: (p[1], p[0] if p[1] % 2 == 0 else -p[0]))]
    else:
        raise ValueError("problem must be 3 or 4")
    if variant in {"adaptive", "improved", "deferred"}:
        ordered = nearest_order(points)
        return improve_open_route(ordered) if variant == "deferred" else ordered
    if variant != "baseline":
        raise ValueError("variant must be baseline or adaptive")
    return tuple(points)


def _clip_horizontal(vertices, bound, keep_above):
    """Clip a convex polygon/segment/point without discarding degeneracies."""
    if not vertices:
        return []
    result = []
    previous = vertices[-1]
    previous_value = previous[1] - bound
    previous_inside = previous_value >= 0 if keep_above else previous_value <= 0
    for current in vertices:
        value = current[1] - bound
        inside = value >= 0 if keep_above else value <= 0
        if inside != previous_inside:
            t = previous_value / (previous_value - value)
            result.append((previous[0] + t * (current[0] - previous[0]), bound))
        if inside:
            result.append(current)
        previous, previous_value, previous_inside = current, value, inside
    return result


def clearance_grid(vertices, *, bearing_deg=0.0, spacing=28.0, start=(0.0, 0.0)):
    """Cover a candidate polygon by radius-20 m clearance disks.

    Rotate the grid to the first observed bearing, clip one row at a time, and
    retain every square that can intersect the polygon. Each retained cell has
    half-diagonal spacing/sqrt(2) < 20, including its boundary. This works when
    no further radio measurement can be obtained from a directional source.
    """
    if not math.isfinite(spacing) or not 0 < spacing < 20 * math.sqrt(2):
        raise ValueError("spacing must be positive and strictly less than 20*sqrt(2)")
    if not math.isfinite(bearing_deg):
        raise ValueError("bearing_deg must be finite")
    vertices = [Position.coerce(vertex) for vertex in vertices]
    if not vertices:
        return ()
    angle = math.radians(bearing_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotated = [(p.x * cosine + p.y * sine, -p.x * sine + p.y * cosine)
               for p in vertices]
    first = math.floor(min(p[1] for p in rotated) / spacing)
    last = math.floor(max(p[1] for p in rotated) / spacing)
    centers = []
    for row in range(first, last + 1):
        strip = _clip_horizontal(rotated, row * spacing, True)
        strip = _clip_horizontal(strip, (row + 1) * spacing, False)
        if not strip:
            continue
        left = math.floor(min(p[0] for p in strip) / spacing)
        right = math.floor(max(p[0] for p in strip) / spacing)
        for column in range(left, right + 1):
            x, y = (column + 0.5) * spacing, (row + 0.5) * spacing
            centers.append(Position(x * cosine - y * sine,
                                    x * sine + y * cosine))
    return nearest_order(centers, start)
