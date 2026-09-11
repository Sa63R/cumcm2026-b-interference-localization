"""Public Q3 geometry: a query farther than an actual silent observation.

This module does not establish evidence provenance. Its caller must supply a
previous actual no_signal on the same known, stationary, uncleared Q3 source.
The existing conservative outer region supplies the source-containment premise.
"""

from __future__ import annotations

import math

from simulator_client.state import Position


def _out(value):
    return math.nextafter(value, -math.inf), math.nextafter(value, math.inf)


def _add(a, b):
    return _out(a[0] + b[0])[0], _out(a[1] + b[1])[1]


def _sub(a, b):
    return _out(a[0] - b[1])[0], _out(a[1] - b[0])[1]


def _mul(a, b):
    values = [x * y for x in a for y in b]
    return _out(min(values))[0], _out(max(values))[1]


def relative_silence_certificate(region, query, actual_negative, *, margin_m=1e-5):
    """Return a sufficient distance certificate, or None at uncertain borders.

    For d=n-q and m=(n+q)/2, |s-q|^2-|s-n|^2=2*d.(s-m).
    Its minimum over a convex polygon is attained at a vertex. Every scalar
    addition/subtraction/product is enclosed using nextafter in both directions;
    the accepted lower endpoint must also exceed margin_m*upper_bound(|d|).
    This avoids cancellation from subtracting two large squared distances.

    Enclosures concern binary64 input vertices, not uncertainty in how a parent
    region was constructed. We inherit that parent's safe-outer-region contract.
    Empty, identical-point and nonfinite/overflowing calculations fail closed.
    """
    if (isinstance(margin_m, bool) or not isinstance(margin_m, (int, float))
            or not math.isfinite(margin_m) or margin_m <= 0):
        raise ValueError("margin_m must be positive and finite")
    q, n = Position.coerce(query), Position.coerce(actual_negative)
    if not region.vertices or not region.observations or q == n:
        return None
    points = tuple(tuple(float(x) for x in v) for v in region.vertices)
    if any(len(v) != 2 or not all(math.isfinite(x) for x in v) for v in points):
        return None
    qv, nv = (q.x, q.y), (n.x, n.y)
    delta = [_sub((nv[i], nv[i]), (qv[i], qv[i])) for i in range(2)]
    midpoint = [_mul(_add((nv[i], nv[i]), (qv[i], qv[i])), (.5, .5))
                for i in range(2)]
    norm2 = (0., 0.)
    for part in delta:
        bound = max(abs(part[0]), abs(part[1]))
        norm2 = _add(norm2, _mul((bound, bound), (bound, bound)))
    if not math.isfinite(norm2[1]) or norm2[1] <= 0:
        return None
    norm_upper = math.nextafter(math.sqrt(norm2[1]), math.inf)
    required = _mul((margin_m, margin_m), (norm_upper, norm_upper))[1]
    lower = math.inf
    for vertex in points:
        terms = [_mul(delta[i], _sub((vertex[i], vertex[i]), midpoint[i])) for i in range(2)]
        enclosed = _add(terms[0], terms[1])
        if not all(math.isfinite(x) for x in enclosed):
            return None
        lower = min(lower, enclosed[0])
    if not math.isfinite(required) or lower <= required:
        return None
    return {
        "method": "relative_actual_negative",
        "actual_negative_position": [n.x, n.y],
        "outer_region_vertex_count": len(points),
        "affine_lower_bound_m2": lower,
        "required_affine_margin_m2": required,
        "query_negative_distance_upper_m": norm_upper,
        "signed_bisector_distance_lower_m": math.nextafter(lower / norm_upper, -math.inf),
        "margin_m": margin_m,
        "rounding_contract": "outward-nextafter interval operations on binary64 input coordinates",
    }


__all__ = ["relative_silence_certificate"]
