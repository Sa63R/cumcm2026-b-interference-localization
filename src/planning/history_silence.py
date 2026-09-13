"""Additional Q3 silence certificates from a channel's negative history.

This helper receives observation geometry only. The caller must restrict it
to an already detected, uncleared, stationary omnidirectional channel and
must keep deductions separate from physical measurements/coverage evidence.
"""

from fractions import Fraction
import math

from simulator_client.state import Position

from .silence_certificate import certify_silence


def _downward_float(value):
    """Round an exact rational to a binary64 lower bound."""
    rounded = float(value)
    if Fraction(rounded) > value:
        rounded = math.nextafter(rounded, -math.inf)
    return rounded


def certify_history_silence(region, position, margin_m=1e-5):
    """Prove silence using either the public radius cap or a negative point.

    A previous no-signal point n implies R < |x-n| for the same existing Q3
    source. Therefore a query q is silent whenever every x in outer polygon
    C satisfies |x-q| > |x-n|. Their squared-distance difference is affine:

        |x-q|^2 - |x-n|^2 = 2 (q-n) . ((q+n)/2-x).

    Its minimum over the complete convex polygon occurs at a vertex, also
    for a segment or singleton. Dividing by 2|q-n| expresses the condition
    as a signed distance from the perpendicular bisector, in metres.
    ``margin_m`` is a margin in THAT signed distance, not a squared-distance
    margin or the difference between the two Euclidean distances.

    The historical affine calculation is exact for the stored binary64
    coordinates, with a rationally verified norm upper bound and downward
    rounding of the reported lower bound. This does not retroactively make
    the region construction or the existing radius-cap helper exact.
    Historical negatives may themselves be previously certified deductions;
    the caller must preserve their provenance, never create arbitrary ones.
    """
    if (isinstance(margin_m, bool) or not isinstance(margin_m, (int, float))
            or not math.isfinite(margin_m) or margin_m <= 0):
        raise ValueError("margin_m must be positive and finite")
    original = certify_silence(region, position, margin_m)
    if original is not None:
        return original
    if not region.vertices or not region.observations:
        return None

    query = Position.coerce(position)
    qx, qy = Fraction(query.x), Fraction(query.y)
    vertices = tuple((Fraction(x), Fraction(y)) for x, y in region.vertices)
    for negative in getattr(region, "no_signal_positions", ()):
        witness = Position.coerce(negative)
        nx, ny = Fraction(witness.x), Fraction(witness.y)
        dx, dy = qx-nx, qy-ny
        norm_squared = dx*dx + dy*dy
        if norm_squared == 0:
            # Repeating a known negative is logically silent, but supplies no
            # positive bisector margin; this strict helper declines that case.
            continue
        mx, my = (qx+nx)/2, (qy+ny)/2
        half_difference = min(dx*(mx-x) + dy*(my-y) for x, y in vertices)
        if half_difference <= 0:
            continue

        norm_upper = math.hypot(float(dx), float(dy))
        while Fraction(norm_upper)**2 < norm_squared:
            norm_upper = math.nextafter(norm_upper, math.inf)
        lower = _downward_float(half_difference/Fraction(norm_upper))
        if lower <= margin_m:
            continue
        return {
            "method": "history_negative_bisector",
            "witness_no_signal_position": [witness.x, witness.y],
            "signed_bisector_distance_lower_m": lower,
            "squared_distance_difference_lower_m2": _downward_float(2*half_difference),
            "query_witness_distance_upper_m": norm_upper,
            "margin_m": margin_m,
            "margin_scope": "signed_distance_from_query_witness_perpendicular_bisector",
            "outer_region_vertex_count": len(vertices),
            "bound_arithmetic": "exact_binary64_coordinate_rationals_then_outward_rounding",
        }
    return None
