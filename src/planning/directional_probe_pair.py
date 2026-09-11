"""Two Q4 probes jointly certified by one *actual direction* observation.

Let p be a positive station, s its same stationary uncleared source, r=|p-s|,
and delta the true bearing minus the reported theta, |delta|<=epsilon<10deg.
The continuous candidate polygon C contains s. Hence

    r >= r_min = max(5, distance(p,C)).

With alpha=45deg, d=.95*r_min*cos(alpha+epsilon), set
q+/-=p+d*u(theta+/-alpha). Both probes are no farther from s than p: expanding
their squared distances suffices since d < 2*r_min*cos(alpha+epsilon).
Their joining segment crosses the p-to-s ray at distance

    t=d*cos(alpha)/cos(delta) < r_min <= r.

Because |delta|<alpha, that crossing is inside the q-/q+ segment as well as
[p,s]. The unknown transmitting closed half-plane contains p and s, so contains
the crossing. Linearity implies at least one endpoint is in the half-plane.
Both endpoints are inside the reception disk, so at least one receives.

This does NOT guarantee the first probe, localization accuracy, or clearance.
The caller must reserve two measurements; if first is positive, second can be
discarded. If first is silent, second retains its guarantee only for this same
uncleared stationary source and the unchanged genuine anchor observation.
The proof is real arithmetic; float coordinates/tolerances are not an interval
arithmetic certificate. Never clip an out-of-bounds candidate: reject it.
"""

from dataclasses import dataclass
import math

from geometry import convex_hull, point
from simulator_client.state import Position


ALPHA_DEG = 45.0
STEP_MARGIN = .95


@dataclass(frozen=True)
class DirectionalProbePair:
    first: Position
    second: Position
    anchor: Position
    anchor_bearing_deg: float
    error_deg: float
    distance_lower_bound_m: float
    step_m: float
    complete_pair_cost_s: float


def _key(p):
    return tuple(round(v, 6) for v in point(p))


def _segment_distance(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    length2 = dx*dx+dy*dy
    if not length2:
        return math.dist(p, a)
    fraction = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length2))
    closest = a[0]+fraction*dx, a[1]+fraction*dy
    return math.dist(p, closest)


def distance_to_convex_region(position, vertices):
    """Distance to the closed convex hull, including point/segment cases.

    Taking a hull tolerates repeated vertices and either traversal orientation.
    Actual candidate regions are already convex; this function neither mutates
    them nor replaces their live representation. Empty geometry is invalid.
    """
    p = point(position)
    hull = convex_hull(point(v) for v in vertices)
    if not hull:
        raise ValueError("distance needs a nonempty finite candidate region")
    if len(hull) == 1:
        return math.dist(p, hull[0])
    if len(hull) == 2:
        return _segment_distance(p, hull[0], hull[1])
    edges = tuple(zip(hull, hull[1:]+hull[:1]))
    cross = [(b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]) for a, b in edges]
    if all(c >= 0. for c in cross) or all(c <= 0. for c in cross):
        return 0.
    return min(_segment_distance(p, a, b) for a, b in edges)


def choose_directional_probe_pair(region, current, observed_positions, max_anchors=4):
    """Return ``(DirectionalProbePair | None, log)`` using observations only.

    Examine at most four distinct real direction anchors nearest to current,
    then rank by travel(current,first)+travel(first,second)+two measurements.
    This is an explicit complete-pair cost, not expected localization cost.
    Entry channel switching is omitted (identical for this source's choices).
    Both points must be fresh at the existing six-decimal coordinate precision.
    ``observed_positions`` supplies no reception sign; no old silent point is
    used to infer that one candidate alone is safe.
    """
    if type(max_anchors) is not int or not 1 <= max_anchors <= 4:
        raise ValueError("max_anchors must be an integer in [1,4]")
    current = Position.coerce(current)
    observed = {_key(p) for p in observed_positions}
    log = {"certificate": "at_least_one_of_two_receives_same_stationary_uncleared_source",
           "arithmetic_scope": "real_geometry_theorem_with_float_coordinates_not_interval_proof",
           "first_individually_guaranteed": False, "clearance_guaranteed": False,
           "required_available_measurements": 2, "alpha_deg": ALPHA_DEG,
           "step_margin": STEP_MARGIN, "max_anchors": max_anchors,
           "anchor_selection": "nearest_distinct_actual_direction_stations_then_complete_pair_cost",
           "switching_s_in_score": 0., "candidates": [], "selected": None, "reason": None}
    if not region.vertices or not region.observations:
        log["reason"] = "empty_region_or_no_actual_direction_anchor"
        return None, log
    vertices = tuple(point(v) for v in region.vertices)
    anchors = {}
    for index, observation in enumerate(region.observations):
        p = Position.coerce(observation.position)
        observed.add(_key(p))
        theta, epsilon = observation.bearing_deg, observation.error_deg
        if not (math.isfinite(theta) and math.isfinite(epsilon) and 0. <= epsilon < 10.):
            continue
        candidate = (epsilon, theta % 360., index, p)
        key = (p.x, p.y)
        if key not in anchors or candidate[:3] < anchors[key][:3]:
            anchors[key] = candidate
    chosen_anchors = sorted(anchors.values(), key=lambda item: (
        current.distance_to(item[3]), item[3].x, item[3].y, item[0], item[1]))[:max_anchors]
    log["distinct_eligible_anchors"] = len(anchors)
    log["anchors_examined"] = len(chosen_anchors)
    eligible = []
    for epsilon, theta, observation_index, anchor in chosen_anchors:
        raw_distance = distance_to_convex_region(anchor, vertices)
        scale = max(1., abs(anchor.x), abs(anchor.y), *(abs(v) for p in vertices for v in p))
        # Subtract a downward engineering guard before using a computed lower
        # distance bound. The much larger .95 step margin is also retained.
        guard = 1e-7+128*math.ulp(scale)
        lower = max(5., raw_distance-guard)
        item = {"anchor": [anchor.x, anchor.y], "anchor_observation_index": observation_index,
                "anchor_bearing_deg": theta, "error_deg": epsilon,
                "raw_region_distance_m": raw_distance, "distance_guard_m": guard,
                "distance_lower_bound_m": lower, "valid": False}
        log["candidates"].append(item)
        if lower > min(1500., region.reception_radius):
            item["reason"] = "anchor_region_inconsistent_with_reception_upper_bound"
            continue
        step = STEP_MARGIN*lower*math.cos(math.radians(ALPHA_DEG+epsilon))
        item["step_m"] = step
        item["radius_sufficient_condition_ratio"] = step/(2*lower*math.cos(math.radians(ALPHA_DEG+epsilon)))
        item["ray_intersection_to_source_distance_ratio_upper"] = (
            step*math.cos(math.radians(ALPHA_DEG))/math.cos(math.radians(epsilon))/lower)
        try:
            pair = tuple(Position(anchor.x+step*math.cos(math.radians(theta+sign*ALPHA_DEG)),
                                  anchor.y+step*math.sin(math.radians(theta+sign*ALPHA_DEG)))
                         for sign in (-1., 1.))
        except ValueError:
            item["reason"] = "candidate_exceeds_coordinate_bound_no_clipping"
            continue
        if _key(pair[0]) == _key(pair[1]) or any(_key(p) in observed for p in pair):
            item["reason"] = "pair_not_two_distinct_fresh_positions"
            continue
        first, second = sorted(pair, key=lambda p: (current.distance_to(p), p.x, p.y))
        first_travel = current.distance_to(first)/5
        between_travel = first.distance_to(second)/5
        cost = first_travel+between_travel+10.
        item.update(valid=True, reason="two_probe_reception_certificate",
                    first=[first.x, first.y], second=[second.x, second.y],
                    movement_to_first_s=first_travel, movement_between_s=between_travel,
                    measurement_s=10., complete_pair_cost_s=cost)
        value = DirectionalProbePair(first, second, anchor, theta, epsilon, lower, step, cost)
        eligible.append((cost, first.x, first.y, second.x, second.y, len(log["candidates"])-1, value))
    if not eligible:
        log["reason"] = "no_fresh_eligible_pair"
        return None, log
    selected = min(eligible, key=lambda item: item[:-1])
    log.update(selected=selected[-2], reason="selected", complete_pair_cost_s=selected[0])
    return selected[-1], log


__all__ = ["DirectionalProbePair", "distance_to_convex_region", "choose_directional_probe_pair"]
