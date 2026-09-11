"""Observation-only audit of Q4 strict-range skip witnesses.

The generic physical/coverage audit must still run. In particular this helper
does not grant a single absence station for any skipped event. It imports no
strategy, simulator, source truth or the production distance routine.
"""

from collections import defaultdict
import math

from localization import CandidateRegion


def _require(value, message):
    if not value:
        raise ValueError(message)


def _point(value):
    result = (value["x"], value["y"]) if isinstance(value, dict) else tuple(value)
    _require(len(result) == 2 and all(isinstance(v, (float, int)) and math.isfinite(v) for v in result),
             "Invalid range-event position")
    return result


def _raw_distance(q, vertices):
    # CandidateRegion returns a counterclockwise convex outer polygon. Replay
    # independently via oriented half-planes and closest points on its edges.
    _require(bool(vertices), "Range witness has an empty observed region")
    edges = list(zip(vertices, vertices[1:]+vertices[:1]))
    if len(vertices) >= 3:
        signed_area = math.fsum(a[0]*b[1]-a[1]*b[0] for a, b in edges)
        cross = [(b[0]-a[0])*(q[1]-a[1])-(b[1]-a[1])*(q[0]-a[0]) for a, b in edges]
        if signed_area > 0 and min(cross) >= 0:
            return 0.
        if signed_area < 0 and max(cross) <= 0:
            return 0.
    distances = []
    for a, b in edges:
        delta = (b[0]-a[0], b[1]-a[1])
        numerator = math.fsum((q[i]-a[i])*delta[i] for i in (0, 1))
        denominator = math.fsum(v*v for v in delta)
        t = min(1., max(0., numerator/denominator)) if denominator else 0.
        foot = tuple(a[i]+t*delta[i] for i in (0, 1))
        distances.append(math.dist(q, foot))
    return min(distances)


def audit_range_prefix(record):
    """Return a JSON-safe successful audit, or raise ValueError on bad evidence."""
    summary = record.get("summary") or {}
    parameters = summary.get("strategy_parameters", {})
    _require(not parameters.get("inferred_no_signal_constraints"), "Q4 range skips must not become inferred negative constraints")
    actual = [a for a in record["history"] if a["action"] in ("/measure", "/clear")]
    reports = summary.get("action_history", [])
    _require(len(reports) == len(actual), "Range replay requires aligned physical/action histories")
    events = defaultdict(list)
    for event in parameters.get("range_skipped_scans", []):
        count = event["after_actual_action_count"]
        _require(type(count) is int and 0 <= count <= len(actual), "Range event references an invalid prefix")
        events[count].append(event)
    regions, live, cleared, checks = {}, set(), set(), []
    def check(prefix):
        for event in events.pop(prefix, []):
            channel = event["channel"]
            _require(type(channel) is int and channel in live-cleared and channel in regions,
                     "Range skip does not concern a real-positive live channel")
            region = regions[channel]
            _require(bool(region.observations), "Range witness lacks a positive bearing observation")
            point = _point(event["position"])
            _require(event.get("reason") == "positive_region_beyond_max_reception_radius", "Invalid range-skip reason")
            lower, margin = event["distance_lower_bound_m"], event["margin_m"]
            _require(isinstance(lower, (int, float)) and math.isfinite(lower), "Nonfinite claimed range distance")
            _require(margin == 1e-5 and lower > 1500.+margin, "Range skip lacks its strict safety margin")
            raw = _raw_distance(point, region.vertices)
            _require(lower <= raw+1e-7, "Claimed range lower bound exceeds independently rebuilt region distance")
            _require(raw > 1500.+margin, "Observed region does not exclude reception")
            _require(event["positive_observation_count"] == len(region.observations), "Range positive-prefix count mismatch")
            checks.append({"channel": channel, "after_actual_action_count": prefix,
                           "position": list(point), "distance_lower_bound_m": lower,
                           "independent_region_distance_m": raw})
    check(0)
    for count, (action, reported) in enumerate(zip(actual, reports), 1):
        channel, kind = action["channel"], action["action"]
        point = _point(action["position"])
        response = action["response"]
        outcome = response["measure_result" if kind == "/measure" else "clear_result"]
        _require(reported["action"] == kind[1:] and reported["channel"] == channel and
                 _point(reported["position"]) == point and reported["result"] == outcome,
                 "Range replay received a nonphysical/fabricated action report")
        if kind == "/measure" and channel not in cleared:
            if outcome == "direction":
                _require(reported["bearing_deg"] == response["svd_deg"], "Range replay bearing differs from physical feedback")
                regions.setdefault(channel, CandidateRegion()).observe(point, response["svd_deg"])
                live.add(channel)
            elif outcome == "near":
                live.add(channel)
            # Ordinary no_signal and skipped events add no geometric constraint.
        elif kind == "/clear" and outcome == "success":
            cleared.add(channel)
        check(count)
    _require(not events, "Unprocessed range events")
    return {"passed": True, "range_skips_verified": len(checks), "events": checks,
            "inferred_observation_count": 0, "inferred_absence_credits": 0,
            "arithmetic_scope": "Positive-only convex geometry with conservative float margins; no interval claim"}
