"""Q4 probe candidates certified by the convex hull of real positive stations.

For a fixed, uncleared source, its reception set is a disk intersected with a
closed half-plane (or just a disk). It is convex. Every convex combination of
its actual positive stations therefore receives, even when R is unknown and
some stations are farther than 1000 m from the source. This argument must not
be applied to estimated source positions, negative stations or cleared sources.

The certificate concerns reception only, in exact real arithmetic. Coordinates
and witness reconstruction use ordinary floating arithmetic, not intervals.
Ranking uses <=5 nominal source supports and zero new angular error; it is not
a posterior, a worst-case contraction bound, or a certified clearing decision.
The caller must retain real-observation clear certificates and optical fallback.
"""

import math

from geometry import convex_hull, point
from simulator_client.state import Position


def _key(value):
    return tuple(round(x, 6) for x in value)


def _candidates(positive, observed, maximum):
    hull = convex_hull(positive)
    if len(hull) < 2:
        return []
    proposed, seen = [], set(observed) | {_key(p) for p in positive}

    def add(weights, kind):
        # Each sparse witness refers only to coordinates of actual positives.
        total = math.fsum(weights.values())
        weights = {p: w / total for p, w in weights.items() if w > 0}
        q = tuple(math.fsum(w * p[j] for p, w in weights.items()) for j in (0, 1))
        key = _key(q)
        if key in seen:
            return
        seen.add(key)
        witness = [{"position": list(p), "weight": w} for p, w in sorted(weights.items())]
        proposed.append((q, {"construction": kind, "weights": witness}))

    mean_weights = {p: 1.0 / len(hull) for p in hull}
    add(mean_weights, "hull_vertex_mean")
    edges = list(zip(hull, hull[1:] + hull[:1])) if len(hull) > 2 else [(hull[0], hull[1])]
    for fraction in (0.25, 0.5, 0.75):
        for a, b in edges:
            add({a: 1.0 - fraction, b: fraction}, "hull_edge_" + str(fraction))
    if len(hull) > 2:
        for vertex in hull:
            weights = {p: w * 0.5 for p, w in mean_weights.items()}
            weights[vertex] += 0.5
            add(weights, "halfway_vertex_to_hull_mean")
    if len(proposed) <= maximum:
        return proposed
    # Spread the bounded set across the complete deterministic proposal list.
    indices = [i * (len(proposed) - 1) // (maximum - 1) for i in range(maximum)] if maximum > 1 else [0]
    return [proposed[i] for i in indices]


def _supports(region):
    center = region.enclosing_disk().center
    vertices = region.vertices
    count = min(4, len(vertices))
    values = [center] + [tuple(0.25 * center[j] + 0.75 * vertices[i * len(vertices) // count][j]
                              for j in (0, 1)) for i in range(count)]
    return list(dict.fromkeys(values))


def _score(region, current, q, supports):
    outcomes = []
    for source in supports:
        distance = math.dist(q, source)
        if distance <= 5.0:
            radius, approach_m, ready, bearing = 5.0, 0.0, True, None
        else:
            bearing = math.degrees(math.atan2(source[1] - q[1], source[0] - q[0])) % 360
            posterior = region.copy().observe(q, bearing)
            if not posterior.vertices:
                return None
            circle = posterior.enclosing_disk()
            radius, ready = circle.radius, circle.radius <= 19.9
            # When ready this is distance to the conservative safe-clear disk.
            # Otherwise it is only approach-to-centre in the nominal proxy.
            safe_radius = max(0.0, 19.9 - radius)
            approach_m = max(0.0, math.dist(q, circle.center) - safe_radius)
        tail_s = approach_m / 5.0 + 5.0
        if not ready:
            tail_s += 6.0 + 0.5 * max(0.0, radius - 19.9) / 5.0
        outcomes.append({"support": list(source), "nominal_bearing_deg": bearing,
                         "nominal_near": bearing is None, "radius_m": radius,
                         "clear_approach_m": approach_m, "nominal_clearable": ready,
                         "tail_proxy_s": tail_s})
    movement_s = math.dist(current, q) / 5.0
    return {"score_s": movement_s + 5.0 + math.fsum(o["tail_proxy_s"] for o in outcomes) / len(outcomes),
            "movement_s": movement_s, "measurement_s": 5.0,
            "sampled_all_clearable": all(o["nominal_clearable"] for o in outcomes),
            "nominal_outcomes": outcomes}


def choose_positive_hull_probe(region, current, observed_positions, max_candidates=24):
    """Return ``(Position | None, log)`` without modifying the live region.

    Only ``region.observations`` supply positive witnesses. ``observed_positions``
    may include negative measurements and is used exclusively to exclude repeat
    coordinates (at the strategy's existing six-decimal deduplication precision).
    Priority is all nominal supports clearable after one measure, then full
    proxy score and coordinate ties. This priority is NOT an actual guarantee.
    Channel-switch cost is omitted: this API lacks channel state, and that cost
    is identical for all candidates of this same requested channel.
    """
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= 24:
        raise ValueError("max_candidates must be an integer in 1..24")
    current = point(current)
    observed = {_key(point(p)) for p in observed_positions}
    positive = sorted(set(point(o.position) for o in region.observations))
    log = {"score_kind": "positive_hull_finite_nominal_observation_proxy",
           "reception_certificate": "convex_combination_of_same_uncleared_source_positive_stations",
           "certificate_arithmetic": "real_arithmetic_theorem_with_float_coordinates_not_interval_proof",
           "switching_s_in_score": 0.0, "switching_note": "same-channel candidate-independent cost omitted",
           "positive_stations": [list(p) for p in positive], "candidates": [],
           "hypotheses": 0, "selected": None, "position": None, "score_s": None}
    if not region.vertices or len(positive) < 2:
        log["reason"] = "empty_region_or_fewer_than_two_distinct_positive_stations"
        return None, log
    candidates = _candidates(positive, observed, max_candidates)
    if not candidates:
        log["reason"] = "no_fresh_convex_combination"
        return None, log
    work = region.copy()
    supports = _supports(work)
    log["hypotheses"] = len(supports)
    for q, witness in candidates:
        scored = _score(work, current, q, supports)
        item = {"position": list(q), **witness, "valid_nominal_score": scored is not None}
        if scored is not None:
            item.update(scored)
        log["candidates"].append(item)
    eligible = [(i, c) for i, c in enumerate(log["candidates"]) if c["valid_nominal_score"]]
    if not eligible:
        log["reason"] = "no_consistent_nominal_support_update"
        return None, log
    index, selected = min(eligible, key=lambda pair: (not pair[1]["sampled_all_clearable"],
                                                      pair[1]["score_s"], *pair[1]["position"]))
    log.update(selected=index, position=selected["position"], score_s=selected["score_s"], reason="selected")
    return Position(*selected["position"]), log
