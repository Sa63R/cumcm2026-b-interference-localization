"""Exact branch-and-bound evaluation of the existing finite radius score.

The score and candidate/source nodes match StateSearch's v1 radius proxy.
Only nonnegative posterior penalties are omitted in lower bounds. Thus this
speeds up a fixed heuristic without claiming to improve its decisions.
"""

import math
import time

from simulator_client.state import Position


def choose_radius_probe(region, current, first_bearing, observed, weight=0.5, *, extra_points=()):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("nonnegative finite weight is required for the score bound")
    started = time.perf_counter()
    circle = region.enclosing_disk()
    center = Position(*circle.center)
    theta = math.radians(first_bearing)
    perpendicular = (-math.sin(theta), math.cos(theta))
    candidates = [center]
    for fraction in (0.5, 1.0):
        anchor = Position(current.x + fraction * (center.x - current.x),
                          current.y + fraction * (center.y - current.y))
        for offset in (-150.0, -50.0, 50.0, 150.0):
            candidates.append(Position(anchor.x + offset * perpendicular[0],
                                       anchor.y + offset * perpendicular[1]))
    # Preserve every original point. Optional additions only enlarge the
    # declared finite action set; exact coordinate duplicates add no action.
    seen = {(p.x, p.y) for p in candidates}
    for point in extra_points:
        point = Position.coerce(point)
        if (point.x, point.y) not in seen:
            candidates.append(point)
            seen.add((point.x, point.y))
    vertices = region.vertices
    selected = [vertices[i * len(vertices) // min(4, len(vertices))]
                for i in range(min(4, len(vertices)))]
    hypotheses = [center] + [Position(0.75 * x + 0.25 * center.x,
                                      0.75 * y + 0.25 * center.y) for x, y in selected]
    legal = []
    for point in candidates:
        if (round(point.x, 6), round(point.y, 6)) in observed:
            continue
        if any(point.distance_to(Position(*vertex)) > 1000 for vertex in vertices):
            continue
        distances = [point.distance_to(source) for source in hypotheses]
        travel = current.distance_to(point) / 5
        lower = travel + sum(d / 5 for d in distances) / len(hypotheses)
        legal.append((lower, point.x, point.y, point, travel, distances))
    best = (math.inf, math.inf, math.inf, None)
    evaluated = pruned = updates = 0
    for lower, _, _, point, travel, distances in sorted(legal):
        # Strict conservative tolerance preserves ties and original (x,y)
        # ordering. Finite candidate scores are only compared when complete.
        if lower > best[0] + 1e-9:
            pruned += 1
            continue
        total, complete = 0.0, True
        for i, (source, d) in enumerate(zip(hypotheses, distances)):
            if d <= 5:
                total += d / 5
            else:
                bearing = math.degrees(math.atan2(source.y - point.y, source.x - point.x)) % 360
                hypothetical = region.copy().observe(point, bearing)
                updates += 1
                if not hypothetical.vertices:
                    total = math.inf
                    break
                radius = hypothetical.enclosing_disk().radius
                total += d / 5 + (6.0 if radius > 19.9 else 0.0) + weight * max(0.0, radius - 19.9) / 5
            partial_lower = travel + (total + sum(d / 5 for d in distances[i + 1:])) / len(hypotheses)
            if partial_lower > best[0] + 1e-9:
                pruned += 1
                complete = False
                break
        if complete:
            evaluated += 1
            candidate = (travel + total / len(hypotheses), point.x, point.y, point)
            if candidate[:3] < best[:3]:
                best = candidate
    if best[3] is None and legal:
        # Match v1's min tuple even for the degenerate all-infinite case.
        point = min((item[3] for item in legal), key=lambda p: (p.x, p.y))
        best = (math.inf, point.x, point.y, point)
    return best[3], {"candidates": len(legal), "proposed_candidates": len(candidates),
        "hypotheses": len(hypotheses),
        "score_s": best[0], "position": [best[3].x, best[3].y] if best[3] else None,
        "score_kind": "finite_nominal_observation_surrogate_exact_pruning",
        "evaluated_candidates": evaluated, "pruned_candidates": pruned,
        "geometry_updates": updates, "runtime_s": time.perf_counter() - started}
