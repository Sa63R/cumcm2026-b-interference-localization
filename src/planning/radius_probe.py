"""Exact branch-and-bound evaluation of the existing finite radius score.

The score and candidate/source nodes match StateSearch's v1 radius proxy.
Only nonnegative posterior penalties are omitted in lower bounds. Thus this
speeds up a fixed heuristic without claiming to improve its decisions.
"""

import math
import time

from simulator_client.state import Position


class ProbeGeometryMemo:
    """Exact posterior geometry shared by two scores of one public region.

    This is deliberately a caller-owned, short-lived cache, not a policy/global
    cache. Values are only None (empty polygon) or immutable binary64 radii.
    Ordered histories and vertices are part of the binding because Q3 negative
    feedback affects hypothetical positive updates even without changing C.
    """

    def __init__(self):
        self._region = None
        self._signature = None
        self._values = {}

    def bind(self, region):
        def coords(value):
            return tuple(float(v).hex() for v in value)

        signature = (
            type(region), tuple(coords(v) for v in region.vertices),
            tuple((coords(o.position), float(o.bearing_deg).hex(), float(o.error_deg).hex())
                  for o in region.observations),
            tuple(coords(v) for v in getattr(region, "no_signal_positions", ())),
            tuple(float(getattr(region, name)).hex() for name in
                  ("prior_radius", "disk_sides", "error_deg", "reception_radius")),
        )
        if self._region is not region or self._signature != signature:
            self._region, self._signature = region, signature
            self._values.clear()

    def posterior_radius(self, point, bearing):
        # Hex keys distinguish signed zero; no coordinate rounding or bins.
        key = (float(point.x).hex(), float(point.y).hex(), float(bearing).hex())
        if key not in self._values:
            hypothetical = self._region.copy().observe(point, bearing)
            self._values[key] = (hypothetical.enclosing_disk().radius
                                 if hypothetical.vertices else None)
        return self._values[key]


def choose_radius_probe(region, current, first_bearing, observed, weight=0.5, *,
                        extra_points=(), geometry_memo=None):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("nonnegative finite weight is required for the score bound")
    started = time.perf_counter()
    circle = region.enclosing_disk()
    if geometry_memo is not None:
        geometry_memo.bind(region)
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
                if geometry_memo is None:
                    hypothetical = region.copy().observe(point, bearing)
                    radius = hypothetical.enclosing_disk().radius if hypothetical.vertices else None
                else:
                    radius = geometry_memo.posterior_radius(point, bearing)
                updates += 1
                if radius is None:
                    total = math.inf
                    break
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
