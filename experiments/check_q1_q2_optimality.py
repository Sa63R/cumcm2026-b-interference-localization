"""Reproduce Q1 implementation limits and a Q2 same-objective counterexample.

Diagnostic only: no solver changes, simulator calls, or hidden scene data.
"""
import hashlib
import itertools
import json
from pathlib import Path

import geometry
import localization
from localization import CandidateRegion, select_next_point, score_detection_point


def main():
    polygon = ((-2., 0.), (-1., -1e-14), (1., -1e-14),
               (2., 0.), (1., 5e-15), (-1., 5e-15))
    computed_diameter = geometry.polygon_diameter(polygon)
    exact_diameter = max(geometry.distance(a, b)
                         for a, b in itertools.combinations(polygon, 2))
    tiny_points = ((0., 0.), (1e-9, 0.))
    circle = geometry.minimum_enclosing_circle(tiny_points)
    exact_radius = geometry.distance(*tiny_points) / 2
    region = CandidateRegion().observe((0, 0), 0)
    selected = select_next_point(region, (0, 0))
    alternative = (833., 551.)
    choice = score_detection_point(region, alternative, (0, 0))
    safe_distance = max(geometry.distance(alternative, v) for v in region.vertices)
    safety_polygon = region.guaranteed_detection_region().vertices
    inside_inner = all(geometry.cross(a, b, alternative) > 0
                       for a, b in zip(safety_polygon, safety_polygon[1:] + safety_polygon[:1]))
    result = {
        "source_sha256": {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                          for name, module in (("geometry", geometry), ("localization", localization))},
        "q1_thin_polygon": {
            "vertices": polygon, "reported_diameter": computed_diameter,
            "independent_pairwise_diameter": exact_diameter,
            "counterexample_present": computed_diameter != exact_diameter,
            "note": "Extreme 1.5e-14 metre thickness; does not establish errors in completed official practice cases.",
        },
        "q1_tiny_circle": {
            "points": tiny_points, "reported_radius": circle.radius,
            "exact_two_point_radius": exact_radius,
            "counterexample_present": circle.radius > exact_radius,
        },
        "q2_proxy_counterexample": {
            "first_point": [0, 0], "first_bearing_deg": 0,
            "selected_point": selected.position, "selected_score": selected.score,
            "selected_movement_m": selected.movement_m,
            "alternative_point": alternative, "alternative_score": choice.score,
            "alternative_movement_m": choice.movement_m,
            "max_distance_to_source_vertices_m": safe_distance,
            "strictly_inside_current_safe_inner_polygon": inside_inner,
            "improvement_fraction": (selected.score - choice.score) / selected.score,
            "counterexample_present": safe_distance <= 1000 and inside_inner and choice.score < selected.score,
            "scope": "Same implemented proxy objective over continuous safe region; no claim of improved true worst posterior radius.",
        },
    }
    target = Path("results/validation/q1_q2_optimality.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
