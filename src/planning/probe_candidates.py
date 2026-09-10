"""Finite observation-geometry action sets, not hidden-world samples."""

import math

from simulator_client.state import Position


def long_axis(region):
    vertices = region.vertices
    a, b = max(((a, b) for a in vertices for b in vertices),
               key=lambda pair: (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2)
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    axis = ((b[0] - a[0]) / length, (b[1] - a[1]) / length) if length else (1, 0)
    perpendicular = (-axis[1], axis[0])
    center = Position(*region.enclosing_disk().center)
    values = [(x - center.x) * axis[0] + (y - center.y) * axis[1] for x, y in vertices]
    return center, axis, perpendicular, min(values), max(values)


def geometry_candidates(region, *, mode, old_best):
    center, axis, perpendicular, low, high = long_axis(region)
    width = high - low
    points = []
    if mode == "axis_quantile":
        # Outer quartiles can be rejected by guaranteed reception. The inner
        # 3/8 and 5/8 positions remain useful for long first-bearing strips.
        scale = min(200.0, max(20.0, width / 8))
        for q in (.25, .375, .625, .75):
            along = low + q * width
            for across in (-scale, 0.0, scale):
                points.append(Position(center.x + along * axis[0] + across * perpendicular[0],
                                       center.y + along * axis[1] + across * perpendicular[1]))
        for across in (-scale, scale):
            points.append(Position(center.x + across * perpendicular[0],
                                   center.y + across * perpendicular[1]))
    elif mode == "local_refine":
        along_step = min(125.0, max(20.0, width / 12))
        across_step = min(50.0, max(10.0, width / 30))
        for along in (-along_step, 0.0, along_step):
            for across in (-across_step, 0.0, across_step):
                if along or across:
                    points.append(Position(old_best.x + along * axis[0] + across * perpendicular[0],
                                           old_best.y + along * axis[1] + across * perpendicular[1]))
    else:
        raise ValueError("mode must be axis_quantile or local_refine")
    return points, {"long_axis_width_m": width, "axis": list(axis),
                    "extra_points_proposed": len(points)}
