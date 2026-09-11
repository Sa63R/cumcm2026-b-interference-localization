"""Conservative Q4 position pruning using one fixed radius and orientation.

Only geometry and supplied observations are consumed. A fixed complete grid is
used, never samples standing in for a coverage proof. Floating trigonometry has
outward guards; polygon clipping and hull orientation use rational arithmetic.
This is an outer relaxation, not an exact hidden-state feasibility solver.
"""
from fractions import Fraction
import math
import time

from geometry import minimum_enclosing_circle, point


GRID_SIZE = 16
BBOX_MARGIN_M = 1e-4
DISTANCE_MARGIN_M = 1e-7
FORCED_MARGIN_M = 1e-4
NEAR_POINT_M = 1e-7
ANGLE_MARGIN_RAD = 1e-10
INTERSECTION_MARGIN_RAD = 1e-12
MAX_CONSTRAINT_WORK = 65536
MAX_VERTICES = 256
TAU = 2. * math.pi


def _fraction_point(p):
    return Fraction(p[0]), Fraction(p[1])


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _exact_hull(points):
    points = sorted(set(points))
    if len(points) < 3:
        return tuple(points)
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return tuple(lower[:-1]+upper[:-1])


def _clip_box_exact(vertices, box):
    """Closed convex polygon intersection with an exact float-endpoint box."""
    result = tuple(vertices)
    xmin, ymin, xmax, ymax = map(Fraction, box)
    for axis, value, lower in ((0, xmin, True), (0, xmax, False),
                               (1, ymin, True), (1, ymax, False)):
        if not result:
            break
        output = []
        for a, b in zip(result, result[1:]+result[:1]):
            ain = a[axis] >= value if lower else a[axis] <= value
            bin = b[axis] >= value if lower else b[axis] <= value
            if ain != bin:
                t = (value-a[axis])/(b[axis]-a[axis])
                output.append((a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])))
            if bin:
                output.append(b)
        result = tuple(dict.fromkeys(output))
    return result


def _round_box(p):
    """A floating box proven to enclose an exact rational intersection point."""
    xs, ys = [], []
    for target, coordinate in ((xs, p[0]), (ys, p[1])):
        f = float(coordinate)
        low, high = math.nextafter(f, -math.inf), math.nextafter(f, math.inf)
        if not math.isfinite(low) or not math.isfinite(high):
            raise ArithmeticError("Intersection conversion overflow")
        if not Fraction(low) <= coordinate <= Fraction(high):
            raise ArithmeticError("Intersection rounding box is not outward")
        target.extend((low, high))
    return (xs[0], ys[0], xs[1], ys[1])


def _corners(box):
    a, b, c, d = box
    return ((a, b), (c, b), (c, d), (a, d))


def _distance_min(p, box):
    a, b, c, d = box
    return math.hypot(max(a-p[0], 0., p[0]-c), max(b-p[1], 0., p[1]-d))


def _distance_max(p, box):
    return max(math.hypot(p[0]-q[0], p[1]-q[1]) for q in _corners(box))


def _normalize(intervals):
    intervals = [(max(0., a), min(TAU, b)) for a, b in intervals if a <= b]
    if any(a == 0. for a, _ in intervals):
        intervals.append((TAU, TAU))
    if any(b == TAU for _, b in intervals):
        intervals.append((0., 0.))
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]+INTERSECTION_MARGIN_RAD:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return tuple(merged)


def _arc_intervals(start, width):
    if width >= TAU-INTERSECTION_MARGIN_RAD:
        return ((0., TAU),)
    start %= TAU
    end = start+width
    if end <= TAU:
        return _normalize(((start, end),))
    return _normalize(((start, TAU), (0., end-TAU)))


def _intersect(left, right):
    result = []
    for a, b in left:
        for c, d in right:
            low, high = max(a, c), min(b, d)
            if low <= high:
                result.append((low, high))
            elif low-high <= INTERSECTION_MARGIN_RAD:
                # A near-tangent numerical gap is retained, never called empty.
                result.append((high, low))
    return _normalize(result)


def _allowed_orientation(observer, box, *, positive, distance_guard):
    """Union over locations in a box, not intersection over box corners."""
    dmin = max(0., _distance_min(observer, box)-distance_guard)
    if dmin <= NEAR_POINT_M:
        return ((0., TAU),), dict(reason="near_or_inside_box", dmin_lower_m=dmin,
                                 direction_arc=None, engine_margin_rad=0.)
    angles = sorted(math.atan2(observer[1]-y, observer[0]-x) % TAU
                    for x, y in _corners(box))
    gaps = [angles[i+1]-angles[i] for i in range(3)]+[angles[0]+TAU-angles[-1]]
    i = max(range(4), key=lambda j: gaps[j])
    start, width = angles[(i+1) % 4], TAU-gaps[i]
    if width >= math.pi:
        # An exterior convex box should subtend <pi. Ambiguity is only relaxed.
        return ((0., TAU),), dict(reason="wide_direction_fallback", dmin_lower_m=dmin,
                                 direction_arc=[start, width], engine_margin_rad=0.)
    engine = math.asin(min(1., 1e-12*max(1., 1./dmin))) if positive else 0.
    margin = ANGLE_MARGIN_RAD+engine
    offset = 0. if positive else math.pi
    intervals = _arc_intervals(start+offset-math.pi/2-margin, width+math.pi+2*margin)
    return intervals, dict(reason="outward_box_direction_union", dmin_lower_m=dmin,
        direction_arc=[start, width], engine_margin_rad=engine,
        angular_margin_rad=ANGLE_MARGIN_RAD)


def _disk(vertices):
    if not vertices:
        return None
    circle = minimum_enclosing_circle(vertices)
    radius = max(circle.radius, *(math.hypot(x-circle.center[0], y-circle.center[1])
                                  for x, y in vertices))
    radius = math.nextafter(radius, math.inf)
    return dict(center=list(circle.center), radius_m=radius,
                ready_under_19_9_m=radius <= 19.9)


def joint_visibility_outer(vertices, positive_positions, negative_positions):
    """Return ``(outer_vertices, evidence)`` without mutating the canonical C.

    Valid input coordinates are interpreted as exact floating-point values for
    clipping. Geometry/budget/degeneracy failures return the unchanged C. Invalid
    non-finite canonical input is rejected since no valid fallback exists.
    """
    began = time.perf_counter()
    original = tuple(point(p) for p in vertices)
    log = dict(method="q4_joint_visibility_fixed_grid_v1", passed=False,
        grid_size=GRID_SIZE, canonical_vertices=[list(p) for p in original],
        positive_positions=[], negative_positions=[], cells=[], bbox=None,
        parameters=dict(bbox_margin_m=BBOX_MARGIN_M, distance_margin_m=DISTANCE_MARGIN_M,
            forced_margin_m=FORCED_MARGIN_M, near_point_m=NEAR_POINT_M,
            angle_margin_rad=ANGLE_MARGIN_RAD, intersection_margin_rad=INTERSECTION_MARGIN_RAD,
            max_constraint_work=MAX_CONSTRAINT_WORK, max_vertices=MAX_VERTICES),
        old_disk=None, new_disk=None, old_vertices_excluded=[], became_ready=False)

    def finish(outer, status, fallback=None):
        log.update(status=status, fallback_reason=fallback, passed=True,
            output_vertices=[list(p) for p in outer])
        if log["old_disk"] is not None:
            log["new_disk"] = log["old_disk"] if outer == original else _disk(outer)
            log["became_ready"] = (not log["old_disk"]["ready_under_19_9_m"]
                                    and log["new_disk"]["ready_under_19_9_m"])
        log["runtime_s"] = time.perf_counter()-began
        return outer, log

    try:
        positives = tuple(point(p) for p in positive_positions)
        negatives = tuple(point(p) for p in negative_positions)
        log["positive_positions"] = [list(p) for p in positives]
        log["negative_positions"] = [list(p) for p in negatives]
        if len(original) > MAX_VERTICES:
            return finish(original, "fallback", "vertex_budget")
        log["old_disk"] = _disk(original)
        canonical = _exact_hull(tuple(_fraction_point(p) for p in original))
        if len(canonical) < 3:
            return finish(original, "fallback", "degenerate_canonical_region")
        # The contract is a convex polygon, not arbitrary unsorted point data.
        area = sum(_cross((Fraction(0), Fraction(0)), a, b)
                   for a, b in zip(map(_fraction_point, original),
                                   map(_fraction_point, original[1:]+original[:1])))
        hull_area = sum(_cross((Fraction(0), Fraction(0)), a, b)
                        for a, b in zip(canonical, canonical[1:]+canonical[:1]))
        if abs(area) != hull_area:
            return finish(original, "fallback", "nonconvex_or_unordered_canonical_region")
        if not positives:
            return finish(original, "fallback", "no_positive_evidence")
        if GRID_SIZE**2*(len(positives)+len(negatives)) > MAX_CONSTRAINT_WORK:
            return finish(original, "fallback", "constraint_budget")
        scale = max(1., *(abs(x) for p in original+positives+negatives for x in p))
        guard = DISTANCE_MARGIN_M+128*math.ulp(scale)
        log["distance_guard_m"] = guard
        xmin = math.nextafter(min(p[0] for p in original)-BBOX_MARGIN_M, -math.inf)
        xmax = math.nextafter(max(p[0] for p in original)+BBOX_MARGIN_M, math.inf)
        ymin = math.nextafter(min(p[1] for p in original)-BBOX_MARGIN_M, -math.inf)
        ymax = math.nextafter(max(p[1] for p in original)+BBOX_MARGIN_M, math.inf)
        xs = [xmin+(xmax-xmin)*i/GRID_SIZE for i in range(GRID_SIZE+1)]
        ys = [ymin+(ymax-ymin)*i/GRID_SIZE for i in range(GRID_SIZE+1)]
        xs[0], xs[-1], ys[0], ys[-1] = xmin, xmax, ymin, ymax
        if not all(a < b for edges in (xs, ys) for a, b in zip(edges, edges[1:])):
            return finish(original, "fallback", "collapsed_grid")
        log.update(bbox=[xmin, ymin, xmax, ymax], x_edges=xs, y_edges=ys)
        cloud, deleted = [], 0
        for iy in range(GRID_SIZE):
            for ix in range(GRID_SIZE):
                box = (xs[ix], ys[iy], xs[ix+1], ys[iy+1])
                clipped = _clip_box_exact(canonical, box)
                cell = dict(id=iy*GRID_SIZE+ix, ix=ix, iy=iy, bbox=list(box),
                    removed=False, forced_negative_indices=[], constraints=[],
                    orientation_intersection=[[0., TAU]], retained_intersection_boxes=[])
                log["cells"].append(cell)
                if not clipped:
                    cell["reason"] = "outside_canonical_region_exact"
                    continue
                minimums = [_distance_min(p, box) for p in positives]
                rlower = max(1000., *minimums)-guard
                uppers = [_distance_max(n, box)+guard for n in negatives]
                forced = [j for j, upper in enumerate(uppers) if upper < rlower-FORCED_MARGIN_M]
                cell.update(positive_min_distances_m=minimums, R_min_lower_m=rlower,
                            negative_max_distances_upper_m=uppers, forced_negative_indices=forced)
                allowed = ((0., TAU),)
                if forced:
                    for kind, points, indices in (("positive", positives, range(len(positives))),
                                                   ("forced_negative", negatives, forced)):
                        for j in indices:
                            intervals, certificate = _allowed_orientation(
                                points[j], box, positive=kind == "positive", distance_guard=guard)
                            allowed = _intersect(allowed, intervals)
                            cell["constraints"].append(dict(kind=kind, observation_index=j,
                                allowed_intervals=[list(v) for v in intervals], **certificate))
                    cell["orientation_intersection"] = [list(v) for v in allowed]
                if forced and not allowed:
                    cell.update(removed=True, reason="forced_directional_orientation_empty")
                    deleted += 1
                    continue
                cell["reason"] = "omni_branch_retained" if not forced else "orientation_outer_nonempty"
                for rational in clipped:
                    outward = _round_box(rational)
                    cell["retained_intersection_boxes"].append(list(outward))
                    cloud.extend(_corners(outward))
        log["deleted_intersecting_cells"] = deleted
        log["retained_intersecting_cells"] = sum(bool(c["retained_intersection_boxes"]) for c in log["cells"])
        if not cloud:
            return finish(original, "fallback", "all_intersecting_cells_deleted")
        if not deleted:
            return finish(original, "unchanged")
        rational_hull = _exact_hull(tuple(_fraction_point(p) for p in cloud))
        if len(rational_hull) < 3:
            return finish(original, "fallback", "degenerate_output")
        outer = tuple((float(x), float(y)) for x, y in rational_hull)
        if any(_fraction_point(p) != q for p, q in zip(outer, rational_hull)):
            raise ArithmeticError("Hull output vertices were not exact input floats")
        log["old_vertices_excluded"] = [i for i, p in enumerate(original)
            if any(_cross(a, b, _fraction_point(p)) < 0
                   for a, b in zip(rational_hull, rational_hull[1:]+rational_hull[:1]))]
        return finish(outer, "outer_refined")
    except (ArithmeticError, ValueError, TypeError, IndexError) as error:
        # No partial partition or incomplete certificate can affect the caller.
        log["error_type"] = type(error).__name__
        return finish(original, "fallback", "geometry_exception")
