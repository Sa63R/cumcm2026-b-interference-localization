"""V1c: move fixed-order COVER tasks inside their certified feasible regions.

This module has no policy state and reads no simulator truth.  Its caller must
enable it only while all detected sources are cleared and unknown channels
remain, and must resume normal planning if a measurement discovers a source.
Anchor identities, responsibility cells, channel tasks and visit order are
unchanged.  The objective is an open route, with no return to its start.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

from geometry import HalfPlane, clip_polygon, disk_polygon, distance, point
from strategies.q3_fresh import COVER, MARGIN, covered


# Larger than the original certificate tolerance, including boundary-roundoff
# in circle intersections and projections.  This is geometric safety slack,
# not a measurement scheduling or source-existence threshold.
POSITION_MARGIN = 1e-3
_RADIUS = 1000.0 - POSITION_MARGIN
_NUMERIC_TOL = 1e-8


def _anchor_point(anchor):
    p = point(anchor)
    for original in COVER:
        if distance(p, original) < 1e-8:
            return original
    raise ValueError("A tail task must retain one of the original COVER anchors")


@lru_cache(maxsize=7)
def responsibility_polygon(anchor):
    """Outer arena polygon intersected with the anchor's fixed Voronoi cell.

    Identical halfplanes to ``covered(..., anchor, cell=True)`` are used.
    The circumscribed arena polygon conservatively contains the true disk.
    """
    anchor = _anchor_point(anchor)
    polygon = disk_polygon((0, 0), 1800, 64, outer=True)
    for other in COVER:
        if distance(anchor, other) < 1e-9:
            continue
        dx, dy = other[0] - anchor[0], other[1] - anchor[1]
        polygon = clip_polygon(polygon, HalfPlane(
            dx, dy, dx * (other[0] + anchor[0]) / 2
            + dy * (other[1] + anchor[1]) / 2))
    if not polygon:
        raise ValueError("Empty COVER responsibility polygon")
    return tuple(polygon)


def certifies_cell(position, anchor):
    """Verify every outer vertex and the original continuous certificate."""
    anchor = _anchor_point(anchor)
    try:
        q = point(position)
    except (ValueError, TypeError):
        return False
    return (all(distance(q, vertex) <= 1000 - MARGIN
                for vertex in responsibility_polygon(anchor))
            and covered((q,), anchor, True))


def open_length(points, robot):
    """Length of the fixed-order open route in metres."""
    total, previous = 0.0, point(robot)
    for q in points:
        total += distance(previous, q)
        previous = q
    return total


@dataclass(frozen=True)
class _DiskIntersection:
    centers: tuple
    seed: tuple
    corners: tuple

    def feasible(self, q):
        return all(distance(q, v) <= _RADIUS + _NUMERIC_TOL
                   for v in self.centers)

    def project(self, target):
        """Euclidean projection, by complete 2-D boundary enumeration.

        A nearest point is the target, a radial projection on one active
        circle, or an intersection of at least two circle boundaries.
        Pair intersections are cached when the region is constructed.
        """
        if self.feasible(target):
            return target
        candidates = [self.seed, *self.corners]
        for center in self.centers:
            d = distance(target, center)
            if d <= _RADIUS:
                continue
            q = (center[0] + _RADIUS * (target[0] - center[0]) / d,
                 center[1] + _RADIUS * (target[1] - center[1]) / d)
            if self.feasible(q):
                candidates.append(q)
        return min(candidates, key=lambda q: distance(q, target))

    def segment_point(self, left, right, current):
        """A feasible point on a segment, or None if its intersection is empty."""
        dx, dy = right[0] - left[0], right[1] - left[1]
        length2 = dx * dx + dy * dy
        if length2 < 1e-20:
            return left if self.feasible(left) else None
        lower, upper = 0.0, 1.0
        for center in self.centers:
            x, y = left[0] - center[0], left[1] - center[1]
            linear = x * dx + y * dy
            constant = x * x + y * y - _RADIUS * _RADIUS
            discriminant = linear * linear - length2 * constant
            if discriminant < 0:
                return None
            root = math.sqrt(discriminant)
            lower = max(lower, (-linear - root) / length2)
            upper = min(upper, (-linear + root) / length2)
            if lower > upper:
                return None
        preferred = ((current[0] - left[0]) * dx
                     + (current[1] - left[1]) * dy) / length2
        t = min(upper, max(lower, preferred))
        q = (left[0] + t * dx, left[1] + t * dy)
        return q if self.feasible(q) else None


@lru_cache(maxsize=7)
def _region(anchor):
    centers = responsibility_polygon(anchor)
    provisional = _DiskIntersection(centers, anchor, ())
    if not provisional.feasible(anchor):
        raise ValueError("Original anchor is not feasible with the safety margin")
    corners = []
    for i, a in enumerate(centers):
        for b in centers[i + 1:]:
            d = distance(a, b)
            if not 1e-9 < d <= 2 * _RADIUS:
                continue
            h = math.sqrt(max(0.0, _RADIUS * _RADIUS - d * d / 4))
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            for sign in (-1, 1):
                q = (mid[0] + sign * h * (b[1] - a[1]) / d,
                     mid[1] - sign * h * (b[0] - a[0]) / d)
                if provisional.feasible(q):
                    corners.append(q)
    return _DiskIntersection(centers, anchor, tuple(corners))


def _minimize_block(region, left, right, current):
    if right is None:
        return region.project(left)
    # On the segment between neighbours, the triangle inequality is exact.
    segment = region.segment_point(left, right, current)
    if segment is not None:
        return segment
    q = current
    for _ in range(24):
        dl, dr = distance(q, left), distance(q, right)
        if min(dl, dr) < 1e-10:
            break  # A nonsmooth focus: preserve the existing feasible point.
        # Constrained Weiszfeld/majorization step for two neighbour distances.
        # Projection minimizes a tight quadratic upper bound, so accepting
        # only a true distance improvement keeps the original objective safe.
        target = ((dr * left[0] + dl * right[0]) / (dl + dr),
                  (dr * left[1] + dl * right[1]) / (dl + dr))
        candidate = region.project(target)
        gain = dl + dr - distance(left, candidate) - distance(candidate, right)
        if gain <= 1e-7:
            break
        q = candidate
    return q


def optimize_tail(points, robot):
    """Return one safe position for each input COVER anchor, in the same order.

    Pure-Python convex block descent with exact disk-intersection projections.
    Iteration limits make this an improvement heuristic, not an asserted
    optimum of the convex program.  The original feasible route is retained
    and returned if the result is worse, nonfinite, or fails certification.
    No channel data, action ordering, or policy termination is modified.
    """
    anchors = tuple(_anchor_point(p) for p in points)
    robot = point(robot)
    if not all(math.isfinite(x) for x in robot):
        raise ValueError("Robot position must be finite")
    if not anchors:
        return ()
    regions = tuple(_region(a) for a in anchors)
    result = list(anchors)
    original_length = open_length(anchors, robot)
    for sweep in range(24):
        previous_length = open_length(result, robot)
        indices = range(len(result)) if sweep % 2 == 0 else reversed(range(len(result)))
        for j in indices:
            left = robot if j == 0 else result[j - 1]
            right = result[j + 1] if j + 1 < len(result) else None
            result[j] = _minimize_block(regions[j], left, right, result[j])
        if previous_length - open_length(result, robot) < 1e-5:
            break
    answer = tuple(result)
    if (open_length(answer, robot) > original_length
            or any(not certifies_cell(q, a) for q, a in zip(answer, anchors))):
        return anchors
    return answer
