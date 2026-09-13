"""Conservative bearing localization and observation-only active sensing."""

from __future__ import annotations

from dataclasses import dataclass
import math

from geometry import (Circle, HalfPlane, Point, bearing_halfplanes, clip_polygon,
                      disk_halfplanes, disk_polygon, distance, minimum_enclosing_circle,
                      point, polygon_area, polygon_diameter)


@dataclass(frozen=True)
class BearingObservation:
    position: Point
    bearing_deg: float
    error_deg: float = 1.005

    def __post_init__(self):
        object.__setattr__(self, "position", point(self.position))
        bearing_halfplanes(self.position, self.bearing_deg, self.error_deg)


class CandidateRegion:
    """An outer polygon containing every source consistent with observations.

    The true source is constrained to the radius-1800 arena and, after each
    positive bearing measurement, a radius-1500 disk around the observer.
    Circular constraints use circumscribed polygons, never inscribed ones.
    ``no_signal`` provides no update here: its meaning depends on source type.
    """

    def __init__(self, prior_radius: float = 1800.0, disk_sides: int = 128,
                 error_deg: float = 1.005, reception_radius: float = 1500.0):
        if not math.isfinite(error_deg) or not 0 < error_deg < 90:
            raise ValueError("error_deg must be in (0,90)")
        if not math.isfinite(reception_radius) or reception_radius <= 0:
            raise ValueError("reception_radius must be positive")
        self.prior_radius = prior_radius
        self.disk_sides = disk_sides
        self.error_deg = error_deg
        self.reception_radius = reception_radius
        self.vertices = disk_polygon((0.0, 0.0), prior_radius, disk_sides, outer=True)
        self.observations: list[BearingObservation] = []
        self._circle: Circle | None = None

    @property
    def status(self) -> str:
        return "bounded" if self.vertices else "empty"

    @property
    def area(self) -> float:
        return polygon_area(self.vertices)

    @property
    def diameter(self) -> float | None:
        return polygon_diameter(self.vertices) if self.vertices else None

    @property
    def approximation_excess_m(self) -> float:
        """Largest radial excess of an individual disk's outer polygon.

        This is not a Hausdorff error bound for an intersection of disks/wedges.
        """
        return max(self.prior_radius, self.reception_radius) * (1 / math.cos(math.pi / self.disk_sides) - 1)

    def copy(self) -> "CandidateRegion":
        other = object.__new__(CandidateRegion)
        other.__dict__ = dict(self.__dict__)
        other.observations = list(self.observations)
        return other

    def observe(self, position, bearing_deg: float) -> "CandidateRegion":
        observation = BearingObservation(point(position), bearing_deg, self.error_deg)
        constraints = bearing_halfplanes(observation.position, observation.bearing_deg, observation.error_deg)
        constraints += disk_halfplanes(observation.position, self.reception_radius, self.disk_sides, outer=True)
        vertices = self.vertices
        for hp in constraints:
            vertices = clip_polygon(vertices, hp)
            if not vertices:
                break
        self.vertices = vertices
        self.observations.append(observation)
        self._circle = None
        return self

    def enclosing_disk(self) -> Circle:
        if self._circle is None:
            self._circle = minimum_enclosing_circle(self.vertices)
        return self._circle

    def contains(self, value, tolerance: float = 1e-7) -> bool:
        value = point(value)
        if not self.vertices:
            return False
        if len(self.vertices) == 1:
            return distance(self.vertices[0], value) <= tolerance
        if len(self.vertices) == 2:
            a, b = self.vertices
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            if length == 0:
                return distance(a, value) <= tolerance
            along = ((value[0] - a[0]) * dx + (value[1] - a[1]) * dy) / length
            perpendicular = abs(dx * (value[1] - a[1]) - dy * (value[0] - a[0])) / length
            return -tolerance <= along <= length + tolerance and perpendicular <= tolerance
        for a, b in zip(self.vertices, self.vertices[1:] + self.vertices[:1]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            if dx * (value[1] - a[1]) - dy * (value[0] - a[0]) < -tolerance * math.hypot(dx, dy):
                return False
        return True

    def guaranteed_detection_region(self, radius: float = 1000.0, sides: int = 64) -> "DetectionRegion":
        return guaranteed_detection_region(self, radius, sides)


@dataclass(frozen=True)
class DetectionRegion:
    """Intersection of radius-r disks centred at every candidate-source vertex.

    ``vertices`` form a conservative inner polygon; ``witness`` belongs to the
    exact intersection when nonempty. Inner-polygon emptiness alone does not
    prove that the exact curved region is empty.
    """

    vertices: tuple[Point, ...]
    source_vertices: tuple[Point, ...]
    radius: float
    status: str  # empty, nonempty, or inconsistent_source_region
    witness: Point | None

    def contains(self, position, tolerance: float = 1e-7) -> bool:
        position = point(position)
        return bool(self.source_vertices) and all(distance(position, v) <= self.radius + tolerance
                                                 for v in self.source_vertices)


def guaranteed_detection_region(region: CandidateRegion, radius: float = 1000.0,
                                sides: int = 64) -> DetectionRegion:
    """Certify a second omni-source reception location without hidden truth.

    For a convex polygon C, max_{x in C}|q-x| is attained at a vertex. Hence
    intersection_{v in vertices(C)} B(v,r) is the exact safe set for C.
    It is nonempty iff the minimum enclosing circle radius of C is <= r.
    """
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("detection radius must be positive")
    if not region.vertices:
        return DetectionRegion((), (), radius, "inconsistent_source_region", None)
    mec = region.enclosing_disk()
    if mec.radius > radius:
        return DetectionRegion((), region.vertices, radius, "empty", None)
    inner = disk_polygon(region.vertices[0], radius, sides, outer=False)
    for vertex in region.vertices[1:]:
        for hp in disk_halfplanes(vertex, radius, sides, outer=False):
            inner = clip_polygon(inner, hp)
            if not inner:
                break
        if not inner:
            break
    return DetectionRegion(inner, region.vertices, radius, "nonempty", mec.center)


@dataclass(frozen=True)
class NextPointChoice:
    position: Point
    score: float
    predicted_radius_m: float
    movement_m: float
    guaranteed_reception: bool
    min_intersection_angle_deg: float
    scoring_hypotheses: int
    score_kind: str = "sampled_worst_linearized_error_plus_travel"


def _source_hypotheses(region: CandidateRegion, limit: int = 12) -> tuple[Point, ...]:
    vertices = region.vertices
    if len(vertices) > limit:
        selected = tuple(vertices[round(i * (len(vertices) - 1) / (limit - 1))] for i in range(limit))
    else:
        selected = vertices
    center = region.enclosing_disk().center
    return selected + (center,)


def score_detection_point(region: CandidateRegion, candidate, current_position,
                          movement_weight: float = 0.02) -> NextPointChoice:
    """Sampled minimax first-order error proxy + movement time penalty.

    Proxy = tan(eps)*(r1+r2)/max(sin(intersection_angle),0.02).
    This is a ranking heuristic, not a rigorous worst-case enclosure radius.
    It needs only observed bearings and the current feasible polygon; no source
    distribution is assumed and no ground-truth positions are accepted.
    """
    if not region.vertices or not region.observations:
        raise ValueError("one positive bearing and a nonempty region are required")
    if movement_weight < 0 or not math.isfinite(movement_weight):
        raise ValueError("movement_weight must be nonnegative")
    candidate, current_position = point(candidate), point(current_position)
    first = region.observations[0].position
    hypotheses = _source_hypotheses(region)
    worst, min_angle = 0.0, 90.0
    eps = math.tan(math.radians(region.error_deg))
    for source in hypotheses:
        r1, r2 = distance(first, source), distance(candidate, source)
        if r2 <= 5.0:
            uncertainty, angle = 5.0, 90.0
        elif r1 <= 1e-7:
            # The wedge apex is a conservative artefact; the actual first
            # response was direction, so a source within 5 m is excluded.
            continue
        else:
            determinant = abs((source[0] - first[0]) * (source[1] - candidate[1])
                              - (source[1] - first[1]) * (source[0] - candidate[0]))
            sine = min(1.0, determinant / (r1 * r2))
            angle = math.degrees(math.asin(sine))
            uncertainty = eps * (r1 + r2) / max(sine, 0.02)
        worst, min_angle = max(worst, uncertainty), min(min_angle, angle)
    movement = distance(candidate, current_position)
    return NextPointChoice(candidate, worst + movement_weight * movement / 5.0,
                           worst, movement,
                           all(distance(candidate, v) <= 1000.0 + 1e-7 for v in region.vertices),
                           min_angle, len(hypotheses))


def second_point_candidates(region: CandidateRegion, current_position, *,
                            min_step: float = 25.0, require_guaranteed: bool = True) -> tuple[Point, ...]:
    """Generate reproducible candidate locations from the first observation.

    Exact safe-region vertices are circular arcs; the returned candidates use
    a conservative inner polygon, its witness, and transverse offsets around
    the enclosing-circle centre. All retained safe candidates are independently
    checked against every source-region vertex.
    """
    if not math.isfinite(min_step) or min_step < 0:
        raise ValueError("min_step must be finite and nonnegative")
    if not region.observations or not region.vertices:
        return ()
    current_position = point(current_position)
    safe = guaranteed_detection_region(region)
    if require_guaranteed and safe.status != "nonempty":
        return ()
    center = region.enclosing_disk().center
    bearing = math.radians(region.observations[0].bearing_deg)
    forward = (math.cos(bearing), math.sin(bearing))
    transverse = (-forward[1], forward[0])
    candidates = [center]
    if safe.witness is not None:
        candidates.append(safe.witness)
    polygon = safe.vertices
    for i in range(min(16, len(polygon))):
        candidates.append(polygon[i * len(polygon) // min(16, len(polygon))])
    for step in (50.0, 150.0, 300.0, 500.0):
        for sign in (-1, 1):
            candidates.append((center[0] + sign * step * transverse[0],
                               center[1] + sign * step * transverse[1]))
    previous = tuple(observation.position for observation in region.observations)
    result = []
    for candidate in candidates:
        if distance(candidate, current_position) < min_step:
            continue
        if any(distance(candidate, old) < min_step for old in previous):
            continue
        if require_guaranteed and not safe.contains(candidate):
            continue
        if not any(distance(candidate, old) < 1e-5 for old in result):
            result.append(candidate)
    return tuple(result)


def select_next_point(region: CandidateRegion, current_position, *, min_step: float = 25.0,
                      movement_weight: float = 0.02, require_guaranteed: bool = True) -> NextPointChoice | None:
    candidates = second_point_candidates(region, current_position, min_step=min_step,
                                          require_guaranteed=require_guaranteed)
    if not candidates:
        return None
    return min((score_detection_point(region, candidate, current_position, movement_weight)
                for candidate in candidates), key=lambda choice: (choice.score, choice.movement_m, choice.position))


__all__ = ["BearingObservation", "CandidateRegion", "DetectionRegion", "NextPointChoice",
           "guaranteed_detection_region", "second_point_candidates", "score_detection_point", "select_next_point"]
