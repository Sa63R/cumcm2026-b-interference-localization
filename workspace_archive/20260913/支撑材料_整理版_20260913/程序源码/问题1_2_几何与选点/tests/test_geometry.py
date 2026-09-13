"""Geometric edge cases and independent exhaustive small-set oracles."""

import itertools
import math
import random

import pytest

from geometry import (Circle, HalfPlane, bearing_halfplanes, clip_polygon, convex_hull,
                      disk_halfplanes, disk_polygon, distance, halfplane_intersection,
                      minimum_enclosing_circle, point, polygon_area, polygon_diameter)


def _brute_circle(points):
    """Enumerate the 1/2/3 support points; independent of incremental MEC."""
    candidates = [(p, 0.0) for p in points]
    for p, q in itertools.combinations(points, 2):
        center = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        candidates.append((center, distance(p, q) / 2))
    for p, q, r in itertools.combinations(points, 3):
        # Solve the two perpendicular-bisector equations by elimination.
        a, b = 2 * (q[0] - p[0]), 2 * (q[1] - p[1])
        c, d = 2 * (r[0] - p[0]), 2 * (r[1] - p[1])
        u = q[0] ** 2 + q[1] ** 2 - p[0] ** 2 - p[1] ** 2
        v = r[0] ** 2 + r[1] ** 2 - p[0] ** 2 - p[1] ** 2
        determinant = a * d - b * c
        if abs(determinant) > 1e-12:
            center = ((u * d - b * v) / determinant, (a * v - u * c) / determinant)
            candidates.append((center, distance(center, p)))
    return min((item for item in candidates if all(distance(item[0], p) <= item[1] + 1e-7
                                                  for p in points)), key=lambda item: item[1])


@pytest.mark.parametrize("bearing", [0, 0.01, 90, 179.99, 270, 359.99, -360, 720])
def test_bearing_wedge_orientation_wraparound_and_forward_ray(bearing):
    observer = (123.0, -87.0)
    planes = bearing_halfplanes(observer, bearing, 1.005)
    for error in (-1.005, -1.0, 0, 1.0, 1.005):
        angle = math.radians(bearing + error)
        source = (observer[0] + 500 * math.cos(angle), observer[1] + 500 * math.sin(angle))
        assert all(hp.contains(source) for hp in planes)
    for error in (-1.01, 1.01, 180):
        angle = math.radians(bearing + error)
        source = (observer[0] + 500 * math.cos(angle), observer[1] + 500 * math.sin(angle))
        assert not all(hp.contains(source) for hp in planes)


def test_halfplane_normalization_and_invalid_input():
    assert HalfPlane(30, 40, 100) == HalfPlane(.6, .8, 2)
    for coefficients in ((0, 0, 1), (math.inf, 0, 1), (1, 0, math.nan)):
        with pytest.raises(ValueError):
            HalfPlane(*coefficients)
    for coordinates in ((1,), (1, 2, 3), (math.inf, 0)):
        with pytest.raises(ValueError):
            point(coordinates)
    for error in (0, 90, math.nan):
        with pytest.raises(ValueError):
            bearing_halfplanes((0, 0), 0, error)


def test_halfplane_intersection_empty_unbounded_point_segment_and_area():
    empty = halfplane_intersection([HalfPlane(1, 0, 0), HalfPlane(-1, 0, -1)])
    assert (empty.status, empty.feasible_point, empty.diameter) == ("empty", None, None)
    for planes in ([], [HalfPlane(1, 0, 3)], [HalfPlane(1, 0, 1), HalfPlane(-1, 0, 0)],
                   [HalfPlane(1, 0, 0), HalfPlane(-1, 0, 0)]):
        result = halfplane_intersection(planes)
        assert result.status == "unbounded"
        assert result.diameter == math.inf
        assert all(hp.contains(result.feasible_point) for hp in planes)
    planes = [HalfPlane(1, 0, 0), HalfPlane(-1, 0, 0), HalfPlane(0, 1, 2), HalfPlane(0, -1, 1)]
    line = halfplane_intersection(planes)
    assert (line.status, line.dimension, line.diameter) == ("bounded", 1, 3)
    singleton = halfplane_intersection(planes + [HalfPlane(0, 1, -1)])
    assert (singleton.status, singleton.dimension, singleton.diameter) == ("bounded", 0, 0)
    square = halfplane_intersection([HalfPlane(1, 0, 1), HalfPlane(-1, 0, 0),
                                    HalfPlane(0, 1, 1), HalfPlane(0, -1, 0)])
    assert (square.status, square.dimension) == ("bounded", 2)
    assert polygon_area(square.vertices) == pytest.approx(1)
    assert square.diameter == pytest.approx(math.sqrt(2))


def test_almost_parallel_boundaries_and_small_full_dimensional_region():
    result = halfplane_intersection([HalfPlane(-1, 0, 0), HalfPlane(1, 0, 1000),
                                    HalfPlane(0, -1, 0), HalfPlane(-1e-10, 1, 1)])
    assert result.status == "bounded"
    assert result.dimension == 2
    assert max(p[1] for p in result.vertices) == pytest.approx(1.0000001)
    tiny = halfplane_intersection([HalfPlane(-1, 0, 0), HalfPlane(0, -1, 0),
                                  HalfPlane(1, 1, .0001)])
    assert tiny.dimension == 2
    assert polygon_area(tiny.vertices) == pytest.approx(5e-9)


def test_clipping_degeneracy_and_translated_area():
    square = ((0, 0), (2, 0), (2, 2), (0, 2))
    segment = clip_polygon(square, HalfPlane(1, 0, 0))
    assert set(segment) == {(0, 0), (0, 2)}
    assert clip_polygon(segment, HalfPlane(0, 1, 0)) == ((0.0, 0.0),)
    assert clip_polygon(segment, HalfPlane(0, 1, -1)) == ()
    shifted = tuple((x + 2_000_000, y - 2_000_000) for x, y in square)
    assert polygon_area(shifted) == 4
    assert polygon_diameter(shifted) == pytest.approx(math.sqrt(8))


def test_hull_diameter_and_mec_degenerate_sets():
    points = [(3, 4), (0, 0), (6, 8), (3, 4)]
    assert convex_hull(points) == ((0.0, 0.0), (6.0, 8.0))
    assert polygon_diameter(points) == 10
    assert minimum_enclosing_circle(points) == Circle((3, 4), 5)
    assert minimum_enclosing_circle([(3, 4)]) == Circle((3, 4), 0)
    assert polygon_diameter([(3, 4)]) == 0
    for operation in (polygon_diameter, minimum_enclosing_circle):
        with pytest.raises(ValueError):
            operation([])


def test_diameter_circle_counterexample_also_invalidates_40m_rule():
    side = 36.0
    triangle = ((0, 0), (side, 0), (side / 2, side * math.sqrt(3) / 2))
    assert polygon_diameter(triangle) == pytest.approx(side)
    diameter_circle = Circle((side / 2, 0), side / 2)
    assert not diameter_circle.contains(triangle[2])
    mec = minimum_enclosing_circle(triangle)
    assert mec.radius == pytest.approx(side / math.sqrt(3))
    assert mec.radius > 20
    assert all(mec.contains(vertex) for vertex in triangle)


def test_counterexample_is_realizable_by_three_one_degree_bearing_wedges():
    source = (18, 6 * math.sqrt(3))
    planes = []
    for rotation in (0, 120, 240):
        angle = math.radians(rotation)
        dx, dy = -900 - source[0], -source[1]
        observer = (source[0] + dx * math.cos(angle) - dy * math.sin(angle),
                    source[1] + dx * math.sin(angle) + dy * math.cos(angle))
        planes.extend(bearing_halfplanes(observer, 1 + rotation, 1))
        assert distance(observer, source) < 1000
    result = halfplane_intersection(planes)
    assert result.status == "bounded"
    assert len(result.vertices) == 3
    assert result.diameter == pytest.approx(36)
    assert polygon_area(result.vertices) == pytest.approx(324 * math.sqrt(3))
    assert minimum_enclosing_circle(result.vertices).radius == pytest.approx(12 * math.sqrt(3))
    assert all(hp.contains(source) for hp in planes)


def test_diameter_calipers_and_mec_against_exhaustive_oracles():
    rng = random.Random(7412026)
    for count in range(3, 11):
        for _ in range(12):
            points = [(rng.uniform(-1500, 1500), rng.uniform(-1500, 1500)) for _ in range(count)]
            expected_diameter = max(distance(p, q) for p, q in itertools.combinations(points, 2))
            assert polygon_diameter(points) == pytest.approx(expected_diameter, abs=1e-7)
            expected_center, expected_radius = _brute_circle(points)
            result = minimum_enclosing_circle(points)
            assert result.radius == pytest.approx(expected_radius, abs=1e-7)
            assert distance(result.center, expected_center) < 1e-6
            assert all(result.contains(p) for p in points)


@pytest.mark.parametrize("sides", [8, 16, 64, 128])
def test_disk_approximations_are_outer_and_inner(sides):
    center, radius = (17, -29), 1000
    outer = disk_polygon(center, radius, sides)
    inner = disk_polygon(center, radius, sides, outer=False)
    assert all(distance(center, p) == pytest.approx(radius / math.cos(math.pi / sides)) for p in outer)
    assert all(distance(center, p) == pytest.approx(radius) for p in inner)
    for k in range(361):
        angle = math.radians(k)
        on_circle = (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))
        assert all(hp.contains(on_circle) for hp in disk_halfplanes(center, radius, sides))
    assert all(all(hp.contains(p) for hp in disk_halfplanes(center, radius, sides, outer=False)) for p in inner)


@pytest.mark.parametrize("sides", [True, 7, 8.5, "128"])
def test_disk_side_count_validation(sides):
    with pytest.raises(ValueError):
        disk_polygon((0, 0), 1000, sides)


@pytest.mark.parametrize("radius", [-1, math.inf, math.nan])
def test_circle_radius_validation(radius):
    with pytest.raises(ValueError):
        Circle((0, 0), radius)
