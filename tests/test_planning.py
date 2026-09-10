"""Geometric coverage checks include exact boundaries and degenerate sets."""

import math
import random

import pytest

from geometry import convex_hull
from planning import clearance_grid, coverage_points, improve_open_route, nearest_order


def test_omnidirectional_seven_points_cover_the_entire_disk():
    points = coverage_points(3)
    assert len(points) == len(set(points)) == 7
    assert (points[0].x, points[0].y) == (0, 0)
    exact_bound = math.sqrt(1800**2 + 1500**2 - 2 * 1800 * 1500 * math.cos(math.pi / 6))
    assert exact_bound < 902 < 1000
    # Sector edges and the centre/ring Voronoi vertex are the extrema used
    # in the proof, plus a polar mesh checking the actual generated plan.
    radii = [0, 1500 / math.sqrt(3), 1000, 1800]
    radii += [1800 * i / 30 for i in range(31)]
    for radius in radii:
        for degree in range(0, 360, 2):
            x, y = radius * math.cos(math.radians(degree)), radius * math.sin(math.radians(degree))
            assert min(math.hypot(x - p.x, y - p.y) for p in points) <= exact_bound + 1e-8
    boundary_midpoint = (1800 * math.cos(math.pi / 6), 1800 * math.sin(math.pi / 6))
    assert min(math.hypot(boundary_midpoint[0] - p.x, boundary_midpoint[1] - p.y)
               for p in points) == pytest.approx(exact_bound)


def test_directional_grid_retains_all_vertices_of_intersecting_cells():
    points = coverage_points(4, variant="baseline")
    coordinates = {(p.x, p.y) for p in points}
    assert len(points) == len(coordinates) == 45
    assert max(math.hypot(p.x, p.y) for p in points) > 1800
    for i in range(-3, 3):
        for j in range(-3, 3):
            nearest_x = max(650 * i, min(0, 650 * (i + 1)))
            nearest_y = max(650 * j, min(0, 650 * (j + 1)))
            if math.hypot(nearest_x, nearest_y) <= 1800:
                assert {(650 * i, 650 * j), (650 * (i + 1), 650 * j),
                        (650 * i, 650 * (j + 1)),
                        (650 * (i + 1), 650 * (j + 1))} <= coordinates


@pytest.mark.parametrize("variant,maximum", [("adaptive", 650 * math.sqrt(2)),
                                            ("triangular", 990.0)])
def test_directional_cover_includes_outward_sources_and_halfplane_edges(variant, maximum):
    points = coverage_points(4, variant=variant)
    rng = random.Random(20260910)
    sources = [(1800 * math.cos(math.radians(angle)), 1800 * math.sin(math.radians(angle)))
               for angle in range(0, 360, 3)]
    sources += [(650 * i, 650 * j) for i in range(-2, 3) for j in range(-2, 3)]
    for _ in range(100):
        radius, angle = 1800 * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
        sources.append((radius * math.cos(angle), radius * math.sin(angle)))
    for x, y in sources:
        for degree in range(0, 360, 5):
            nx, ny = math.cos(math.radians(degree)), math.sin(math.radians(degree))
            visible = [p for p in points if nx * (p.x - x) + ny * (p.y - y) >= -1e-9]
            assert min(math.hypot(p.x - x, p.y - y) for p in visible) <= maximum + 1e-8


def test_triangular_cover_contains_the_cell_for_every_sampled_source():
    points = coverage_points(4, variant="triangular")
    spacing, height = 990.0, 990 * math.sqrt(3) / 2
    indices = {(round(p.x / spacing - p.y / (2 * height)), round(p.y / height)) for p in points}
    assert len(points) == len(indices) == 31
    assert max(math.hypot(p.x, p.y) for p in points) > 1800
    # Independently invert lattice coordinates, pick the containing triangle,
    # and check all three vertices, including sources on the target boundary.
    for radius in (0, 1, 990, 1500, 1799.999, 1800):
        for degree in range(360):
            x, y = radius * math.cos(math.radians(degree)), radius * math.sin(math.radians(degree))
            v = y / height
            u = x / spacing - v / 2
            i, j = math.floor(u), math.floor(v)
            if u - i + v - j <= 1:
                triangle = ((i, j), (i + 1, j), (i, j + 1))
            else:
                triangle = ((i + 1, j + 1), (i, j + 1), (i + 1, j))
            assert set(triangle) <= indices


def test_open_two_opt_preserves_points_start_and_never_increases_distance():
    def length(route, start):
        coordinates = [start] + [(p.x, p.y) for p in route]
        return sum(math.dist(a, b) for a, b in zip(coordinates, coordinates[1:]))

    rng = random.Random(6)
    for count in (0, 1, 2, 8, 31, 45):
        points = [(rng.uniform(-2000, 2000), rng.uniform(-2000, 2000)) for _ in range(count)]
        start = (170, 260)
        original = nearest_order(points, start)
        improved = improve_open_route(original, start)
        assert len(improved) == len(original)
        assert set(improved) == set(original)
        assert length(improved, start) <= length(original, start) + 1e-8
        assert improve_open_route(improved, start) == improved
    deferred = coverage_points(4, variant="deferred")
    adaptive = coverage_points(4, variant="adaptive")
    assert set(deferred) == set(adaptive)
    assert length(deferred, (0, 0)) < length(adaptive, (0, 0))


@pytest.mark.parametrize("bearing", [0, 1.005, 30, 89.999, 180, 359.999])
def test_optical_cover_contains_sampled_polygon_boundary_and_interior(bearing):
    vertices = convex_hull([(-170, -11), (-80, -30), (450, 16), (100, 46)])
    points = clearance_grid(vertices, bearing_deg=bearing, start=(250, -50))
    assert points and len(points) == len(set(points))
    sampled = list(vertices)
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        sampled.extend(((1 - t / 100) * a[0] + t / 100 * b[0],
                        (1 - t / 100) * a[1] + t / 100 * b[1]) for t in range(101))
    rng = random.Random(32)
    for _ in range(100):
        weights = [rng.random() for _ in vertices]
        total = sum(weights)
        sampled.append(tuple(sum(w * v[k] for w, v in zip(weights, vertices)) / total for k in (0, 1)))
    for x, y in sampled:
        assert min(math.hypot(p.x - x, p.y - y) for p in points) <= 28 / math.sqrt(2) + 1e-8


@pytest.mark.parametrize("vertices", [[], [(0, 0)], [(28, 28)], [(-28, -28)],
                                       [(0, 0), (280, 0)], [(0, 0), (0, -280)],
                                       [(-50, -50), (50, 50)]])
def test_optical_cover_keeps_points_segments_and_grid_boundaries(vertices):
    points = clearance_grid(vertices)
    if not vertices:
        assert points == ()
        return
    a, b = vertices[0], vertices[-1]
    for i in range(101):
        x, y = (a[k] + (b[k] - a[k]) * i / 100 for k in (0, 1))
        assert min(math.hypot(p.x - x, p.y - y) for p in points) < 20


def test_ordering_changes_only_route_not_cover():
    for problem in (3, 4):
        baseline = coverage_points(problem, variant="baseline")
        adaptive = coverage_points(problem, variant="adaptive")
        assert set(baseline) == set(adaptive)
        assert adaptive == coverage_points(problem, variant="improved")
        assert adaptive == nearest_order(baseline)
    assert [(p.x, p.y) for p in nearest_order([(1, 0), (-1, 0), (0, 2)])] == [(-1, 0), (1, 0), (0, 2)]


@pytest.mark.parametrize("spacing", [0, -1, 20 * math.sqrt(2), 29, math.inf, math.nan])
def test_optical_spacing_must_preserve_strict_twenty_metre_margin(spacing):
    with pytest.raises(ValueError):
        clearance_grid([(0, 0)], spacing=spacing)


def test_invalid_plan_parameters_are_rejected():
    with pytest.raises(ValueError):
        coverage_points(2)
    with pytest.raises(ValueError):
        coverage_points(3, variant="unknown")
    with pytest.raises(ValueError):
        clearance_grid([(0, 0)], bearing_deg=math.inf)
