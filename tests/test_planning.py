"""Geometric coverage checks include exact boundaries and degenerate sets."""

import math
import random
import itertools
from collections import Counter

import pytest

from geometry import convex_hull
from planning import clearance_grid, coverage_points, improve_open_route, nearest_order
from planning.coverage import omni_coverage_points
from planning.routing import exact_open_route
from simulator_client.state import Position


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


def _open_route_length(points, start=(0.0, 0.0)):
    coordinates = [Position.coerce(start)] + [Position.coerce(p) for p in points]
    return sum(a.distance_to(b) for a, b in zip(coordinates, coordinates[1:]))


@pytest.mark.parametrize("radius", [1123, 1150, 1200, 1500, 900 * math.sqrt(3), 1732])
def test_tunable_omni_cover_attains_the_proved_worst_distance(radius):
    points = omni_coverage_points(radius)
    assert len(points) == len(set(points)) == 7
    assert points[0] == Position(0, 0)
    assert all(math.hypot(p.x, p.y) == pytest.approx(radius) for p in points[1:])
    rho = max(radius / math.sqrt(3),
              math.sqrt(1800**2 + radius**2 - math.sqrt(3) * 1800 * radius))
    assert rho < 1000 - .019
    # Independent extrema: at each sector bisector, inspect the outer arena
    # and the interior origin/ring Voronoi junction. One attains the bound.
    extreme_distances = []
    for degrees in range(30, 360, 60):
        for distance_from_origin in (1800, radius / math.sqrt(3)):
            position = Position(distance_from_origin * math.cos(math.radians(degrees)),
                                distance_from_origin * math.sin(math.radians(degrees)))
            extreme_distances.append(min(position.distance_to(p) for p in points))
    assert max(extreme_distances) == pytest.approx(rho, abs=1e-8)
    # Actual generated positions, not only the analytic formula, are checked
    # over the polar domain and exact angular sector boundaries.
    for r in [1800 * k / 25 for k in range(26)] + [radius / math.sqrt(3)]:
        for degrees in range(0, 360, 3):
            source = Position(r * math.cos(math.radians(degrees)), r * math.sin(math.radians(degrees)))
            assert min(source.distance_to(p) for p in points) <= rho + 1e-8
    assert _open_route_length(points) == pytest.approx(6 * radius)


def test_tunable_omni_default_and_engineering_tradeoff():
    assert omni_coverage_points() == omni_coverage_points(1200)
    expected = {1150: 988.5114204360128, 1200: 968.9015717043836}
    for radius, worst in expected.items():
        points = omni_coverage_points(radius)
        source = Position(1800 * math.cos(math.pi / 6), 1800 * math.sin(math.pi / 6))
        assert min(source.distance_to(p) for p in points) == pytest.approx(worst)
        assert (9000 - _open_route_length(points)) / 5 == pytest.approx(420 if radius == 1150 else 360)
    # This new entry point deliberately does not change legacy Q3 defaults.
    assert set(coverage_points(3)) == set(omni_coverage_points(1500))


@pytest.mark.parametrize("radius", [1122.999999, 1732.000001, 0, -1, 1800,
                                    True, False, "1200", None, math.nan, math.inf, -math.inf])
def test_tunable_omni_rejects_unsafe_or_invalid_radius(radius):
    with pytest.raises(ValueError):
        omni_coverage_points(radius)


def test_outer_boundary_check_alone_would_miss_an_internal_coverage_hole():
    radius = 1800
    outer_distance = math.sqrt(1800**2 + radius**2 - math.sqrt(3) * 1800 * radius)
    assert outer_distance < 1000
    assert radius / math.sqrt(3) > 1000
    with pytest.raises(ValueError):
        omni_coverage_points(radius)


def test_exact_open_route_matches_independent_permutations():
    rng = random.Random(918203)
    for count in range(7):
        for _ in range(5):
            points = [Position(rng.uniform(-1800, 1800), rng.uniform(-1800, 1800))
                      for _ in range(count)]
            start = Position(rng.uniform(-1800, 1800), rng.uniform(-1800, 1800))
            before = list(points)
            route = exact_open_route(points, start)
            optimum = min((_open_route_length(order, start) for order in itertools.permutations(points)),
                          default=0.0)
            assert _open_route_length(route, start) == pytest.approx(optimum, abs=1e-8)
            assert Counter(route) == Counter(points)
            assert points == before
            assert route == exact_open_route(reversed(points), start)


def test_exact_open_route_has_free_endpoint_and_fixed_arbitrary_start():
    points = [(0, 0), (100, 0), (200, 0)]
    start = (250, 0)
    route = exact_open_route(points, start)
    assert route == (Position(200, 0), Position(100, 0), Position(0, 0))
    assert _open_route_length(route, start) == 250
    assert exact_open_route([], start) == ()
    assert exact_open_route([(10, 20)], start) == (Position(10, 20),)


def test_exact_open_route_handles_duplicates_and_ties_deterministically():
    points = [(0, 0), (1, 0), (-1, 0), (0, 0)]
    reference = exact_open_route(points)
    for permutation in itertools.permutations(points):
        assert exact_open_route(permutation) == reference
    assert Counter(reference) == Counter(Position(*p) for p in points)
    assert _open_route_length(reference) == 3


def test_exact_open_route_six_ring_points_reaches_the_geometric_lower_bound():
    ring = omni_coverage_points(1200)[1:]
    route = exact_open_route(ring, (0, 0))
    assert set(route) == set(ring)
    assert _open_route_length(route) == pytest.approx(7200)


def test_exact_open_route_validates_size_and_coordinates():
    with pytest.raises(ValueError):
        exact_open_route([(i, 0) for i in range(7)])
    for points, start in [([(math.nan, 0)], (0, 0)), ([], (math.inf, 0)),
                          ([(True, 0)], (0, 0)), ([(0,)], (0, 0))]:
        with pytest.raises(ValueError):
            exact_open_route(points, start)
