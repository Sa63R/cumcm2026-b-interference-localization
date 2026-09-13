import math

import pytest

from planning.clear_region import nearest_clear_point, nearest_near_clear_point
from simulator_client.state import Position


@pytest.mark.parametrize("query,expected", [((0, 10), (0, 4)), ((10, 0), (2, 0)),
                                            ((0, 3), (0, 3))])
def test_analytic_lens_projection(query, expected):
    p = nearest_clear_point([(-3, 0), (3, 0)], query, radius=5)
    assert (p.x, p.y) == pytest.approx(expected, abs=1e-7)
    assert max(p.distance_to(Position(-3, 0)), p.distance_to(Position(3, 0))) <= 5


def test_tangent_and_duplicate_constraints():
    p = nearest_clear_point([(-5, 0), (5, 0), (-5, 0)], (0, 10), radius=5)
    assert p == Position(0, 0)
    assert nearest_clear_point([(-6, 0), (6, 0)], (0, 10), radius=5) is None
    assert nearest_clear_point([], (0, 0)) is None


def test_third_disk_filters_otherwise_valid_lens_tip():
    p = nearest_clear_point([(-3, 0), (3, 0), (0, -3)], (0, 10), radius=5)
    assert (p.x, p.y) == pytest.approx((0, 2), abs=1e-7)


def test_rigid_transform_equivariance():
    for angle in (0.17, 1.4, 3.1):
        def transform(p):
            x, y = p
            return (1137 + x * math.cos(angle) - y * math.sin(angle),
                    -720 + x * math.sin(angle) + y * math.cos(angle))
        p = nearest_clear_point([transform((-3, 0)), transform((3, 0))],
                                transform((0, 10)), radius=5)
        assert (p.x, p.y) == pytest.approx(transform((0, 4)), abs=1e-6)


def test_near_clear_bound_includes_opposite_extreme_source():
    p = nearest_near_clear_point((0, 0), (30, 0))
    assert (p.x, p.y) == pytest.approx((14.9, 0), abs=1e-7)
    assert p.distance_to(Position(-5, 0)) <= 19.9
    assert nearest_near_clear_point((0, 0), (4, 0)) == Position(4, 0)


def test_nearest_point_is_no_farther_than_verified_old_construction():
    vertices = [(-3, 0), (3, 0)]
    current = Position(0, 10)
    old = Position(0, 2)
    p = nearest_clear_point(vertices, current, radius=5, fallback=old)
    assert current.distance_to(p) < current.distance_to(old) - 1.9


@pytest.mark.parametrize("radius", [0, -1, float('nan'), float('inf'), True])
def test_invalid_radius_rejected(radius):
    with pytest.raises(ValueError):
        nearest_clear_point([(0, 0)], (0, 0), radius=radius)
