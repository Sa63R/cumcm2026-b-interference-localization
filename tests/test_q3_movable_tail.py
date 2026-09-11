import math
import random

import pytest

from geometry import distance
from strategies.q3_fresh import COVER, MARGIN, covered
from strategies.q3_movable_tail import (
    POSITION_MARGIN, _region, certifies_cell, open_length, optimize_tail,
    responsibility_polygon,
)


def test_fixed_responsibility_polygons_are_covered_by_original_anchors():
    assert POSITION_MARGIN > MARGIN
    for anchor in COVER:
        vertices = responsibility_polygon(anchor)
        assert vertices
        assert max(distance(anchor, v) for v in vertices) < 1000 - POSITION_MARGIN
        assert certifies_cell(anchor, anchor)


def test_feasible_displacement_has_full_continuous_cover_certificate():
    anchors = COVER[1:] + COVER[:1]
    moved = optimize_tail(anchors, (0, 0))
    assert len(moved) == len(anchors)
    assert any(distance(q, anchor) > 10 for q, anchor in zip(moved, anchors))
    for q, anchor in zip(moved, anchors):
        assert certifies_cell(q, anchor)
        assert max(distance(q, v) for v in responsibility_polygon(anchor)) <= 1000 - MARGIN
    assert covered(moved)
    assert open_length(moved, (0, 0)) < open_length(anchors, (0, 0)) - 100


def test_empty_and_single_task_open_endpoint():
    assert optimize_tail([], (0, 0)) == ()
    anchor = COVER[1]
    nearest, = optimize_tail([anchor], (0, 0))
    assert distance(nearest, (0, 0)) < distance(anchor, (0, 0))
    assert open_length([nearest], (0, 0)) == distance(nearest, (0, 0))
    # Starting inside the feasible set should incur no travel or return leg.
    same, = optimize_tail([anchor], nearest)
    assert same == nearest


@pytest.mark.parametrize("seed", range(8))
def test_varied_subsets_and_orders_never_lengthen_route(seed):
    rng = random.Random(seed)
    anchors = rng.sample(COVER, 1 + seed % 7)
    angle = rng.random() * 2 * math.pi
    robot = (1750 * math.cos(angle), 1750 * math.sin(angle))
    moved = optimize_tail(anchors, robot)
    assert open_length(moved, robot) <= open_length(anchors, robot)
    assert all(certifies_cell(q, a) for q, a in zip(moved, anchors))
    assert moved == optimize_tail(anchors, robot)


def test_projection_satisfies_variational_inequality():
    # For a Euclidean projection q, (target-q).(x-q) <= 0 for every feasible x.
    for anchor in COVER:
        region = _region(anchor)
        for target in ((-2200, 1700), (1700, -2100), (0, 0)):
            q = region.project(target)
            assert region.feasible(q)
            for feasible in (anchor, *region.corners):
                dot = ((target[0] - q[0]) * (feasible[0] - q[0])
                       + (target[1] - q[1]) * (feasible[1] - q[1]))
                assert dot <= 1e-4


def test_noncover_tasks_and_nonfinite_robot_are_rejected():
    with pytest.raises(ValueError, match="COVER"):
        optimize_tail([(25, 10)], (0, 0))
    with pytest.raises(ValueError, match="finite"):
        optimize_tail([COVER[0]], (math.nan, 0))
    assert not certifies_cell((math.nan, 0), COVER[0])
    assert not certifies_cell((1800, 1800), COVER[0])
