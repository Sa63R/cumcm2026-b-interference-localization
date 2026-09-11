"""The pair guarantee is tested directly as geometry, not via hidden-policy input."""

import copy
from dataclasses import asdict
import json
import math
import random

import pytest

from localization import CandidateRegion
from planning.directional_probe_pair import choose_directional_probe_pair, distance_to_convex_region
from simulator_client.state import Position


def bearing(observer, source):
    return math.degrees(math.atan2(source[1]-observer[1], source[0]-observer[0])) % 360


def dot(normal, p, source):
    return sum(normal[j]*(p[j]-source[j]) for j in (0, 1))


def validate_pair(pair, source, radius, normal):
    p = pair.anchor.x, pair.anchor.y
    r = math.dist(p, source)
    assert r <= radius+1e-8
    assert dot(normal, p, source) >= -1e-8
    q = [(point.x, point.y) for point in (pair.first, pair.second)]
    assert all(math.dist(point, source) <= r+1e-8
               and math.dist(point, source) <= radius+1e-8 for point in q)
    assert any(dot(normal, point, source) >= -1e-8 for point in q)
    # Independently reconstruct the shared ray/segment witness. Its position
    # comes from test truth solely to verify the theorem, never from the API.
    delta = math.radians((bearing(p, source)-pair.anchor_bearing_deg+180) % 360-180)
    t = pair.step_m*math.cos(math.pi/4)/math.cos(delta)
    assert 0 < t < pair.distance_lower_bound_m+1e-8 <= r+1e-8
    fraction = .5*(1+math.tan(delta)/math.tan(math.pi/4))
    assert 0 < fraction < 1


@pytest.mark.parametrize("radius", [1000., 1500.])
@pytest.mark.parametrize("relative_error", [-1.005, 0., 1.005])
@pytest.mark.parametrize("normal_offset", [-90., -89.999999, 0., 89.999999, 90.])
def test_bearing_radius_and_halfplane_boundaries(radius, relative_error, normal_offset):
    source = (123., -87.)
    true_angle = 359.8
    theta = math.radians(true_angle)
    anchor = (source[0]-radius*math.cos(theta), source[1]-radius*math.sin(theta))
    region = CandidateRegion().observe(anchor, (true_angle+relative_error) % 360)
    pair, log = choose_directional_probe_pair(region, (20., 40.), [anchor])
    assert pair is not None and log["required_available_measurements"] == 2
    n_angle = math.radians(true_angle+180+normal_offset)
    validate_pair(pair, source, radius, (math.cos(n_angle), math.sin(n_angle)))
    assert log["candidates"][log["selected"]]["radius_sufficient_condition_ratio"] == pytest.approx(.475)
    assert log["candidates"][log["selected"]]["ray_intersection_to_source_distance_ratio_upper"] < 1.


@pytest.mark.parametrize("distance", [5.+1e-10, 5.000001, 10., 999.999999])
def test_direction_response_near_boundary_does_not_use_false_zero_apex_distance(distance):
    anchor, source = (0., 0.), (distance, 0.)
    region = CandidateRegion().observe(anchor, 0.)
    assert region.contains(anchor)  # The conservative wedge retains a false apex.
    pair, _ = choose_directional_probe_pair(region, anchor, [anchor])
    assert pair is not None and pair.distance_lower_bound_m == 5.
    assert pair.first != pair.second and pair.first != Position(*anchor)
    for normal in ((-1., 0.), (0., 1.), (0., -1.)):
        validate_pair(pair, source, 1000., normal)


def test_random_legal_halfdisks_always_have_one_receiving_endpoint():
    rng = random.Random(923511)
    for _ in range(150):
        source = rng.uniform(-500, 500), rng.uniform(-500, 500)
        theta = rng.uniform(0, 360)
        r = rng.uniform(5.000001, 1500.)
        angle = math.radians(theta)
        anchor = source[0]-r*math.cos(angle), source[1]-r*math.sin(angle)
        error = rng.uniform(-1.005, 1.005)
        region = CandidateRegion().observe(anchor, (theta+error) % 360)
        pair, _ = choose_directional_probe_pair(region, (rng.uniform(-1000, 1000), 80.), [anchor])
        assert pair is not None
        n_angle = math.radians(theta+180+rng.uniform(-90, 90))
        validate_pair(pair, source, max(1000., r), (math.cos(n_angle), math.sin(n_angle)))


@pytest.mark.parametrize("position, vertices, expected", [
    ((0., 0.), ((3., 4.),), 5.),
    ((0., 0.), ((3., -2.), (3., 2.)), 3.),
    ((3., 0.), ((3., -2.), (3., 2.)), 0.),
    ((0., 0.), ((2., -1.), (4., -1.), (4., 1.), (2., 1.)), 2.),
    ((3., 0.), ((2., -1.), (4., -1.), (4., 1.), (2., 1.)), 0.),
    ((0., 0.), ((2., 1.), (4., 1.), (4., -1.), (2., -1.)), 2.),
    ((1., 1.), ((0., 0.), (2., 0.), (0., 2.)), 0.),
])
def test_point_segment_inside_boundary_and_reversed_polygon_distance(position, vertices, expected):
    assert distance_to_convex_region(position, vertices) == pytest.approx(expected)


@pytest.mark.parametrize("vertices, lower", [
    (((900., -5.), (1100., -5.), (1100., 5.), (900., 5.)), 900.),
    (((900., -5.), (900., 5.)), 900.),
    (((1000., 0.),), 1000.),
])
def test_observation_region_excludes_anchor_apex_and_raises_safe_step(vertices, lower):
    region = CandidateRegion().observe((0., 0.), 0.)
    region.vertices = vertices
    pair, log = choose_directional_probe_pair(region, (0., 0.), [(0., 0.)])
    assert pair is not None
    assert lower-1e-5 < pair.distance_lower_bound_m < lower
    assert pair.step_m > 500.
    for source in vertices:
        # The rectangular corners fit the genuine bearing wedge.
        theta = bearing((0., 0.), source)
        for offset in (-90., 0., 90.):
            n_theta = math.radians(theta+180+offset)
            validate_pair(pair, source, 1500., (math.cos(n_theta), math.sin(n_theta)))
    assert log["candidates"][0]["raw_region_distance_m"] == pytest.approx(lower)


def test_only_one_real_positive_produces_pair_but_first_can_be_silent():
    source, anchor = (100., 0.), (0., 0.)
    normal = (-.005, 1.)
    region = CandidateRegion().observe(anchor, 0.)
    old_center = region.enclosing_disk().center
    assert dot(normal, anchor, source) > 0.
    assert dot(normal, old_center, source) < 0.  # The old center is on the back side.
    pair, log = choose_directional_probe_pair(region, (0., -100.), [anchor])
    assert pair is not None and dot(normal, (pair.first.x, pair.first.y), source) < 0.
    assert dot(normal, (pair.second.x, pair.second.y), source) > 0.
    assert not log["first_individually_guaranteed"] and not log["clearance_guaranteed"]


def test_full_pair_must_be_fresh_even_if_observed_point_might_have_been_silent():
    region = CandidateRegion().observe((0., 0.), 0.)
    original, _ = choose_directional_probe_pair(region, (0., 0.), [(0., 0.)])
    for seen in (original.first, original.second):
        pair, log = choose_directional_probe_pair(region, (0., 0.), [(0., 0.), seen])
        assert pair is None
        assert log["candidates"][0]["reason"] == "pair_not_two_distinct_fresh_positions"


def test_nearest_four_anchors_complete_pair_ranking_and_no_live_mutation():
    source = (800., 200.)
    positives = [(0., 0.), (0., 100.), (0., 200.), (100., 0.), (100., 100.), (100., 200.)]
    region = CandidateRegion()
    for p in positives:
        region.observe(p, bearing(p, source))
    before = copy.deepcopy(region.__dict__)
    result1 = choose_directional_probe_pair(region, (40., 20.), positives)
    result2 = choose_directional_probe_pair(region, (40., 20.), positives)
    assert result1 == result2 and result1[0] is not None
    assert region.__dict__ == before and region._circle is None
    pair, log = result1
    assert log["anchors_examined"] == 4 and log["distinct_eligible_anchors"] == 6
    eligible = [c for c in log["candidates"] if c["valid"]]
    expected = min(eligible, key=lambda c:(c["complete_pair_cost_s"], *c["first"], *c["second"]))
    assert [pair.first.x, pair.first.y] == expected["first"]
    assert pair.complete_pair_cost_s == pytest.approx(10.+math.dist((40., 20.), expected["first"])/5
                                                   +math.dist(expected["first"], expected["second"])/5)
    json.dumps({"pair": asdict(pair), "log": log}, allow_nan=False)


def test_error_domain_and_empty_history_fail_closed():
    no_history = CandidateRegion()
    assert choose_directional_probe_pair(no_history, (0., 0.), [])[0] is None
    wide = CandidateRegion(error_deg=10.).observe((0., 0.), 0.)
    pair, log = choose_directional_probe_pair(wide, (0., 0.), [])
    assert pair is None and log["anchors_examined"] == 0
    empty = CandidateRegion().observe((0., 0.), 0.)
    empty.vertices = ()
    assert choose_directional_probe_pair(empty, (0., 0.), [])[0] is None
    with pytest.raises(ValueError):
        distance_to_convex_region((0., 0.), ())


@pytest.mark.parametrize("limit", [0, 5, True, 1.5])
def test_invalid_anchor_budget_rejected(limit):
    with pytest.raises(ValueError):
        choose_directional_probe_pair(CandidateRegion(), (0., 0.), [], limit)


def test_out_of_bounds_probe_is_rejected_without_clipping():
    # Deliberately artificial geometry exercises API-coordinate protection;
    # it does not purport to be a physically feasible arena-source history.
    from localization import BearingObservation
    region = CandidateRegion()
    region.observations = [BearingObservation((1_999_999., 0.), 0.)]
    region.vertices = ((1_999_999., 0.), (2_000_001., -.01), (2_000_001., .01))
    pair, log = choose_directional_probe_pair(region, (0., 0.), [])
    assert pair is None
    assert log["candidates"][0]["reason"] == "candidate_exceeds_coordinate_bound_no_clipping"
