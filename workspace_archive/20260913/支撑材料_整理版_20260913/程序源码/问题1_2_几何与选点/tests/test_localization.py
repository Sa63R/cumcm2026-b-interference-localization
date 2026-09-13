"""Conservatism, reception certificates, and executable observation-only choices."""

import math
import random

import pytest

from geometry import Circle, distance
from localization import (CandidateRegion, guaranteed_detection_region,
                          score_detection_point, second_point_candidates, select_next_point)


def _bearing(observer, source):
    return math.degrees(math.atan2(source[1] - observer[1], source[0] - observer[0])) % 360


def _region_with_vertices(vertices):
    region = CandidateRegion()
    region.vertices = tuple(vertices)
    return region


def test_successive_extreme_errors_keep_truth_and_shrink_region():
    rng = random.Random(317)
    for _ in range(30):
        theta, radius = rng.uniform(0, 2 * math.pi), rng.uniform(0, 1800)
        source = (radius * math.cos(theta), radius * math.sin(theta))
        region = CandidateRegion()
        previous_area, previous_radius = region.area, region.enclosing_disk().radius
        for index in range(6):
            theta, radius = rng.uniform(0, 2 * math.pi), rng.uniform(10, 1500)
            observer = (source[0] + radius * math.cos(theta), source[1] + radius * math.sin(theta))
            reported = round((_bearing(observer, source) + (-1 if index % 2 else 1)) % 360, 2) % 360
            region.observe(observer, reported)
            assert region.contains(source)
            assert region.enclosing_disk().contains(source)
            assert region.area <= previous_area + 1e-6
            assert region.enclosing_disk().radius <= previous_radius + 1e-6
            previous_area, previous_radius = region.area, region.enclosing_disk().radius


def test_wraparound_and_reception_boundary_are_conservative():
    region = CandidateRegion().observe((0, 0), 359.99)
    assert region.contains((1500, 0))
    assert region.contains((1499, 0))
    assert not region.contains((-100, 0))
    assert not region.contains((1600, 0))
    assert region.approximation_excess_m == pytest.approx(1800 * (1 / math.cos(math.pi / 128) - 1))


def test_copy_and_circle_cache_invalidate_independently():
    original = CandidateRegion().observe((0, 0), 0)
    cached = original.enclosing_disk()
    duplicate = original.copy()
    duplicate.observe((600, 500), 270)
    assert len(original.observations) == 1
    assert len(duplicate.observations) == 2
    assert original.enclosing_disk() is cached
    assert duplicate.enclosing_disk().radius < cached.radius
    assert original.vertices != duplicate.vertices


def test_inconsistent_measurements_fail_closed():
    region = CandidateRegion().observe((0, 0), 0).observe((-100, 0), 180)
    assert region.status == "empty"
    assert region.diameter is None
    assert not region.contains((0, 0))
    with pytest.raises(ValueError):
        region.enclosing_disk()
    assert guaranteed_detection_region(region).status == "inconsistent_source_region"
    assert select_next_point(region, (0, 0)) is None


def test_point_and_segment_membership_use_meter_tolerance():
    region = _region_with_vertices(((0, 0), (1000, 0)))
    assert region.contains((500, 0))
    assert not region.contains((500, .001), tolerance=1e-7)
    assert not region.contains((1001, 0))
    region.vertices = ((10, 20),)
    assert region.contains((10, 20))
    assert not region.contains((10.001, 20))


def test_guaranteed_region_nonempty_iff_mec_fits_and_vertices_are_safe():
    region = _region_with_vertices(((-600, -200), (600, -200), (600, 200), (-600, 200)))
    safe = guaranteed_detection_region(region)
    assert safe.status == "nonempty"
    assert safe.witness is not None and safe.contains(safe.witness)
    assert safe.vertices
    assert all(safe.contains(p) for p in safe.vertices)
    assert not safe.contains((900, 0))
    # Every convex combination of source vertices also lies within 1000 m.
    for candidate in safe.vertices[::3]:
        for weight in (0, .1, .5, .9, 1):
            source = (1200 * weight - 600, 400 * weight - 200)
            assert distance(candidate, source) <= 1000 + 1e-7
    huge = _region_with_vertices(((-1001, 0), (1001, 0)))
    assert guaranteed_detection_region(huge).status == "empty"


def test_empty_inner_polygon_does_not_mean_empty_exact_safe_region():
    endpoint = (600.0, 800.0)  # exact 1000 m norm, away from polygon vertices
    region = _region_with_vertices((endpoint, (-endpoint[0], -endpoint[1])))
    # The exact intersection of two externally tangent disks is {origin},
    # while the fixed-angle inscribed polygons do not reach that tangency.
    safe = guaranteed_detection_region(region, sides=8)
    assert safe.status == "nonempty"
    assert safe.vertices == ()
    assert safe.contains(safe.witness)
    assert distance(safe.witness, (0, 0)) < 1e-7


def test_second_point_candidates_are_safe_fresh_and_deterministic():
    region = CandidateRegion().observe((0, 0), 0)
    candidates = second_point_candidates(region, (0, 0))
    assert len(candidates) > 2
    assert candidates == second_point_candidates(region, (0, 0))
    assert all(distance(p, (0, 0)) >= 25 for p in candidates)
    assert all(distance(p, source) <= 1000 + 1e-7 for p in candidates for source in region.vertices)
    choice = select_next_point(region, (0, 0))
    assert choice.position in candidates
    assert choice.guaranteed_reception
    assert choice.score_kind == "sampled_worst_linearized_error_plus_travel"
    assert choice.score == min(score_detection_point(region, p, (0, 0)).score for p in candidates)
    assert abs(choice.position[1]) > 50  # chosen point is transverse to first ray
    assert choice == select_next_point(region, (0, 0))


def test_selected_point_reduces_region_in_fixed_independent_example():
    source = (1000, 0)
    region = CandidateRegion().observe((0, 0), 0)
    choice = select_next_point(region, (0, 0))
    selected = region.copy().observe(choice.position, _bearing(choice.position, source))
    collinear = region.copy().observe((500, 0), 0)
    assert selected.enclosing_disk().radius < collinear.enclosing_disk().radius
    assert selected.contains(source)


def test_no_observation_or_unavailable_safe_region_returns_no_choice():
    assert select_next_point(CandidateRegion(), (0, 0)) is None
    region = CandidateRegion().observe((0, 0), 0)
    assert second_point_candidates(region, (0, 0), min_step=1e9) == ()
    region.vertices = ((-1200, 0), (1200, 0))
    region._circle = None
    assert select_next_point(region, (0, 0)) is None
    assert second_point_candidates(region, (0, 0), require_guaranteed=False)


@pytest.mark.parametrize("radius", [0, -1, math.inf, math.nan])
def test_reception_radius_validation(radius):
    with pytest.raises(ValueError):
        CandidateRegion(reception_radius=radius)
    with pytest.raises(ValueError):
        guaranteed_detection_region(CandidateRegion(), radius)


@pytest.mark.parametrize("min_step", [-1, math.inf, math.nan])
def test_minimum_step_validation(min_step):
    with pytest.raises(ValueError):
        second_point_candidates(CandidateRegion(), (0, 0), min_step=min_step)


def test_score_requires_observation_and_valid_travel_weight():
    with pytest.raises(ValueError):
        score_detection_point(CandidateRegion(), (500, 100), (0, 0))
    region = CandidateRegion().observe((0, 0), 0)
    for weight in (-1, math.inf, math.nan):
        with pytest.raises(ValueError):
            score_detection_point(region, (500, 100), (0, 0), weight)
