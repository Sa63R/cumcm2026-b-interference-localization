"""Q3-only convex inference from legal, independently constructed responses."""

import math
import random

import pytest

from localization import CandidateRegion, select_next_point
from localization.omni import OmniCandidateRegion
from simulator_client.state import Position


def bearing(observer, source, error=0.0):
    angle = math.degrees(math.atan2(source[1] - observer[1], source[0] - observer[0]))
    return round((angle + error) % 360, 2) % 360


def apply_events(region, events):
    for event in events:
        if event[0] == "direction":
            region.observe(event[1], event[2])
        else:
            region.observe_no_signal(event[1])
    return region


def test_positive_negative_bisector_has_the_correct_sign_and_keeps_boundary():
    region = OmniCandidateRegion().observe((0, 0), 0)
    previous_radius = region.enclosing_disk().radius
    assert region.observe_no_signal((2000, 0)) is region
    # |s| <= R < |s-(2000,0)| implies x < 1000; the safe closure is x <= 1000.
    assert max(x for x, _ in region.vertices) == pytest.approx(1000)
    assert region.contains((500, 0))
    assert region.contains((1000, 0))
    assert not region.contains((1001, 0))
    assert region.enclosing_disk().radius < previous_radius


def test_negative_before_first_positive_is_retained_without_guessing_absence():
    region = OmniCandidateRegion()
    original_vertices = region.vertices
    assert region.observe_no_signal((2000, 0)) is region
    assert region.vertices == original_vertices
    assert region.observations == []
    assert region.no_signal_positions == [(2000.0, 0.0)]
    region.observe((0, 0), 0)
    after_positive = OmniCandidateRegion().observe((0, 0), 0).observe_no_signal((2000, 0))
    assert region.vertices == after_positive.vertices


def test_unknown_radius_does_not_justify_a_1500_m_negative_exclusion():
    # A valid R=1000 source is visible at distance 900 and silent at 1200.
    source = (0, 0)
    region = OmniCandidateRegion().observe((-900, 0), 0)
    region.observe_no_signal((1200, 0))
    assert math.dist(source, (1200, 0)) < 1500
    assert region.contains(source)


def test_positive_range_information_can_be_stronger_than_minimum_radius_alone():
    # These observations are valid for s=(0,0), R=1450. The joint comparison
    # gives x <= 30, despite not being told R or a measured distance.
    region = OmniCandidateRegion().observe((-1400, 0), 0)
    previous_area = region.area
    region.observe_no_signal((1460, 0))
    assert max(x for x, _ in region.vertices) == pytest.approx(30)
    assert region.contains((0, 0))
    assert region.area < previous_area


@pytest.mark.parametrize("radius", [1000.0, 1000.000001, 1234.5, 1500.0])
def test_extreme_bearing_errors_and_unknown_radii_keep_true_sources(radius):
    rng = random.Random(4026)
    for case in range(12):
        angle, radial = rng.uniform(0, 2 * math.pi), 1800 * math.sqrt(rng.random())
        source = (radial * math.cos(angle), radial * math.sin(angle))
        events = []
        for index, fraction in enumerate((0.01, 0.4, 0.85, 1.0)):
            direction = rng.uniform(0, 2 * math.pi)
            distance = radius * fraction
            observer = (source[0] + distance * math.cos(direction),
                        source[1] + distance * math.sin(direction))
            events.append(("direction", observer,
                           bearing(observer, source, -1 if index % 2 else 1)))
        for margin in (0.001, 1.0, 300.0, 900.0):
            direction = rng.uniform(0, 2 * math.pi)
            distance = radius + margin
            observer = (source[0] + distance * math.cos(direction),
                        source[1] + distance * math.sin(direction))
            assert math.dist(source, observer) > radius
            events.append(("no_signal", observer))
        rng.shuffle(events)
        region = OmniCandidateRegion()
        previous_area = region.area
        for event in events:
            apply_events(region, [event])
            assert region.contains(source), (radius, case, event)
            assert region.enclosing_disk().contains(source)
            assert region.area <= previous_area + 1e-6
            previous_area = region.area


def test_multiple_observation_orders_give_equivalent_regions():
    source = (400, 0)
    events = [("direction", (0, 0), bearing((0, 0), source, 1)),
              ("direction", (-800, 0), bearing((-800, 0), source, -1)),
              ("direction", (0, -800), bearing((0, -800), source, 1)),
              ("no_signal", (2000, 0)),
              ("no_signal", (1200, 1000)),
              ("no_signal", (1200, -1000))]
    positive = [event for event in events if event[0] == "direction"]
    negative = [event for event in events if event[0] == "no_signal"]
    orders = [events, list(reversed(events)), negative + positive,
              [events[i] for i in (3, 0, 4, 1, 5, 2)]]
    regions = [apply_events(OmniCandidateRegion(), sequence) for sequence in orders]
    first = regions[0]
    for region in regions:
        assert region.contains(source)
        assert len(region.observations) == 3
        assert len(region.no_signal_positions) == 3
        assert region.area == pytest.approx(first.area, abs=1e-6)
        assert region.enclosing_disk().radius == pytest.approx(first.enclosing_disk().radius, abs=1e-6)
        assert all(first.contains(vertex, tolerance=1e-6) for vertex in region.vertices)
        assert all(region.contains(vertex, tolerance=1e-6) for vertex in first.vertices)


def test_copy_preserves_subtype_and_independently_invalidates_cached_circle():
    original = OmniCandidateRegion().observe((0, 0), 0)
    original.observe_no_signal((-2000, 0))
    cached = original.enclosing_disk()
    duplicate = original.copy()
    assert isinstance(duplicate, OmniCandidateRegion)
    assert duplicate.enclosing_disk() is cached
    duplicate.observe_no_signal((2000, 0))
    assert original.enclosing_disk() is cached
    assert duplicate.enclosing_disk().radius < cached.radius
    assert original.no_signal_positions == [(-2000.0, 0.0)]
    assert duplicate.no_signal_positions == [(-2000.0, 0.0), (2000.0, 0.0)]
    duplicate.observe((0, -400), 45)
    assert len(original.observations) == 1
    assert len(duplicate.observations) == 2
    original.observe_no_signal((2000, 0))
    assert (2000.0, 0.0) in original.no_signal_positions  # Copies do not share a dedup set.


def test_repeated_negative_coordinate_is_idempotent_and_accepts_position():
    region = OmniCandidateRegion().observe((0, 0), 0)
    region.observe_no_signal(Position(2000, 0))
    vertices, circle = region.vertices, region.enclosing_disk()
    region.observe_no_signal((2000, 0))
    assert region.vertices == vertices
    assert region.enclosing_disk() is circle
    assert region.no_signal_positions == [(2000.0, 0.0)]


@pytest.mark.parametrize("negative_first", [False, True])
def test_same_point_positive_and_negative_is_an_explicit_contradiction(negative_first):
    region = OmniCandidateRegion()
    events = [("direction", (0, 0), 0), ("no_signal", (0, 0))]
    apply_events(region, list(reversed(events)) if negative_first else events)
    assert region.status == "empty"
    assert region.diameter is None
    assert not region.contains((100, 0))
    assert select_next_point(region, (0, 0)) is None
    with pytest.raises(ValueError):
        region.enclosing_disk()


def test_conflicting_bearing_and_distance_information_can_empty_the_region():
    # These bearings alone admit the origin. Silence at their midpoint adds
    # x<=-50 from the first positive and x>=50 from the second: impossible.
    region = OmniCandidateRegion().observe((-100, 0), 0).observe((100, 0), 180)
    assert region.contains((0, 0))
    region.observe_no_signal((0, 0))
    assert region.status == "empty"
    copied = region.copy().observe_no_signal((1000, 0))
    assert copied.status == "empty"


def test_distance_comparison_is_q3_only_and_cannot_use_directional_shadow():
    # This is a deliberate Q4 counterexample, not valid input for this class:
    # a source at the origin facing east is visible at (500,0), but shadowed
    # at (-100,0), even though the shadowed point is closer.
    base = CandidateRegion().observe((500, 0), 180)
    assert base.contains((0, 0))
    inappropriate = OmniCandidateRegion().observe((500, 0), 180)
    inappropriate.observe_no_signal((-100, 0))
    assert not inappropriate.contains((0, 0))


@pytest.mark.parametrize("position", [(math.nan, 0), (0, math.inf), (1, 2, 3)])
def test_invalid_negative_positions_fail_before_mutating_observations(position):
    region = OmniCandidateRegion().observe((0, 0), 0)
    vertices = region.vertices
    with pytest.raises(ValueError):
        region.observe_no_signal(position)
    assert region.no_signal_positions == []
    assert region.vertices == vertices
