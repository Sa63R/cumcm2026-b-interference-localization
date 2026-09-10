"""Exact cache invalidation under movement and new legal source observations."""

from localization.omni import OmniCandidateRegion
from research_rl.controller import DeepRLSearch, geometry_features, polygon_shape
from research_rl.joint_scan import JointScanRLSearch
from simulation import LocalResearchSimulator, random_scenario
from simulator_client.state import Position
from tests.test_strategy import ObservationOnlyClient


def controller_with_observed_region():
    simulator = LocalResearchSimulator(random_scenario(3, 100601))
    controller = JointScanRLSearch(ObservationOnlyClient(simulator.client()), lambda f,c,t: t)
    region = OmniCandidateRegion().observe(Position(0, 0), 0.0)
    controller.regions[1] = region
    controller.detected.add(1)
    controller.first_bearings[1] = 0.0
    controller.observed_positions[1] = {(0.0, 0.0)}
    return controller, region


def test_geometry_cache_matches_direct_values_after_positive_and_negative_updates():
    controller, region = controller_with_observed_region()
    point = Position(500, 100)
    for action in (None, "positive", "negative"):
        if action == "positive":
            region.observe(Position(500, 500), 315.0)
        elif action == "negative":
            region.observe_no_signal(Position(-1400, 0))
        assert region.vertices
        direct = geometry_features(region, point, polygon_shape(region.vertices), 0.0)
        first = controller._geometric_features(1, point)
        second = controller._geometric_features(1, point)
        assert list(first) == direct
        assert first is second
        assert controller._region_area(1) == region.area


def test_geometry_cache_invalidation_includes_error_and_first_bearing():
    controller, region = controller_with_observed_region()
    point = Position(500, 500)
    before = controller._geometric_features(1, point)
    region.error_deg = 0.5
    controller.first_bearings[1] = 15.0
    after = controller._geometric_features(1, point)
    direct = geometry_features(region, point, polygon_shape(region.vertices), 15.0)
    assert list(after) == direct
    assert after != before


def test_source_candidate_cache_tracks_move_and_observed_coordinate_changes():
    controller, region = controller_with_observed_region()
    for action in (None, "move", "observed", "limit", "near"):
        if action == "move":
            controller.client.state.position = Position(200, 300)
        elif action == "observed":
            candidate = DeepRLSearch._source_candidates(controller, 1)[0]
            controller.observed_positions[1].add((round(candidate.point.x, 6), round(candidate.point.y, 6)))
        elif action == "limit":
            controller.probe_counts[1] = controller.max_active_probes
        elif action == "near":
            controller.near_points[1] = Position(7, 8)
        direct = DeepRLSearch._source_candidates(controller, 1)
        first = controller._source_candidates(1)
        second = controller._source_candidates(1)
        assert list(first) == direct
        assert first is second


def test_policy_feature_mutation_does_not_poison_geometry_cache():
    controller, region = controller_with_observed_region()
    point = Position(500, 100)
    geometry = controller._geometric_features(1, point)
    copied_row = [1.0, 2.0] + list(geometry)
    copied_row[2:] = [99.0] * len(geometry)
    assert list(controller._geometric_features(1, point)) == geometry_features(region, point, polygon_shape(region.vertices), 0.0)
