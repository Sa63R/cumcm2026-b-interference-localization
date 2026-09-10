"""Truth-isolation, safety, coverage and fallback-time tests without PyTorch."""

import math
import random

import pytest

from research_rl import run_rl_search
from research_rl.controller import (CONTEXT_DIM, FEATURE_DIM, FEATURE_DIMS,
    DeepRLSearch, geometry_features, polygon_shape, GEOMETRY_FEATURE_NAMES)
from localization.omni import OmniCandidateRegion
from simulator_client.state import Position
from simulation import LocalResearchSimulator, Scenario, Source, difficult_scenarios, random_scenario
from tests.test_strategy import ObservationOnlyClient


@pytest.mark.parametrize("scenario", difficult_scenarios(3) + [random_scenario(3, s) for s in range(100001, 100006)], ids=lambda s: s.case_id)
@pytest.mark.parametrize("selector", ["teacher", "random"])
def test_observation_only_policy_completes_and_accounts_for_all_actions(scenario, selector):
    rng = random.Random(91)
    observed_shapes = []

    def policy(features, context, teacher):
        observed_shapes.append((len(features), len(context)))
        assert len(context) == CONTEXT_DIM
        assert all(len(row) == FEATURE_DIM for row in features)
        assert all(math.isfinite(x) for row in features for x in row)
        return teacher if selector == "teacher" else rng.randrange(len(features))

    simulator = LocalResearchSimulator(scenario)
    result = run_rl_search(ObservationOnlyClient(simulator.client()), policy=policy)
    evaluation = simulator.evaluation()
    assert evaluation["all_cleared"]
    assert evaluation["failed_clear_count"] == 0
    assert result.completion_certified_under_model
    assert result.virtual_time_s == evaluation["virtual_time_s"]
    assert result.accepted_actions == evaluation["action_count"]
    assert sum(result.time_breakdown.values()) == pytest.approx(result.virtual_time_s, abs=0.001)
    assert result.learning["decisions"] == len(observed_shapes)
    assert result.learning["decisions"] == sum(result.learning["action_counts"].values())
    assert simulator.observation_history()[-1]["action"] == "/exit"


def test_decision_limit_finishes_inherited_policy_and_charges_tail_time():
    simulator = LocalResearchSimulator(random_scenario(3, 100014))
    result = run_rl_search(ObservationOnlyClient(simulator.client()),
                           policy=lambda f, c, t: t, max_decisions=1)
    assert simulator.evaluation()["all_cleared"]
    assert result.learning["decisions"] == 1
    assert result.learning["fallback_counts"]["decision_limit"] == 1
    assert result.learning["fallback_virtual_time_s"] > 0
    assert result.learning["fallback_actions"] > 0


def test_final_sixteenth_clear_is_recorded_even_when_certificate_stops_session():
    sources = tuple(Source(c, math.cos(c), math.sin(c), 1000) for c in range(1, 17))
    simulator = LocalResearchSimulator(Scenario("rl-sixteen-near", 3, 100002, sources))
    costs = []
    search = DeepRLSearch(ObservationOnlyClient(simulator.client()), lambda f, c, t: t,
                          recorder=lambda *args: costs.append(args[-1]))
    result = search.run()
    assert simulator.evaluation()["all_cleared"]
    assert len(costs) == 16
    assert sum(costs) + result.learning["initial_scan_virtual_time_s"] == pytest.approx(result.virtual_time_s)


def test_early_action_limit_does_not_certify_success():
    simulator = LocalResearchSimulator(random_scenario(3, 100015))
    result = run_rl_search(simulator.client(), policy=lambda f, c, t: t, max_actions=22)
    assert not result.completion_certified_under_model
    assert result.completion_reason == "action_budget"
    assert simulator.observation_history()[-1]["action"] == "/exit"


def test_q4_rejected_before_session_entry():
    simulator = LocalResearchSimulator(random_scenario(4, 100017))
    with pytest.raises(ValueError, match="Q3 only"):
        run_rl_search(simulator.client(), problem=4, policy=lambda f, c, t: t)
    assert simulator.observation_history() == []


def test_invalid_policy_index_is_rejected_and_session_exits():
    simulator = LocalResearchSimulator(random_scenario(3, 100018))
    with pytest.raises(ValueError, match="invalid candidate"):
        run_rl_search(simulator.client(), policy=lambda f, c, t: len(f))
    assert simulator.observation_history()[-1]["action"] == "/exit"


def rectangle_region(half_length=500, half_width=10, rotation=0):
    region = OmniCandidateRegion()
    theta = math.radians(rotation)
    c, s = math.cos(theta), math.sin(theta)
    region.vertices = tuple((x * c - y * s, x * s + y * c)
        for x, y in [(-half_length, -half_width), (half_length, -half_width),
                     (half_length, half_width), (-half_length, half_width)])
    return region


def test_geometry_reception_certificate_equals_maximum_vertex_distance():
    region = rectangle_region()
    shape = polygon_shape(region.vertices)
    for position in (Position(0, 0), Position(0, 900), Position(-1000, 0)):
        values = dict(zip(GEOMETRY_FEATURE_NAMES, geometry_features(region, position, shape, 0)))
        maximum = max(math.hypot(x - position.x, y - position.y) for x, y in region.vertices)
        assert values["max_vertex_distance"] * 3600 == pytest.approx(maximum)
        assert values["guaranteed_reception"] == float(maximum <= 1000)
        # Independently sample convex combinations: the vertex certificate
        # must bound distances everywhere inside the conservative polygon.
        for a in (0, 0.3, 0.7, 1):
            for b in (0, 0.4, 1):
                distance = math.hypot(-500 + a * 1000 - position.x,
                                      -10 + b * 20 - position.y)
                assert distance <= maximum + 1e-9


def test_cross_bearing_proxy_recognizes_long_thin_uncertainty():
    region = rectangle_region()
    shape = polygon_shape(region.vertices)
    along = dict(zip(GEOMETRY_FEATURE_NAMES, geometry_features(region, Position(-1000, 0), shape, 0)))
    across = dict(zip(GEOMETRY_FEATURE_NAMES, geometry_features(region, Position(0, 500), shape, 0)))
    assert shape["major"] == pytest.approx(1000)
    assert shape["minor"] == pytest.approx(20)
    assert across["posterior_trace_ratio_proxy"] < along["posterior_trace_ratio_proxy"]
    assert along["crossing_angle_sin"] == pytest.approx(0)
    assert across["crossing_angle_sin"] == pytest.approx(1)
    center = dict(zip(GEOMETRY_FEATURE_NAMES, geometry_features(region, Position(0, 0), shape, 0)))
    assert center["posterior_trace_ratio_proxy"] == 1
    assert center["posterior_area_ratio_proxy"] == 1


def test_shape_and_information_scalars_are_rotation_invariant():
    original = rectangle_region()
    rotated = rectangle_region(rotation=37)
    theta = math.radians(37)
    q = Position(-300, 700)
    rotated_q = Position(q.x * math.cos(theta) - q.y * math.sin(theta),
                         q.x * math.sin(theta) + q.y * math.cos(theta))
    before = dict(zip(GEOMETRY_FEATURE_NAMES, geometry_features(original, q, polygon_shape(original.vertices), 0)))
    after = dict(zip(GEOMETRY_FEATURE_NAMES, geometry_features(rotated, rotated_q, polygon_shape(rotated.vertices), 37)))
    for key in GEOMETRY_FEATURE_NAMES:
        if key not in {"axis_cos2", "axis_sin2"}:
            assert before[key] == pytest.approx(after[key], abs=1e-7)


def test_v1_feature_prefix_and_teacher_actions_remain_compatible():
    observations = {}
    reports = []
    for version in ("v1", "v2"):
        rows = []
        def policy(features, context, teacher):
            assert len(features[0]) == FEATURE_DIMS[version]
            rows.append(([row[:24] for row in features], context, teacher))
            return teacher
        simulator = LocalResearchSimulator(random_scenario(3, 100019))
        reports.append(run_rl_search(ObservationOnlyClient(simulator.client()),
                                      policy=policy, feature_version=version))
        observations[version] = rows
    assert observations["v1"] == observations["v2"]
    assert reports[0].virtual_time_s == reports[1].virtual_time_s
