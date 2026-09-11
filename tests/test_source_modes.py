"""Nominal service is fully charged and cannot manufacture a live certificate."""

from dataclasses import asdict
import copy
import math

import pytest

from geometry import distance
from localization.omni import OmniCandidateRegion
from planning.source_modes import predict_source_mode
from simulation import LocalResearchSimulator
from simulation.cases import Scenario, Source
from simulator_client.state import Position


def strip(x=1000., half_width=100., half_height=2.):
    region = OmniCandidateRegion().observe((0., 0.), 0.)
    region.vertices = ((x-half_width, -half_height), (x+half_width, -half_height),
                       (x+half_width, half_height), (x-half_width, half_height))
    region._circle = None
    return region


def test_near_feedback_charges_one_measure_one_switch_and_one_clear():
    region = strip(half_width=.5, half_height=.5)
    result = predict_source_mode(region, (1000., 0.), 0.)
    assert result.valid and result.hypotheses == 3
    assert result.cost_s == 11.
    assert result.measurements_mean == 1.
    assert result.exit_position == Position(1000., 0.)
    for support in result.support_results:
        assert support.reason == "near"
        assert [a.action for a in support.actions] == ["measure", "clear"]
        assert [a.cost_s for a in support.actions] == [6., 5.]
        assert sum(a.movement_s for a in support.actions) == 0.


def test_cost_includes_complete_continuation_and_closest_certified_exit():
    result = predict_source_mode(strip(), (1000., 100.), 0., max_probes=6)
    assert result.valid
    assert result.hypotheses == 3
    for support in result.support_results:
        actions = support.actions
        assert actions[-1].action == "clear"
        assert support.cost_s == sum(a.cost_s for a in actions)
        assert support.cost_s >= 11. + max(0., Position(1000., 100.).distance_to(
            support.source_position)-20.)/5 - 1e-8
        assert sum(a.switching_s for a in actions) == 1.
        assert sum(a.measurement_s for a in actions) == 5.*support.measurements
        assert actions[-1].optical_s == 3. and actions[-1].removal_s == 2.
        replay = strip()
        previous = Position(1000., 100.)
        for action in actions:
            assert action.movement_s == pytest.approx(previous.distance_to(action.position)/5)
            if action.action == "measure" and action.result == "direction":
                replay.observe(action.position, action.bearing_deg)
            previous = action.position
        assert all(distance((actions[-1].position.x, actions[-1].position.y), v)
                   <= 19.9+1e-7 for v in replay.vertices)
        circle = replay.enclosing_disk()
        before_clear = actions[-2].position
        assert actions[-1].movement_s == pytest.approx(max(0.,
            before_clear.distance_to(Position(*circle.center))-(19.9-circle.radius))/5)
    assert result.cost_s == pytest.approx(sum(s.cost_s for s in result.support_results)/3)
    assert result.exit_position.x == pytest.approx(sum(s.exit_position.x for s in result.support_results)/3)


@pytest.mark.parametrize("supports", [3, 5])
def test_live_geometry_histories_negative_set_and_empty_mec_cache_never_mutate(supports):
    region = OmniCandidateRegion().observe((0., 0.), 0.).observe_no_signal((-800., 0.))
    region._circle = None
    before = copy.deepcopy(region.__dict__)
    observed = frozenset({(1., 10.)})
    result = predict_source_mode(region, (850., 150.), 0., observed, supports=supports)
    assert result.valid
    assert region.__dict__ == before
    assert region._circle is None
    assert observed == frozenset({(1., 10.)})
    assert 0 < result.hypotheses <= supports
    for support in result.support_results:
        lower, upper = support.radius_interval_m
        assert 1000. <= lower <= upper <= 1500.
        assert all(support.source_position.distance_to(Position(*o.position)) <= upper
                   for o in region.observations)
        assert all(support.source_position.distance_to(Position(*p)) > lower
                   for p in region.no_signal_positions)


def test_same_radius_sign_compatibility_rejects_relaxed_but_impossible_nodes():
    # This artificial outer region has distance(source, negative)<1000, so no
    # allowed reception radius could explain that historical no_signal.
    region = strip()
    region.no_signal_positions.append((1000., 100.))
    region._no_signal_set.add((1000., 100.))
    result = predict_source_mode(region, (1000., 200.), 0.)
    assert not result.valid
    assert result.reason == "no_radius_and_history_compatible_support"
    assert math.isinf(result.cost_s) and result.exit_position is None


def test_failed_nominal_budget_is_not_a_cheap_terminal():
    region = OmniCandidateRegion().observe((0., 0.), 0.)
    result = predict_source_mode(region, (750., 0.), 0., max_probes=1)
    assert not result.valid and math.isinf(result.cost_s)
    assert result.exit_position is None
    assert result.hypotheses > 0 and result.measurements_mean == 1.
    assert any(not s.valid for s in result.support_results)
    assert all(math.isinf(s.cost_s) for s in result.support_results if not s.valid)
    assert "probe_budget_exhausted_without_certificate" in result.reason


@pytest.mark.parametrize("kind, expected", [
    ("empty", "empty_region"), ("prior", "positive_bearing_history_required"),
    ("far_probe", "first_probe_not_guaranteed_reception"),
    ("repeat", "first_probe_already_observed"),
    ("outside", "no_radius_and_history_compatible_support"),
    ("nan_vertices", "invalid_observation_geometry"),
    ("zero_budget", "no_probe_budget"), ("nan_bearing", "invalid_first_bearing"),
])
def test_invalid_regions_and_unusable_entry_are_explicit(kind, expected):
    region, entry, kwargs = strip(), (1000., 100.), {}
    if kind == "empty":
        region.vertices = ()
    elif kind == "prior":
        region = OmniCandidateRegion()
    elif kind == "far_probe":
        entry = (-1000., 0.)
    elif kind == "repeat":
        kwargs["observed_positions"] = {(1000., 100.)}
    elif kind == "outside":
        region = strip(2000., 10., 1.)
        entry = (2000., 100.)
    elif kind == "nan_vertices":
        region.vertices = ((math.nan, 0.),)
    elif kind == "zero_budget":
        kwargs["max_probes"] = 0
    elif kind == "nan_bearing":
        kwargs["first_bearing_deg"] = math.nan
    result = predict_source_mode(region, entry, kwargs.pop("first_bearing_deg", 0.), **kwargs)
    assert not result.valid and result.reason == expected
    assert math.isinf(result.cost_s) and result.exit_position is None


@pytest.mark.parametrize("kwargs", [{"max_probes": True}, {"max_probes": -1},
                                     {"supports": 4}, {"uncertainty_weight": -.5},
                                     {"uncertainty_weight": math.inf}])
def test_invalid_configuration_does_not_start_prediction(kwargs):
    with pytest.raises(ValueError):
        predict_source_mode(strip(), (1000., 100.), 0., **kwargs)


def test_nominal_same_channel_oracle_replay_matches_every_charge_and_feedback():
    # These are tiny explicitly constructed oracle checks, not a training or
    # benchmark batch. Nine other channels merely satisfy simulator case size;
    # only channel 2 is queried and it cannot benefit from shared observations.
    region = OmniCandidateRegion().observe((0., 0.), 0.)
    first = Position(750., 150.)
    result = predict_source_mode(region, first, 0.)
    assert result.valid and result.measurements_mean > 1.
    for index, support in enumerate(result.support_results):
        radius = sum(support.radius_interval_m)/2
        target = Source(2, support.source_position.x, support.source_position.y, radius)
        others = tuple(Source(channel, 0., 0., 1000.) for channel in range(3, 12))
        scenario = Scenario(f"source-mode-oracle-{index}", 3, 100121, (target,)+others, "zero")
        simulator = LocalResearchSimulator(scenario)
        client = simulator.client()
        client.enter()
        for action in support.actions:
            response = (client.measure(action.position, 2) if action.action == "measure"
                        else client.clear(action.position, 2))
            if action.action == "measure":
                assert response["measure_result"] == action.result
                if action.result == "direction":
                    assert response["svd_deg"] == action.bearing_deg
            else:
                assert response["clear_result"] == "success"
        client.exit()
        oracle = simulator.evaluation()
        # Real first travel is deliberately excluded by the mode contract;
        # simulator and predictor both include entry switch 1->2 exactly once.
        assert oracle["virtual_time_s"]-Position(0., 0.).distance_to(first)/5 == pytest.approx(
            support.cost_s, abs=1e-5)
        assert oracle["failed_clear_count"] == 0
        assert oracle["cleared_channels"] == [2]
        assert all(a["channel"] in (None, 2) for a in simulator.observation_history())


def test_repeated_prediction_is_deterministic_including_support_action_traces():
    region = OmniCandidateRegion().observe((0., 0.), 0.)
    first = predict_source_mode(region, (750., 150.), 0.)
    second = predict_source_mode(region, (750., 150.), 0.)
    assert asdict(first) == asdict(second)
