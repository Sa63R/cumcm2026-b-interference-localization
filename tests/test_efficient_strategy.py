"""Observation-only Q3 integration and independent completion/accounting checks."""

import math

import pytest

from simulation import LocalResearchSimulator, Scenario, Source, difficult_scenarios, random_scenario
from strategies import run_search
from tests.test_strategy import ObservationOnlyClient


def _run(scenario, **kwargs):
    sim = LocalResearchSimulator(scenario)
    report = run_search(ObservationOnlyClient(sim.client()), variant="efficient", **kwargs)
    return sim, report


def _assert_complete(sim, report):
    history = sim.observation_history()
    assert history[0]["action"] == "/enter"
    assert history[-1]["action"] == "/exit"
    # Ground truth is available to the evaluator only after the policy exits.
    evaluation = sim.evaluation()
    assert evaluation["all_cleared"]
    assert report.completion_certified_under_model
    assert report.as_dict()["all_cleared"] is True
    assert report.cleared_channels == evaluation["cleared_channels"]
    assert report.cleared_count == evaluation["source_total"]
    assert report.unresolved_channels == []
    assert report.error is None
    assert report.exit_error is None
    assert report.variant == "efficient"
    assert report.problem == 3
    assert report.accepted_actions == evaluation["action_count"] == len(history)
    assert report.measurement_count == evaluation["measurement_count"]
    assert report.virtual_time_s == evaluation["virtual_time_s"]
    assert report.time_breakdown == pytest.approx(evaluation["time_breakdown_s"], abs=1e-4)
    assert sum(report.time_breakdown.values()) == pytest.approx(report.virtual_time_s, abs=1e-4)

    if report.completion_reason == "source_count_upper_bound_reached":
        assert report.cleared_count == 16
        assert history[-2]["action"] == "/clear"
        assert history[-2]["response"]["clear_result"] == "success"
    else:
        assert report.completion_reason == "coverage_exhausted_and_all_detected_cleared"
        assert report.coverage_complete
        assert report.coverage_points_visited == report.coverage_points_total == 7
        # A full-coverage certificate must scan each still-live channel at
        # every guaranteed cover point. Earlier successful clears may replace
        # those measurements because each channel has at most one source.
        clear_times = {
            item["channel"]: item["virtual_time_s"]
            for item in report.action_history
            if item["action"] == "clear" and item["result"] == "success"
        }
        for position in report.coverage_points:
            scans = [item for item in report.action_history
                     if item["phase"] == "coverage" and item["position"] == position]
            assert scans
            finish = max(item["virtual_time_s"] for item in scans)
            scanned = {item["channel"] for item in scans}
            removed = {channel for channel, time in clear_times.items() if time < finish}
            assert scanned | removed == set(range(1, 21))


@pytest.mark.parametrize(
    "scenario", difficult_scenarios(3) + [random_scenario(3, seed) for seed in range(5)],
    ids=lambda scenario: scenario.case_id,
)
def test_difficult_and_random_cases_complete_without_hidden_truth(scenario):
    sim, report = _run(scenario)
    _assert_complete(sim, report)


def test_sixteen_near_origin_sources_stop_at_199_seconds_without_extra_actions():
    sources = tuple(Source(channel, math.cos(channel), math.sin(channel), 1000.0)
                    for channel in range(1, 17))
    scenario = Scenario("q3-sixteen-near-origin", 3, 0, sources, "zero")
    sim, report = _run(scenario)
    _assert_complete(sim, report)
    assert report.completion_reason == "source_count_upper_bound_reached"
    assert not report.coverage_complete
    assert report.coverage_points_visited == 1
    assert report.coverage_points_total == 7
    assert report.measurement_count == 20
    assert report.clear_attempt_count == report.cleared_count == 16
    assert report.accepted_actions == 38  # enter + 20 scans + 16 clears + exit
    assert report.virtual_time_s == 199.0  # 20*5 + 19 switches + 16*(3+2)
    assert report.time_breakdown["movement_s"] == 0
    assert all(item["position"] == [0.0, 0.0] for item in report.action_history)
    history = sim.observation_history()
    successes = [i for i, item in enumerate(history)
                 if item["response"].get("clear_result") == "success"]
    assert successes[-1] == len(history) - 2


@pytest.mark.parametrize("max_actions", [2, 3, 10, 22])
def test_action_budget_reserves_exit_and_never_certifies_partial_search(max_actions):
    sim, report = _run(random_scenario(3, 2), max_actions=max_actions)
    history = sim.observation_history()
    assert history[0]["action"] == "/enter"
    assert history[-1]["action"] == "/exit"
    assert report.accepted_actions == len(history) == max_actions
    assert report.completion_reason == "action_budget"
    assert not report.completion_certified_under_model
    assert not report.coverage_complete
    assert report.as_dict()["all_cleared"] is False


@pytest.mark.parametrize("config", [
    [], "route", True,
    {"unknown_option": 1},
    {"schedule": "unknown"}, {"schedule": []},
    {"ring_radius": 1122.0}, {"ring_radius": 1733.0},
    {"ring_radius": float("nan")}, {"ring_radius": float("inf")},
    {"ring_radius": True}, {"ring_radius": "1200"},
    {"use_negative": 1}, {"dynamic_coverage": "yes"}, {"edge_clear": 0},
    {"detour_limit_m": -1}, {"detour_limit_m": 10001},
    {"detour_limit_m": float("nan")}, {"detour_limit_m": True},
    {"speculative_radius": -1}, {"speculative_radius": 201},
    {"speculative_radius": float("inf")}, {"speculative_radius": True},
])
def test_invalid_efficient_configuration_is_rejected_before_enter(config):
    sim = LocalResearchSimulator(random_scenario(3, 0))
    with pytest.raises(ValueError):
        run_search(ObservationOnlyClient(sim.client()), variant="efficient", efficient_config=config)
    assert sim.observation_history() == []


@pytest.mark.parametrize("kwargs", [{"problem": 4}, {"active_policy": "minimax"}])
def test_unsupported_problem_or_policy_is_rejected_before_enter(kwargs):
    sim = LocalResearchSimulator(random_scenario(kwargs.get("problem", 3), 0))
    with pytest.raises(ValueError):
        run_search(ObservationOnlyClient(sim.client()), variant="efficient", **kwargs)
    assert sim.observation_history() == []


def test_all_no_signal_never_produces_a_vacuous_completion_certificate():
    sim = LocalResearchSimulator(random_scenario(3, 0))
    client = sim.client()
    measure = client.measure

    def broken_measure(position, channel):
        response = measure(position, channel)
        return {key: value for key, value in dict(response, measure_result="no_signal").items()
                if key != "svd_deg"}

    client.measure = broken_measure
    report = run_search(ObservationOnlyClient(client), variant="efficient")
    assert sim.observation_history()[-1]["action"] == "/exit"
    assert report.coverage_complete
    assert report.coverage_points_visited == 7
    assert report.completion_reason == "source_count_inconsistent"
    assert report.detected_channels == report.cleared_channels == []
    assert report.clear_attempt_count == 0
    assert not report.completion_certified_under_model
    assert report.as_dict()["all_cleared"] is False
    assert report.as_dict()["mean_localization_clearance_time_s"] is None
    assert not sim.evaluation()["all_cleared"]


def test_radio_channel_and_full_time_ledger_match_observed_actions():
    sim, report = _run(random_scenario(3, 8))
    _assert_complete(sim, report)
    current_channel, cleared, previous = 1, set(), (0.0, 0.0)
    costs = dict(movement_s=0.0, switching_s=0, detection_s=0, optical_s=0, removal_s=0)
    for action in report.action_history:
        position = action["position"]
        costs["movement_s"] += math.dist(previous, position) / 5.0
        previous = position
        assert 1 <= action["channel"] <= 20
        assert action["channel"] not in cleared
        if action["action"] == "measure":
            costs["switching_s"] += action["channel"] != current_channel
            current_channel = action["channel"]
            costs["detection_s"] += 5
        else:
            assert action["action"] == "clear"
            costs["optical_s"] += 3
            if action["result"] == "success":
                costs["removal_s"] += 2
                cleared.add(action["channel"])
        assert action["virtual_time_s"] == pytest.approx(sum(costs.values()), abs=1e-4)
    assert report.time_breakdown == pytest.approx(costs, abs=1e-4)


@pytest.mark.parametrize("schedule", ["immediate", "detour", "route"])
@pytest.mark.parametrize("use_negative", [True, False])
def test_schedule_and_negative_observation_options_preserve_completion(schedule, use_negative):
    config = {"schedule": schedule, "use_negative": use_negative}
    sim, report = _run(random_scenario(3, 7), efficient_config=config)
    _assert_complete(sim, report)
    assert report.strategy_parameters["schedule"] == schedule
    assert report.strategy_parameters["use_negative"] is use_negative


def test_seventeen_positive_channels_fail_before_upper_bound_certification():
    # Inject a syntactically valid response that contradicts the public source
    # maximum. Previously the strategy cleared 16 sources and certified while
    # leaving its seventeenth detected channel unresolved.
    sources = tuple(Source(channel, 0, 0, 1000) for channel in range(1, 17))
    scenario = Scenario("q3-inconsistent-seventeenth-response", 3, 0, sources, "zero")
    sim = LocalResearchSimulator(scenario)
    client = sim.client()
    original_measure = client.measure

    def inconsistent_measure(position, channel):
        response = original_measure(position, channel)
        return dict(response, measure_result="near") if channel == 17 else response

    client.measure = inconsistent_measure
    report = run_search(ObservationOnlyClient(client), variant="efficient")
    assert report.completion_reason == "source_count_inconsistent"
    assert report.error is not None
    assert report.detected_channels == list(range(1, 18))
    assert not report.completion_certified_under_model
    assert report.as_dict()["all_cleared"] is False
    assert report.clear_attempt_count == report.cleared_count == 0
    assert report.measurement_count == 17
    assert report.accepted_actions == 19  # enter + 17 accepted measurements + exit
    assert sim.observation_history()[-1]["action"] == "/exit"
    assert report.exit_error is None


def test_negative_only_and_near_only_priors_are_not_reported_as_source_estimates():
    sources = tuple(Source(channel, 0, 0, 1000) for channel in range(1, 11))
    sources += (Source(11, 500, 0, 1000),)
    scenario = Scenario("q3-estimate-evidence-filter", 3, 0, sources, "zero")
    sim, report = _run(scenario)
    _assert_complete(sim, report)
    responses = {}
    for action in report.action_history:
        if action["action"] == "measure":
            responses.setdefault(action["channel"], set()).add(action["result"])
    negative_only = {channel for channel, kinds in responses.items() if kinds == {"no_signal"}}
    near_only = {channel for channel, kinds in responses.items()
                 if "near" in kinds and "direction" not in kinds}
    assert negative_only == set(range(12, 21))
    assert near_only == set(range(1, 11))
    assert not negative_only.intersection(report.source_estimates)
    assert not near_only.intersection(report.source_estimates)
    assert set(report.source_estimates) == {11}  # Genuine bearing-based inference remains.
