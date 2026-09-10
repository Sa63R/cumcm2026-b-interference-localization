"""End-to-end observation-only policies, completion invariants and budgets."""

import pytest

from simulation import LocalResearchSimulator, difficult_scenarios, random_scenario
from simulator_client.errors import SimulatorError
from strategies import run_search


class ObservationOnlyClient:
    """A restricted interface that fails if a policy requests hidden truth."""

    __slots__ = ("_client",)

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        if name not in {"state", "remaining_real_time_s", "pending_request",
                        "enter", "measure", "clear", "exit"}:
            raise AssertionError(f"Policy requested non-observation attribute {name}")
        return getattr(self._client, name)


@pytest.mark.parametrize("problem,variant,policy", [
    (3, "baseline", "center"), (3, "adaptive", "center"),
    (3, "adaptive", "minimax"), (4, "baseline", "center"),
    (4, "adaptive", "center"), (4, "deferred", "center"),
    (4, "triangular", "center"),
])
def test_boundary_sources_are_all_cleared_without_truth_access(problem, variant, policy):
    sim = LocalResearchSimulator(difficult_scenarios(problem)[0])
    report = run_search(ObservationOnlyClient(sim.client()), problem=problem,
                        variant=variant, active_policy=policy)
    evaluation = sim.evaluation()  # Truth is accessed only after /exit.
    assert evaluation["all_cleared"]
    assert report.completion_certified_under_model
    assert report.coverage_complete
    assert report.coverage_points_visited == report.coverage_points_total
    assert report.unresolved_channels == []
    assert report.cleared_count == evaluation["cleared_total"]
    assert report.virtual_time_s == evaluation["virtual_time_s"]
    assert report.accepted_actions == evaluation["action_count"]
    assert sum(report.time_breakdown.values()) == pytest.approx(report.virtual_time_s, abs=1e-4)
    history = sim.observation_history()
    assert history[0]["action"] == "/enter"
    assert history[-1]["action"] == "/exit"
    # Each channel is scanned at every cover point, unless a prior successful
    # /clear has already established that its unique source is removed.
    clear_times = {item["channel"]: item["virtual_time_s"] for item in report.action_history
                   if item["action"] == "clear" and item["result"] == "success"}
    for position in report.coverage_points:
        scans = [item for item in report.action_history
                 if item["phase"] == "coverage" and item["position"] == position]
        scanned = {item["channel"] for item in scans}
        finish = max(item["virtual_time_s"] for item in scans)
        assert scanned | {channel for channel, when in clear_times.items() if when < finish} == set(range(1, 21))


@pytest.mark.parametrize("problem,variant", [(3, "adaptive"), (4, "triangular")])
def test_alternating_extreme_errors_do_not_exclude_real_sources(problem, variant):
    sim = LocalResearchSimulator(difficult_scenarios(problem)[4])
    result = run_search(ObservationOnlyClient(sim.client()), problem=problem, variant=variant)
    assert result.completion_certified_under_model
    assert sim.evaluation()["all_cleared"]


@pytest.mark.parametrize("max_actions", [2, 3, 10, 22])
def test_action_budget_reserves_explicit_exit_and_never_certifies_partial_work(max_actions):
    sim = LocalResearchSimulator(random_scenario(3, 2))
    report = run_search(sim.client(), max_actions=max_actions)
    assert report.completion_reason == "action_budget"
    assert not report.completion_certified_under_model
    assert not report.coverage_complete
    assert report.accepted_actions <= max_actions
    assert sim.observation_history()[-1]["action"] == "/exit"


def test_virtual_budget_exits_before_exceeding_limit():
    sim = LocalResearchSimulator(random_scenario(3, 5), max_virtual_duration_s=10)
    report = run_search(sim.client())
    assert report.completion_reason == "virtual_budget"
    assert report.virtual_time_s < 10
    assert not report.completion_certified_under_model
    assert sim.observation_history()[-1]["action"] == "/exit"


def test_real_deadline_reserve_exits_without_measurement():
    sim = LocalResearchSimulator(random_scenario(3, 5), max_real_duration_s=1.5)
    report = run_search(sim.client())
    assert report.completion_reason == "real_deadline"
    assert report.measurement_count == 0
    assert not report.completion_certified_under_model
    assert sim.observation_history()[-1]["action"] == "/exit"


def test_all_no_signal_is_model_inconsistency_not_vacuous_success():
    sim = LocalResearchSimulator(random_scenario(3, 0))
    client = sim.client()
    measure = client.measure

    def broken_measure(position, channel):
        response = measure(position, channel)
        return {key: value for key, value in dict(response, measure_result="no_signal").items()
                if key != "svd_deg"}

    client.measure = broken_measure
    report = run_search(client)
    assert report.coverage_complete
    assert report.completion_reason == "source_count_inconsistent"
    assert not report.completion_certified_under_model
    assert report.as_dict()["all_cleared"] is False
    assert report.as_dict()["mean_localization_clearance_time_s"] is None


def test_failed_clear_is_never_counted_as_success():
    sim = LocalResearchSimulator(random_scenario(3, 3))
    client = sim.client()
    original_clear = client.clear

    def failed_clear(position, channel):
        return dict(original_clear(position, channel), clear_result="no_target_in_range")

    client.clear = failed_clear
    report = run_search(client, variant="baseline")
    assert report.clear_attempt_count > 0
    assert report.cleared_count == 0
    assert report.unresolved_channels
    assert report.completion_reason == "unresolved_source"
    assert not report.completion_certified_under_model


def test_clear_does_not_change_radio_channel_and_cleared_channels_are_not_scanned():
    sim = LocalResearchSimulator(random_scenario(3, 8))
    report = run_search(sim.client())
    current_channel, cleared, switches = 1, set(), 0
    for action in report.action_history:
        if action["action"] == "measure":
            assert action["channel"] not in cleared
            switches += action["channel"] != current_channel
            current_channel = action["channel"]
        elif action["result"] == "success":
            cleared.add(action["channel"])
    assert report.time_breakdown["switching_s"] == switches


def test_deferred_variants_complete_shared_scans_before_localization():
    for variant in ("deferred", "triangular"):
        sim = LocalResearchSimulator(random_scenario(4, 7))
        report = run_search(sim.client(), problem=4, variant=variant)
        phases = [item["phase"] for item in report.action_history]
        first_localization = next(i for i, phase in enumerate(phases) if phase != "coverage")
        assert "coverage" not in phases[first_localization:]
        assert report.measurement_count >= 20 * report.coverage_points_total
        assert report.completion_certified_under_model


def test_protocol_error_exits_when_no_request_outcome_is_pending():
    sim = LocalResearchSimulator(random_scenario(3, 9))
    client = sim.client()

    def fail(*args):
        raise SimulatorError("injected transport failure")

    client.measure = fail
    report = run_search(client)
    assert report.completion_reason == "protocol_error"
    assert not report.completion_certified_under_model
    assert sim.observation_history()[-1]["action"] == "/exit"


def test_pending_unknown_request_prevents_a_fresh_exit():
    sim = LocalResearchSimulator(random_scenario(3, 9))
    client = sim.client()

    def fail(*args):
        client._pending = object()
        raise SimulatorError("outcome is unknown")

    client.measure = fail
    report = run_search(client)
    assert report.completion_reason == "protocol_error"
    assert sim.observation_history()[-1]["action"] == "/enter"
    sim.finish_for_evaluation()


@pytest.mark.parametrize("kwargs", [dict(problem=2), dict(variant="missing"),
                                    dict(max_actions=True), dict(max_actions=1),
                                    dict(max_active_probes=-1), dict(max_active_probes=31),
                                    dict(max_active_probes=1.5), dict(active_policy="truth"),
                                    dict(problem=3, variant="triangular")])
def test_invalid_strategy_options_fail_before_entering(kwargs):
    sim = LocalResearchSimulator(random_scenario(3, 0))
    with pytest.raises(ValueError):
        run_search(sim.client(), **kwargs)
    assert sim.observation_history() == []
