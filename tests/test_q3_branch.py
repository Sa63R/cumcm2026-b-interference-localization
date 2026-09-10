"""Continuation tests use only constructed hypotheses and public observations."""

import copy
from dataclasses import asdict
import math
import time

import pytest

from simulation import LocalResearchSimulator, Scenario, Source
from simulation.q3_branch import make_q3_branch
from simulator_client.errors import DeadlineExceeded, SessionError
from simulator_client.state import SourceState


def scenario(*, error_mode="zero", problem=3):
    sources = (Source(1, 100, 0, 1000, 0 if problem == 4 else None),) + tuple(
        Source(channel, -1500, 0, 1200) for channel in range(2, 11))
    return Scenario("branch-unit-hypothesis", problem, 73, sources, error_mode)


def active(case=None, **kwargs):
    client = LocalResearchSimulator(case or scenario(), **kwargs).client()
    client.enter()
    return client


def record(client, history, action, position, channel):
    response = getattr(client, action)(position, channel)
    item = {"action": action, "position": list(position), "channel": channel,
            "phase": "unit", "virtual_time_s": response["virtual_time_s"],
            "result": response["measure_result" if action == "measure" else "clear_result"]}
    if item["result"] == "direction":
        item["bearing_deg"] = response["svd_deg"]
    history.append(item)
    return response


def test_branch_continues_all_five_costs_and_preserves_current_channel():
    client = active()
    history = []
    record(client, history, "measure", (300, 400), 1)
    record(client, history, "measure", (300, 400), 2)
    record(client, history, "clear", (300, 0), 3)
    assert client.state.virtual_time_s == 194
    branch = make_q3_branch(scenario(), client.state, history)
    assert branch.state.snapshot() | {"real_deadline": None} == client.state.snapshot() | {"real_deadline": None}
    assert branch.measure((300, 0), 2)["virtual_time_s"] == 199
    assert branch.clear((100, 0), 1)["virtual_time_s"] == 244
    assert branch.state.current_channel == 2
    assert asdict(branch.state.time_breakdown) == {
        "movement_s": 220, "switching_s": 1, "detection_s": 15,
        "optical_s": 6, "removal_s": 2}
    assert branch.state.accepted_actions == client.state.accepted_actions + 2
    assert branch.exit()["virtual_time_s"] == 244


def test_branches_and_inputs_are_independent():
    client = active()
    history = []
    record(client, history, "measure", (0, 0), 1)
    original_state = client.state.snapshot()
    original_history = copy.deepcopy(history)
    first = make_q3_branch(scenario(), client.state, history)
    second = make_q3_branch(scenario(), client.state, history)
    first.clear((100, 0), 1)
    first.state.sources[1].failed_clear_count += 99
    history[0]["bearing_deg"] = 20
    assert second.measure((0, 0), 1)["svd_deg"] == 0
    assert client.state.snapshot() == original_state
    assert second.state.sources[1].status == "detected"
    assert second.state.sources[1].failed_clear_count == 0
    assert original_history[0]["bearing_deg"] == 0
    first.exit()
    assert second.state.session == "active"


def test_historical_bearing_overrides_new_seed_error_but_only_at_exact_point():
    client = active()
    history = []
    record(client, history, "measure", (0, 0), 1)
    branch = make_q3_branch(scenario(error_mode="positive_extreme"), client.state, history)
    assert branch.measure((-0.0, 0.0), 1)["svd_deg"] == 0
    assert branch.measure((0.00000001, 0), 1)["svd_deg"] == 1
    assert branch.measure((0.00000001, 0), 1)["svd_deg"] == 1
    assert branch.measure((0, 0), 1)["svd_deg"] == 0


def test_no_signal_and_near_anchors_are_retained_with_fixed_hypothesis():
    client = active()
    history = []
    record(client, history, "measure", (-1000, 0), 1)
    record(client, history, "measure", (99, 0), 1)
    branch = make_q3_branch(scenario(error_mode="negative_extreme"), client.state, history)
    assert branch.measure((-1000, 0), 1)["measure_result"] == "no_signal"
    assert branch.measure((99, 0), 1)["measure_result"] == "near"
    assert branch.measure((-900, 0), 1)["measure_result"] == "direction"
    assert branch.measure((-900.00001, 0), 1)["measure_result"] == "no_signal"


def test_clear_state_dominates_pre_clear_anchor_and_post_clear_absence():
    client = active()
    history = []
    record(client, history, "measure", (0, 0), 1)
    record(client, history, "clear", (100, 0), 1)
    record(client, history, "measure", (0, 0), 1)
    branch = make_q3_branch(scenario(), client.state, history)
    assert branch.state.sources[1] == client.state.sources[1]
    response = branch.measure((0, 0), 1)
    assert response["measure_result"] == "no_signal"
    assert "svd_deg" not in response
    assert branch.clear((100, 0), 1)["clear_result"] == "no_target_in_range"
    assert branch.state.cleared_count == 1


def test_branch_removal_also_overrides_old_anchor():
    client = active()
    history = []
    record(client, history, "measure", (0, 0), 1)
    branch = make_q3_branch(scenario(), client.state, history)
    branch.clear((100, 0), 1)
    assert branch.measure((0, 0), 1)["measure_result"] == "no_signal"


def test_no_automatic_exit_after_last_hypothesized_source():
    case = scenario()
    client = active(case)
    history = []
    for source in case.sources[:-1]:
        record(client, history, "clear", (source.x, source.y), source.channel)
    branch = make_q3_branch(case, client.state, history)
    source = case.sources[-1]
    assert branch.clear((source.x, source.y), source.channel)["clear_result"] == "success"
    assert branch.state.cleared_count == 10
    assert branch.state.session == "active"
    assert branch.measure((0, 0), 20)["measure_result"] == "no_signal"
    assert branch.exit()["exit_reason"] == "user_exit"
    assert not hasattr(branch, "evaluation")
    assert not hasattr(branch, "scenario")
    assert not hasattr(branch, "sources")


def test_remaining_virtual_budget_is_not_refreshed():
    client = active(max_virtual_duration_s=12)
    history = []
    record(client, history, "measure", (0, 0), 1)
    branch = make_q3_branch(scenario(), client.state, history)
    assert branch.state.max_virtual_duration_s - branch.state.virtual_time_s == 7
    assert branch.measure((0, 0), 1)["virtual_time_s"] == 10
    assert branch.measure((0, 0), 1)["virtual_time_s"] == 15
    with pytest.raises(DeadlineExceeded):
        branch.measure((0, 0), 1)


def test_branch_real_deadline_is_local_and_does_not_extend_input():
    client = active()
    client.state.real_deadline = -500
    before = time.monotonic()
    branch = make_q3_branch(scenario(), client.state, [])
    after = time.monotonic()
    assert before + 1200 <= branch.state.real_deadline <= after + 1200
    assert client.state.real_deadline == -500
    assert branch.measure((0, 0), 1)["accepted"]
    with pytest.raises(SessionError):
        branch.enter()


def test_microsecond_rounding_does_not_rebill_history():
    client = active()
    history = []
    for _ in range(20):
        record(client, history, "measure", (1, 1), 1)
        record(client, history, "measure", (0, 0), 1)
    branch = make_q3_branch(scenario(), client.state, history)
    old = client.state.virtual_time_s
    old_components = asdict(client.state.time_breakdown)
    assert asdict(branch.state.time_breakdown) == old_components
    response = branch.measure((1, 1), 1)
    assert response["virtual_time_s"] == pytest.approx(old + round(math.sqrt(2) / 5, 6) + 5)
    assert branch.state.time_breakdown.movement_s == pytest.approx(old_components["movement_s"] + math.sqrt(2) / 5)


@pytest.mark.parametrize("session", ["new", "exited", "real_timeout"])
def test_non_active_state_rejected(session):
    state = active().state
    state.session = session
    with pytest.raises(ValueError, match="active"):
        make_q3_branch(scenario(), state, [])


def test_q4_rejected():
    with pytest.raises(ValueError, match="Question 3"):
        make_q3_branch(scenario(problem=4), active().state, [])


@pytest.mark.parametrize("field,value", [("current_channel", 0), ("current_channel", True),
                                        ("virtual_time_s", -1), ("virtual_time_s", math.nan),
                                        ("max_virtual_duration_s", None), ("max_virtual_duration_s", 0),
                                        ("accepted_actions", -1), ("accepted_actions", True)])
def test_bad_basic_state_rejected(field, value):
    state = active().state
    setattr(state, field, value)
    with pytest.raises(ValueError):
        make_q3_branch(scenario(), state, [])


def test_missing_observed_source_and_bad_accounting_rejected():
    state = active().state
    state.sources[20] = SourceState(status="detected")
    with pytest.raises(ValueError, match="omits"):
        make_q3_branch(scenario(), state, [])
    state.sources.clear()
    state.time_breakdown.movement_s = 1
    with pytest.raises(ValueError, match="breakdown"):
        make_q3_branch(scenario(), state, [])


def test_conflicting_active_point_feedback_rejected():
    client = active()
    history = []
    record(client, history, "measure", (0, 0), 1)
    history.append(dict(history[0], bearing_deg=0.5))
    with pytest.raises(ValueError, match="conflicting"):
        make_q3_branch(scenario(), client.state, history)


def test_clear_history_cannot_override_uncleared_public_state():
    client = active()
    history = [{"action": "clear", "position": [100, 0], "channel": 1, "result": "success"}]
    with pytest.raises(ValueError, match="contradicts"):
        make_q3_branch(scenario(), client.state, history)
