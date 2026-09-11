from types import SimpleNamespace

import pytest

from experiments.audit_q3_fresh_choices import (
    ReplayMismatch, assert_same_observation, counterfactual_cost, resolve_selected, summarize,
)
from strategies.q3_fresh_stepper import Action


def test_duplicate_labels_resolved_using_recorded_actual_action():
    first = Action("measure", (100, 0), 1, "rollout_probe")
    second = Action("measure", (200, 0), 2, "rollout_probe")
    options = [("probe_other_source", (first,)), ("probe_other_source", (second,))]
    recorded = dict(action="measure", position=[200, 0], channel=2, phase="rollout_probe")
    index, chosen = resolve_selected(options, dict(selected="probe_other_source"), recorded)
    assert index == 1 and chosen == (second,)
    with pytest.raises(ReplayMismatch):
        resolve_selected(options + [options[1]], dict(selected="probe_other_source"), recorded)


def test_observation_replay_checks_feedback_and_elapsed_cost():
    row = dict(action="measure", channel=2, phase="cover", result="direction",
               position=[0, 0], bearing_deg=90, virtual_time_s=5)
    assert_same_observation(row, row.copy(), 0)
    with pytest.raises(ReplayMismatch, match="bearing_deg"):
        assert_same_observation(dict(row, bearing_deg=91), row, 0)
    with pytest.raises(ReplayMismatch, match="virtual_time_s"):
        assert_same_observation(dict(row, virtual_time_s=6), row, 0)


def test_counterfactual_requires_certified_exit_and_disables_tail(monkeypatch):
    action = Action("measure", (0, 0), 1, "cover")
    fake_state = SimpleNamespace(virtual_time_s=25.0, session="active")
    client = SimpleNamespace(state=fake_state)
    monkeypatch.setattr("experiments.audit_q3_fresh_choices.make_actual_branch", lambda *a: client)
    class Fork:
        def __init__(self):
            self.history = []
            self.movable_tail = True
            self.tail_plan = [1]
        def run(self, **kwargs):
            assert self.movable_tail is False and self.tail_plan == []
            fake_state.virtual_time_s += 12
            fake_state.session = "exited"
            return dict(completion_certified=True, confirmation_tail_s=3)
    controller = SimpleNamespace(client=client, pending=action, history=[], clone=lambda c: Fork())
    result = counterfactual_cost(controller, None, (action,))
    assert result["complete"] and result["remaining_s"] == 12
    def fail(*a):
        raise TimeoutError("deliberate incomplete branch")
    monkeypatch.setattr("experiments.audit_q3_fresh_choices.make_actual_branch", fail)
    result = counterfactual_cost(controller, None, (action,))
    assert result["complete"] is False and result["remaining_s"] is None


def test_invalid_replay_deltas_are_excluded_from_summary():
    rows = [dict(replay_valid=False, audits=[dict(usable=False, actual_delta_s=-1000)])]
    report = summarize(rows)
    assert report["usable_audits"] == 0 and report["mean_actual_delta_s"] is None
    assert report["attempted_audits"] == 1 and report["incomplete_or_invalid_audits"] == 1
