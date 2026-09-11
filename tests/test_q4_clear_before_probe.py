"""Scripted real controller prefixes; no simulator, scene or network access."""
import copy
from types import SimpleNamespace

import pytest

from simulator_client.state import Position
from strategies.q4_clear_before_probe import Q4ClearBeforeProbe, run_q4_clear_before_probe
from strategies.q4_range_scheduling import Q4RangeScheduling
from strategies.q4_r2_scheduling import _ServiceSliceExpired
from strategies.search import _StopSearch


class Replies:
    def __init__(self):
        self.state = SimpleNamespace(position=Position(0, 0), current_channel=1,
            virtual_time_s=0., max_virtual_duration_s=360000., sources={})
        self.calls = []
        self.measure_replies = []
        self.clear_replies = ["success"]
        self.reject_kind = None
        self.remaining_real_time_s = None
        self.before_measure = None

    def measure(self, p, c):
        if self.before_measure:
            self.before_measure()
        kind, bearing = self.measure_replies.pop(0)
        result = self._call("measure", p, c, kind)
        if bearing is not None:
            result["svd_deg"] = bearing
        return result

    def clear(self, p, c):
        return self._call("clear", p, c, self.clear_replies.pop(0))

    def _call(self, action, p, c, result):
        self.calls.append((action, p, c))
        if self.reject_kind == action:
            return {"accepted": False}
        cost = round(self.state.position.distance_to(p)/5*1e6)/1e6
        cost += 5 if action == "measure" or result == "success" else 3
        if action == "measure":
            cost += c != self.state.current_channel
            self.state.current_channel = c
        self.state.position = p
        self.state.virtual_time_s += cost
        return {"accepted": True, "virtual_time_s": self.state.virtual_time_s,
                "measure_result" if action == "measure" else "clear_result": result}


def make(monkeypatch):
    import planning.q4_directional_cover as cover
    monkeypatch.setattr(cover, "certified_cover_points", lambda profile:
        ((Position(0, 0), Position(100, 0)), {"passed": True, "test_only": True}))
    client = Replies()
    policy = Q4ClearBeforeProbe(client, 20000, 6, max_expansions=0)
    return policy, client


def prime(policy, client):
    client.measure_replies = [("direction", 0.), ("direction", 90.)]
    policy._perform("measure", Position(-1000, 0), 1, "coverage")
    policy._perform("measure", Position(0, -1000), 1, "coverage")
    disk = policy.regions[1].enclosing_disk()
    assert 19.9 < disk.radius <= 40.
    return Position.coerce(disk.center)


def pending_measure(policy, point, channel=1):
    """Call the interception hook at the actual resolver's pending-action scope."""
    policy._probe_resolving_channel = channel
    try:
        return policy._perform("measure", point, channel, "active_localization")
    finally:
        policy._probe_resolving_channel = None


def test_success_exits_parent_resolve_without_fake_measure_and_runs_finally(monkeypatch):
    policy, client = make(monkeypatch)
    center = prime(policy, client)
    before = copy.deepcopy(policy.regions[1].vertices)
    policy.pending_pair = ("test",)
    assert policy._resolve(1) is True
    assert client.calls[-1] == ("clear", center, 1)
    assert len(client.calls) == 3 and policy.report.measurement_count == 2
    assert policy.regions[1].vertices == before and 1 not in policy.near_points
    assert policy.cleared == policy.speculative_attempted == {1}
    assert policy.pending_pair is None and policy._probe_resolving_channel is None
    event = policy.clear_before_probe_log[0]
    assert event["status"] == "cleared" and event["executed"]
    assert event["after_actual_action_count"] == 2 and event["end_actual_action_count"] == 3
    assert event["decision_wall_s"] <= event["runtime_s"]


def test_failure_then_original_near_response_drives_real_near_clear(monkeypatch):
    policy, client = make(monkeypatch)
    center = prime(policy, client)
    client.clear_replies = ["no_target_in_range", "success"]
    client.measure_replies = [("near", None)]
    assert policy._resolve(1)
    assert client.calls[-3:] == [("clear", center, 1), ("measure", center, 1), ("clear", center, 1)]
    assert [a["phase"] for a in policy.report.action_history[-3:]] == [
        "speculative_clear_before_probe", "active_localization", "near_clear"]
    event = policy.clear_before_probe_log[0]
    assert event["status"] == "failed_then_measured" and event["end_actual_action_count"] == 4
    assert policy.report.measurement_count == 3 and policy.cleared == {1}


def test_miss_does_not_change_region_freshness_or_tuning_before_real_measure(monkeypatch):
    policy, client = make(monkeypatch)
    center = prime(policy, client)
    client.state.current_channel = 2
    region = copy.deepcopy(policy.regions[1].__dict__)
    observed = copy.deepcopy(policy.observed_positions)
    def before_measure():
        assert policy.regions[1].__dict__ == region
        assert policy.observed_positions == observed
        assert not policy.cleared and not policy.near_points
        assert client.state.current_channel == 2
    client.before_measure = before_measure
    client.clear_replies = ["no_target_in_range"]
    client.measure_replies = [("no_signal", None)]
    t = client.state.virtual_time_s
    movement = client.state.position.distance_to(center)/5
    response = pending_measure(policy, center)
    assert response["measure_result"] == "no_signal"
    assert policy.regions[1].__dict__ == region
    assert client.state.virtual_time_s-t == pytest.approx(movement+9, abs=1e-6)
    assert client.state.current_channel == 1 and policy.speculative_attempted == {1}


def test_once_per_source_persists_across_resolver_scopes(monkeypatch):
    policy, client = make(monkeypatch)
    center = prime(policy, client)
    client.clear_replies = ["no_target_in_range"]
    client.measure_replies = [("no_signal", None), ("no_signal", None)]
    pending_measure(policy, center)
    pending_measure(policy, center)
    assert [c[0] for c in client.calls[2:]] == ["clear", "measure", "measure"]
    assert len(policy.clear_before_probe_log) == 1


@pytest.mark.parametrize("reason", ["action_budget", "virtual_budget", "service_budget"])
def test_insufficient_failure_continuation_budget_preserves_original_measure(monkeypatch, reason):
    policy, client = make(monkeypatch)
    center = prime(policy, client)
    client.state.position = center
    if reason == "action_budget":
        policy.max_actions = policy.actions + 2  # Original measure plus exit fit.
    elif reason == "virtual_budget":
        client.state.max_virtual_duration_s = client.state.virtual_time_s + 7.
    else:
        policy.service_deadline = client.state.virtual_time_s + 7.
    client.measure_replies = [("no_signal", None)]
    response = pending_measure(policy, center)
    assert response["measure_result"] == "no_signal" and client.calls[-1][0] == "measure"
    assert not policy.speculative_attempted and len(client.calls) == 3
    event = policy.clear_before_probe_log[0]
    assert event["status"] == "skipped_budget" and reason in event["budget"]["skip_reasons"]
    assert event["end_actual_action_count"] == event["after_actual_action_count"] == 2


def test_failure_budget_counts_original_enter_action_if_present(monkeypatch):
    policy, client = make(monkeypatch)
    center = prime(policy, client)
    policy.actions += 1  # Mimic _Search.run's accepted enter before this prefix.
    policy.max_actions = 5
    client.measure_replies = [("no_signal", None)]
    pending_measure(policy, center)
    budget = policy.clear_before_probe_log[0]["budget"]
    assert budget["policy_action_count"] == 3 and budget["remaining_actions"] == 2
    assert budget["skip_reasons"] == ["action_budget"]


def test_service_budget_refusal_still_uses_original_exception(monkeypatch):
    policy, client = make(monkeypatch)
    center = prime(policy, client)
    policy.service_deadline = client.state.virtual_time_s + 1.
    with pytest.raises(_ServiceSliceExpired):
        policy._resolve(1)
    assert len(client.calls) == 2 and not policy.speculative_attempted
    assert policy.pending_pair is None and policy._probe_resolving_channel is None


def test_realtime_deadline_is_not_swallowed_or_bypassed(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client)
    client.remaining_real_time_s = 1.
    with pytest.raises(_StopSearch, match="real_deadline"):
        policy._resolve(1)
    assert len(client.calls) == 2 and not policy.speculative_attempted
    assert policy.clear_before_probe_log[0]["budget"]["skip_reasons"] == ["real_deadline"]


@pytest.mark.parametrize("rejected", ["clear", "measure"])
def test_rejection_retains_only_actual_accepted_prefix(monkeypatch, rejected):
    policy, client = make(monkeypatch)
    prime(policy, client)
    client.reject_kind = rejected
    client.clear_replies = ["no_target_in_range"]
    client.measure_replies = [("no_signal", None)]
    with pytest.raises(_StopSearch, match="request_rejected"):
        policy._resolve(1)
    event = policy.clear_before_probe_log[0]
    clear_was_accepted = rejected == "measure"
    assert event["executed"] == clear_was_accepted
    assert (1 in policy.speculative_attempted) == clear_was_accepted
    assert event["end_actual_action_count"] == 2 + clear_was_accepted
    assert event["status"] == "interrupted" and not policy.cleared
    assert policy.pending_pair is None and policy._probe_resolving_channel is None


@pytest.mark.parametrize("change", ["coverage", "outside_resolve", "off_center", "large_radius", "small_radius"])
def test_only_original_active_center_candidate_is_intercepted(monkeypatch, change):
    policy, client = make(monkeypatch)
    center = prime(policy, client)
    phase = "active_localization"
    policy._probe_resolving_channel = 1
    if change == "coverage":
        phase = "coverage"
    elif change == "outside_resolve":
        policy._probe_resolving_channel = None
    elif change == "off_center":
        center = Position(center.x+1e-8, center.y)
    else:
        r = 40.000001 if change == "large_radius" else 19.9
        monkeypatch.setattr(policy.regions[1], "enclosing_disk", lambda:
            SimpleNamespace(center=(center.x, center.y), radius=r))
    client.measure_replies = [("no_signal", None)]
    policy._perform("measure", center, 1, phase)
    assert len(client.calls) == 3 and client.calls[-1][0] == "measure"
    assert not policy.clear_before_probe_log


def test_real_success_of_sixteenth_source_reaches_original_cap_stop(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client)
    policy.cleared = set(range(2, 17))
    policy.detected.update(policy.cleared)
    policy.points = []
    with pytest.raises(_StopSearch, match="source_count_upper_bound_reached"):
        policy._execute_plan()
    assert policy.cleared == set(range(1, 17))
    assert policy.report.completion_certified_under_model
    assert len(client.calls) == 3 and policy.clear_before_probe_log[0]["status"] == "cleared"


def test_inherited_algorithms_and_configuration_are_unchanged(monkeypatch):
    policy, client = make(monkeypatch)
    for name in ("_scan", "_next_probe", "_execute_plan", "_early_candidate", "_early_service", "_clear"):
        assert getattr(Q4ClearBeforeProbe, name) is getattr(Q4RangeScheduling, name)
    assert policy.range_pruning_enabled and policy.mode == "state_pruned"
    assert policy.scheduling_config["service_budget_s"] == 60.


@pytest.mark.parametrize("kwargs", [{"problem": 3}, {"config": "bad"}, {"max_actions": True},
    {"max_active_probes": -1}, {"max_expansions": 10001}])
def test_bad_configuration_fails_before_accessing_client(kwargs):
    with pytest.raises(ValueError):
        run_q4_clear_before_probe(object(), **kwargs)
