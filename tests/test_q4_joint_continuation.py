"""Constructed accepted prefixes only; no generated scenes or hidden sources."""
import copy

import pytest

from simulator_client.state import Position
from strategies import q4_joint_continuation as continuation
from strategies.q4_joint_visibility import Q4JointVisibility
from strategies.q4_r2_scheduling import _ServiceSliceExpired
from experiments.audit_q4_joint_visibility import audit_joint_visibility_certificate
from tests.test_q4_clear_before_probe import Replies
from tests.test_q4_joint_visibility_strategy import prime, shrink


def make(monkeypatch, active=6):
    monkeypatch.setattr("planning.q4_directional_cover.certified_cover_points", lambda profile:
        ((Position(0, 0), Position(100, 0)), {"passed": True, "test_only": True}))
    client = Replies()
    policy = continuation.Q4JointContinuation(client, 20000, active, max_expansions=0)
    return policy, client


def fake_helper(monkeypatch, target, fraction=.9, fallback=False):
    calls = []
    def fake(vertices, positive, negative):
        calls.append(copy.deepcopy((vertices, positive, negative)))
        output = tuple(vertices) if fallback else shrink(vertices, fraction)
        return output, dict(passed=True, status="fallback" if fallback else "outer_refined",
            old_vertices_excluded=[0] if not fallback and fraction < 1 else [],
            output_vertices=[list(p) for p in output], test_only=True)
    monkeypatch.setattr(target, fake)
    return calls


def context(policy):
    value = policy._start_joint(1)
    policy._joint_context = value
    return value


def miss(policy, client, p=(-2100., -2200.), phase="active_localization", channel=1):
    client.measure_replies = [("no_signal", None)]
    return policy._perform("measure", Position(*p), channel, phase)


def test_only_one_new_module_and_inherited_action_algorithms(monkeypatch):
    policy, _ = make(monkeypatch)
    for name in ("run", "_scan", "_execute_plan", "_resolve", "_next_probe", "_clear", "_check_budget"):
        assert getattr(continuation.Q4JointContinuation, name) is getattr(Q4JointVisibility, name)
    assert policy.joint_config == "probe_optical"
    assert policy.continuation_limit == 120
    assert policy.report.strategy_parameters["joint_visibility_continuation_limits"] == {
        "per_resolver": 6, "per_session": 120, "helper_constraint_work": 65536}


def test_accepted_miss_updates_only_aux_after_radio_and_pending_completion(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client)
    fake_helper(monkeypatch, "strategies.q4_joint_visibility.joint_visibility_outer", .95)
    calls = fake_helper(monkeypatch, "strategies.q4_joint_continuation.joint_visibility_outer", .8)
    ctx = context(policy)
    original = copy.deepcopy(policy.regions[1].__dict__)
    old_aux = copy.deepcopy(ctx["region"].__dict__)
    point = policy._next_probe(1, 0)
    pending = ctx["pending_probe"]
    before = len(client.calls)
    response = miss(policy, client, (point.x, point.y))
    assert response["measure_result"] == "no_signal"
    assert len(client.calls) == before + 1 and pending["status"] == "measured"
    assert ctx["pending_probe"] is None
    assert calls[0][0] == old_aux["vertices"]
    assert calls[0][2][-1] == [point.x, point.y]
    assert policy.regions[1].__dict__ == original
    assert ctx["region"].vertices != old_aux["vertices"]
    assert not ctx["event"]["aux_updates"]
    event = policy.continuation_log[0]
    assert event["status"] == "applied" and event["input_source"] == "auxiliary"
    assert event["trigger_actual_action_index"] + 1 == event["after_actual_action_count"] == len(policy.report.action_history)
    assert event["end_actual_action_count"] == event["after_actual_action_count"]
    assert event["budget"]["session_calls_after"] == event["budget"]["resolver_calls_after"] == 1


def test_entry_without_aux_can_create_one_after_first_real_miss(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client, negative=False)
    ctx = context(policy)
    assert ctx["region"] is None and ctx["event"]["skip_reason"] == "no_negative_evidence"
    calls = fake_helper(monkeypatch, "strategies.q4_joint_continuation.joint_visibility_outer", .9)
    original = copy.deepcopy(policy.regions[1].__dict__)
    miss(policy, client)
    assert len(calls) == 1 and ctx["region"] is not None
    assert policy.continuation_log[-1]["input_source"] == "canonical"
    assert policy.regions[1].__dict__ == original


def test_recursive_basis_and_full_pn_include_prior_refresh(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client, negative=False)
    calls = fake_helper(monkeypatch, "strategies.q4_joint_continuation.joint_visibility_outer", .99)
    ctx = context(policy)
    miss(policy, client)
    first = policy.continuation_log[-1]
    miss(policy, client, (-2300., -2200.))
    second = policy.continuation_log[-1]
    assert len(calls) == 2 and second["input_source"] == "auxiliary"
    assert second["input_vertices"] == first["output_aux_vertices"]
    assert second["basis"]["previous_applied_event_id"] == first["id"]
    assert second["basis"]["previous_applied_prefix"] == first["after_actual_action_count"]
    assert len(calls[-1][1]) == 2 and len(calls[-1][2]) == 2
    assert ctx["last_applied_continuation_id"] == second["id"]


def test_real_bearing_between_refreshes_is_retained_in_basis_and_region(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client, negative=False)
    calls = fake_helper(monkeypatch, "strategies.q4_joint_continuation.joint_visibility_outer", .99)
    ctx = context(policy)
    miss(policy, client)
    client.measure_replies = [("direction", 0.)]
    policy._perform("measure", Position(-1000, 0), 1, "active_localization")
    update_prefix = len(policy.report.action_history)
    expected_input = copy.deepcopy(ctx["region"].vertices)
    miss(policy, client, (-2300., -2300.))
    event = policy.continuation_log[-1]
    assert calls[-1][0] == expected_input
    assert event["basis"]["bearing_update_prefixes"] == [update_prefix]
    assert len(event["positive_positions"]) == len(ctx["region"].observations) == 3
    assert len(ctx["event"]["aux_updates"]) == 1


def test_resolver_limit_counts_actual_helper_calls_not_decisions(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client, negative=False)
    calls = fake_helper(monkeypatch, "strategies.q4_joint_continuation.joint_visibility_outer", 1.)
    ctx = context(policy)
    for i in range(7):
        miss(policy, client, (-2100.-i, -2200.))
    assert len(calls) == ctx["continuation_calls"] == policy.continuation_calls == 6
    assert policy.continuation_log[-1]["status"] == "resolver_work_budget"
    assert policy.continuation_log[-1]["budget"]["session_calls_after"] == 6


def test_no_boundary_reduction_leaves_full_physical_trace_equal_to_static_optical(monkeypatch):
    policy, client = make(monkeypatch)
    base_client = Replies()
    base = Q4JointVisibility(base_client, 20000, 6, max_expansions=0, config="probe_optical")
    fake_helper(monkeypatch, "strategies.q4_joint_continuation.joint_visibility_outer", 1.)
    for controller, replies in ((policy, client), (base, base_client)):
        prime(controller, replies, negative=False)
        replies.clear_replies = ["no_target_in_range", "success"]
        replies.measure_replies = [("no_signal", None), ("near", None)]
        assert controller._resolve(1)
    assert client.calls == base_client.calls
    assert policy.report.action_history == base.report.action_history
    assert policy.regions[1].__dict__ == base.regions[1].__dict__
    assert policy.continuation_log[0]["status"] == "no_boundary_reduction"


def test_helper_exception_is_logged_and_preserves_parent_cleanup(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client, negative=False)
    def broken(*args):
        raise RuntimeError("constructed helper failure")
    monkeypatch.setattr("strategies.q4_joint_continuation.joint_visibility_outer", broken)
    client.clear_replies = ["no_target_in_range"]
    client.measure_replies = [("no_signal", None)]
    with pytest.raises(RuntimeError, match="constructed"):
        policy._resolve(1)
    assert policy._joint_context is None and policy._probe_resolving_channel is None
    assert policy.continuation_log[-1]["status"] == "interrupted"
    assert policy.continuation_calls == 1
    assert policy.joint_resolvers[-1]["status"] == "interrupted"


@pytest.mark.parametrize("kind", ["coverage_miss", "direction", "near", "clear", "outside_resolver"])
def test_nontrigger_actions_never_refresh(monkeypatch, kind):
    policy, client = make(monkeypatch)
    prime(policy, client, negative=False)
    context(policy)
    calls = fake_helper(monkeypatch, "strategies.q4_joint_continuation.joint_visibility_outer")
    if kind == "coverage_miss":
        miss(policy, client, phase="coverage")
    elif kind == "outside_resolver":
        policy._joint_context = None
        miss(policy, client)
    elif kind == "clear":
        client.clear_replies = ["no_target_in_range"]
        policy._perform("clear", Position(1, 2), 1, "guaranteed_clearance")
    else:
        client.measure_replies = [(kind, 90. if kind == "direction" else None)]
        policy._perform("measure", Position(0, -500), 1, "active_localization")
    assert not calls and not policy.continuation_log


@pytest.mark.parametrize("budget", ["resolver", "session"])
def test_deterministic_work_limit_preserves_existing_aux(monkeypatch, budget):
    policy, client = make(monkeypatch)
    prime(policy, client)
    fake_helper(monkeypatch, "strategies.q4_joint_visibility.joint_visibility_outer", .95)
    calls = fake_helper(monkeypatch, "strategies.q4_joint_continuation.joint_visibility_outer")
    ctx = context(policy)
    ctx["region"].enclosing_disk()  # Ready checks may populate this lazy cache.
    original = copy.deepcopy(ctx["region"].__dict__)
    if budget == "resolver":
        ctx["continuation_calls"] = policy.max_active_probes
    else:
        policy.continuation_calls = policy.continuation_limit
    miss(policy, client)
    assert not calls and policy.continuation_log[-1]["status"] == budget + "_work_budget"
    assert ctx["region"].__dict__ == original
    assert not policy.continuation_log[-1]["helper_called"]


@pytest.mark.parametrize("fallback", [False, True])
def test_no_improvement_keeps_original_aux(monkeypatch, fallback):
    policy, client = make(monkeypatch)
    prime(policy, client)
    fake_helper(monkeypatch, "strategies.q4_joint_visibility.joint_visibility_outer", .95)
    fake_helper(monkeypatch, "strategies.q4_joint_continuation.joint_visibility_outer", 1., fallback=fallback)
    ctx = context(policy)
    ctx["region"].enclosing_disk()
    original = copy.deepcopy(ctx["region"].__dict__)
    miss(policy, client)
    assert ctx["region"].__dict__ == original
    assert policy.continuation_log[-1]["status"] == ("helper_fallback" if fallback else "no_boundary_reduction")
    assert policy.continuation_calls == 1


def test_failed_measure_does_not_refresh(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client, negative=False)
    context(policy)
    client.reject_kind = "measure"
    with pytest.raises(Exception):
        miss(policy, client)
    assert not policy.continuation_log


def test_real_geometric_helper_certificates_replay_without_source_truth(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client, negative=False)
    ctx = context(policy)
    original = copy.deepcopy(policy.regions[1].__dict__)
    for p in ((-30., 0.), (0., -30.)):
        miss(policy, client, p)
    checked = [audit_joint_visibility_certificate(e["helper_evidence"])
               for e in policy.continuation_log if e["helper_called"]]
    assert checked and all(c["passed"] for c in checked)
    assert policy.regions[1].__dict__ == original
    assert all(e["positive_positions"] == [[-1000., 0.], [0., -1000.]] for e in policy.continuation_log)
    assert ctx["region"] is not policy.regions[1]


def test_r8_failed_clear_then_actual_miss_is_one_continuation(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client, negative=False)
    calls = fake_helper(monkeypatch, "strategies.q4_joint_continuation.joint_visibility_outer", .2)
    client.clear_replies = ["no_target_in_range", "success"]
    client.measure_replies = [("no_signal", None)]
    assert policy._resolve(1)
    assert len(calls) == 1
    assert [a["phase"] for a in policy.report.action_history[-3:]] == [
        "speculative_clear_before_probe", "active_localization", "joint_visibility_clear"]
    assert policy.continuation_log[0]["trigger_actual_action_index"] == 3
    assert policy._joint_context is None and policy._probe_resolving_channel is None


def test_r8_success_bypasses_refresh(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client, negative=False)
    calls = fake_helper(monkeypatch, "strategies.q4_joint_continuation.joint_visibility_outer")
    assert policy._resolve(1)
    assert not calls and not policy.continuation_log and policy.speculative_attempted == {1}


def test_service_interruption_does_not_reset_or_create_physical_budget(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client, negative=False)
    client.clear_replies = ["no_target_in_range"]
    client.measure_replies = [("no_signal", None)]
    fake_helper(monkeypatch, "strategies.q4_joint_continuation.joint_visibility_outer", .9)
    policy.service_deadline = client.state.virtual_time_s + 1.
    before = len(client.calls)
    with pytest.raises(_ServiceSliceExpired):
        policy._resolve(1)
    assert len(client.calls) == before and not policy.continuation_log
    assert policy._joint_context is None and policy._probe_resolving_channel is None


@pytest.mark.parametrize("kwargs", [{"problem": 3}, {"config": "probe"}, {"max_active_probes": True}, {"max_expansions": -1}])
def test_invalid_entrypoint_rejects_before_client_access(kwargs):
    with pytest.raises(ValueError):
        continuation.run_q4_joint_continuation(object(), **kwargs)
