"""Scripted controller tests; geometry certificates have their own full suite.

Selected tests inject explicit auxiliary polygons to isolate controller state
handling. Such test-only helper certificates are not scientific evidence.
"""
import copy

import pytest

from planning import clearance_grid
from simulator_client.state import Position
from strategies.q4_clear_before_probe import Q4ClearBeforeProbe
from strategies.q4_joint_visibility import Q4JointVisibility, run_q4_joint_visibility
from strategies.q4_r2_scheduling import _ServiceSliceExpired
from strategies.search import _StopSearch
from tests.test_q4_clear_before_probe import Replies


def make(monkeypatch, config="probe", active=6):
    import planning.q4_directional_cover as cover
    monkeypatch.setattr(cover, "certified_cover_points", lambda profile:
        ((Position(0, 0), Position(100, 0)), {"passed": True, "test_only": True}))
    client = Replies()
    policy = Q4JointVisibility(client, 20000, active, config=config, max_expansions=0)
    return policy, client


def prime(policy, client, *, scale=1000., negative=True):
    client.measure_replies = [("direction", 0.), ("direction", 90.)]
    policy._perform("measure", Position(-scale, 0), 1, "coverage")
    policy._perform("measure", Position(0, -scale), 1, "coverage")
    if negative:
        client.measure_replies = [("no_signal", None)]
        policy._perform("measure", Position(-2000, -2000), 1, "coverage")
    return policy.regions[1].enclosing_disk()


def shrink(vertices, fraction):
    anchor = vertices[0]
    return tuple((fraction*x+(1-fraction)*anchor[0], fraction*y+(1-fraction)*anchor[1])
                 for x, y in vertices)


def helper(monkeypatch, fraction=0.3, fallback=False, anchor_min_x=False):
    calls = []
    def fake(vertices, positives, negatives):
        calls.append(copy.deepcopy((vertices, positives, negatives)))
        if anchor_min_x:
            first = min(range(len(vertices)), key=lambda i: vertices[i][0])
            vertices = tuple(vertices[first:])+tuple(vertices[:first])
        outer = tuple(vertices) if fallback else shrink(vertices, fraction)
        return outer, {"passed": True, "status": "fallback" if fallback else "outer_refined",
                       "test_only": True, "output_vertices": outer}
    monkeypatch.setattr("strategies.q4_joint_visibility.joint_visibility_outer", fake)
    return calls


def test_aux_certified_clear_exits_full_parent_resolver_and_preserves_canonical(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client)
    calls = helper(monkeypatch)
    original = copy.deepcopy(policy.regions[1].__dict__)
    policy.pending_pair = ("sentinel",)
    assert policy._resolve(1)
    assert len(calls) == 1 and policy.regions[1].__dict__ == original
    assert policy.report.action_history[-1]["phase"] == "joint_visibility_clear"
    assert policy.cleared == {1} and not policy.speculative_attempted
    assert not policy.clear_before_probe_log and not policy.joint_grids
    assert policy.pending_pair is None and policy._probe_resolving_channel is None and policy._joint_context is None
    assert policy.joint_probes[0]["kind"] == "clear" and policy.joint_probes[0]["status"] == "cleared"
    assert policy.joint_resolvers[0]["end_actual_action_count"] == 4


def test_aux_probe_uses_same_five_point_family_without_expanding_r8_predicate(monkeypatch):
    policy, client = make(monkeypatch)
    canonical = prime(policy, client)
    helper(monkeypatch, .9)
    client.measure_replies = [("near", None)]
    assert policy._resolve(1)
    probe = policy.joint_probes[0]
    assert len(probe["candidates"]) == 5 and probe["selected"] == 0
    assert probe["canonical_point"] == list(canonical.center) and probe["point"] != probe["canonical_point"]
    assert probe["executed_measure"] and not probe["r8_clear_before_measure"]
    assert not policy.clear_before_probe_log and not policy.speculative_attempted
    assert [a["phase"] for a in policy.report.action_history[-2:]] == ["active_localization", "near_clear"]


def test_aux_positive_updates_are_real_and_negative_does_not_recompute_helper(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client)
    calls = helper(monkeypatch, 1.)
    context = policy._start_joint(1)
    policy._joint_context = context
    canonical = policy.regions[1]
    aux = context["region"]
    assert aux is not canonical and aux.observations is not canonical.observations
    before = tuple(aux.vertices)
    client.measure_replies = [("no_signal", None), ("direction", 90.)]
    policy._perform("measure", Position(-2200, -2200), 1, "active_localization")
    assert tuple(aux.vertices) == before and not context["event"]["aux_updates"]
    policy._perform("measure", Position(0, -500), 1, "active_localization")
    assert len(calls) == 1 and len(context["event"]["aux_updates"]) == 1
    assert len(aux.observations) == len(canonical.observations) == 3
    assert context["event"]["aux_updates"][0]["after_actual_action_count"] == 5
    assert tuple(aux.vertices) != before and not policy.joint_contradictions


def test_r8_local_success_exception_is_still_handled_when_aux_center_unchanged(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client)
    helper(monkeypatch, 1.)
    assert policy._resolve(1)
    probe = policy.joint_probes[0]
    assert probe["r8_clear_before_measure"] and probe["status"] == "cleared_by_r8"
    assert not probe["executed_measure"]
    assert policy.speculative_attempted == {1} and policy.clear_before_probe_log[0]["status"] == "cleared"
    assert policy._probe_resolving_channel is None and policy._joint_context is None


def test_fallback_helper_preserves_original_r8_clear_path(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client)
    calls = helper(monkeypatch, fallback=True)
    assert policy._resolve(1)
    assert len(calls) == 1 and not policy.joint_probes
    assert policy.report.action_history[-1]["phase"] == "speculative_clear_before_probe"
    assert policy.joint_resolvers[0]["skip_reason"] == "helper_fallback"


@pytest.mark.parametrize("kind", ["ready", "no_negative", "near", "cleared"])
def test_ineligible_resolver_does_not_call_helper(monkeypatch, kind):
    policy, client = make(monkeypatch)
    prime(policy, client, scale=300. if kind == "ready" else 1000., negative=kind != "no_negative")
    if kind == "near":
        client.measure_replies = [("near", None)]
        policy._perform("measure", Position(0, 0), 1, "coverage")
    elif kind == "cleared":
        policy._clear(Position(0, 0), 1, "guaranteed_clearance")
    calls = helper(monkeypatch)
    assert policy._resolve(1)
    assert not calls and not policy.joint_probes


def test_canonical_ready_after_real_measure_keeps_original_certificate_priority(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client)
    helper(monkeypatch, 1.)
    context = policy._start_joint(1)
    policy._joint_context = context
    client.measure_replies = [("direction", 90.)]
    policy._perform("measure", Position(0, -200), 1, "coverage")
    assert policy._ready(1)
    policy._joint_context = None
    assert policy._resolve(1)
    assert policy.report.action_history[-1]["phase"] == "certified_clear"


def test_probe_mode_keeps_canonical_optical_fallback(monkeypatch):
    policy, client = make(monkeypatch, "probe", active=0)
    prime(policy, client)
    helper(monkeypatch, .9)
    expected = clearance_grid(policy.regions[1].vertices, bearing_deg=policy.first_bearings[1], start=client.state.position)
    assert policy._resolve(1)
    assert client.calls[-1] == ("clear", expected[0], 1)
    assert policy.report.action_history[-1]["phase"] == "guaranteed_clearance" and not policy.joint_grids


def test_joint_optical_executes_only_real_prefix_of_complete_logged_grid(monkeypatch):
    policy, client = make(monkeypatch, "probe_optical", active=0)
    prime(policy, client)
    helper(monkeypatch, .9)
    before = copy.deepcopy(policy.regions[1].vertices)
    client.clear_replies = ["no_target_in_range", "success"]
    assert policy._resolve(1)
    grid = policy.joint_grids[0]
    assert grid["status"] == "cleared" and grid["actual_grid_actions"] == 2
    assert [list(p.__dict__.values()) for _, p, _ in client.calls[-2:]] == grid["grid"][:2]
    assert all(a["phase"] == "joint_visibility_optical" for a in policy.report.action_history[-2:])
    assert policy.regions[1].vertices == before and not policy.joint_contradictions
    assert policy._joint_context is None and policy._probe_resolving_channel is None


def test_joint_grid_exhaustion_retains_contradiction_and_original_grid_fallback(monkeypatch):
    policy, client = make(monkeypatch, "probe_optical", active=0)
    prime(policy, client)
    helper(monkeypatch, .9)
    canonical_first = clearance_grid(policy.regions[1].vertices, bearing_deg=policy.first_bearings[1], start=client.state.position)[0]
    count = len(clearance_grid(shrink(policy.regions[1].vertices, .9), bearing_deg=policy.first_bearings[1], start=client.state.position))
    client.clear_replies = ["no_target_in_range"]*count+["success"]
    assert policy._resolve(1)
    assert policy.joint_grids[0]["status"] == "exhausted_without_success"
    assert policy.joint_grids[0]["actual_grid_actions"] == count
    assert policy.joint_contradictions[0]["reason"] == "exhausted_without_success"
    assert client.calls[-1] == ("clear", canonical_first, 1)
    assert policy.report.action_history[-1]["phase"] == "guaranteed_clearance"


@pytest.mark.parametrize("config,active", [("probe", 6), ("probe_optical", 0)])
def test_service_budget_exception_preserves_zero_action_prefix_and_all_finally(monkeypatch, config, active):
    policy, client = make(monkeypatch, config, active)
    prime(policy, client)
    helper(monkeypatch, .3)
    policy.pending_pair = ("sentinel",)
    policy.service_deadline = client.state.virtual_time_s+1.
    with pytest.raises(_ServiceSliceExpired):
        policy._resolve(1)
    assert len(client.calls) == 3 and not policy.cleared
    assert policy._joint_context is None and policy._probe_resolving_channel is None and policy.pending_pair is None
    assert policy.joint_resolvers[0]["status"] == "interrupted"
    event = (policy.joint_probes if active else policy.joint_terminal_clears)[0]
    assert event["status"] == "interrupted" and event["end_actual_action_count"] == 3


@pytest.mark.parametrize("config", ["probe", "probe_optical"])
def test_real_helper_ready_certificate_used_even_without_probe_budget(monkeypatch, config):
    policy, client = make(monkeypatch, config, active=0)
    prime(policy, client, negative=False)
    client.measure_replies = [("no_signal", None), ("no_signal", None)]
    policy._perform("measure", Position(-30, 0), 1, "coverage")
    policy._perform("measure", Position(0, -30), 1, "coverage")
    assert policy.regions[1].enclosing_disk().radius > 19.9
    assert policy._resolve(1)
    event = policy.joint_terminal_clears[0]
    assert event["radius_m"] < 19.9 and event["status"] == "cleared"
    assert event["end_actual_action_count"]-event["after_actual_action_count"] == 1
    assert policy.report.action_history[-1]["phase"] == "joint_visibility_clear"
    assert not policy.joint_probes and not policy.joint_grids
    assert policy._joint_context is None and policy._probe_resolving_channel is None


def test_last_real_probe_can_create_terminal_aux_certificate(monkeypatch):
    policy, client = make(monkeypatch, active=1)
    client.measure_replies = [("direction", 0.), ("no_signal", None)]
    policy._perform("measure", Position(-1000, 0), 1, "coverage")
    policy._perform("measure", Position(-2000, -2000), 1, "coverage")
    helper(monkeypatch, .04, anchor_min_x=True)
    client.measure_replies = [("direction", 0.)]
    assert policy._resolve(1)
    assert len(policy.joint_probes) == 1 and policy.joint_probes[0]["radius_m"] > 19.9
    assert policy.joint_probes[0]["executed_measure"]
    assert policy.regions[1].enclosing_disk().radius > 19.9
    assert len(policy.joint_resolvers[0]["aux_updates"]) == 1
    assert policy.joint_terminal_clears[0]["radius_m"] <= 19.9
    assert [a["phase"] for a in policy.report.action_history[-2:]] == ["active_localization", "joint_visibility_clear"]


def test_rejected_joint_clear_does_not_change_source_or_fake_success(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client)
    helper(monkeypatch)
    client.reject_kind = "clear"
    with pytest.raises(_StopSearch, match="request_rejected"):
        policy._resolve(1)
    assert not policy.cleared and len(policy.report.action_history) == 3
    assert policy.joint_probes[0]["status"] == "interrupted"
    assert not policy.joint_contradictions and policy._joint_context is None


def test_real_helper_observation_inputs_exclude_clear_miss_and_post_clear_silence(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client)
    client.clear_replies = ["no_target_in_range"]
    policy._clear(Position(100, 100), 1, "guaranteed_clearance")
    context = policy._start_joint(1)
    evidence = context["event"]["helper_evidence"]
    assert evidence["negative_positions"] == [[-2000, -2000]]
    assert evidence["positive_positions"] == [[-1000, 0], [0, -1000]]
    assert evidence["canonical_vertices"] == [list(p) for p in policy.regions[1].vertices]
    client.clear_replies = ["success"]
    policy._clear(Position(0, 0), 1, "guaranteed_clearance")
    client.measure_replies = [("no_signal", None)]
    policy._perform("measure", Position(0, 0), 1, "coverage")
    assert len(policy.joint_radio[1]) == 3


def test_inherited_global_algorithms_and_sixteenth_actual_clear_stop(monkeypatch):
    policy, client = make(monkeypatch)
    prime(policy, client)
    helper(monkeypatch)
    for name in ("_ready", "_target", "_scan", "_execute_plan", "_early_candidate", "_early_service"):
        assert getattr(Q4JointVisibility, name) is getattr(Q4ClearBeforeProbe, name)
    policy.cleared = set(range(2, 17))
    policy.detected.update(policy.cleared)
    policy.points = []
    with pytest.raises(_StopSearch, match="source_count_upper_bound_reached"):
        policy._execute_plan()
    assert policy.cleared == set(range(1, 17)) and policy.report.completion_certified_under_model


@pytest.mark.parametrize("kwargs", [{"config": "bad"}, {"problem": 3}, {"max_actions": True},
    {"max_active_probes": -1}, {"max_expansions": 10001}])
def test_input_validation_precedes_client_access(kwargs):
    with pytest.raises(ValueError):
        run_q4_joint_visibility(object(), **kwargs)
