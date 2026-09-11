"""Real reply and semi-MDP billing contracts, using small public fixtures."""
import math
import time

import pytest

from q4_rl.bundle_controller import (
    Q4BundleSearch, BundleCandidate, run_q4_bundle, feature_schema,
    FEATURE_SCHEMA_VERSION, GLOBAL_DIM, CANDIDATE_DIM,
)
from simulator_client.state import Position
from strategies.search import _StopSearch
from tests.test_q4_rl_micro_controller import ScriptedClient


@pytest.fixture
def make(monkeypatch):
    import planning.q4_directional_cover as cover
    monkeypatch.setattr(cover, "certified_cover_points", lambda profile:
        ((Position(100, 0), Position(200, 0)), {"passed": True, "test_only": True}))
    def factory(**kwargs):
        client = ScriptedClient()
        return Q4BundleSearch(client, max_expansions=0, **kwargs), client
    return factory


def test_schema_and_single_bundle_feature_cost_matches_real_receipts(make):
    search, client = make()
    client.state.current_channel = 7
    candidates = search._candidates()
    assert len(candidates) == 2
    assert candidates[0].channels == (7, *range(1, 7), *range(8, 21))
    global_features, rows = search._features(candidates)
    assert len(global_features) == GLOBAL_DIM == 10
    assert all(len(row) == CANDIDATE_DIM == 18 for row in rows)
    assert feature_schema()["version"] == FEATURE_SCHEMA_VERSION
    assert rows[0][5] == 139/1000
    assert rows[0][7] == rows[0][16] == 1.
    assert rows[0][17] == 0.
    search._execute_candidate(candidates[0])
    assert [call[2] for call in client.calls] == list(candidates[0].channels)
    assert client.state.virtual_time_s == rows[0][5]*1000
    assert search.cover_ledger[candidates[0].point] == set(range(1, 21))
    assert search.cover_ledger[candidates[1].point] == set()
    assert search.detected == set(range(1, 11))
    assert search.report.learning["bundle_events"][0]["status"] == "group_finished"


def test_next_request_observes_previous_actual_feedback(make):
    search, client = make()
    seen = []
    def reply(point, channel):
        assert search.cover_ledger[point] == set(seen)
        assert search.detected == {c for c in seen if c <= 10}
        seen.append(channel)
        return {"measure_result": "near" if channel <= 10 else "no_signal"}
    client.reply = reply
    search._execute_candidate(search._candidates()[0])
    assert seen == list(range(1, 21))
    assert search.negative_counts[20] == 1


def test_sixteenth_observation_stops_group_but_does_not_claim_cleared(make):
    search, client = make()
    client.reply = lambda p, c: {"measure_result": "near" if c <= 16 else "no_signal"}
    search._execute_candidate(search._candidates()[0])
    assert len(client.calls) == 16
    assert search.detected == set(range(1, 17))
    assert not search.cleared and not search.report.completion_certified_under_model
    assert search.report.learning["discovery_certified"]
    assert search.report.learning["bundle_events"][-1]["status"] == "discovery_certified"
    assert all(c.kind == "service" for c in search._candidates())
    assert all((search.points[0], c) not in search.actual_measurements for c in range(17, 21))


def test_known_channels_not_grouped_and_geometry_features_are_public_means(make):
    search, client = make()
    search._perform("measure", Position(0, 0), 1, "fixture")
    candidates = search._candidates()
    scans = [candidate for candidate in candidates if candidate.kind == "scan_bundle"]
    assert all(1 not in candidate.channels for candidate in scans)
    services = [candidate for candidate in candidates if candidate.kind == "service"]
    assert len(services) == 1 and services[0].channels == (1,)
    global_features, rows = search._features(candidates)
    row = rows[candidates.index(services[0])]
    assert row[5] == row[16] == 0.
    assert row[17] == .005
    assert row[8] == row[9] == 1.
    assert math.isclose(row[10], 5/1800)
    assert all(math.isfinite(value) for values in [global_features]+rows for value in values)


@pytest.mark.parametrize("mode,reason", [
    ("action", "action_budget"), ("virtual", "virtual_budget"),
    ("administrative", "training_deadline"), ("rejection", "request_rejected"),
])
def test_partial_group_budget_interrupt_and_rejection_preserve_exact_receipts(make, mode, reason):
    search, client = make(max_actions=5 if mode == "action" else 20000)
    if mode == "virtual":
        client.state.max_virtual_duration_s = 33.
    if mode == "rejection":
        client.reject_measure = True
    if mode == "administrative":
        def reply(point, channel):
            search.action_deadline_epoch = time.time()-1
            return {"measure_result": "no_signal"}
        client.reply = reply
    report = search.run()
    assert report.completion_reason == reason
    assert not report.completion_certified_under_model
    assert client.state.session == "exited"
    event = report.learning["bundle_events"][0]
    actual = [item["channel"] for item in report.action_history if item["action"] == "measure"]
    assert event["measured_channels"] == actual
    assert search.cover_ledger[search.points[0]] == set(actual)
    assert event["status"] == "interrupted"
    assert event["cost_s"] == report.virtual_time_s
    transitions = report.learning["transitions"]
    assert len(transitions) == 1
    assert sum(row["cost_s"] for row in transitions) == report.virtual_time_s
    assert transitions[-1]["terminal"]
    if mode == "rejection":
        assert not actual and transitions[0]["cost_s"] == 0.


def test_decision_limit_fallback_and_terminal_tail_remain_in_last_transition(make):
    search, client = make(max_decisions=1)
    client.exit_cost = 7.25
    report = search.run()
    assert report.completion_certified_under_model
    assert report.cleared_count == 10
    assert report.learning["fallback_reason"] == "decision_limit"
    assert report.learning["fallback_cost_s"] > 0
    assert report.learning["uncovered_cost_s"] == 7.25
    transition = report.learning["transitions"][0]
    assert transition["cost_s"] == report.virtual_time_s
    assert transition["fallback_cost_s"] == report.learning["fallback_cost_s"]
    assert transition["terminal"]


def test_complete_rule_requires_real_clears_and_covers_only_unknown_channels(make):
    search, client = make()
    report = search.run()
    assert report.completion_certified_under_model and report.learning["training_success"]
    assert report.cleared_count == 10
    assert report.learning["failed_clear_count"] == 0
    assert report.learning["action_counts"] == {"scan_bundle": 2, "service": 10}
    assert report.learning["fallback_cost_s"] == 0
    assert report.learning["completion_certificate"].startswith("per_unknown_channel")
    assert sum(row["cost_s"] for row in report.learning["transitions"]) == report.virtual_time_s
    assert len([call for call in client.calls if call[0] == "clear"]) == 10
    assert search.cover_ledger[search.points[1]] == set(range(11, 21))


def test_failed_clear_is_retained_and_never_becomes_success(make):
    search, client = make()
    search._perform("measure", Position(0, 0), 1, "fixture")
    client.clear_reply = lambda p, c: "no_target_in_range"
    candidate = next(c for c in search._candidates() if c.kind == "service")
    search._execute_candidate(candidate)
    assert 1 not in search.cleared and 1 in search.blocked
    assert search.report.action_history[-1]["result"] == "no_target_in_range"
    assert not search.report.completion_certified_under_model


def test_invalid_forged_or_stale_groups_cannot_claim_credits(make):
    search, client = make()
    for candidate in (BundleCandidate("scan_bundle", Position(123, 456), (1,)),
                      BundleCandidate("scan_bundle", search.points[0], (1, 1)),
                      BundleCandidate("scan_bundle", search.points[0], ())):
        with pytest.raises(_StopSearch, match="invalid_scan_bundle"):
            search._execute_candidate(candidate)
    assert not client.calls and not search.actual_measurements
    with pytest.raises(ValueError, match="problem=4"):
        run_q4_bundle(client, problem=3)
