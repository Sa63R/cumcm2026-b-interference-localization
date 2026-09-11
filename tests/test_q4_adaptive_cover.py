"""Exact geometry and scripted ledger checks; no scenario/official simulator."""
import copy
import time

import pytest

from planning.q4_directional_cover import certified_cover_points
from q4_rl.adaptive_cover import (AdaptiveCoverLedger, certify, verify_exact_certificate,
                                 replay_adaptive_artifact)
from q4_rl.adaptive_controller import Q4AdaptiveRLSearch
from q4_rl.controller import Q4RLSearch


@pytest.fixture(scope="module")
def fixed():
    return certified_cover_points("compact_22")[0]


def ledger(fixed):
    return AdaptiveCoverLedger(fixed, episode_wall_s=30., decision_wall_s=5.)


def record(manager, history, channel, point, result="no_signal", action="measure"):
    item = dict(action=action, channel=channel, position=list(point), result=result)
    history.append(item)
    manager.observe(len(history)-1, action, point, channel, result)


def test_actual_opportunity_replaces_fixed_obligation_but_future_points_do_not_certify_absence(fixed):
    manager = ledger(fixed)
    before = copy.deepcopy(manager.pending[11])
    # Merely forecasting an arbitrary-position measurement earns no credit.
    assert manager.proposal(11, (0., 0.), [(20., 0.)]) is not None
    assert manager.pending[11] == before
    assert manager.absent() == set()
    history = []
    record(manager, history, 11, (20., 0.))
    assert manager.remove(11, (0., 0.))
    assert (0., 0.) not in manager.pending[11]
    assert 11 not in manager.absent()
    assert replay_adaptive_artifact(history, manager.artifact())["passed"]
    for point in sorted(manager.pending[11]):
        record(manager, history, 11, point)
    assert 11 in manager.absent()
    audit = replay_adaptive_artifact(history, manager.artifact())
    assert audit["passed"] and audit["certified_absent"] == [11]


def test_failed_deletion_is_transactional_and_budget_exhaustion_earns_no_credit(fixed):
    manager = ledger(fixed)
    before, roots = copy.deepcopy(manager.pending), dict(manager.roots)
    assert not manager.remove(11, (0., 0.))
    assert manager.pending == before and manager.roots == roots
    manager.decision_search_wall_s = manager.decision_wall_s
    assert manager.proposal(11, (0., 0.), [(20., 0.)]) is None
    assert manager.attempts[-1]["status"] == "budget_exhausted"
    assert manager.pending == before and manager.roots == roots


def test_two_individually_optional_supports_cannot_both_be_deleted(fixed):
    augmented = list(fixed)+[(20., 0.)]
    manager, history = ledger(augmented), []
    # Either the original origin or its certified 20m substitute suffices.
    assert manager.proposal(11, (0., 0.)) is not None
    assert manager.proposal(11, (20., 0.)) is not None
    record(manager, history, 11, (5000., 0.))  # Actual, but cannot fill the central coverage hole.
    assert manager.remove(11, (0., 0.))
    before, root = set(manager.pending[11]), manager.roots[11]
    assert not manager.remove(11, (20., 0.))
    assert manager.pending[11] == before and manager.roots[11] == root
    assert replay_adaptive_artifact(history, manager.artifact())["passed"]


def test_exact_verifier_rejects_missing_leaf_invalid_support_and_bad_hash(fixed):
    certificate, _ = certify(fixed)
    broken = copy.deepcopy(certificate)
    broken["leaves"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        verify_exact_certificate(broken)
    broken = copy.deepcopy(certificate)
    leaf = next(v for v in broken["leaves"] if v["kind"] == "covered")
    leaf["stations"] = [0]
    with pytest.raises(ValueError):
        verify_exact_certificate(broken)
    manager = ledger(fixed)
    artifact = manager.artifact()
    root = next(iter(artifact["proofs"]))
    artifact = copy.deepcopy(artifact)
    artifact["proofs"][root]["points"][0][0] += 1
    with pytest.raises(ValueError, match="hash"):
        replay_adaptive_artifact([], artifact)


def test_wrong_channel_nearby_point_or_unexecuted_point_cannot_support_replacement(fixed):
    manager, history = ledger(fixed), []
    record(manager, history, 11, (20., 0.))
    assert manager.remove(11, (0., 0.))
    artifact = manager.artifact()
    wrong = copy.deepcopy(history)
    wrong[0]["channel"] = 12
    with pytest.raises(ValueError, match="unmeasured"):
        replay_adaptive_artifact(wrong, artifact)
    nearby = copy.deepcopy(history)
    nearby[0]["position"][0] += 1e-10
    with pytest.raises(ValueError, match="unmeasured"):
        replay_adaptive_artifact(nearby, artifact)
    with pytest.raises(ValueError, match="triggering"):
        replay_adaptive_artifact([], artifact)


def test_false_actual_absence_claim_is_rejected_even_with_valid_planned_cover(fixed):
    manager, history = ledger(fixed), []
    record(manager, history, 11, (20., 0.))
    assert manager.remove(11, (0., 0.))
    artifact = copy.deepcopy(manager.artifact())
    artifact["certified_absent"] = [11]
    with pytest.raises(ValueError, match="absence"):
        replay_adaptive_artifact(history, artifact)


def test_deadline_rejects_uncached_search_without_certification(fixed):
    points = [(p.x+0.000123, p.y) for p in fixed]
    certificate, diagnostic = certify(points, deadline=time.perf_counter()-1)
    assert certificate is None and diagnostic["status"] == "budget_exhausted"


def test_disabled_switch_preserves_original_controller_public_features(fixed):
    from tests.test_q4_rl_controller import Replies
    first = Q4RLSearch(Replies(), max_expansions=0)
    second = Q4AdaptiveRLSearch(Replies(), adaptive_cover=False, max_expansions=0)
    a, b = first._candidates(), second._candidates()
    assert a == b
    assert first._features(a) == second._features(b)
    assert second.report.learning["algorithm"] == first.report.learning["algorithm"]


def test_ledger_without_any_replacement_preserves_known_channel_scan_actions(fixed):
    from tests.test_q4_rl_controller import Replies
    first = Q4RLSearch(Replies(), max_expansions=0, record_transitions=False).run()
    second = Q4AdaptiveRLSearch(Replies(), replacement_actions=False,
        max_expansions=0, record_transitions=False).run()
    assert first.action_history == second.action_history
    assert first.virtual_time_s == second.virtual_time_s
    assert not second.learning["adaptive_cover"]["events"]
    assert second.learning["adaptive_replay_audit"]["complete"]


def test_new_candidate_has_prior_exact_proof_but_no_actual_credit(fixed):
    from tests.test_q4_rl_controller import Replies
    from simulator_client.state import Position
    client = Replies()
    client.state.position = Position(20., 0.)
    search = Q4AdaptiveRLSearch(client, max_expansions=0,
        cover_decision_wall_s=5., cover_episode_wall_s=30.)
    candidates = search._candidates()
    assert any(point == Position(20., 0.) for point, channel in search._replacement_map)
    assert search.adaptive.absent() == set()
    assert all((0., 0.) in search.adaptive.pending[c] for c in range(1, 21))
    assert search.report.learning["feature_schema"]["version"] != "q4-joint-scan-service-v1"
    assert len(search._features(candidates)[0]) == 10
