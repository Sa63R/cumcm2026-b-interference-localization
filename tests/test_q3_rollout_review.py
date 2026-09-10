"""Independent checks for cheap incomplete branches and planning reserves."""

import copy
import time
from types import SimpleNamespace

import pytest

from simulation import LocalResearchSimulator, Scenario, Source
from simulation.q3_branch import make_q3_branch
from strategies import q3_belief
from strategies.efficient import EfficientSearch
from strategies.rollout import RolloutSearch, _Continuation
from tests.test_strategy import ObservationOnlyClient


def prepared_policy():
    hypothesis = Scenario("review-ten-near", 3, 51,
                          tuple(Source(c, 0, 0, 1000) for c in range(1, 11)), "zero")
    client = LocalResearchSimulator(hypothesis).client()
    client.enter()
    policy = RolloutSearch(ObservationOnlyClient(client), 20000, 6,
                           {"particles": 1, "candidates": 2, "max_searches": 1})
    policy.actions = 1
    policy._scan(policy.points[0])
    return hypothesis, client, policy, list(policy.points[1:])


def test_lower_cost_incomplete_alternative_cannot_displace_complete_baseline(monkeypatch):
    hypothesis, real_client, policy, remaining = prepared_policy()
    before = copy.deepcopy(real_client.state)
    history_before = copy.deepcopy(policy.report.action_history)
    baseline = EfficientSearch._next_task(policy, remaining)
    monkeypatch.setattr(q3_belief, "sample_worlds", lambda *args, **kwargs: [hypothesis])
    calls = []

    def fake_run(branch):
        calls.append(branch.first_task)
        # The alternative is almost free only because its work was truncated.
        complete = len(calls) == 1
        branch.client.exit()
        return SimpleNamespace(completion_certified_under_model=complete,
                               completion_reason="coverage_exhausted_and_all_detected_cleared"
                               if complete else "action_budget",
                               error=None, exit_error=None,
                               virtual_time_s=before.virtual_time_s + (1000 if complete else 1))

    monkeypatch.setattr(_Continuation, "run", fake_run)
    assert policy._next_task(remaining) == baseline
    assert len(calls) == 2 and calls[0] != calls[1]
    assert policy.report.planning["changed_decisions"] == 0
    assert policy.report.planning["fallback_counts"]["incomplete_continuation"] == 1
    assert real_client.state == before
    assert policy.report.action_history == history_before
    real_client.exit()


@pytest.mark.parametrize("remaining_seconds", [0.1, 5.0])
def test_real_time_reserve_skips_sampling_and_branch_search(monkeypatch, remaining_seconds):
    _, real_client, policy, remaining = prepared_policy()
    baseline = EfficientSearch._next_task(policy, remaining)
    before = copy.deepcopy(real_client.state)

    def forbidden(*args, **kwargs):
        raise AssertionError("Sampling must not start inside the real-time reserve")

    monkeypatch.setattr(q3_belief, "sample_worlds", forbidden)
    monkeypatch.setattr(_Continuation, "run", forbidden)
    real_client.state.real_deadline = real_client._clock() + remaining_seconds
    expected_deadline = real_client.state.real_deadline
    assert policy._next_task(remaining) == baseline
    assert policy.report.planning["searches"] == 0
    assert policy.report.planning["candidate_evaluations"] == 0
    assert policy.report.planning["fallback_counts"]["compute_budget"] == 1
    before.real_deadline = expected_deadline
    assert real_client.state == before
    real_client.exit()


def test_continuation_compute_timeout_preserves_exit_and_does_not_certify():
    hypothesis, real_client, policy, remaining = prepared_policy()
    baseline = EfficientSearch._next_task(policy, remaining)
    before = copy.deepcopy(real_client.state)
    with make_q3_branch(hypothesis, real_client.state, policy.report.action_history) as client:
        branch = _Continuation(policy, ObservationOnlyClient(client), remaining,
                               baseline, time.perf_counter() - 1)
        report = branch.run()
        assert client.state.session == "exited"
        assert report.virtual_time_s == before.virtual_time_s
        assert report.accepted_actions == policy.actions + 1  # explicit exit only
        assert report.action_history == []
        assert report.completion_reason == "rollout_compute_budget"
        assert not report.completion_certified_under_model
        assert not report.coverage_complete
        assert report.exit_error is None
    assert real_client.state == before
    real_client.exit()


def test_continuation_inherits_remaining_action_budget_instead_of_resetting_it():
    hypothesis, real_client, policy, remaining = prepared_policy()
    policy.max_actions = policy.actions + 1  # only /exit remains affordable
    baseline = EfficientSearch._next_task(policy, remaining)
    before = copy.deepcopy(real_client.state)
    with make_q3_branch(hypothesis, real_client.state, policy.report.action_history) as client:
        report = _Continuation(policy, client, remaining, baseline,
                               time.perf_counter() + 10).run()
        assert client.state.session == "exited"
        assert report.completion_reason == "action_budget"
        assert not report.completion_certified_under_model
        assert report.accepted_actions == policy.max_actions
        assert report.virtual_time_s == before.virtual_time_s
        assert report.action_history == []
    assert real_client.state == before
    real_client.exit()
