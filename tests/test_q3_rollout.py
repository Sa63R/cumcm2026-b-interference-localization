"""Q3 planning must preserve observations, certification and real action costs."""

import copy
import math
import time

import pytest

from simulation import LocalResearchSimulator, Scenario, Source, random_scenario
from strategies import run_search
from strategies.efficient import EfficientSearch
from strategies.rollout import RolloutSearch, _Continuation
from tests.test_strategy import ObservationOnlyClient


def run(seed, **config):
    sim = LocalResearchSimulator(random_scenario(3, seed))
    report = run_search(ObservationOnlyClient(sim.client()), variant="rollout",
                        rollout_config=config)
    return sim, report


@pytest.mark.parametrize("config", [
    {"max_searches": 0}, {"max_candidate_evaluations": 0}, {"max_planning_s": 0},
])
def test_disabled_planning_is_exactly_frozen_efficient(config):
    reference = LocalResearchSimulator(random_scenario(3, 42))
    base = run_search(ObservationOnlyClient(reference.client()), variant="efficient")
    sim, report = run(42, **config)
    assert report.action_history == base.action_history
    assert report.virtual_time_s == base.virtual_time_s
    assert report.planning["candidate_evaluations"] == 0
    assert report.completion_certified_under_model and sim.evaluation()["all_cleared"]


@pytest.mark.parametrize("options", [
    {"problem": 4}, {"active_policy": "minimax"}, {"efficient_config": {}},
    {"rollout_config": {"particles": 0}}, {"rollout_config": {"particles": True}},
    {"rollout_config": {"candidates": 1}}, {"rollout_config": {"max_planning_s": math.inf}},
    {"rollout_config": {"seed": -1}}, {"rollout_config": {"new_option": 1}},
])
def test_invalid_or_q4_configuration_never_enters(options):
    sim = LocalResearchSimulator(random_scenario(3, 42))
    with pytest.raises(ValueError):
        run_search(ObservationOnlyClient(sim.client()), variant="rollout", **options)
    assert sim.observation_history() == []


def test_rollout_config_cannot_silently_affect_q4():
    sim = LocalResearchSimulator(random_scenario(4, 42))
    with pytest.raises(ValueError):
        run_search(sim.client(), problem=4, variant="triangular", rollout_config={})
    assert sim.observation_history() == []


def test_sampling_failure_uses_baseline_without_real_probe(monkeypatch):
    from strategies import q3_belief

    def failed(*args, **kwargs):
        raise q3_belief.BeliefSamplingError("injected empty proposal pool")

    monkeypatch.setattr(q3_belief, "sample_worlds", failed)
    reference = LocalResearchSimulator(random_scenario(3, 43))
    base = run_search(reference.client(), variant="efficient")
    sim, report = run(43)
    assert report.action_history == base.action_history
    assert report.planning["fallback_counts"]["belief_sampling"] > 0
    assert report.planning["candidate_evaluations"] == 0
    assert sim.evaluation()["all_cleared"]


def test_continuation_charges_only_remaining_work_and_keeps_parent_isolated():
    from simulation.q3_branch import make_q3_branch

    scenario = random_scenario(3, 44)
    sim = LocalResearchSimulator(scenario)
    client = sim.client()
    client.enter()
    parent = EfficientSearch(ObservationOnlyClient(client), 20000, 6, None)
    parent.actions = 1
    parent._scan(parent.points[0])
    state_before = copy.deepcopy(client.state)
    history_before = copy.deepcopy(parent.report.action_history)
    remaining = parent.points[1:]
    first = parent._next_task(remaining)
    # The evaluator supplies truth ONLY here to test a deterministic fork.
    # Production policy constructs hypotheses from observation history instead.
    with make_q3_branch(scenario, client.state, history_before) as branch_client:
        branch = _Continuation(parent, ObservationOnlyClient(branch_client), remaining,
                               first, time.perf_counter() + 30)
        result = branch.run()
    assert client.state == state_before
    assert parent.report.action_history == history_before
    assert result.completion_certified_under_model
    assert result.virtual_time_s > state_before.virtual_time_s
    assert result.coverage_points_visited == 7 or result.cleared_count == 16
    # Baseline from the same start must reach the same final time, including
    # localization, scans of absent channels, and accepted explicit exit.
    full = LocalResearchSimulator(scenario)
    reference = run_search(full.client(), variant="efficient")
    assert result.virtual_time_s == reference.virtual_time_s
    assert result.accepted_actions == reference.accepted_actions
    assert result.action_history == reference.action_history[len(history_before):]
    client.exit()


def test_branch_does_not_stop_when_all_ten_hypothetical_sources_are_cleared():
    from simulation.q3_branch import make_q3_branch

    scenario = Scenario("ten-at-origin", 3, 0,
                        tuple(Source(c, 0, 0, 1000) for c in range(1, 11)), "zero")
    sim = LocalResearchSimulator(scenario)
    client = sim.client()
    client.enter()
    parent = EfficientSearch(client, 20000, 6, None)
    parent.actions = 1
    parent._scan(parent.points[0])
    for channel in range(1, 11):
        assert parent._resolve(channel)
    start_time = client.state.virtual_time_s
    remaining = parent.points[1:]
    with make_q3_branch(scenario, client.state, parent.report.action_history) as branch_client:
        result = _Continuation(parent, branch_client, remaining, parent._next_task(remaining),
                               time.perf_counter() + 30).run()
    assert result.virtual_time_s > start_time + 6 * 10 * 5
    assert result.measurement_count == 60
    assert result.coverage_complete and result.coverage_points_visited == 7
    assert result.completion_reason == "coverage_exhausted_and_all_detected_cleared"
    client.exit()


@pytest.mark.parametrize("seed", [45, 46])
def test_real_rollout_is_observation_only_complete_and_bounded(seed):
    sim, report = run(seed, particles=2, max_searches=2, max_candidate_evaluations=12)
    assert report.problem == 3 and report.variant == "rollout"
    assert report.completion_certified_under_model
    assert sim.evaluation()["all_cleared"]
    assert 0 < report.planning["candidate_evaluations"] <= 12
    assert report.planning["searches"] <= 2
    assert report.accepted_actions == len(sim.observation_history())
    assert report.measurement_count == sim.evaluation()["measurement_count"]
    assert sum(report.time_breakdown.values()) == pytest.approx(report.virtual_time_s, abs=1e-4)
    assert sim.observation_history()[-1]["action"] == "/exit"
    # Hypothetical actions are not in the real session or its time ledger.
    assert report.virtual_time_s == sim.evaluation()["virtual_time_s"]
    data = report.as_dict()
    assert data["planning"]["algorithm"] == "root_macro_task_monte_carlo_rollout"
    assert data["strategy_parameters"]["particles"] == 2


def test_equal_legal_histories_produce_equal_choices_despite_hidden_world_changes():
    # All sources are outside reception range at the origin in both worlds.
    # Policies get identical histories but very different unobserved locations.
    histories, decisions = [], []
    for offset in (0.0, 0.17):
        scenario = Scenario("hidden-rotation", 3, 0,
            tuple(Source(c, 1750 * math.cos(c + offset),
                         1750 * math.sin(c + offset), 1000) for c in range(1, 11)))
        sim = LocalResearchSimulator(scenario)
        client = sim.client()
        client.enter()
        policy = RolloutSearch(ObservationOnlyClient(client), 20000, 6,
                               {"particles": 1, "max_searches": 1})
        policy.actions = 1
        policy._scan(policy.points[0])
        histories.append(copy.deepcopy(policy.report.action_history))
        decisions.append(policy._next_task(policy.points[1:]))
        client.exit()
    assert histories[0] == histories[1]
    assert decisions[0] == decisions[1]
