"""Truth-isolation, safety, coverage and fallback-time tests without PyTorch."""

import math
import random

import pytest

from research_rl import run_rl_search
from research_rl.controller import CONTEXT_DIM, FEATURE_DIM, DeepRLSearch
from simulation import LocalResearchSimulator, Scenario, Source, difficult_scenarios, random_scenario
from tests.test_strategy import ObservationOnlyClient


@pytest.mark.parametrize("scenario", difficult_scenarios(3) + [random_scenario(3, s) for s in range(100001, 100006)], ids=lambda s: s.case_id)
@pytest.mark.parametrize("selector", ["teacher", "random"])
def test_observation_only_policy_completes_and_accounts_for_all_actions(scenario, selector):
    rng = random.Random(91)
    observed_shapes = []

    def policy(features, context, teacher):
        observed_shapes.append((len(features), len(context)))
        assert len(context) == CONTEXT_DIM
        assert all(len(row) == FEATURE_DIM for row in features)
        assert all(math.isfinite(x) for row in features for x in row)
        return teacher if selector == "teacher" else rng.randrange(len(features))

    simulator = LocalResearchSimulator(scenario)
    result = run_rl_search(ObservationOnlyClient(simulator.client()), policy=policy)
    evaluation = simulator.evaluation()
    assert evaluation["all_cleared"]
    assert evaluation["failed_clear_count"] == 0
    assert result.completion_certified_under_model
    assert result.virtual_time_s == evaluation["virtual_time_s"]
    assert result.accepted_actions == evaluation["action_count"]
    assert sum(result.time_breakdown.values()) == pytest.approx(result.virtual_time_s, abs=0.001)
    assert result.learning["decisions"] == len(observed_shapes)
    assert result.learning["decisions"] == sum(result.learning["action_counts"].values())
    assert simulator.observation_history()[-1]["action"] == "/exit"


def test_decision_limit_finishes_inherited_policy_and_charges_tail_time():
    simulator = LocalResearchSimulator(random_scenario(3, 100014))
    result = run_rl_search(ObservationOnlyClient(simulator.client()),
                           policy=lambda f, c, t: t, max_decisions=1)
    assert simulator.evaluation()["all_cleared"]
    assert result.learning["decisions"] == 1
    assert result.learning["fallback_counts"]["decision_limit"] == 1
    assert result.learning["fallback_virtual_time_s"] > 0
    assert result.learning["fallback_actions"] > 0


def test_final_sixteenth_clear_is_recorded_even_when_certificate_stops_session():
    sources = tuple(Source(c, math.cos(c), math.sin(c), 1000) for c in range(1, 17))
    simulator = LocalResearchSimulator(Scenario("rl-sixteen-near", 3, 100002, sources))
    costs = []
    search = DeepRLSearch(ObservationOnlyClient(simulator.client()), lambda f, c, t: t,
                          recorder=lambda *args: costs.append(args[-1]))
    result = search.run()
    assert simulator.evaluation()["all_cleared"]
    assert len(costs) == 16
    assert sum(costs) + result.learning["initial_scan_virtual_time_s"] == pytest.approx(result.virtual_time_s)


def test_early_action_limit_does_not_certify_success():
    simulator = LocalResearchSimulator(random_scenario(3, 100015))
    result = run_rl_search(simulator.client(), policy=lambda f, c, t: t, max_actions=22)
    assert not result.completion_certified_under_model
    assert result.completion_reason == "action_budget"
    assert simulator.observation_history()[-1]["action"] == "/exit"


def test_q4_rejected_before_session_entry():
    simulator = LocalResearchSimulator(random_scenario(4, 100017))
    with pytest.raises(ValueError, match="Q3 only"):
        run_rl_search(simulator.client(), problem=4, policy=lambda f, c, t: t)
    assert simulator.observation_history() == []


def test_invalid_policy_index_is_rejected_and_session_exits():
    simulator = LocalResearchSimulator(random_scenario(3, 100018))
    with pytest.raises(ValueError, match="invalid candidate"):
        run_rl_search(simulator.client(), policy=lambda f, c, t: len(f))
    assert simulator.observation_history()[-1]["action"] == "/exit"
