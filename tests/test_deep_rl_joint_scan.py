"""v3 unknown-channel control, exact teacher replay, and coverage certificates."""

import math
import random

import pytest

from research_rl import run_rl_search
from research_rl.controller import FEATURE_DIMS
from research_rl.joint_scan import JointScanRLSearch
from simulation import LocalResearchSimulator, Scenario, Source, random_scenario, difficult_scenarios
from strategies import run_search
from tests.test_strategy import ObservationOnlyClient


@pytest.mark.parametrize("scenario", difficult_scenarios(3) + [random_scenario(3, s) for s in range(100201, 100207)], ids=lambda s: s.case_id)
@pytest.mark.parametrize("selector", ["teacher", "random"])
def test_joint_policy_completes_with_observation_only_ledger(scenario, selector):
    rng = random.Random(91)
    def policy(features, context, teacher):
        assert all(len(row) == FEATURE_DIMS["v3"] for row in features)
        assert all(math.isfinite(x) for row in features for x in row)
        return teacher if selector == "teacher" else rng.randrange(len(features))
    simulator = LocalResearchSimulator(scenario)
    result = run_rl_search(ObservationOnlyClient(simulator.client()), policy=policy,
                           feature_version="v3", max_decisions=256)
    truth = simulator.evaluation()
    assert truth["all_cleared"]
    assert truth["failed_clear_count"] == 0
    assert result.completion_certified_under_model
    assert result.virtual_time_s == truth["virtual_time_s"]
    assert result.learning["initial_scan_virtual_time_s"] == 0
    assert sum(result.time_breakdown.values()) == pytest.approx(result.virtual_time_s, abs=0.001)
    assert result.learning["decisions"] <= 140 + 16 * 7
    assert result.learning["fallback_counts"].get("decision_limit", 0) == 0
    # Verify actual observations independently, rather than trusting the ledger.
    if result.cleared_count != 16:
        for point in result.coverage_points:
            scanned = {a["channel"] for a in result.action_history
                       if a["action"] == "measure" and a["position"] == point}
            assert scanned | set(result.cleared_channels) == set(range(1, 21))


@pytest.mark.parametrize("seed", range(100211, 100221))
def test_teacher_reproduces_efficient_action_sequence_exactly(seed):
    scenario = random_scenario(3, seed)
    first = LocalResearchSimulator(scenario)
    efficient = run_search(first.client(), variant="efficient")
    second = LocalResearchSimulator(scenario)
    joint = run_rl_search(second.client(), policy=lambda f,c,t: t, feature_version="v3")
    def actions(report):
        return [(r["action"], r["channel"], r["position"], r["result"]) for r in report.action_history]
    assert actions(efficient) == actions(joint)
    assert efficient.virtual_time_s == joint.virtual_time_s


def test_actor_can_interrupt_initial_scan_and_avoid_last_four_empty_channels():
    scenario = Scenario("joint-sixteen-near", 3, 100230,
        tuple(Source(c, math.cos(c), math.sin(c), 1000) for c in range(1, 17)))
    simulator = LocalResearchSimulator(scenario)
    def immediate_clear(features, context, teacher):
        clears = [i for i, row in enumerate(features) if row[2] == 1]
        if clears:
            return clears[0]
        here = [i for i, row in enumerate(features) if row[44] == 1 and row[50] == 1]
        return min(here, key=lambda i: features[i][45])
    result = run_rl_search(ObservationOnlyClient(simulator.client()), policy=immediate_clear,
                           feature_version="v3")
    assert simulator.evaluation()["all_cleared"]
    assert result.virtual_time_s == 175
    assert result.measurement_count == 16
    assert result.learning["interrupted_scans"] > 0
    assert result.completion_reason == "source_count_upper_bound_reached"


def test_initial_action_is_not_forced_to_origin_or_channel_one():
    simulator = LocalResearchSimulator(random_scenario(3, 100231))
    def policy(features, context, teacher):
        if context[9] == 0:
            return len(features) - 1
        return teacher
    result = run_rl_search(simulator.client(), policy=policy, feature_version="v3")
    assert result.action_history[0]["position"] != [0.0, 0.0]
    assert result.action_history[0]["channel"] == 20
    assert simulator.evaluation()["all_cleared"]


def test_decision_limit_fills_only_pending_pairs_and_counts_full_tail():
    simulator = LocalResearchSimulator(random_scenario(3, 100232))
    costs = []
    controller = JointScanRLSearch(ObservationOnlyClient(simulator.client()), lambda f,c,t: t,
        max_decisions=1, recorder=lambda *args: costs.append(args[-1]))
    result = controller.run()
    assert simulator.evaluation()["all_cleared"]
    assert len(costs) == 1
    assert sum(costs) + result.learning["fallback_virtual_time_s"] == pytest.approx(result.virtual_time_s)
    scan_pairs = [(tuple(a["position"]), a["channel"]) for a in result.action_history if a["phase"] == "coverage"]
    assert len(scan_pairs) == len(set(scan_pairs))


def test_scan_features_distinguish_same_point_channels_after_one_observation():
    simulator = LocalResearchSimulator(random_scenario(3, 100233))
    controller = JointScanRLSearch(simulator.client(), lambda f,c,t: t)
    controller.client.enter()
    remaining = list(controller.points)
    candidates = controller._candidates(remaining)
    assert len(candidates) == 140
    features, _ = controller._features(candidates, remaining)
    assert features[0][45] != features[1][45]
    assert features[0][53:] == [1] * 7
    controller._execute_candidate(candidates[0], remaining)
    updated = controller._candidates(remaining)
    assert not any(c.kind == "cover" and c.point == candidates[0].point
                   and c.channel == candidates[0].channel for c in updated)
    other_point = next(i for i, c in enumerate(updated)
                      if c.kind == "cover" and c.channel == candidates[0].channel)
    features, _ = controller._features(updated, remaining)
    assert features[other_point][53] == 0
    controller.client.exit()


def test_budget_stops_without_claiming_coverage_or_success():
    simulator = LocalResearchSimulator(random_scenario(3, 100234))
    result = run_rl_search(simulator.client(), policy=lambda f,c,t: t,
                           feature_version="v3", max_actions=3)
    assert result.accepted_actions == 3
    assert result.completion_reason == "action_budget"
    assert not result.coverage_complete
    assert not result.completion_certified_under_model
    assert simulator.observation_history()[-1]["action"] == "/exit"


def test_invalid_all_negative_endpoint_cannot_certify_zero_sources():
    simulator = LocalResearchSimulator(random_scenario(3, 100235))
    client = simulator.client()
    original = client.measure
    def broken_measure(position, channel):
        response = original(position, channel)
        return {key: value for key, value in dict(response, measure_result="no_signal").items()
                if key != "svd_deg"}
    client.measure = broken_measure
    result = run_rl_search(ObservationOnlyClient(client), policy=lambda f,c,t: t, feature_version="v3")
    assert result.coverage_complete
    assert not result.completion_certified_under_model
    assert result.completion_reason == "source_count_inconsistent"
