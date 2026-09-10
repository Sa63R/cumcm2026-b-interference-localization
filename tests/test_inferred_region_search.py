"""The two independent mechanisms retain their frozen implementations and ledger."""

import pytest

from localization.omni import OmniCandidateRegion
from planning.silence_certificate import certify_silence
from simulation import LocalResearchSimulator, difficult_scenarios, random_scenario
from simulator_client.state import Position
from strategies.inferred_state_search import InferredSilenceSearch
from strategies.inferred_region_state_search import InferredMeanPointSearch, run_inferred_mean_point_state_search
from strategies.refined_state_search import RefinedStateSearch
from strategies.region_state_search import RegionStateSearch, run_region_state_search
from tests.test_inferred_silence import BASE
from tests.test_strategy import ObservationOnlyClient


def test_combination_reuses_frozen_methods_without_new_probe_or_scan_logic():
    assert InferredMeanPointSearch._next_task is RegionStateSearch._next_task
    assert InferredMeanPointSearch._scan is InferredSilenceSearch._scan
    assert InferredMeanPointSearch._next_probe is RefinedStateSearch._next_probe
    assert InferredMeanPointSearch._resolve is RegionStateSearch._resolve


@pytest.mark.parametrize("seed", [111001, 111002])
def test_disabled_combination_is_identical_to_original_mean_point(seed):
    scenario = random_scenario(3, seed)
    original = run_region_state_search(ObservationOnlyClient(LocalResearchSimulator(scenario).client()),
                                      config=BASE, mode="mean_point")
    disabled = run_inferred_mean_point_state_search(ObservationOnlyClient(LocalResearchSimulator(scenario).client()),
                                                   config=BASE, enabled=False)
    assert original.action_history == disabled.action_history
    assert original.virtual_time_s == disabled.virtual_time_s
    assert disabled.strategy_parameters["inferred_no_signal_constraints"] == []


@pytest.mark.parametrize("scenario", difficult_scenarios(3) + [random_scenario(3, 111001)])
def test_legal_inference_physical_coverage_and_clearance_are_preserved(scenario):
    sim = LocalResearchSimulator(scenario)
    report = run_inferred_mean_point_state_search(ObservationOnlyClient(sim.client()), config=BASE)
    evaluation = sim.evaluation()
    assert evaluation["all_cleared"] and report.completion_certified_under_model
    assert evaluation["failed_clear_count"] == 0
    assert evaluation["measurement_count"] == sum(a["action"] == "measure" for a in report.action_history)
    assert evaluation["virtual_time_s"] == report.virtual_time_s
    known, cleared, regions, actual = set(), set(), {}, {}
    constraints = report.strategy_parameters["inferred_no_signal_constraints"]
    cursor = 0
    for index, action in enumerate(report.action_history):
        while cursor < len(constraints) and constraints[cursor]["after_actual_action_count"] == index:
            inferred = constraints[cursor]
            c = inferred["channel"]
            assert c in known and c not in cleared and not inferred["physical_measurement"]
            certificate = certify_silence(regions[c], inferred["position"])
            assert certificate and certificate["distance_lower_m"] == inferred["distance_lower_m"]
            regions[c].observe_no_signal(inferred["position"])
            cursor += 1
        c = action["channel"]
        p = Position(*action["position"])
        if action["action"] == "measure":
            region = regions.setdefault(c, OmniCandidateRegion())
            actual.setdefault(c, {})[tuple(action["position"])] = action["result"]
            if action["result"] == "direction":
                region.observe(p, action["bearing_deg"])
                known.add(c)
            elif action["result"] == "near":
                known.add(c)
            elif action["result"] == "no_signal":
                region.observe_no_signal(p)
        elif action["action"] == "clear":
            if action["phase"] == "certified_clear":
                assert all(p.distance_to(Position(*v)) <= 19.9 + 1e-6 for v in regions[c].vertices)
            if action["result"] == "success":
                cleared.add(c)
    assert cursor == len(constraints)
    if len(known) < 16:
        for c in set(range(1, 21)) - known:
            assert all(actual[c].get(tuple(p)) == "no_signal" for p in report.coverage_points)
    assert all(not any(a["phase"] == "coverage" and a["channel"] == i["channel"]
                       and a["position"] == i["position"] for a in report.action_history) for i in constraints)
    assert report.strategy_parameters["region_travel_mode"] == "mean_point"


@pytest.mark.parametrize("kwargs", [{"problem":4}, {"enabled":1}, {"max_actions":True}, {"max_active_probes":31}])
def test_bad_combination_options_cannot_start_an_environment(kwargs):
    with pytest.raises(ValueError):
        run_inferred_mean_point_state_search(None, **kwargs)
