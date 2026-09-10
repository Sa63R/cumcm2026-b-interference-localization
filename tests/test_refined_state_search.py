"""The enlarged finite score improves monotonically; actual safety is separate."""

import json
from pathlib import Path

import pytest

from localization.omni import OmniCandidateRegion
from simulation import LocalResearchSimulator, difficult_scenarios, random_scenario
from simulator_client.state import Position
from strategies.refined_state_search import run_refined_state_search
from tests.test_strategy import ObservationOnlyClient


BASE = json.loads((Path(__file__).resolve().parents[1] /
                   "experiments/state_search_candidate_v1.json").read_text())["kwargs"]["config"]


@pytest.mark.parametrize("mode", ["axis_quantile", "local_refine"])
@pytest.mark.parametrize("scenario", difficult_scenarios(3) + [random_scenario(3, 100257)])
def test_nested_probe_candidates_preserve_real_clearance_certificates(scenario, mode):
    sim = LocalResearchSimulator(scenario)
    report = run_refined_state_search(ObservationOnlyClient(sim.client()), config=BASE, mode=mode)
    evaluation = sim.evaluation()
    assert evaluation["all_cleared"] and report.completion_certified_under_model
    assert evaluation["failed_clear_count"] == 0
    for log in report.strategy_parameters["probe_search_log"]:
        assert log["score_improvement_s"] >= -1e-9
        assert log["candidates"] >= log["baseline_candidates"]
        assert log["proposed_candidates"] <= 23
    # Independently reconstruct actual measurement regions at clear time.
    regions, near = {}, {}
    for item in report.action_history:
        c, p = item["channel"], Position(*item["position"])
        if item["action"] == "measure":
            region = regions.setdefault(c, OmniCandidateRegion())
            if item["result"] == "direction":
                region.observe(p, item["bearing_deg"])
            elif item["result"] == "no_signal":
                region.observe_no_signal(p)
            elif item["result"] == "near":
                near[c] = p
        elif item["phase"] == "certified_clear":
            assert all(p.distance_to(Position(*v)) <= 19.9 + 1e-6 for v in regions[c].vertices)
        elif item["phase"] == "near_clear":
            assert p == near[c]
