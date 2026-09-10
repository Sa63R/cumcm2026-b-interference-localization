"""Analytic full-disk cover and legal-history adaptive-layout checks."""

import json
from pathlib import Path

import pytest

from planning.disk_cover import disk_cover_radius
from planning.ring_layout import ring_cover_radius, ring_sites
from simulation import LocalResearchSimulator, difficult_scenarios, random_scenario
from simulator_client.state import Position
from strategies.layout_state_search import run_layout_state_search
from tests.test_strategy import ObservationOnlyClient


BASE = json.loads((Path(__file__).resolve().parents[1] /
                   "experiments/state_search_candidate_v1.json").read_text())["kwargs"]["config"]


@pytest.mark.parametrize("radius", [1123, 1150, 1250, 1732])
def test_phase_invariant_formula_matches_full_voronoi_witnesses(radius):
    for phase in (0, 7.3, 15, 40, -987.5):
        sites = ring_sites(radius, phase)
        assert disk_cover_radius(sites) == pytest.approx(ring_cover_radius(radius), abs=1e-6)
        assert ring_cover_radius(radius) < 1000


@pytest.mark.parametrize("scenario", difficult_scenarios(3) + [random_scenario(3, 100129)])
def test_layout_selection_uses_only_observations_and_retains_discovery_certificate(scenario):
    sim = LocalResearchSimulator(scenario)
    report = run_layout_state_search(ObservationOnlyClient(sim.client()), config=BASE,
        layout_radii=(1123, 1150, 1250), reselect_before_outer=True,
        layout_max_expansions=10)
    evaluation = sim.evaluation()
    assert evaluation["all_cleared"] and report.completion_certified_under_model
    assert evaluation["failed_clear_count"] == 0
    assert report.error is None and report.exit_error is None
    for log in report.strategy_parameters["layout_log"]:
        assert log["frozen_union_lower_bound_s"] <= log["frozen_union_upper_bound_s"] + 1e-7
        for candidate in log["candidates"]:
            if candidate["cover_radius_m"] is not None:
                assert candidate["cover_radius_m"] < 1000
    if len(report.cleared_channels) < 16:
        for channel in set(range(1, 21)) - set(report.cleared_channels):
            negatives = [Position(*item["position"]) for item in report.action_history
                         if item["action"] == "measure" and item["channel"] == channel
                         and item["result"] == "no_signal"]
            assert disk_cover_radius(negatives) <= 1000 + 1e-6


@pytest.mark.parametrize("options", [{"problem": 4}, {"layout_radii": [1122]},
    {"phase_step_deg": 10.0}, {"phase_step_deg": True}, {"reselect_before_outer": 1}])
def test_invalid_layout_configuration_rejects_before_actions(options):
    sim = LocalResearchSimulator(random_scenario(3, 100129))
    with pytest.raises(ValueError):
        run_layout_state_search(sim.client(), **options)
    assert not sim.observation_history()
