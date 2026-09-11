"""Collection refinements preserve legal feedback, completeness and budgets."""

import pytest

from geometry import disk_polygon
from localization import CandidateRegion
from practice_control.refinement import _RefinedTriangularSearch, run_collection_search
from simulation import LocalResearchSimulator, Scenario, Source, random_scenario
from tests.test_strategy import ObservationOnlyClient


@pytest.mark.parametrize("problem,variant", [(3, "efficient"), (4, "triangular")])
@pytest.mark.parametrize("seed", [2, 3])
def test_refinement_keeps_every_action_and_completes_without_truth(problem, variant, seed):
    sim = LocalResearchSimulator(random_scenario(problem, seed))
    report = run_collection_search(ObservationOnlyClient(sim.client()), problem=problem, variant=variant)
    evaluation = sim.evaluation()  # Accessible only after the policy exits.
    assert evaluation["all_cleared"]
    assert report.completion_certified_under_model
    assert report.accepted_actions == evaluation["action_count"]
    assert report.measurement_count == evaluation["measurement_count"]
    assert report.virtual_time_s == evaluation["virtual_time_s"]
    assert len(report.action_history) == report.accepted_actions - 2
    extra = [a for a in report.action_history if a["phase"].startswith("dataset_refine_")]
    metadata = report.strategy_parameters["dataset_refinement"]
    assert len(extra) == metadata["extra_measurements"] > 0
    assert len(extra) <= 32
    assert all(count <= 2 for count in metadata["per_channel"].values())
    assert all(a["action"] == "measure" for a in extra)


def test_already_near_sources_need_no_extra_measurement():
    scenario = Scenario("collection-near", 3, 1,
                        tuple(Source(c, 1.0, 1.0, 1000.0) for c in range(1, 17)), "zero")
    sim = LocalResearchSimulator(scenario)
    report = run_collection_search(ObservationOnlyClient(sim.client()), problem=3, variant="efficient")
    assert sim.evaluation()["all_cleared"]
    assert report.strategy_parameters["dataset_refinement"]["extra_measurements"] == 0
    assert report.measurement_count == 20


def test_directional_silence_stops_refinement_and_keeps_legal_clear():
    sources = (Source(1, 10.0, 0.0, 1000.0, 0.0),) + tuple(
        Source(c, 100.0 * c, 0.0, 1000.0) for c in range(2, 11))
    sim = LocalResearchSimulator(Scenario("collection-backface", 4, 1, sources, "zero"))
    client = ObservationOnlyClient(sim.client())
    client.enter()
    search = _RefinedTriangularSearch(client, 4, "triangular", 20000, 6, "center")
    # A synthetic legal prior containing the source; the controller sees only
    # this region and the public client, never the source's chosen orientation.
    region = CandidateRegion()
    region.vertices = disk_polygon((10.0, 0.0), 10.0, outer=True)
    search.regions[1] = region
    search.first_bearings[1] = 180.0
    assert search._clear((0.0, 0.0), 1, "guaranteed_clearance")
    assert [a["result"] for a in search.report.action_history] == ["no_signal", "success"]
    assert search.refinement["extra_measurements"] == 1
    assert search.refinement["no_signal_responses"] == 1
    client.exit()


@pytest.mark.parametrize("maximum", [2, 3, 22, 40])
def test_small_action_budget_keeps_exit(maximum):
    sim = LocalResearchSimulator(random_scenario(3, 2))
    report = run_collection_search(ObservationOnlyClient(sim.client()), problem=3,
                                   variant="efficient", max_actions=maximum)
    assert sim.observation_history()[-1]["action"] == "/exit"
    assert report.accepted_actions <= maximum


@pytest.mark.parametrize("kwargs", [
    dict(problem=3, variant="triangular"), dict(problem=4, variant="efficient"),
    dict(problem=True, variant="efficient"), dict(problem=3, variant="efficient", max_actions=True),
    dict(problem=3, variant="efficient", max_actions=1),
])
def test_invalid_settings_rejected_before_enter(kwargs):
    sim = LocalResearchSimulator(random_scenario(3, 2))
    with pytest.raises(ValueError):
        run_collection_search(ObservationOnlyClient(sim.client()), **kwargs)
    assert sim.observation_history() == []
