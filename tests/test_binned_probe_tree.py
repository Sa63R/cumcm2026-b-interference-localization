"""Circular observation-bin likelihoods and continuous bounded-noise checks."""

import pytest

from localization.omni import OmniCandidateRegion
from planning.binned_probe_tree import BinnedProbeTree
from planning.probe_tree import WeightedSource
from simulator_client.state import Position


@pytest.mark.parametrize("width", [.5, 1.0])
def test_binning_wraps_zero_and_preserves_each_sources_prior_mass(width):
    region = OmniCandidateRegion().observe((0, 0), 0)
    support = (WeightedSource(Position(900, -2), .2),
               WeightedSource(Position(950, 1), .3),
               WeightedSource(Position(1000, 4), .5))
    tree = BinnedProbeTree(bin_width_deg=width)
    branches = tree._branches(region, Position(0, 0), support)
    assert sum(chance for chance, _, _ in branches) == pytest.approx(1)
    masses = {}
    for chance, updated, posterior in branches:
        assert sum(source.weight for source in posterior) == pytest.approx(1)
        assert updated.error_deg == 1.005 + width / 2
        for source in posterior:
            masses[source.position] = masses.get(source.position, 0) + chance * source.weight
            assert updated.contains(source.position, tolerance=1e-5)
    assert masses == pytest.approx({source.position: source.weight for source in support})
    assert region.error_deg == 1.005  # Predictions never coarsen live geometry.
    assert tree.singleton_branches < tree.branch_count


def test_binned_pruning_matches_fully_enumerated_two_step_tree():
    region = OmniCandidateRegion().observe((0, 0), 25).observe((250, -200), 41)
    settings = dict(bin_width_deg=.5, depth=2, supports=6, candidates=3,
                    inner_candidates=3, max_expansions=100000, first_bearing=25)
    _, pruned = BinnedProbeTree(**settings).choose(region, Position(250, -200), set())
    _, full = BinnedProbeTree(**settings, use_bounds=False).choose(region, Position(250, -200), set())
    assert pruned["exact_for_declared_tree"] and full["exact_for_declared_tree"]
    assert pruned["finite_model_cost_s"] == pytest.approx(full["finite_model_cost_s"], abs=1e-7)


@pytest.mark.parametrize("case_index", [0, 4, 6])
def test_coarse_predictions_use_fine_actual_data_and_complete_hard_cases(case_index):
    from simulation import LocalResearchSimulator, difficult_scenarios
    from strategies.binned_state_search import run_binned_state_search
    from tests.test_strategy import ObservationOnlyClient
    sim = LocalResearchSimulator(difficult_scenarios(3)[case_index])
    result = run_binned_state_search(ObservationOnlyClient(sim.client()), config={
        "max_expansions": 100, "scan_source_s": 0,
        "opportunistic_measurements": True, "active_probe_search": True,
        "probe_depth": 1, "probe_supports": 6})
    evaluation = sim.evaluation()
    assert evaluation["all_cleared"] and result.completion_certified_under_model
    assert evaluation["failed_clear_count"] == 0
    assert result.measurement_count == evaluation["measurement_count"]
    assert any(log.get("observation_bin_deg") == .5 for log in result.strategy_parameters["probe_search_log"])
