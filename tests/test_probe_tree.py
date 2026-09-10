"""Finite-tree numerical optimality and source-state compression invariants."""

import pytest

from geometry import polygon_area
from localization.omni import OmniCandidateRegion
from planning.probe_tree import ProbeTree, WeightedSource, polygon_quadrature
from simulator_client.state import Position


def _region():
    region = OmniCandidateRegion()
    region.observe((0, 0), 25)
    region.observe((250, -200), 41)
    assert region.vertices
    return region


def test_polygon_quadrature_preserves_area_centroid_and_total_probability():
    region = OmniCandidateRegion()
    region.vertices = ((0., 0.), (90., 0.), (0., 30.))
    for n in (1, 2, 3, 8):
        support = polygon_quadrature(region, n)
        assert len(support) == n
        assert sum(s.weight for s in support) == pytest.approx(1)
        assert sum(s.weight * s.position.x for s in support) == pytest.approx(30)
        assert sum(s.weight * s.position.y for s in support) == pytest.approx(10)
        assert all(region.contains(s.position) for s in support)


def test_observation_grouping_preserves_probability_and_bayes_mass():
    region = _region()
    support = polygon_quadrature(region, 4)
    tree = ProbeTree(noise_nodes=3)
    point = Position(*region.enclosing_disk().center)
    branches = tree._branches(region, point, support)
    assert sum(chance for chance, _, _ in branches) == pytest.approx(1)
    reconstructed = {}
    for chance, updated, posterior in branches:
        assert sum(source.weight for source in posterior) == pytest.approx(1)
        for source in posterior:
            reconstructed[source.position] = reconstructed.get(source.position, 0) + chance * source.weight
            assert updated is None or updated.contains(source.position, tolerance=1e-5)
    assert reconstructed == pytest.approx({s.position: s.weight for s in support})


@pytest.mark.parametrize("depth,noise", [(1, 1), (1, 3), (2, 1)])
def test_oracle_pruning_matches_unpruned_finite_tree(depth, noise):
    region = _region()
    current = Position(250, -200)
    settings = dict(depth=depth, noise_nodes=noise, candidates=5, inner_candidates=3,
                    max_expansions=100000, first_bearing=25)
    _, bounded = ProbeTree(**settings).choose(region, current, set())
    _, exhaustive = ProbeTree(**settings, use_bounds=False).choose(region, current, set())
    assert bounded["exact_for_declared_tree"] and exhaustive["exact_for_declared_tree"]
    assert bounded["finite_model_cost_s"] == pytest.approx(exhaustive["finite_model_cost_s"], abs=1e-7)
    assert bounded["oracle_lower_bound_s"] <= bounded["finite_model_cost_s"] + 1e-7


def test_more_depth_and_nested_actions_do_not_worsen_exact_fixed_model():
    region = _region()
    current = Position(250, -200)
    values = {}
    for depth, actions in ((1, 3), (1, 5), (2, 5)):
        _, log = ProbeTree(depth=depth, candidates=actions, inner_candidates=3,
                           first_bearing=25, max_expansions=100000).choose(region, current, set())
        assert log["exact_for_declared_tree"]
        values[depth, actions] = log["finite_model_cost_s"]
    assert values[1, 5] <= values[1, 3] + 1e-7
    assert values[2, 5] <= values[1, 5] + 1e-7


def test_budgeted_search_still_returns_finite_feasible_upper_cost():
    region = _region()
    current = Position(250, -200)
    _, limited = ProbeTree(depth=2, first_bearing=25, max_expansions=0).choose(region, current, set())
    _, exact = ProbeTree(depth=2, first_bearing=25, max_expansions=100000).choose(region, current, set())
    assert limited["finite_model_cost_s"] >= exact["finite_model_cost_s"] - 1e-7
    assert limited["oracle_lower_bound_s"] <= exact["finite_model_cost_s"] + 1e-7
    assert not limited["exact_for_declared_tree"]


@pytest.mark.parametrize("depth,noise", [(1, 1), (1, 3), (2, 1)])
@pytest.mark.parametrize("case_index", [0, 4, 6])
def test_finite_tree_controller_keeps_real_clearance_certificates(depth, noise, case_index):
    from simulation import LocalResearchSimulator, difficult_scenarios
    from strategies.state_search import run_state_search
    from tests.test_strategy import ObservationOnlyClient
    sim = LocalResearchSimulator(difficult_scenarios(3)[case_index])
    result = run_state_search(ObservationOnlyClient(sim.client()), config={
        "max_expansions": 100, "scan_source_s": 0, "opportunistic_measurements": True,
        "active_probe_search": True, "probe_model": "optical_tree",
        "probe_depth": depth, "probe_noise_nodes": noise})
    evaluation = sim.evaluation()
    assert evaluation["all_cleared"] and result.completion_certified_under_model
    assert evaluation["failed_clear_count"] == 0
    assert result.virtual_time_s == evaluation["virtual_time_s"]
    regions, near = {}, {}
    for item in result.action_history:
        channel, p = item["channel"], Position(*item["position"])
        if item["action"] == "measure":
            region = regions.setdefault(channel, OmniCandidateRegion())
            if item["result"] == "direction":
                region.observe(p, item["bearing_deg"])
            elif item["result"] == "no_signal":
                region.observe_no_signal(p)
            elif item["result"] == "near":
                near[channel] = p
        elif item["phase"] == "certified_clear":
            assert all(p.distance_to(Position(*v)) <= 19.9 + 1e-6 for v in regions[channel].vertices)
        elif item["phase"] == "near_clear":
            assert p == near[channel]
