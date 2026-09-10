"""Independent brute-force and legal-observation checks for subset search."""

from itertools import permutations
import math
import random

import pytest

from planning.state_route import RouteTask, solve_state_route
from planning.disk_cover import disk_cover_radius
from planning.coverage import omni_coverage_points
from simulation import LocalResearchSimulator, Scenario, Source, difficult_scenarios, random_scenario
from simulator_client.state import Position
from strategies.state_search import run_state_search
from tests.test_strategy import ObservationOnlyClient


def explicit_cost(tasks, order, start):
    point, result = start, 0.0
    remaining_sources = sum(task.is_source for task in tasks)
    for i in order:
        task = tasks[i]
        result += point.distance_to(task.position) / 5 + task.service_s
        if task.is_source:
            remaining_sources -= 1
        else:
            result += 6 * remaining_sources
        point = task.position
    return result


@pytest.mark.parametrize("seed,n", [(1, 3), (2, 5), (3, 7), (4, 8)])
def test_compressed_search_matches_independent_permutation_enumeration(seed, n):
    rng = random.Random(seed)
    start = Position(170, -90)
    tasks = [RouteTask(Position(rng.uniform(-800, 800), rng.uniform(-800, 800)),
                       i % 3 != 0, rng.uniform(0, 20)) for i in range(n)]
    optimum = min(explicit_cost(tasks, order, start) for order in permutations(range(n)))
    result = solve_state_route(tasks, start, max_expansions=100000)
    assert result.exact
    assert result.cost_s == pytest.approx(optimum, abs=1e-7)
    assert result.lower_bound_s == pytest.approx(optimum, abs=1e-7)
    assert explicit_cost(tasks, result.order, start) == pytest.approx(result.cost_s)
    limited = solve_state_route(tasks, start, max_expansions=2)
    assert limited.lower_bound_s <= optimum + 1e-7 <= limited.cost_s + 1e-7


def test_scan_order_cost_can_outweigh_shorter_geometric_path():
    tasks = [RouteTask(Position(0, 0), False, 0),
             RouteTask(Position(5, 0), True, 0)]
    result = solve_state_route(tasks)
    assert result.order == (1, 0)
    assert result.cost_s == 2


@pytest.mark.parametrize("scenario", difficult_scenarios(3) + [random_scenario(3, 2000)])
def test_policy_completes_through_restricted_observation_interface(scenario):
    sim = LocalResearchSimulator(scenario)
    report = run_state_search(ObservationOnlyClient(sim.client()),
                              config={"max_expansions": 100})
    evaluation = sim.evaluation()
    assert evaluation["all_cleared"]
    assert report.completion_certified_under_model
    assert evaluation["failed_clear_count"] == 0
    assert report.virtual_time_s == evaluation["virtual_time_s"]
    assert report.error is None and report.exit_error is None
    assert sim.observation_history()[-1]["action"] == "/exit"
    for plan in report.strategy_parameters["planning_log"]:
        assert plan["lower_bound_s"] <= plan["cost_s"] + 1e-7
        assert plan["expanded"] <= 100


def test_action_budget_exits_and_does_not_certify_partial_clearance():
    sim = LocalResearchSimulator(random_scenario(3, 2000))
    report = run_state_search(sim.client(), max_actions=3)
    assert not report.completion_certified_under_model
    assert report.completion_reason == "action_budget"
    assert sim.observation_history()[-1]["action"] == "/exit"


def test_q4_is_rejected_without_any_action():
    sim = LocalResearchSimulator(random_scenario(4, 2000))
    with pytest.raises(ValueError):
        run_state_search(sim.client(), problem=4)
    assert not sim.observation_history()


def test_cover_radius_matches_analytic_seven_site_formula():
    for ring in (1123, 1150, 1500, 1732):
        expected = max(ring / math.sqrt(3), math.sqrt(1800**2 + ring**2 - math.sqrt(3) * 1800 * ring))
        assert disk_cover_radius(omni_coverage_points(ring)) == pytest.approx(expected, abs=1e-6)


def test_cover_witnesses_detect_interior_hole_even_when_boundary_is_covered():
    ring = [Position(1800 * math.cos(i * math.pi / 3),
                     1800 * math.sin(i * math.pi / 3)) for i in range(6)]
    boundary_worst = max(min(Position(1800 * math.cos(i / 1000 * 2 * math.pi),
                                     1800 * math.sin(i / 1000 * 2 * math.pi)).distance_to(p)
                             for p in ring) for i in range(1000))
    assert boundary_worst < 1000
    assert disk_cover_radius(ring) >= 1800 - 1e-6


def test_finite_voronoi_superset_dominates_dense_disk_sample():
    rng = random.Random(73)
    for _ in range(8):
        sites = [Position(rng.uniform(-1500, 1500), rng.uniform(-1500, 1500)) for _ in range(7)]
        claimed = disk_cover_radius(sites)
        for _ in range(2000):
            r, theta = 1800 * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
            p = Position(r * math.cos(theta), r * math.sin(theta))
            assert min(p.distance_to(site) for site in sites) <= claimed + 1e-6


@pytest.mark.parametrize("scenario", difficult_scenarios(3) + [random_scenario(3, 2001)])
def test_replacement_cover_uses_measured_channels_and_keeps_certificate(scenario):
    sim = LocalResearchSimulator(scenario)
    report = run_state_search(ObservationOnlyClient(sim.client()), config={
        "max_expansions": 100, "scan_source_s": 0,
        "replace_coverage": True, "opportunistic_measurements": True})
    evaluation = sim.evaluation()
    assert evaluation["all_cleared"] and report.completion_certified_under_model
    assert evaluation["failed_clear_count"] == 0
    histories = report.action_history
    cleared = set(report.cleared_channels)
    absent = set(range(1, 21)) - cleared
    # No hidden source count is used online; after exit, verify every truly
    # absent channel independently has an actual no-signal full-disk cover.
    if len(cleared) < 16:
        for channel in absent:
            negatives = [Position(*item["position"]) for item in histories
                         if item["action"] == "measure" and item["channel"] == channel
                         and item["result"] == "no_signal"]
            assert disk_cover_radius(negatives) <= 1000 + 1e-6
    for replacement in report.strategy_parameters["coverage_replacements"]:
        assert replacement["certified_cover_radius_m"] <= 1000 - 1e-5
        for channel in replacement["channels_measured"]:
            assert any(item["channel"] == channel and item["phase"] == "replacement_discovery"
                       and item["position"] == replacement["position"] for item in histories)


@pytest.mark.parametrize("scenario", difficult_scenarios(3))
def test_active_discrete_probe_search_never_installs_hypothetical_observations(scenario):
    sim = LocalResearchSimulator(scenario)
    report = run_state_search(ObservationOnlyClient(sim.client()), config={
        "max_expansions": 100, "scan_source_s": 0, "active_probe_search": True,
        "opportunistic_measurements": True, "probe_uncertainty_weight": 0.5})
    evaluation = sim.evaluation()
    assert evaluation["all_cleared"] and report.completion_certified_under_model
    assert evaluation["failed_clear_count"] == 0
    assert report.virtual_time_s == evaluation["virtual_time_s"]
    assert report.measurement_count == evaluation["measurement_count"]
    # Reconstruct every clearance certificate from independent legal history,
    # never from hypothetical candidate-region copies used only for ranking.
    from localization.omni import OmniCandidateRegion
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


def test_known_sixteen_sources_do_not_trigger_pointless_replacement_scans():
    scenario = Scenario("review-known16-cluster", 3, 73,
                        tuple(Source(c, 1490.0, 0.0, 1500.0) for c in range(1, 17)),
                        error_mode="zero")
    sim = LocalResearchSimulator(scenario)
    result = run_state_search(sim.client(), config={"max_expansions": 100,
                                                    "replace_coverage": True})
    assert sim.evaluation()["all_cleared"]
    assert not result.strategy_parameters["coverage_replacements"]
    assert not any(item["phase"] == "replacement_discovery" for item in result.action_history)
