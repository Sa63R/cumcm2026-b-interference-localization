import math
import random

import pytest

from planning.disk_waypoint import two_leg_disk_waypoint
from simulation import LocalResearchSimulator, Scenario, Source, random_scenario
from strategies.geometric_joint import run_joint_search
from strategies.geometric_route_clear import run_route_clear_search
from tests.test_strategy import ObservationOnlyClient


@pytest.mark.parametrize('p,q,c,r,expected,length', [
    ((-4., 0.), (4., 0.), (0., 0.), 1., (-1., 0.), 8.),
    ((-4., 2.), (4., 2.), (0., 0.), 1., (0., 1.), 2*math.sqrt(17)),
    ((3., 4.), (3., 4.), (0., 0.), 1., (.6, .8), 8.),
    ((.2, .3), (4., 2.), (0., 0.), 1., (.2, .3), math.hypot(3.8, 1.7)),
    ((-4., 1.), (4., 1.), (0., 0.), 1., (0., 1.), 8.),
    ((-4., 1.), (4., 1.), (0., 0.), 0., (0., 0.), 2*math.sqrt(17)),
    ((-4., 0.), (.5, 0.), (0., 0.), 1., (-1., 0.), 4.5),
])
def test_waypoint_matches_independent_analytic_cases(p, q, c, r, expected, length):
    result = two_leg_disk_waypoint(p, q, c, r)
    assert result['position'] == pytest.approx(expected, abs=1e-6)
    assert result['local_length_m'] == pytest.approx(length, abs=1e-8)
    assert result['gap_m'] <= 1e-6


def test_numerical_optimizer_has_independent_support_gap_and_dense_feasible_check():
    rng = random.Random(5401)
    for _ in range(160):
        center = tuple(rng.uniform(-1800, 1800) for _ in (0, 1))
        radius = rng.uniform(.001, 20.)
        endpoints = [tuple(center[j]+rng.uniform(-2500, 2500) for j in (0, 1)) for _ in (0, 1)]
        p, q = endpoints
        result = two_leg_disk_waypoint(p, q, center, radius)
        z = result['position']
        assert math.dist(z, center) <= radius+1e-10
        objective = math.dist(z, p)+math.dist(z, q)
        g = tuple((z[j]-p[j])/math.dist(z, p)+(z[j]-q[j])/math.dist(z, q) for j in (0, 1))
        bound = objective+sum(g[j]*(center[j]-z[j]) for j in (0, 1))-radius*math.hypot(*g)
        assert objective-max(bound, math.dist(p, q)) <= 1e-5
        grid = [(center[0]+radius*math.cos(i*math.tau/1440),
                 center[1]+radius*math.sin(i*math.tau/1440)) for i in range(1440)]
        grid_cost = min(math.dist(point, p)+math.dist(point, q) for point in grid)
        assert objective <= grid_cost+1e-7
        assert bound <= grid_cost+1e-7


def test_disabled_route_clear_is_exactly_original_joint_trajectory():
    case = random_scenario(3, 105001)
    before_sim, after_sim = LocalResearchSimulator(case), LocalResearchSimulator(case)
    before = run_joint_search(ObservationOnlyClient(before_sim.client()))
    after = run_route_clear_search(ObservationOnlyClient(after_sim.client()), route_clear=False)
    assert before.action_history == after.action_history
    assert before.virtual_time_s == after.virtual_time_s
    assert not after.route_clear_decisions


def test_route_clear_retains_real_certificate_sharing_and_full_accounting():
    sim = LocalResearchSimulator(random_scenario(3, 105001))
    report = run_route_clear_search(ObservationOnlyClient(sim.client()))
    assert sim.observation_history()[-1]['action'] == '/exit'
    actual = sim.evaluation()  # Only after termination.
    assert actual['all_cleared'] and actual['failed_clear_count'] == 0
    assert report.completion_certified_under_model
    assert report.virtual_time_s == actual['virtual_time_s']
    assert sum(report.time_breakdown.values()) == pytest.approx(report.virtual_time_s, abs=1e-4)
    assert report.route_clear_decisions
    for choice in report.route_clear_decisions:
        assert math.dist(choice['actual_position'], choice['center'])+choice['region_radius_m'] <= 19.9+1e-9
        assert choice['channel'] != choice['successor_channel']
        if choice['accepted']:
            assert choice['local_length_m'] <= choice['legacy_local_length_m']+1e-9
            assert choice['gap_m'] <= 1e-5
    shares = [a for a in report.action_history if a['phase'] == 'geometric_shared_observation']
    assert len(shares) == report.joint_observations['shared_measurements']


def test_sixteen_actual_clears_still_stop_and_budget_does_not_certify_early():
    sources = tuple(Source(c, math.cos(c), math.sin(c), 1000.) for c in range(1, 17))
    sim = LocalResearchSimulator(Scenario('route-clear-sixteen', 3, 0, sources, 'zero'))
    report = run_route_clear_search(ObservationOnlyClient(sim.client()))
    assert report.completion_reason == 'source_count_upper_bound_reached'
    assert report.cleared_count == 16 and report.virtual_time_s == 199.
    assert sim.observation_history()[-2]['action'] == '/clear'
    short = LocalResearchSimulator(random_scenario(3, 105001))
    incomplete = run_route_clear_search(ObservationOnlyClient(short.client()), max_actions=25)
    assert incomplete.accepted_actions == 25
    assert not incomplete.completion_certified_under_model
    assert short.observation_history()[-1]['action'] == '/exit'


@pytest.mark.parametrize('kwargs', [dict(problem=4), dict(route_clear=1), dict(max_actions=True)])
def test_invalid_options_rejected_before_enter(kwargs):
    sim = LocalResearchSimulator(random_scenario(3, 105001))
    with pytest.raises(ValueError):
        run_route_clear_search(ObservationOnlyClient(sim.client()), **kwargs)
    assert sim.observation_history() == []
