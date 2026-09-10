import copy
import math

import pytest

from localization.omni import OmniCandidateRegion
from simulation import LocalResearchSimulator, Scenario, Source, random_scenario
from strategies.geometric_joint import polygon_quadrature, run_joint_search, shared_observation_value
from tests.test_strategy import ObservationOnlyClient


def test_geometric_value_is_observation_only_and_does_not_update_region():
    region = OmniCandidateRegion().observe((0., 0.), 0.)
    before = copy.deepcopy(region.__dict__)
    value = shared_observation_value(region, (750., 500.))
    assert value and value['guaranteed_reception']
    assert value['travel_saving_proxy_m'] >= 0
    # The only permitted cache change is computing the original MEC.
    assert region.vertices == before['vertices']
    assert region.observations == before['observations']
    assert region.no_signal_positions == before['no_signal_positions']
    assert shared_observation_value(region, (-1000., 0.)) is None
    assert all(region.contains(p) for p in polygon_quadrature(region.vertices))


@pytest.mark.parametrize('config', [None, {'active_points': False}, {'after_clear': False}])
def test_joint_observations_preserve_clearance_and_accounting(config):
    sim = LocalResearchSimulator(random_scenario(3, 103001))
    report = run_joint_search(ObservationOnlyClient(sim.client()), joint_config=config)
    assert sim.observation_history()[-1]['action'] == '/exit'
    evaluation = sim.evaluation()  # Only after policy termination.
    assert evaluation['all_cleared'] and report.completion_certified_under_model
    assert evaluation['failed_clear_count'] == 0
    assert report.virtual_time_s == evaluation['virtual_time_s']
    assert sum(report.time_breakdown.values()) == pytest.approx(report.virtual_time_s, abs=1e-4)
    observed, current = set(), 1
    sharing_count = 0
    for action in report.action_history:
        if action['action'] == 'measure':
            pair = (tuple(action['position']), action['channel'])
            if action['phase'] == 'geometric_shared_observation':
                assert pair not in observed
                sharing_count += 1
            observed.add(pair)
            current = action['channel']
    assert sharing_count == report.joint_observations['shared_measurements']
    for decision in report.joint_observations['decisions']:
        assert decision['estimated_net_gain_s'] >= 5
        assert decision['actual_measure_cost_s'] in (5, 6)
        assert decision['response'] in ('direction', 'near')
    assert report.joint_observations['actual_sharing_time_s'] == sum(
        d['actual_measure_cost_s'] for d in report.joint_observations['decisions'])


def test_sixteen_actual_clears_stop_without_sharing_after_the_last_clear():
    sources = tuple(Source(c, math.cos(c), math.sin(c), 1000.) for c in range(1, 17))
    sim = LocalResearchSimulator(Scenario('joint-sixteen-near', 3, 0, sources, 'zero'))
    report = run_joint_search(ObservationOnlyClient(sim.client()))
    assert report.cleared_count == 16 and report.completion_reason == 'source_count_upper_bound_reached'
    assert sim.observation_history()[-2]['action'] == '/clear'
    assert report.virtual_time_s == 199.


@pytest.mark.parametrize('kwargs', [dict(problem=4), dict(max_actions=True),
    dict(joint_config={'active_points': 1}), dict(joint_config={'minimum_net_gain_s': float('nan')}),
    dict(joint_config={'max_shared_per_stop': -1})])
def test_invalid_options_rejected_before_enter(kwargs):
    sim = LocalResearchSimulator(random_scenario(3, 103001))
    with pytest.raises(ValueError):
        run_joint_search(ObservationOnlyClient(sim.client()), **kwargs)
    assert sim.observation_history() == []


def test_action_budget_still_reserves_exit_without_false_completeness():
    sim = LocalResearchSimulator(random_scenario(3, 103001))
    report = run_joint_search(ObservationOnlyClient(sim.client()), max_actions=25)
    assert report.accepted_actions == 25
    assert sim.observation_history()[-1]['action'] == '/exit'
    assert not report.completion_certified_under_model
