import copy
import math
import statistics
import time

import pytest

from localization.omni import OmniCandidateRegion
from simulation import LocalResearchSimulator, random_scenario
from simulator_client.state import Position
from strategies.geometric_joint import JointObservationConfig, run_joint_search
from strategies.geometric_probe_cost import (_HypothesisClient, observation_worlds,
    primary_cost_samples, run_probe_cost_search, secondary_credit)
from tests.test_strategy import ObservationOnlyClient


def test_explicit_primary_cost_includes_move_switch_measure_and_success_once():
    region = OmniCandidateRegion().observe((0., 0.), 0.)
    before = copy.deepcopy(region.__dict__)
    samples = primary_cost_samples(region, [((500., 0.), 1250., 0)],
        Position(0., 0.), 1, 2, Position(500., 0.), {(0., 0.)})
    assert samples[0]['cost_s'] == 111.
    assert samples[0]['components_s'] == dict(movement_s=100., switching_s=1.,
        detection_s=5., optical_s=3., removal_s=2.)
    assert region.__dict__ == before


def test_hypothesis_fail_clear_cost_and_coordinate_noise_are_consistent():
    region = OmniCandidateRegion().observe((0., 0.), 0.)
    a = _HypothesisClient(region, ((700., 0.), 1250., 0), (0., 0.), 1, math.inf)
    b = _HypothesisClient(region, ((700., 0.), 1250., 0), (0., 0.), 1, math.inf)
    a.measure((100., 50.), 1)
    assert a.measure((300., 30.), 1) == b.measure((300., 30.), 1)
    assert a.measure((300., 30.), 1) == b.measure((300., 30.), 1)
    old = a.state.virtual_time_s
    result = a.clear((300., 30.), 2)
    assert result['clear_result'] == 'no_target_in_range'
    assert a.state.virtual_time_s-old == pytest.approx(3.)
    assert a.state.current_channel == 1  # Clears never tune the receiver.


def test_constructed_long_wedge_rewards_parallax_but_small_region_rejects_detour():
    wedge = OmniCandidateRegion().observe((0., 0.), 0.)
    worlds = observation_worlds(wedge)
    def cost(region, worlds, current, q, seen):
        samples = primary_cost_samples(region, worlds, Position(*current), 1, 1, Position(*q), seen)
        assert samples is not None
        return statistics.fmean(sample['cost_s'] for sample in samples)
    straight = cost(wedge, worlds, (0., 0.), (750., 0.), {(0., 0.)})
    transverse = cost(wedge, worlds, (0., 0.), (750., 150.), {(0., 0.)})
    assert transverse < straight-20.
    small = OmniCandidateRegion().observe((-1400., 0.), 0.).observe((0., -1400.), 90.)
    center = small.enclosing_disk().center
    assert small.enclosing_disk().radius > 19.9
    small_worlds = observation_worlds(small)
    seen = {(-1400., 0.), (0., -1400.)}
    direct = cost(small, small_worlds, (-100., 0.), center, seen)
    detour = cost(small, small_worlds, (-100., 0.), (center[0], center[1]+50.), seen)
    assert detour > direct+10.


def test_secondary_credit_cancels_when_sharing_disabled_and_preserves_observations():
    region = OmniCandidateRegion().observe((0., 0.), 0.)
    original = copy.deepcopy(region.__dict__)
    worlds = observation_worlds(region, 3)
    trace = [dict(action='measure', position=(750., 150.)),
             dict(action='measure', position=(900., 100.)),
             dict(action='clear', position=(950., 100.), result='success')]
    value = secondary_credit(region, worlds, trace, 1, 2, {(0., 0.)},
        JointObservationConfig(after_clear=False, active_points=False))
    assert value['mean_credit_s'] == 0.
    assert value['credits_s'] == [0.]*len(worlds)
    assert value['mean_shared_count'] == 0.
    assert region.__dict__ == original


def test_secondary_same_trace_same_worlds_have_identical_replacement_credit():
    region = OmniCandidateRegion().observe((0., 0.), 0.)
    worlds = observation_worlds(region, 3)
    trace = [dict(action='measure', position=(750., 500.)),
             dict(action='measure', position=(750., 500.)),
             dict(action='clear', position=(750., 500.), result='success')]
    a = secondary_credit(region, worlds, trace, 1, 2, {(0., 0.)}, JointObservationConfig())
    b = secondary_credit(region, worlds, trace, 1, 2, {(0., 0.)}, JointObservationConfig())
    assert a == b
    # Once observed, repeated stops do not pay or earn a second measurement.
    assert a['mean_shared_count'] <= 1.
    assert a['mean_credit_s']-b['mean_credit_s'] == 0.


def test_failed_primary_clear_does_not_create_a_fake_shared_observation():
    region = OmniCandidateRegion().observe((0., 0.), 0.)
    trace = [dict(action='clear', position=(750., 500.), result='no_target_in_range'),
             dict(action='clear', position=(-1500., -1500.), result='success')]
    value = secondary_credit(region, observation_worlds(region, 3), trace, 1, 2,
                             {(0., 0.)}, JointObservationConfig())
    assert value['mean_shared_count'] == 0.
    assert value['mean_credit_s'] == 0.


def test_disabled_or_zero_compute_budget_preserves_original_joint_actions():
    case = random_scenario(3, 107001)
    original = run_joint_search(ObservationOnlyClient(LocalResearchSimulator(case).client()))
    for kwargs in (dict(probe_mode='disabled'), dict(max_planning_s=0.)):
        report = run_probe_cost_search(ObservationOnlyClient(LocalResearchSimulator(case).client()), **kwargs)
        assert report.action_history == original.action_history
        assert report.virtual_time_s == original.virtual_time_s


@pytest.mark.parametrize('mode', ['single', 'cross'])
def test_first_probe_execution_still_uses_observation_only_clearance_and_full_ledger(mode):
    sim = LocalResearchSimulator(random_scenario(3, 107001))
    report = run_probe_cost_search(ObservationOnlyClient(sim.client()), probe_mode=mode)
    assert sim.observation_history()[-1]['action'] == '/exit'
    result = sim.evaluation()
    assert result['all_cleared'] and result['failed_clear_count'] == 0
    assert report.completion_certified_under_model
    assert report.virtual_time_s == result['virtual_time_s']
    assert sum(report.time_breakdown.values()) == pytest.approx(report.virtual_time_s, abs=1e-4)
    decisions = report.first_probe_planning['decisions']
    assert len({d['channel'] for d in decisions}) == len(decisions)
    for decision in decisions:
        assert decision['candidates'][0]['position'] == decision['baseline']
        for candidate in decision['candidates']:
            if 'secondary_credit' in candidate:
                assert candidate['score_s'] == pytest.approx(candidate['primary_mean_cost_s']-candidate['incremental_credit_s'])


@pytest.mark.parametrize('kwargs', [dict(problem=4), dict(probe_mode='unknown'),
    dict(max_planning_s=float('inf')), dict(max_actions=True)])
def test_invalid_options_fail_before_start(kwargs):
    sim = LocalResearchSimulator(random_scenario(3, 107001))
    with pytest.raises(ValueError):
        run_probe_cost_search(ObservationOnlyClient(sim.client()), **kwargs)
    assert sim.observation_history() == []
