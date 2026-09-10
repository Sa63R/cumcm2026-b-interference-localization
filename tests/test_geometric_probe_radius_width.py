import math
from fractions import Fraction

import pytest

from localization.omni import OmniCandidateRegion
from simulation import LocalResearchSimulator, random_scenario
from strategies import geometric_probe_cost as probe
from strategies.geometric_joint import GeometricJointSearch
from tests.test_strategy import ObservationOnlyClient


def test_radius_marginal_uses_all_positive_and_negative_constraints_without_mutation():
    region = OmniCandidateRegion().observe((0., 0.), 0.).observe((2000., 0.), 180.)
    region.observe_no_signal((-100., 0.))
    worlds = (((1250., 0.), 1300., 0), ((1250., 0.), 1300., 1))
    before = region.__dict__.copy()
    # R must be >=1250 and <1350: one fifth of its 500 m prior interval.
    assert probe.radius_width_weights(region, worlds) == pytest.approx((.2-2e-10,)*2)
    assert region.__dict__ == before
    assert probe.radius_width_weights(region, (((1600., 0.), 1500., 0),)) == (0.,)


def test_weights_preserve_worlds_and_three_error_phases_including_zero_width(monkeypatch):
    region = OmniCandidateRegion().observe((0., 0.), 0.)
    monkeypatch.setattr(probe, 'polygon_quadrature', lambda vertices, limit: ((500., 0.), (1250., 0.), (1500., 0.)))
    worlds = probe.observation_worlds(region)
    assert len(worlds) == 9
    assert probe.radius_width_weights(region, worlds) == (1.,)*3+(.5,)*3+(0.,)*3
    assert worlds == probe.observation_worlds(region)
    for index in range(0, 9, 3):
        assert [world[2] for world in worlds[index:index+3]] == [0, 1, 2]


def test_radial_area_integration_matches_independent_exact_conditional_mean():
    # Only the direction status at origin is conditioned on, not a numerical
    # bearing. Polar area contributes r; integrating R contributes w(r).
    F = Fraction
    den = F(1000**2-5**2, 2)+F(1500*(1500**2-1000**2), 2*500)-F(1500**3-1000**3, 3*500)
    num = F(1000**3-5**3, 3)+F(1500*(1500**3-1000**3), 3*500)-F(1500**4-1000**4, 4*500)
    assert num/den == F(162499990, 189997)
    region = OmniCandidateRegion().observe((0., 0.), 0.)
    radii = [5.+(i+.5)*1495./20000 for i in range(20000)]
    weights = probe.radius_width_weights(region, tuple(((r, 0.), 1250., 0) for r in radii))
    numerical = math.fsum(r*r*w for r, w in zip(radii, weights))/math.fsum(r*w for r, w in zip(radii, weights))
    assert numerical == pytest.approx(float(num/den), abs=1e-5)
    assert float(num/den) == pytest.approx(855.2766096306784)


def test_all_zero_mass_falls_back_to_original_probe_without_evaluating_tail(monkeypatch):
    client = LocalResearchSimulator(random_scenario(3, 107001)).client()
    policy = probe.GeometricProbeCostSearch(client, 10000, 6, None, 'single', 120., 'radius_width')
    policy.regions[1] = OmniCandidateRegion().observe((0., 0.), 0.)
    policy.first_bearings[1] = 0.
    original = GeometricJointSearch._next_probe(policy, 1, 0)
    monkeypatch.setattr(probe, 'observation_worlds', lambda region: (((1500., 0.), 1500., 0),))
    monkeypatch.setattr(probe, 'primary_cost_samples', lambda *args: pytest.fail('Zero-mass hypothesis evaluated'))
    assert policy._next_probe(1, 0) == original
    assert policy.report.first_probe_planning['fallbacks'] == ['zero_radius_mass']
    assert policy.report.first_probe_planning['planning_s'] > 0.


def test_radius_weighted_execution_remains_observation_only_and_logs_weighted_costs():
    sim = LocalResearchSimulator(random_scenario(3, 107001))
    report = probe.run_probe_cost_search(ObservationOnlyClient(sim.client()), hypothesis_weighting='radius_width')
    result = sim.evaluation()
    assert result['all_cleared'] and result['failed_clear_count'] == 0
    assert report.completion_certified_under_model
    assert report.virtual_time_s == result['virtual_time_s']
    assert sum(report.time_breakdown.values()) == pytest.approx(report.virtual_time_s, abs=1e-4)
    decisions = report.first_probe_planning['decisions']
    assert decisions and any(min(d['radius_weights']) < max(d['radius_weights']) for d in decisions)
    for decision in decisions:
        weights = decision['radius_weights']
        assert 0 < decision['effective_world_count'] <= len(weights)+1e-8
        for candidate in decision['candidates']:
            assert len(weights) == len(candidate['primary_costs_s'])
            expected = math.fsum(w*c for w, c in zip(weights, candidate['primary_costs_s']))/math.fsum(weights)
            assert candidate['primary_mean_cost_s'] == pytest.approx(expected)
            assert candidate['incremental_credit_s'] == 0.


@pytest.mark.parametrize('kwargs', [dict(hypothesis_weighting='posterior'),
    dict(probe_mode='cross', hypothesis_weighting='radius_width'),
    dict(probe_mode='disabled', hypothesis_weighting='radius_width')])
def test_weighting_validation_precedes_enter(kwargs):
    sim = LocalResearchSimulator(random_scenario(3, 107001))
    with pytest.raises(ValueError):
        probe.run_probe_cost_search(ObservationOnlyClient(sim.client()), **kwargs)
    assert sim.observation_history() == []
