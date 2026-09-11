import copy
import math
import time
from types import SimpleNamespace

import pytest

from localization import CandidateRegion
from planning.q4_observation_tree import (Hypothesis, ObservationTree,
    bearing_bin_likelihoods, make_belief, observation_branches, unique_prefix)
from simulator_client.state import Position
from strategies.q4_observation_search import Q4ObservationSearch, run_q4_observation_search
from strategies.q4_range_scheduling import Q4RangeScheduling
from strategies.q4_range_pruning import Q4RangePruningSearch
from strategies.q4_r2_scheduling import Q4R2Scheduling


def rectangle(x0=-100., x1=100., y0=100., y1=200.):
    region = CandidateRegion()
    region.vertices = [(x0,y0), (x1,y0), (x1,y1), (x0,y1)]
    return region


def armed_tree(depth=2):
    tree = ObservationTree(depth=depth, max_cpu_s=10.)
    tree.deadline, tree.nodes = time.perf_counter()+10., 0
    return tree


@pytest.mark.parametrize('angle', [0., .001, 9.8, 10., 179.9, 359.9])
def test_integrated_likelihood_conserves_mass_across_wrap_and_bins(angle):
    values = bearing_bin_likelihoods(angle)
    assert sum(values.values()) == pytest.approx(1.)
    assert all(0 <= k < 36 and 0 < p <= 1 for k,p in values.items())
    if angle == 0:
        assert values == pytest.approx({35:.5, 0:.5})


def test_fixed_same_coordinate_error_is_not_counted_as_independent_evidence():
    row = {'action':'measure', 'channel':4, 'position':[0.,0.],
           'result':'direction', 'bearing_deg':90.}
    prefix = unique_prefix([row, dict(row), {**row,'channel':5}], 4)
    assert len(prefix) == 1
    region = rectangle(-5,5,100,200)
    assert make_belief(region, prefix) == make_belief(region, unique_prefix([row],4))


def test_positive_negative_soft_weights_change_but_live_geometry_is_untouched():
    region = rectangle(-3,3,100,200)
    before = copy.deepcopy(region.__dict__)
    observation = {'position':(0.,0.), 'result':'direction', 'bearing_deg':90.}
    positive = make_belief(region, (observation,))
    silent = make_belief(region, ({**observation,'result':'no_signal'},))
    def omni_mass(values):
        return sum(h.weight for h in values if h.orientation is None)
    assert omni_mass(positive) > omni_mass(silent)
    assert sum(h.weight for h in positive) == pytest.approx(1.)
    assert all(h.weight > 0 for h in positive)
    assert region.__dict__ == before
    assert len({h.position for h in positive}) == 24


def test_binned_posterior_shares_particles_and_conservative_geometry_includes_bin_edges():
    region = rectangle(-1000,1000,-1000,1000)
    tree = armed_tree()
    for bin_index in (0,17,35):
        updated = tree._updated_region(region, (0.,0.), ('bearing',bin_index))
        for delta in (-1.005,0.,10.+1.005):
            angle = math.radians(bin_index*10+delta)
            assert updated.contains((500*math.cos(angle),500*math.sin(angle)), tolerance=1e-6)
    hypotheses = (Hypothesis((100.,10.),1000.,None,.4),
                  Hypothesis((200.,20.),1500.,None,.6))
    branches = observation_branches(hypotheses, (0.,0.))
    assert len(branches) == 1 and len(branches[0].particles) == 2
    assert sum(h.weight for h in branches[0].particles) == pytest.approx(1.)
    assert branches[0].spatial_ess == pytest.approx(1/(.4**2+.6**2))


def test_real_feedback_branches_choose_different_common_second_probe():
    region = rectangle(-400,400,100,600)
    hypotheses = tuple(Hypothesis(p,1500.,None,.25)
                       for p in ((80.,100.),(240.,300.),(-80.,100.),(-240.,300.)))
    tree = armed_tree()
    value, logs = tree._action_value(region, hypotheses, (0.,0.), (0.,0.), set(),90.,2)
    assert math.isfinite(value) and value > 10
    expanded = [b for b in logs if b['reason'] == 'shared_second_probe']
    assert len(expanded) == 2
    assert all(b['particles'] == 2 and b['spatial_ess'] == pytest.approx(2.) for b in expanded)
    assert expanded[0]['continuation'] != expanded[1]['continuation']
    assert expanded[0]['continuation'][0]*expanded[1]['continuation'][0] < 0
    # Recompute E[cost | feedback] for each shared candidate. The min is outside
    # the particle expectation, exactly as used by the actual tree.
    for branch, log in zip(observation_branches(hypotheses,(0.,0.)), logs):
        child = tree._updated_region(region,(0.,0.),branch.outcome)
        candidates = tree._candidates(child,{(0.,0.)},90.)
        common = [tree._action_value(child,branch.particles,(0.,0.),q,{(0.,0.)},90.,1)[0]
                  for q in candidates]
        assert log['continuation_cost_s'] == pytest.approx(min(common))


def test_low_spatial_ess_cannot_gain_clairvoyant_continuation():
    tree = armed_tree()
    region = rectangle(-200,200,100,300)
    hypotheses = (Hypothesis((80.,100.),1000.,None,1.),)
    value, logs = tree._action_value(region,hypotheses,(0.,0.),(0.,0.),set(),90.,2)
    assert value > 10 and logs[0]['reason'] == 'low_spatial_ess_leaf'
    assert logs[0]['continuation'] is None
    child = tree._updated_region(region,(0.,0.),logs[0]['outcome'])
    assert logs[0]['continuation_cost_s'] == pytest.approx(tree._leaf(child,(0.,0.),90.))


def test_common_action_prevents_average_of_per_particle_perfect_choices():
    class KnownTableTree(ObservationTree):
        def _updated_region(self, region, observer, outcome):
            return region.copy()
        def _candidates(self, *args, **kwargs):
            return ((1.,1.),(-1.,1.))
        def _action_value(self, region, particles, observer, action, observed, bearing, depth):
            if depth == 1:
                # Two hypotheses require opposite exits. Each would pay 1 with
                # hidden-truth action selection; any common action costs 51.
                value = sum(h.weight*(1. if h.position[0]*action[0] > 0 else 101.)
                            for h in particles)
                return value, []
            return super()._action_value(region,particles,observer,action,observed,bearing,depth)
    tree = KnownTableTree(depth=2,max_cpu_s=10.)
    tree.deadline, tree.nodes = time.perf_counter()+10., 0
    # Both return silence at observer=0; the feedback cannot distinguish them.
    particles = (Hypothesis((2000.,100.),1000.,None,.5),
                 Hypothesis((-2000.,100.),1000.,None,.5))
    value, log = tree._action_value(rectangle(),particles,(0.,0.),(0.,0.),set(),90.,2)
    assert value == pytest.approx(5.+51.)
    assert log[0]['continuation_cost_s'] == 51.


def test_first_channel_switch_is_charged_once_and_does_not_change_selection():
    region = rectangle(-30,30,100,200)
    arguments = (region,(0.,0.),(0.,150.),())
    a, log_a = ObservationTree(max_cpu_s=2.).choose(*arguments,first_bearing=90.)
    b, log_b = ObservationTree(max_cpu_s=2.).choose(*arguments,first_bearing=90.,initial_switch_s=1.)
    assert not log_a['fallback'] and not log_b['fallback'] and a == b
    assert log_b['predicted_local_cost_s'] == pytest.approx(log_a['predicted_local_cost_s']+1.)


def test_noncompleted_leaf_pays_complete_common_optical_route_and_travel():
    tree = armed_tree()
    region = rectangle(-40,40,100,180)
    near = tree._leaf(region,(0.,100.),90.)
    far = tree._leaf(region,(0.,-1000.),90.)
    assert near > 5 and far > near + 150


def test_empty_particles_and_timeout_return_baseline_without_clearance(monkeypatch):
    region = rectangle()
    before = copy.deepcopy(region.__dict__)
    monkeypatch.setattr('planning.q4_observation_tree.make_belief',lambda *args: ())
    point, log = ObservationTree().choose(region,(0.,0.),(0.,150.),(),first_bearing=90.)
    assert point == (0.,150.) and log['fallback']
    assert log['reason'] == 'particle_exhaustion'
    assert region.__dict__ == before
    monkeypatch.undo()
    point, log = ObservationTree(max_cpu_s=1e-12).choose(
        region,(0.,0.),(0.,150.),(),first_bearing=90.)
    assert point == (0.,150.) and log['reason'] == 'cpu_budget'


@pytest.mark.parametrize('attribute,reason', [('max_nodes','node_budget'),
                                            ('max_grid','optical_grid_budget')])
def test_computation_cap_discards_partial_scores_instead_of_favoring_first(attribute,reason):
    tree = ObservationTree(max_cpu_s=2.)
    setattr(tree,attribute,0)
    selected,log = tree.choose(rectangle(),(0.,0.),(0.,150.),(),first_bearing=90.)
    assert selected == (0.,150.) and log['fallback'] and log['reason'] == reason
    assert 'predicted_local_cost_s' not in log


def test_no_same_point_future_measurements_and_no_input_mutation():
    region = rectangle(-30,30,100,200)
    before = copy.deepcopy(region.__dict__)
    prefix = ({'position':(0.,150.),'result':'no_signal','bearing_deg':None},)
    point, log = ObservationTree(depth=1,max_cpu_s=2.).choose(
        region,(0.,0.),(-25.,150.),prefix,first_bearing=90.)
    assert not log['fallback']
    assert all(score['position'] != (0.,150.) for score in log['candidate_scores'])
    assert region.__dict__ == before


class ObservedStateOnly:
    def __init__(self):
        self.state = SimpleNamespace(sources={},position=Position(0.,0.),current_channel=1)
    def __getattr__(self, name):
        raise AssertionError(f'Forbidden client attribute: {name}')


def test_strategy_only_uses_legal_prefix_and_retains_cover_schedule_resolver(monkeypatch):
    policy = Q4ObservationSearch(ObservedStateOnly(),20000,6,depth=1)
    policy.regions[1] = rectangle(-30,30,100,200)
    policy.first_bearings[1] = 90.
    before = copy.deepcopy(policy.regions[1].__dict__)
    point = policy._next_probe(1,0)
    assert isinstance(point,Position)
    # Original baseline itself fills its lazy MEC cache; no vertex/observation
    # update is permitted by the new model.
    assert policy.regions[1].vertices == before['vertices']
    assert policy.regions[1].observations == before['observations']
    assert len(policy.points) == 22
    assert type(policy)._scan is Q4RangePruningSearch._scan
    assert type(policy)._execute_plan is Q4R2Scheduling._execute_plan
    assert type(policy)._resolve is Q4RangeScheduling._resolve
    assert not policy.cleared and not policy.near_points


@pytest.mark.parametrize('kwargs',[{'problem':3},{'depth':0},{'depth':True},{'max_active_probes':True}])
def test_invalid_config_rejected_before_client_access(kwargs):
    with pytest.raises(ValueError):
        run_q4_observation_search(None,**kwargs)
