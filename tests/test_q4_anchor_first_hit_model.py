"""Finite first-hit accounting and belief-only construction, no scenarios."""
import copy
import math

import pytest

from localization import CandidateRegion
from planning.q4_anchor_first_hit import (ModelUnavailable,_Work,anchor_candidates,
    branch_region,expected_first_hit_cost,optical_first_hit,radio_prefix,rank_anchor_probes)


def region_and_prefix():
    r=CandidateRegion().observe((-1000.,0.),0.)
    return r,({'position':(-1000.,0.),'result':'direction','bearing_deg':0.},)


def test_common_order_overlap_absorbs_once_and_stops_at_first_hit():
    route=[(0.,0.),(30.,0.),(1000.,0.)]
    nodes=[(0.,0.),(10.,0.),(30.,0.)];weights=[.25,.25,.5]
    expected=3.+.5*(30./5.+3.)+2.  # Survival cost along one shared route.
    value,hits=expected_first_hit_cost(route,(0,0),nodes,weights)
    assert value==expected==9.5 and hits==[0,0,1]
    assert value < 3.+30./5.+3.+970./5.+3.+2.  # Full route is not charged.


def test_tiny_positive_uncovered_mass_is_never_dropped():
    with pytest.raises(ModelUnavailable,match='uncovered_positive_mass'):
        expected_first_hit_cost([(0,0)],(0,0),[(0,0),(1000,0)],[1.-1e-12,1e-12])
    cost,hits=expected_first_hit_cost([(0,0)],(0,0),[(0,0),(1000,0)],[1.,0.])
    assert cost==5. and hits==[0,None]


@pytest.mark.parametrize('weights',[[math.nan,1.],[math.inf,0.],[-1.,2.],[0.,0.],[.5]])
def test_invalid_first_hit_mass_rejected(weights):
    with pytest.raises(ModelUnavailable):expected_first_hit_cost([(0,0)],(0,0),[(0,0),(1,0)],weights)


def test_clear_boundary_is_twenty_and_not_measure_near_five():
    cost,hits=expected_first_hit_cost([(0,0)],(0,0),[(20.,0.)],[1.])
    assert cost==5. and hits==[0]
    with pytest.raises(ModelUnavailable):expected_first_hit_cost([(0,0)],(0,0),[(20.000001,0.)],[1.])


@pytest.mark.parametrize('bin_index',[0,359])
def test_bin_region_covers_whole_bin_plus_error_and_does_not_observe_live(bin_index):
    r=CandidateRegion();before=copy.deepcopy(r.__dict__)
    child=branch_region(r,(0,0),('bearing',bin_index))
    for angle in (bin_index-1.005,bin_index+.5,bin_index+2.005):
        p=(1000*math.cos(math.radians(angle)),1000*math.sin(math.radians(angle)))
        assert child.contains(p,1e-6)
    assert child.observations==r.observations and r.__dict__==before
    silent=branch_region(r,(100,0),('no_signal',))
    assert silent.vertices==r.vertices and silent is not r


def test_large_grid_budget_rejects_without_treating_tail_as_zero():
    with pytest.raises(ModelUnavailable,match='grid_budget'):
        optical_first_hit(CandidateRegion(),(0,0),0.,[(0,0)],[1.],_Work())


def test_ready_center_still_checks_positive_node_coverage():
    r=CandidateRegion();r.vertices=((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.));r._circle=None
    value,log=optical_first_hit(r,(10,0),0.,[(0,0)],[1.],_Work())
    assert value==7. and log['kind']=='certified_center' and len(log['route'])==1
    with pytest.raises(ModelUnavailable,match='uncovered_positive_mass'):
        optical_first_hit(r,(10,0),0.,[(100,0)],[1.],_Work())


def test_nearest_real_direction_anchor_and_history_index_tie():
    history=[dict(action='measure',channel=1,position=[0,0],result='direction',bearing_deg=0.),
             dict(action='measure',channel=1,position=[200,0],result='direction',bearing_deg=180.),
             dict(action='measure',channel=2,position=[100,0],result='direction',bearing_deg=90.)]
    candidates,anchor=anchor_candidates(history,1,(100,0),(500,0),set())
    assert anchor['action_index']==0 and candidates==((500.,0.),(100.,100.),(100.,-100.))
    candidates,anchor=anchor_candidates(history,1,(200,0),(500,0),set())
    assert anchor['action_index']==1


def test_anchor_dedup_freshness_and_original_always_first():
    h=[dict(action='measure',channel=1,position=[0,0],result='direction',bearing_deg=0.)]
    candidates,_=anchor_candidates(h,1,(0,0),(100,100),{(100.,-100.)})
    assert candidates==((100.,100.),)
    with pytest.raises(ModelUnavailable,match='distinct_candidates'):
        r,prefix=region_and_prefix()
        rank_anchor_probes(r,prefix,candidates=candidates,current=(0,0),first_bearing=0.,channel=1,current_channel=1)


def test_exact_fixed_point_history_dedup_and_conflict_reject():
    item=dict(action='measure',channel=1,position=[0,0],result='direction',bearing_deg=0.)
    assert len(radio_prefix([item,item],1))==1
    with pytest.raises(ModelUnavailable,match='conflicting'):
        radio_prefix([item,dict(item,bearing_deg=1.)],1)


def test_real_continuous_belief_scores_complete_same_candidate_set():
    r,prefix=region_and_prefix();original=copy.deepcopy(r.__dict__)
    candidates=(r.enclosing_disk().center,(-900.,100.),(-900.,-100.))
    original=copy.deepcopy(r.__dict__)
    selected,log=rank_anchor_probes(r,prefix,candidates=candidates,current=(-1000,0),
        first_bearing=0.,channel=1,current_channel=1)
    assert selected==1 and len(log['candidate_scores'])==3 and log['work_used']<=262144
    assert log['selected_expected_cost_s']<log['original_expected_cost_s']-1e-9
    assert r.__dict__==original
    for candidate in log['candidate_scores']:
        assert math.fsum(b['mass'] for b in candidate['branches'])==pytest.approx(1.)
        total=candidate['immediate_s']
        for branch in candidate['branches']:
            weights=branch['spatial_weights'];route=branch['route']
            assert math.fsum(weights)==pytest.approx(1.)
            # Independent survival computation on the SINGLE branch route.
            alive=set(i for i,w in enumerate(weights) if w>0);expected=0.;previous=candidate['position']
            for step,q in enumerate(route):
                survival=math.fsum(weights[i] for i in alive)
                expected+=survival*(math.dist(previous,q)/5.+3.)
                caught={i for i in alive if math.dist(log['nodes'][i],q)<=20.}
                expected+=2.*math.fsum(weights[i] for i in caught)
                for i in caught:assert branch['first_hit_indices'][i]==step
                alive-=caught;previous=q
            assert not alive and expected==pytest.approx(branch['first_hit_cost_s'],abs=1e-8)
            total+=branch['mass']*expected
        assert total==pytest.approx(candidate['expected_cost_s'],abs=1e-8)


def test_unavailable_later_candidate_invalidates_whole_scoring(monkeypatch):
    import planning.q4_anchor_first_hit as model
    r,prefix=region_and_prefix();baseline=r.enclosing_disk().center
    predict=model.predict_branches
    def fail_later(belief,observer,**kwargs):
        if tuple(observer)==(-900.,-100.):raise ModelUnavailable('constructed_later_failure')
        return predict(belief,observer,**kwargs)
    monkeypatch.setattr(model,'predict_branches',fail_later)
    with pytest.raises(ModelUnavailable,match='later_failure'):
        rank_anchor_probes(r,prefix,candidates=(baseline,(-900,100),(-900,-100)),
            current=(-1000,0),first_bearing=0.,channel=1,current_channel=1)


def test_work_budget_is_deterministic_whole_decision_fallback(monkeypatch):
    import planning.q4_anchor_first_hit as model
    r,prefix=region_and_prefix()
    monkeypatch.setitem(model.LIMITS,'work_limit',10)
    with pytest.raises(ModelUnavailable):
        rank_anchor_probes(r,prefix,candidates=(r.enclosing_disk().center,(-900,100)),
            current=(-1000,0),first_bearing=0.,channel=1,current_channel=1)


def test_equal_complete_expected_cost_retains_original_candidate():
    r=CandidateRegion().observe((-1000.,0.),0.)
    r.vertices=((0.,0.),);r._circle=None
    prefix=({'position':(-1000.,0.),'result':'direction','bearing_deg':0.},)
    selected,log=rank_anchor_probes(r,prefix,candidates=((10.,0.),(-10.,0.)),current=(0.,0.),
        first_bearing=0.,channel=1,current_channel=1)
    costs=[v['expected_cost_s'] for v in log['candidate_scores']]
    assert costs[0]==pytest.approx(costs[1],abs=1e-9) and selected==0
