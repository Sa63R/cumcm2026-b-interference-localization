from dataclasses import replace
import math
from types import SimpleNamespace
import time

from simulation.cases import random_scenario
from strategies.q3_fresh_stepper import Action
from strategies.q3_round3 import Proposal,Round3Stepper
from strategies.q3_round3_exploration import ExplorationRollout,compatible_witnesses
from tests.test_q3_round3 import prefix,find_cover_boundary
from tests.test_q3_belief import assert_compatible


def test_witnesses_are_compatible_bounded_and_include_lost_visibility():
    policy=SimpleNamespace(history=[],pending=Action('measure',(1200,0),1,'cover'))
    proposal=Proposal('cover_shift',(Action('measure',(1124.7901,0),1,'cover'),),'cover_shift',(1200,0),1)
    worlds,labels=compatible_witnesses(policy,proposal,[random_scenario(3,910100)],time.perf_counter()+5)
    assert 0<len(worlds)<=4 and 'lost_visibility' in labels
    for world in worlds:assert_compatible(world,policy.history)
    world=worlds[labels.index('lost_visibility')]
    assert any(math.dist((s.x,s.y),(1200,0))<=s.reception_radius_m<math.dist((s.x,s.y),(1124.7901,0)) for s in world.sources)


def test_empty_nominal_pool_yields_no_risk_witness():
    world=random_scenario(3,910100)
    # This guard is also exercised by real public count conditioning elsewhere.
    policy=SimpleNamespace(history=[],pending=Action('measure',(1200,0),1,'cover'))
    proposal=Proposal('other',(Action('measure',(-1200,0),1,'cover'),))
    assert compatible_witnesses(policy,proposal,[],time.perf_counter()+1)==([],[])


def test_explicit_directions_are_distinct_and_candidate_count_stays_six():
    p,c=find_cover_boundary();r=ExplorationRollout(evaluate_cover=True,directions=True,events=True)
    directions=r.direction_proposals(p)
    assert len(directions)==2 and directions[0].actions[0].position!=directions[1].actions[0].position
    candidates=r.proposals(p)
    assert len(candidates)<=6 and candidates[0].kind=='baseline'
    assert any(x.kind=='cover_shift' for x in candidates)


def test_key_event_can_bypass_eight_action_cooldown_once():
    p,c=find_cover_boundary();r=ExplorationRollout(events=True)
    r.last_planned_action=len(p.history)-1
    reason=r.event_reason(p)
    assert reason in ('all_detected_cleared','cross_region_departure','leaving_cover_point')
    assert r.event_reason(p) is None


def test_witness_cost_is_separate_and_veto_is_only_explicit_mode(monkeypatch):
    p,c=find_cover_boundary()
    world=random_scenario(3,910100)
    monkeypatch.setattr('strategies.q3_round3_exploration.compatible_witnesses',lambda *a,**k:([world],['fixture']))
    for mode,expected in (('record',True),('veto',False)):
        r=ExplorationRollout(witness_mode=mode)
        monkeypatch.setattr(r,'_evaluate_proposal',lambda p,c,w,d,wi,ci,stage:100 if ci==0 else 350)
        record=dict(best_index=1,confirmation={'mean_s':-20,'standard_error_s':1})
        assert r.risk_check(p,c,[world],time.perf_counter()+10,record)==expected
        assert record['confirmation']['mean_s']==-20
        assert record['risk_max_positive_delta_s']==250
