"""Independent costs, actual scripted radio prefixes and corruption; no cases."""
import copy
import math
from dataclasses import asdict

import pytest

from localization import CandidateRegion
from experiments import audit_q4_anchor_first_hit as audit
from planning import q4_anchor_first_hit as model


def region():
    c=CandidateRegion(); c.observe((0.,0.),0.)
    c.vertices=((400.,-3.),(500.,-3.),(500.,3.),(400.,3.)); c._circle=None
    return c


def test_parent_safety_replay_only_adds_explicit_ownership():
    audit.verify_replay_delta()


def test_overlapping_disks_absorb_only_first_time_and_charge_only_reached_steps():
    route=[(0.,0.),(30.,0.),(60.,0.)]
    nodes=[(10.,0.),(45.,0.),(60.,0.)]; weights=[.5,.25,.25]
    cost,hits=audit.common_route_first_hit(route,(0.,0.),nodes,weights)
    assert hits==[0,1,2]
    assert cost==.5*5.+.25*14.+.25*23.
    assert model.expected_first_hit_cost(route,(0.,0.),nodes,weights)==(cost,hits)


@pytest.mark.parametrize('weight',[1.,1e-12,1e-100])
def test_any_uncovered_positive_mass_cannot_disappear(weight):
    with pytest.raises(ValueError,match='no physical first hit'):
        audit.common_route_first_hit([(0.,0.)],(0.,0.),[(0.,0.),(100.,0.)],[1.-weight,weight])


def test_zero_weight_does_not_fabricate_hit_and_near_cost_is_five():
    assert audit.common_route_first_hit([(0.,0.)],(0.,0.),[(5.,0.),(100.,0.)],[1.,0.])==(5.,[0,None])


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-.01,True])
def test_invalid_weights_rejected(bad):
    with pytest.raises(ValueError): audit.common_route_first_hit([(0.,0.)],(0.,0.),[(0.,0.)],[bad])


def test_common_route_model_and_independent_costs_agree_without_truth():
    c=region(); prefix=[dict(position=(0.,0.),result='direction',bearing_deg=0.)]
    candidates=[(450.,0.),(100.,100.),(100.,-100.)]
    before=copy.deepcopy(c.__dict__)
    selected,expected=audit.reference_score(c,prefix,candidates,(0.,0.),0.,2,1)
    actual,logged=model.rank_anchor_probes(c,prefix,candidates=candidates,current=(0.,0.),first_bearing=0.,channel=2,current_channel=1)
    assert actual==selected
    audit.same(logged,expected)
    assert c.__dict__==before
    assert logged['candidate_scores'][0]['immediate_s']==96.
    _,same_channel=audit.reference_score(c,prefix,candidates,(0.,0.),0.,2,2)
    assert math.isclose(logged['original_expected_cost_s']-same_channel['original_expected_cost_s'],1.)


def test_silent_branch_keeps_geometry_and_direction_widening_is_only_hypothetical():
    c=region(); before=copy.deepcopy(c.__dict__)
    silent=audit.reference_branch(c,(100.,100.),('no_signal',),audit.ModelWork())
    assert silent.__dict__==c.__dict__ and silent is not c
    b=audit.reference_branch(c,(100.,100.),('bearing',344),audit.ModelWork())
    p=model.branch_region(c,(100.,100.),('bearing',344))
    assert b.vertices==p.vertices and c.__dict__==before


def test_exact_position_feedback_repeat_and_conflict():
    h=[dict(action='measure',channel=1,position=[0.,0.],result='direction',bearing_deg=0.)]
    assert len(audit.radio_history(h+h,2,1))==1
    bad=copy.deepcopy(h[0]);bad['result']='no_signal';bad.pop('bearing_deg')
    with pytest.raises(audit.ModelUnavailable,match='conflicting_fixed_position'):
        audit.radio_history(h+[bad],2,1)


def test_anchor_nearest_real_direction_stable_tie_and_six_decimal_freshness():
    h=[dict(action='measure',channel=1,position=[-1.,0.],result='direction',bearing_deg=0.),
       dict(action='measure',channel=1,position=[1.,0.],result='direction',bearing_deg=90.),
       dict(action='measure',channel=1,position=[99.,100.],result='no_signal')]
    values,anchor=audit.reference_candidates(h,len(h),1,(0.,0.),(450.,0.))
    assert anchor['action_index']==0 and values==[(450.,0.),(99.,-100.)]



def controller_record(monkeypatch,mode='near'):
    from strategies.q4_anchor_first_hit import AnchorFirstHit
    from simulator_client.state import Position
    from strategies.search import _StopSearch
    from tests.test_q4_clear_before_probe import Replies
    from tests.test_audit_q4_clear_before_probe import wrap
    monkeypatch.setattr('planning.q4_directional_cover.certified_cover_points',lambda profile:
        ((Position(0.,0.),Position(100.,0.)),{'passed':True,'test_only':True}))
    client=Replies(); policy=AnchorFirstHit(client,20000,6,max_expansions=200)
    client.measure_replies=[('direction',0.)]
    policy._perform('measure',Position(-1000.,0.),1,'constructed_observation')
    if mode=='history_budget':
        for j in range(1,65):
            client.measure_replies=[('direction',0.)]
            policy._perform('measure',Position(-1000.+j*.01,0.),1,'constructed_observation')
    if mode=='original_best':
        client.measure_replies=[('no_signal',None)]
        policy._perform('measure',Position.coerce(policy.regions[1].enclosing_disk().center),2,'constructed_observation')
    canonical=copy.deepcopy(policy.regions[1].vertices)
    if mode=='reject':
        client.measure_replies=[('near',None)];client.reject_kind='measure'
        with pytest.raises(_StopSearch): policy._resolve(1)
    else:
        if mode=='no_signal': client.measure_replies=[('no_signal',None),('near',None)]
        elif mode=='direction': client.measure_replies=[('direction',round(math.degrees(math.atan2(-100.,900.))%360.,2)),('near',None)]
        else: client.measure_replies=[('near',None)]
        client.clear_replies=['success']*8
        assert policy._resolve(1)
    record=wrap(policy,reason='request_rejected' if mode=='reject' else None)
    record['row']['strategy']=audit.LABEL
    record['spec']=dict(entrypoint=audit.ENTRY,kwargs=dict(config=audit.CONFIG,max_actions=20000,max_active_probes=6,max_expansions=200))
    if mode=='reject':
        event=record['summary']['strategy_parameters']['anchor_first_hit_log'][0]
        record['history'].append(dict(action='/measure',channel=1,position=event['selected'],response={'accepted':False}))
    if mode in {'near','no_signal','reject'}: assert policy.regions[1].vertices==canonical
    return record


def prefix(record,monkeypatch):
    monkeypatch.setattr(audit,'verify_source_contract',lambda:56)
    return audit.audit_anchor_first_hit_prefix(record)


@pytest.mark.parametrize('mode',['near','no_signal','direction','reject'])
def test_complete_actual_prefix_inherits_r12_r8_range_and_scheduling(monkeypatch,mode):
    record=controller_record(monkeypatch,mode); original=copy.deepcopy(record)
    result=prefix(record,monkeypatch)
    assert result['anchor_probe_actions']==(0 if mode=='reject' else 1)
    assert all(result[k]['passed'] for k in ('r12','r8','range','scheduling'))
    assert record==original
    if mode=='no_signal': assert result['r12']['continuation_events']==1


@pytest.mark.parametrize('change',['missing','duplicate','wrong_resolver','future','wrong_index','wrong_current',
    'wrong_tuned','fake_anchor','candidate_omit','point','baseline','canonical_radius','model_vertices',
    'node','mass','route','first_hit','cost','work','partial_fallback','wrong_feedback','fake_ready',
    'wrong_phase','counter','cap','fake_parent_owner','no_measure','wire_bearing'])
def test_real_prefix_corruption_rejected(monkeypatch,change):
    record=controller_record(monkeypatch)
    params=record['summary']['strategy_parameters'];e=params['anchor_first_hit_log'][0]
    if change=='missing': params['anchor_first_hit_log'].clear()
    elif change=='duplicate': params['anchor_first_hit_log'].append(copy.deepcopy(e))
    elif change=='wrong_resolver': e['resolver_id']+=1
    elif change=='future': e['after_actual_action_count']+=1
    elif change=='wrong_index': e['index']=1
    elif change=='wrong_current': e['current_position'][0]+=1
    elif change=='wrong_tuned': e['current_channel']=2
    elif change=='fake_anchor': e['anchor']['action_index']+=1
    elif change=='candidate_omit': e['candidates'].pop()
    elif change=='point': e['selected'][0]+=1
    elif change=='baseline': e['baseline'][1]+=1
    elif change=='canonical_radius': e['canonical_radius_m']-=1
    elif change=='model_vertices': e['model_vertices'][0][0]+=1
    elif change=='node': e['model']['nodes'][0][0]+=1
    elif change=='mass': e['model']['candidate_scores'][0]['branches'][0]['mass']+=.01
    elif change=='route': e['model']['candidate_scores'][0]['branches'][0]['route'][0][0]+=1
    elif change=='first_hit': e['model']['candidate_scores'][0]['branches'][0]['first_hit_indices'][0]=999
    elif change=='cost': e['model']['candidate_scores'][0]['expected_cost_s']-=1
    elif change=='work': e['model']['work_used']-=1
    elif change=='partial_fallback': e['changed']=False;e['parent_fallback']=True
    elif change=='wrong_feedback': e['actual_result']='direction'
    elif change=='fake_ready': record['summary']['source_estimates']['1']['vertices']=[[0.,0.]]
    elif change=='wrong_phase': record['summary']['action_history'][1]['phase']='coverage'
    elif change=='counter': e['actual_new_probes_after']=2
    elif change=='cap': params['anchor_first_hit_limits']['actual_new_probe_limit']=41
    elif change=='fake_parent_owner': params['joint_visibility_probe_log'].append(dict(resolver_id=0,after_actual_action_count=1,index=0))
    elif change=='no_measure': e['executed_measure']=False
    elif change=='wire_bearing': record['history'][0]['response']['svd_deg']=1.
    with pytest.raises((ValueError,KeyError,AssertionError)):
        prefix(record,monkeypatch)


def test_wrapper_does_not_accept_false_inherited_check(monkeypatch):
    record=controller_record(monkeypatch)
    monkeypatch.setattr(audit,'audit_range_prefix',lambda r:dict(passed=False))
    with pytest.raises(ValueError,match='Inherited anchor'): prefix(record,monkeypatch)


def test_full_envelope_retains_original_record_for_generic(monkeypatch):
    record=controller_record(monkeypatch); seen=[]
    def generic(value):
        assert value is record;seen.append(value);return dict(passed=True,test_only=True)
    monkeypatch.setattr('experiments.audit_q4_cover.audit_record',generic)
    monkeypatch.setattr(audit,'verify_source_contract',lambda:56)
    result=audit.audit_full(record)
    assert seen and result['passed'] and result['prefix']['anchor_probe_actions']==1


def test_unreviewed_entry_or_configuration_cannot_borrow_parent(monkeypatch):
    record=controller_record(monkeypatch)
    record['spec']['kwargs']['config']='after_active_miss_optical'
    with pytest.raises(ValueError,match='spec'): prefix(record,monkeypatch)


@pytest.mark.parametrize('mode,reason',[('history_budget','model_unavailable'),('original_best','original_best')])
def test_full_real_parent_fallback_owns_its_actions_without_fabricated_replacement(monkeypatch,mode,reason):
    record=controller_record(monkeypatch,mode)
    result=prefix(record,monkeypatch)
    event=record['summary']['strategy_parameters']['anchor_first_hit_log'][0]
    assert result['anchor_probe_actions']==0 and event['status']==reason
    assert event['parent_fallback'] and not event['changed'] and not event['executed_measure']
    assert event['after_actual_action_count']==event['end_actual_action_count']
    assert all(result[k]['passed'] for k in ('r12','r8','range','scheduling'))
    if mode=='history_budget': assert event['model'] is None and event['unavailable_reason']=='history_budget'


def test_exact_frozen_source_and_imported_belief_identity():
    assert audit.verify_source_contract()==56


def test_changed_source_identity_is_rejected_without_modifying_any_source(monkeypatch):
    monkeypatch.setitem(audit.NEW_SOURCE_CONTRACT,'src/planning/q4_conditional_belief.py','0'*64)
    with pytest.raises(ValueError,match='Changed source contract'): audit.verify_source_contract()


def test_complete_entry_runs_actual_source_contract(monkeypatch):
    record=controller_record(monkeypatch)
    result=audit.audit_anchor_first_hit_prefix(record)
    assert result['source_contract_files']==56 and result['anchor_probe_actions']==1
