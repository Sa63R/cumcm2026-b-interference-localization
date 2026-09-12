"""Constructed actual prefixes and tampering, never performance scenarios."""
import copy
import math
from types import SimpleNamespace

import pytest

from localization import CandidateRegion
from experiments import audit_q4_shared_known as audit


def one_region():
    r=CandidateRegion().observe((-1000.,0.),0.)
    return SimpleNamespace(snapshot=lambda n:({1},set(),{}, {1:r}),h=[]),r


def test_nominal_one_measure_can_predict_ready_but_never_mutates_live_region():
    p,r=one_region()
    old=copy.deepcopy(r.vertices); count=len(r.observations)
    candidate=audit.nominal(p,0,1,(-250.,-500.))
    assert candidate['predicted_radius_m']<19.9<r.enclosing_disk().radius
    assert candidate['ratio']==pytest.approx(candidate['predicted_radius_m']/candidate['original_radius_m'])
    assert r.vertices==old and len(r.observations)==count


def test_near_cutoff_is_five_not_twenty_and_max_radius_is_fifteen_hundred():
    p,r=one_region(); x,y=r.enclosing_disk().center
    assert audit.nominal(p,0,1,(x,y+5.)) is None
    assert audit.nominal(p,0,1,(x,y+1500.001)) is None
    # Ten metres is a legitimate directional prediction, not near feedback.
    c=audit.nominal(p,0,1,(x,y+10.))
    assert c is not None and c['center_distance_m']==pytest.approx(10.)


def test_same_point_negative_also_prevents_repeated_fixed_error_observation():
    p,_=one_region()
    p.h=[dict(action='measure',position=[-250.,-500.],channel=1,result='no_signal')]
    assert not audit.fresh_at(p,1,1,(-250.+1e-8,-500.))
    assert audit.fresh_at(p,1,2,(-250.,-500.))
    p.h[0]['action']='clear'
    assert audit.fresh_at(p,1,1,(-250.,-500.))


def test_nonready_is_required_and_bad_geometry_not_a_zero_cost_model():
    p,r=one_region()
    assert audit.nominal(p,0,1,(0.,-500.)) is None
    r.vertices=(); r._circle=None
    assert audit.nominal(p,0,1,(-250.,-500.)) is None


def controller_record(monkeypatch, feedback='no_signal'):
    """Full original calls on a tiny cover-chain fixture, not a scene claim."""
    from tests.test_q4_known_source import ScriptedReplies
    from simulator_client.state import Position
    from strategies.q4_shared_known import Q4SharedKnown
    from strategies.search import _StopSearch
    monkeypatch.setattr('planning.q4_directional_cover.certified_cover_points',lambda profile:
        (tuple(Position(*p) for p in [(-1000.,0.),(500.,0.)]),{'passed':True,'test_only':True}))
    client=ScriptedReplies(); policy=Q4SharedKnown(client,20000,6,max_expansions=200); policy.actions=1
    wire=[dict(action='/enter',response=dict(accepted=True,max_virtual_duration_s=360000.))]
    counts={}; original_measure=client.measure; original_clear=client.clear
    def measure(p,c):
        counts[c]=counts.get(c,0)+1
        current_event=policy.shared_known_observation_log[-1] if policy.shared_known_observation_log else None
        sharing=current_event and current_event['selected_channel']==c and current_event['status']=='planned'
        if sharing:
            if feedback=='rejected': client.reject_kind='measure'; reply=('no_signal',None)
            elif feedback=='nominal':
                selected=next(e for e in current_event['candidates'] if e['channel']==c)
                reply=('direction',selected['nominal_bearing_deg'])
            else: reply=(feedback,None)
        elif c in (1,2): reply=('direction',15. if c==1 else 0.) if counts[c]==1 else ('near',None)
        else: reply=('no_signal',None)
        client.measure_replies=[reply]
        response=original_measure(p,c)
        wire.append(dict(action='/measure',position=[p.x,p.y],channel=c,response=copy.deepcopy(response)))
        return response
    def clear(p,c):
        client.clear_replies=['success']
        response=original_clear(p,c)
        wire.append(dict(action='/clear',position=[p.x,p.y],channel=c,response=copy.deepcopy(response)))
        return response
    monkeypatch.setattr(client,'measure',measure); monkeypatch.setattr(client,'clear',clear)
    try: policy._execute_plan()
    except _StopSearch as e: policy.report.completion_reason=str(e)
    return dict(summary=policy.report.as_dict(),history=wire,row={'strategy':audit.LABEL},
        spec=dict(entrypoint=audit.ENTRY,kwargs=dict(config=audit.CONFIG,max_expansions=200)))


@pytest.mark.parametrize('feedback',['no_signal','nominal','near','rejected'])
def test_real_macro_sharing_retains_actual_feedback_and_all_parent_intervals(monkeypatch,feedback):
    record=controller_record(monkeypatch,feedback)
    result=audit.replay_macros(record)
    expected=0 if feedback=='rejected' else 1
    assert result['shared_observation_actions']==expected
    assert result['shared_opportunities']==(1 if feedback=='rejected' else 2)
    event=record['summary']['strategy_parameters']['shared_known_observation_log'][0]
    assert event['selected_channel']==2 and event['actual_ready_after']==(feedback in {'nominal','near'})
    assert result['completed_scans']==(1 if feedback=='rejected' else 2)


@pytest.mark.parametrize('change',['trigger_index','trigger_channel','trigger_parent','parent_end','early_parent',
    'start','end','position','tuning','known','blocked','seen_before','seen_after','attempted_before',
    'attempted_after','selected','fee','actual_result','actual_bearing','actual_radius','fake_ready',
    'action_budget','virtual_budget','missing_event','duplicate_event','orphan_share','rename_phase'])
def test_trigger_action_identity_truthful_feedback_and_budget_tampering_rejected(monkeypatch,change):
    record=controller_record(monkeypatch)
    params=record['summary']['strategy_parameters']; events=params['shared_known_observation_log']; e=events[0]
    fields={'trigger_index':'trigger_clear_action_index','trigger_channel':'trigger_channel',
        'trigger_parent':'parent_resolver_id','parent_end':'parent_end_action_count','start':'after_actual_action_count',
        'end':'end_actual_action_count','tuning':'current_channel'}
    if change in fields: e[fields[change]]+=1
    elif change=='early_parent': e['early_service_index']=0
    elif change=='position': e['position'][0]+=1
    elif change=='known': e['known_channels'].append(20)
    elif change=='blocked': e['blocked_channels']=[2]
    elif change=='seen_before': e['seen_clear_before']=[1]
    elif change=='seen_after': e['seen_clear_after']=[]
    elif change=='attempted_before': e['attempted_before']=[2]
    elif change=='attempted_after': e['attempted_after']=[]
    elif change=='selected': e['selected_channel']=3
    elif change=='fee': e['actual_cost_s']=0.
    elif change=='actual_result': e['actual_result']='near'
    elif change=='actual_bearing': e['actual_bearing_deg']=e['candidates'][0]['nominal_bearing_deg']
    elif change=='actual_radius': e['actual_radius_after_m']=e['candidates'][0]['predicted_radius_m']
    elif change=='fake_ready': e['actual_ready_after']=True
    elif change=='action_budget': e['budget']['policy_actions_before']+=1
    elif change=='virtual_budget': e['budget']['virtual_limit_s']+=1
    elif change=='missing_event': events.pop(0)
    elif change=='duplicate_event': events.insert(1,copy.deepcopy(e))
    elif change=='orphan_share': events.clear()
    else: record['summary']['action_history'][e['after_actual_action_count']]['phase']='coverage'
    with pytest.raises(ValueError): audit.replay_macros(record)


@pytest.mark.parametrize('change',['omit_candidate','radius','vertices','near_limit','bearing','pred_vertices',
    'pred_radius','ratio','eligible','reason','positive_count','freshness'])
def test_complete_independent_nominal_candidate_evidence_rejects_tampering(monkeypatch,change):
    record=controller_record(monkeypatch)
    e=record['summary']['strategy_parameters']['shared_known_observation_log'][0]; c=e['candidates'][0]
    if change=='omit_candidate': e['candidates']=[]
    elif change=='radius': c['original_radius_m']=30.
    elif change=='vertices': c['original_vertices'][0][0]+=1
    elif change=='near_limit': c['nominal_distance_m']=20.
    elif change=='bearing': c['nominal_bearing_deg']+=1
    elif change=='pred_vertices': c['predicted_vertices'][0][0]+=1
    elif change=='pred_radius': c['predicted_radius_m']=0.
    elif change=='ratio': c['ratio']=0.
    elif change=='eligible': c['eligible']=False
    elif change=='reason': c['reason']='already_ready'
    elif change=='positive_count': c['positive_observation_count']+=1
    else: c['previously_observed']=True
    with pytest.raises(ValueError): audit.replay_macros(record)


def test_share_real_bearing_cannot_be_replaced_by_nominal_in_phase_history(monkeypatch):
    record=controller_record(monkeypatch,'nominal')
    event=record['summary']['strategy_parameters']['shared_known_observation_log'][0]
    record['summary']['action_history'][event['after_actual_action_count']]['bearing_deg']+=5.
    with pytest.raises(ValueError,match='bearing'): audit.replay_macros(record)


def test_no_shared_action_is_allowed_inside_an_old_resolver(monkeypatch):
    record=controller_record(monkeypatch)
    epoch=record['summary']['strategy_parameters']['joint_visibility_resolver_log'][0]
    record['summary']['action_history'][epoch['after_actual_action_count']]['phase']=audit.PHASE
    with pytest.raises(ValueError,match='sharing'): audit.replay_macros(record)


def test_zero_action_stop_needs_correct_recorded_original_budget(monkeypatch):
    record=controller_record(monkeypatch,'rejected')
    event=record['summary']['strategy_parameters']['shared_known_observation_log'][0]
    event['interruption_reason']='action_budget'
    with pytest.raises(ValueError,match='action exhaustion'): audit.replay_macros(record)


@pytest.mark.parametrize('feedback',['no_signal','nominal','near','rejected'])
def test_complete_inherited_safety_checks_receive_real_shared_history(monkeypatch,feedback):
    record=controller_record(monkeypatch,feedback)
    result=audit.audit_shared_known_prefix(record)
    assert result['passed'] and all(result[k]['passed'] for k in ('r12','r8','range','scheduling'))
    assert result['shared_observation_actions']==(0 if feedback=='rejected' else 1)


@pytest.mark.parametrize('child',['audit_joint_continuation_prefix','audit_clear_before_probe_prefix',
                                 'audit_range_prefix','audit_scheduling_prefix'])
def test_inherited_false_is_not_an_implicit_pass(monkeypatch,child):
    record=controller_record(monkeypatch)
    monkeypatch.setattr(audit.base,child,lambda record:{'passed':False})
    with pytest.raises(ValueError,match='Inherited shared-known audit'): audit.audit_shared_known_prefix(record)


def test_shared_and_reused_source_identity_checked_before_spec_normalization(monkeypatch):
    audit.verify_source_contract()
    record=controller_record(monkeypatch)
    monkeypatch.setattr(audit,'BASE_AUDIT_SHA256','0'*64)
    with pytest.raises(ValueError,match='R34 replay'): audit.audit_shared_known_prefix(record)


def test_different_entry_or_config_cannot_borrow_inherited_safety_view(monkeypatch):
    record=controller_record(monkeypatch)
    record['spec']['kwargs']['config']='unknown_channels'
    with pytest.raises(ValueError,match='spec'): audit.audit_shared_known_prefix(record)
