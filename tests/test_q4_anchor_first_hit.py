"""Actual inherited resolver calls with scripted replies, never scenes/socket."""
import copy

import pytest

from simulator_client.state import Position
from strategies.q4_anchor_first_hit import AnchorFirstHit, preview_probe, run_q4_anchor_first_hit
from strategies.q4_joint_continuation import Q4JointContinuation
from strategies.search import _StopSearch
from strategies.q4_r2_scheduling import _ServiceSliceExpired
from tests.test_q4_clear_before_probe import Replies
from tests.test_q4_joint_visibility_strategy import shrink


def make(monkeypatch,active=6):
    monkeypatch.setattr('planning.q4_directional_cover.certified_cover_points',lambda profile:
        ((Position(0,0),Position(100,0)),{'passed':True,'test_only':True}))
    client=Replies()
    return AnchorFirstHit(client,20000,active,max_expansions=0),client


def prime(policy,client,channel=1):
    client.measure_replies=[('direction',0.)]
    policy._perform('measure',Position(-1000,0),channel,'constructed_observation')
    assert policy.regions[channel].enclosing_disk().radius>40.


def test_complete_real_first_probe_with_no_old_selected_forgery(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client)
    before=copy.deepcopy(policy.regions[1].__dict__)
    def before_measure():
        assert policy.regions[1].__dict__==before
        assert not policy.joint_probes and not policy.cleared
    client.before_measure=before_measure
    client.measure_replies=[('direction',270.)]
    assert policy._resolve(1)
    event=policy.anchor_first_hit_log[0]
    assert event['changed'] and event['executed_measure'] and not event['parent_fallback']
    assert event['selected']==[-900.,100.] and event['baseline']!=event['selected']
    assert event['after_actual_action_count']==1 and event['end_actual_action_count']==2
    assert event['status']=='measured' and event['actual_result']=='direction'
    assert event['actual_new_probes_before']==0 and event['actual_new_probes_after']==1
    assert not event['r8_clear_before_measure'] and not policy.clear_before_probe_log
    assert [a['phase'] for a in policy.report.action_history[1:]]==['active_localization','certified_clear']
    assert policy.joint_resolvers[0]['end_actual_action_count']==3
    assert not policy.joint_probes and not policy.joint_grids and policy.actual_new_probes==1
    assert policy._joint_context is policy._probe_resolving_channel is policy._anchor_pending is None


def test_near_response_is_real_not_predicted_geometry(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client)
    before=copy.deepcopy(policy.regions[1].__dict__)
    client.measure_replies=[('near',None)]
    assert policy._resolve(1)
    event=policy.anchor_first_hit_log[0]
    assert event['actual_result']=='near' and event['executed_measure']
    assert policy.regions[1].__dict__==before and 1 in policy.near_points
    assert policy.report.action_history[-1]['phase']=='near_clear'


def test_real_silence_flows_into_existing_continuation_and_next_probe(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client)
    client.measure_replies=[('no_signal',None),('near',None)]
    assert policy._resolve(1)
    events=policy.anchor_first_hit_log
    assert events[0]['changed'] and events[0]['actual_result']=='no_signal'
    assert events[1]['index']==1 and events[1]['status']=='not_first_probe' and events[1]['parent_fallback']
    assert policy.actual_new_probes==1 and len(policy.continuation_log)==1
    assert policy.continuation_log[0]['trigger_actual_action_index']==1
    assert policy.joint_radio[1][1]['result']=='no_signal'
    assert len([a for a in policy.report.action_history if a['action']=='measure'])==3


@pytest.mark.parametrize('gate',['request','actions','virtual','real','service'])
def test_global_or_service_rejection_does_not_consume_actual_new_probe(monkeypatch,gate):
    policy,client=make(monkeypatch);prime(policy,client)
    if gate=='request':client.reject_kind='measure'
    elif gate=='actions':policy.max_actions=policy.actions+1
    elif gate=='virtual':client.state.max_virtual_duration_s=client.state.virtual_time_s+1.
    elif gate=='real':client.remaining_real_time_s=2.
    else:policy.service_deadline=client.state.virtual_time_s+1.
    client.measure_replies=[('near',None)];before=len(client.calls)
    with pytest.raises((_StopSearch,_ServiceSliceExpired)):policy._resolve(1)
    event=policy.anchor_first_hit_log[0]
    assert event['changed'] and event['status']=='interrupted' and not event['executed_measure']
    assert event['after_actual_action_count']==event['end_actual_action_count']==1
    assert event['actual_cost_s']==0. and policy.actual_new_probes==0
    assert len(client.calls)==before+(gate=='request') and not policy.cleared
    assert policy._anchor_pending is policy._joint_context is policy._probe_resolving_channel is None


def test_actual_new_probe_cap_forty_then_untouched_parent(monkeypatch):
    import strategies.q4_anchor_first_hit as module
    policy,client=make(monkeypatch);prime(policy,client)
    policy.actual_new_probes=39;client.measure_replies=[('direction',270.)]
    assert policy._resolve(1) and policy.actual_new_probes==40
    prime(policy,client,2);client.clear_replies=['success'];client.measure_replies=[('near',None)]
    def forbidden(*args,**kwargs):raise AssertionError('model called beyond actual cap')
    monkeypatch.setattr(module,'rank_anchor_probes',forbidden)
    assert policy._resolve(2)
    event=policy.anchor_first_hit_log[-1]
    assert event['status']=='actual_new_probe_limit' and event['parent_fallback'] and not event['changed']
    assert policy.actual_new_probes==40


def test_unavailable_model_calls_parent_only_after_failed_complete_score(monkeypatch):
    import strategies.q4_anchor_first_hit as module
    policy,client=make(monkeypatch);prime(policy,client)
    expected=policy.regions[1].enclosing_disk().center
    def unavailable(*args,**kwargs):raise module.ModelUnavailable('constructed_scoring_failure')
    monkeypatch.setattr(module,'rank_anchor_probes',unavailable)
    client.measure_replies=[('near',None)]
    assert policy._resolve(1)
    event=policy.anchor_first_hit_log[0]
    assert event['status']=='model_unavailable' and event['parent_fallback']
    assert not event['changed'] and policy.actual_new_probes==0
    assert policy.report.action_history[1]['position']==list(expected)


def test_original_best_uses_parent_without_external_action_owner(monkeypatch):
    import strategies.q4_anchor_first_hit as module
    policy,client=make(monkeypatch);prime(policy,client)
    monkeypatch.setattr(module,'rank_anchor_probes',lambda *a,**k:(0,{'test_only':True}))
    client.measure_replies=[('near',None)]
    assert policy._resolve(1)
    event=policy.anchor_first_hit_log[0]
    assert event['status']=='original_best' and event['parent_fallback'] and not event['executed_measure']
    assert event['after_actual_action_count']==event['end_actual_action_count']==1


def test_canonical_radius_below40_retains_real_r8_clear(monkeypatch):
    policy,client=make(monkeypatch)
    for point,bearing in [((-1000,0),0.),((0,-1000),90.)]:
        client.measure_replies=[('direction',bearing)]
        policy._perform('measure',Position(*point),1,'constructed_observation')
    assert 19.9<policy.regions[1].enclosing_disk().radius<40.
    assert policy._resolve(1)
    assert policy.anchor_first_hit_log[0]['status']=='canonical_not_wide'
    assert policy.report.action_history[-1]['phase']=='speculative_clear_before_probe'
    assert policy.actual_new_probes==0 and len(policy.clear_before_probe_log)==1


def set_test_aux(policy,fraction):
    # Explicit controller-only auxiliary fixture, not a claimed safe certificate.
    context=policy._start_joint(1);policy._joint_context=context
    aux=policy.regions[1].copy();aux.vertices=shrink(aux.vertices,fraction);aux._circle=None
    context['region']=aux
    return context


def test_new_probe_real_bearing_still_updates_parent_aux(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client)
    context=set_test_aux(policy,.9)
    point=policy._next_probe(1,0)
    event=policy.anchor_first_hit_log[-1]
    assert event['changed'] and event['model_region_kind']=='auxiliary' and not policy.joint_probes
    client.measure_replies=[('direction',270.)]
    policy._perform('measure',point,1,'active_localization')
    assert event['executed_measure'] and len(context['event']['aux_updates'])==1
    assert context['event']['aux_updates'][0]['after_actual_action_count']==2
    assert context['pending_probe'] is None and policy._anchor_pending is None


def test_ready_aux_clear_keeps_real_parent_event_and_no_model(monkeypatch):
    from strategies.q4_joint_visibility import _JointSourceCleared
    policy,client=make(monkeypatch);prime(policy,client)
    context=set_test_aux(policy,.01)
    assert context['region'].enclosing_disk().radius<=19.9
    with pytest.raises(_JointSourceCleared):policy._next_probe(1,0)
    event=policy.anchor_first_hit_log[0]
    assert event['status']=='parent_clear' and event['parent_fallback'] and event['model'] is None
    assert policy.joint_probes[0]['kind']=='clear' and policy.joint_probes[0]['status']=='cleared'
    assert policy.report.action_history[-1]['phase']=='joint_visibility_clear' and policy.actual_new_probes==0


@pytest.mark.parametrize('auxiliary',[False,True])
def test_pure_preview_exactly_matches_parent_fresh_geometry_without_writes(monkeypatch,auxiliary):
    policy,client=make(monkeypatch);prime(policy,client)
    context=set_test_aux(policy,.9) if auxiliary else None
    region=context['region'] if context else policy.regions[1]
    for taken in range(6):
        before=(copy.deepcopy(policy.report.action_history),len(policy.joint_probes),copy.deepcopy(region.__dict__))
        p=preview_probe(region,0.,client.state.position,policy.observed_positions[1])
        assert before==(policy.report.action_history,len(policy.joint_probes),region.__dict__)
        expected=Q4JointContinuation._next_probe(policy,1,0)
        assert p==expected
        if p is None:break
        policy.observed_positions[1].add((round(p.x,6),round(p.y,6)))
    assert p is None


def test_zero_probe_budget_goes_to_original_complete_optical_grid(monkeypatch):
    policy,client=make(monkeypatch,active=0);prime(policy,client)
    assert policy._resolve(1)
    assert not policy.anchor_first_hit_log and policy.report.action_history[-1]['phase']=='guaranteed_clearance'


def test_all_other_actions_scheduling_and_geometry_are_inherited():
    for name in ('_resolve','_clear','_scan','_execute_plan','_early_candidate','_early_service',
                 '_check_budget','_insertion_budget','_ready','_target','run'):
        assert getattr(AnchorFirstHit,name) is getattr(Q4JointContinuation,name)


@pytest.mark.parametrize('kwargs',[{'problem':3},{'problem':True},{'config':'wrong'},
    {'max_actions':True},{'max_active_probes':31},{'max_expansions':-1}])
def test_invalid_entrypoint_before_client(kwargs):
    with pytest.raises(ValueError):run_q4_anchor_first_hit(object(),**kwargs)
