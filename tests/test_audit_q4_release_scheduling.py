"""Pure released-route algebra and scripted real-prefix tampering tests."""
import copy
import itertools
import math

import pytest

from experiments import audit_q4_release_scheduling as audit


@pytest.mark.parametrize('releases',[(0,0),(0,1),(1,2),(2,2)])
def test_small_released_dag_matches_independent_complete_permutations(releases):
    covers=[(10.,0.),(20.,20.)]; sources=[(5.,-5.),(12.,30.)]
    background=[12.,6.]; weights=[[6.,0.],[0.,6.]]; start=(0.,0.)
    tasks=[('cover',0),('cover',1),('source',0),('source',1)]
    values=[]
    for order in itertools.permutations(tasks):
        if order.index(('cover',0))>order.index(('cover',1)): continue
        visited=0; allowed=True
        for kind,i in order:
            if kind=='cover': visited+=1
            elif visited<releases[i]: allowed=False; break
        if allowed:
            values.append(audit.release_route_cost(covers,sources,start,order,background,weights,releases))
    optimum=audit.small_optimum(covers,sources,start,background,weights,releases)
    assert optimum==pytest.approx(min(values))
    assert audit.base.root_relaxation(covers,sources,start,background)+10.<=optimum+1e-10


def test_future_release_cannot_be_executed_as_first_source():
    with pytest.raises(ValueError,match='before its predicted release'):
        audit.release_route_cost([(0.,0.)],[(0.,0.)],(0.,0.),[('source',0),('cover',0)],[6.],[[6.]],[1])


def test_unserved_source_scan_and_five_second_service_both_count_once():
    c=[(0.,0.)]; s=[(0.,0.)]; start=(0.,0.)
    assert audit.release_route_cost(c,s,start,[('cover',0),('source',0)],[6.],[[6.]],[1])==17.
    assert audit.release_route_cost(c,s,start,[('source',0),('cover',0)],[6.],[[6.]],[0])==11.


@pytest.mark.parametrize('release',[[True],[-1],[2],[],[0,0]])
def test_invalid_release_identity_or_index_rejected(release):
    with pytest.raises(ValueError):
        audit.release_route_cost([(0.,0.)],[(0.,0.)],(0.,0.),[('cover',0),('source',0)],[6.],[[6.]],release)


def controller_record(monkeypatch,*,nominal_reply=False,reject_clear=False,tail_ready=False):
    """Actual calls, fixed replies, three public fixture stations; no scene."""
    from tests.test_q4_known_source import ScriptedReplies
    from simulator_client.state import Position
    from strategies.q4_release_scheduling import Q4ReleaseScheduling
    from strategies.search import _StopSearch
    points=((-1000.,0.),(-250.,-500.),(500.,0.))
    monkeypatch.setattr('planning.q4_directional_cover.certified_cover_points',lambda profile:
        (tuple(Position(*p) for p in points),{'passed':True,'test_only':True}))
    client=ScriptedReplies(); policy=Q4ReleaseScheduling(client,20000,6,max_expansions=200); policy.actions=1
    wire=[dict(action='/enter',response=dict(accepted=True,max_virtual_duration_s=360000.))]
    original_measure,original_clear=client.measure,client.clear
    counts={}
    def measure(p,c):
        counts[c]=counts.get(c,0)+1
        if c==1: reply=('near',None)
        elif c==3 and tail_ready:
            if counts[c]==1: reply=('direction',30.)
            elif (p.x,p.y)==points[-1] or policy._joint_context is not None: reply=('near',None)
            else: reply=('no_signal',None)
        elif c==2:
            if counts[c]==1: reply=('direction',0.)
            elif policy._joint_context is not None: reply=('near',None)
            elif nominal_reply and (p.x,p.y)==points[1]:
                center=policy.regions[2].enclosing_disk().center
                reply=('direction',round(math.degrees(math.atan2(center[1]-p.y,center[0]-p.x))%360,2)%360)
            else: reply=('no_signal',None)
        else: reply=('no_signal',None)
        client.measure_replies=[reply]
        response=original_measure(p,c)
        wire.append(dict(action='/measure',position=[p.x,p.y],channel=c,response=copy.deepcopy(response)))
        return response
    def clear(p,c):
        if reject_clear: client.reject_kind='clear'
        client.clear_replies=['success']
        response=original_clear(p,c)
        wire.append(dict(action='/clear',position=[p.x,p.y],channel=c,response=copy.deepcopy(response)))
        return response
    monkeypatch.setattr(client,'measure',measure); monkeypatch.setattr(client,'clear',clear)
    try: policy._execute_plan()
    except _StopSearch as e: policy.report.completion_reason=str(e)
    return dict(summary=policy.report.as_dict(),history=wire,row={'strategy':audit.LABEL},
        spec=dict(entrypoint=audit.ENTRY,kwargs=dict(config=audit.CONFIG,max_expansions=200)))


def test_prediction_does_not_make_silent_real_source_ready(monkeypatch):
    record=controller_record(monkeypatch)
    result=audit.replay_macros(record)
    assert result['forecast_scheduled_macros']==1 and result['completed_scans']==3
    params=record['summary']['strategy_parameters']; first=params['known_source_plan_log'][0]
    assert first['release_indices']==[0,1] and first['source_services_s']==[5.,5.]
    second=next(s for s in params['known_source_service_log'] if s['selected']['channel']==2)
    assert second['selected']['ready_before'] is False and second['remaining_covers_before']==[]


def test_real_matching_direction_can_authorize_source_before_cover_tail(monkeypatch):
    record=controller_record(monkeypatch,nominal_reply=True)
    result=audit.replay_macros(record)
    assert result['forecast_scheduled_macros']>=1 and result['completed_scans']==3
    service=next(s for s in record['summary']['strategy_parameters']['known_source_service_log']
                 if s['selected']['channel']==2)
    assert service['selected']['ready_before'] is True and service['remaining_covers_before']


def test_zero_action_mixed_plan_is_not_counted_as_a_real_macro(monkeypatch):
    record=controller_record(monkeypatch,reject_clear=True)
    result=audit.replay_macros(record)
    assert result['forecast_scheduled_macros']==0 and result['completed_scans']==1


@pytest.mark.parametrize('change',['release_zero','release_late','target','initial_radius','nominal_bearing',
    'nominal_distance','predicted_radius','step_index','step_position','step_status','step_stop',
    'missing_step','missing_forecast','fake_ready','source_service_zero','predicted_matrix',
    'unknown_removed','expanded','cumulative','cost','false_lower','duplicate_order','selected',
    'alias','service_ready','service_covers','service_prefix','source_macro_missing'])
def test_forecast_route_and_real_service_tampering_rejected(monkeypatch,change):
    record=controller_record(monkeypatch)
    p=record['summary']['strategy_parameters']; e=p['known_source_plan_log'][0]
    f=e['release_forecasts'][1]; step=f['steps'][0]
    if change=='release_zero': e['release_indices'][1]=f['release_index']=0
    elif change=='release_late': e['release_indices'][1]=f['release_index']=2
    elif change=='target': f['target'][0]+=1
    elif change=='initial_radius': f['initial_radius_m']-=1
    elif change=='nominal_bearing': step['bearing_deg']+=.01
    elif change=='nominal_distance': step['nominal_distance_m']+=1
    elif change=='predicted_radius': step['predicted_radius_m']=0.
    elif change=='step_index': step['cover_index']=1
    elif change=='step_position': step['position'][0]+=1
    elif change=='step_status': step['status']='actual_ready'
    elif change=='step_stop': step['stopped']=False
    elif change=='missing_step': f['steps']=[]
    elif change=='missing_forecast': e['release_forecasts'].pop()
    elif change=='fake_ready': f['ready_before']=True
    elif change=='source_service_zero': e['source_services_s']=[0.,0.]
    elif change=='predicted_matrix': e['source_scan_s'][0][1]=0.
    elif change=='unknown_removed': e['background_scan_s'][0]-=6.
    elif change=='expanded': e['max_expansions']+=1
    elif change=='cumulative': e['total_expansions_after']+=1
    elif change=='cost': e['result']['cost_s']-=5.
    elif change=='false_lower': e['result']['lower_bound_s']=e['result']['cost_s']+1.
    elif change=='duplicate_order': e['result']['order']=[['source',0]]*4
    elif change=='selected': e['selected_channel']=2
    elif change=='alias': p['release_scheduling_plan_log']=[]
    else:
        s=p['known_source_service_log'][-1]
        if change=='service_ready': s['selected']['ready_before']=True
        elif change=='service_covers': s['remaining_covers_before']=[[0.,0.]]
        elif change=='service_prefix': s['service_end_action_count']-=1
        else: p['known_source_service_log'].pop()
    with pytest.raises(ValueError): audit.replay_macros(record)


def test_forecast_skips_five_meter_near_and_range_without_fake_observations():
    from types import SimpleNamespace
    from localization import CandidateRegion
    r=CandidateRegion().observe((-1000.,0.),0.)
    center=tuple(r.enclosing_disk().center); old=copy.deepcopy(r.vertices); n=len(r.observations)
    prefix=SimpleNamespace(target=lambda i,c:center,snapshot=lambda i:({1},set(),{}, {1:r}),ready=lambda i,c:False)
    covers=[center,(center[0],center[1]+1501.),(center[0],center[1]-500.)]
    result=audit.forecast(prefix,0,1,covers)
    assert [s['status'] for s in result['steps']]==['skip_near_distance','skip_beyond_reception_radius','predicted_ready']
    assert result['release_index']==3 and len(r.observations)==n and r.vertices==old


def test_no_remaining_covers_releases_nonready_without_invented_radio():
    from types import SimpleNamespace
    from localization import CandidateRegion
    r=CandidateRegion().observe((-1000.,0.),0.); center=tuple(r.enclosing_disk().center)
    p=SimpleNamespace(target=lambda i,c:center,snapshot=lambda i:({1},set(),{}, {1:r}),ready=lambda i,c:False)
    result=audit.forecast(p,0,1,[])
    assert result['release_index']==0 and result['status']=='no_remaining_covers' and result['steps']==[]


@pytest.mark.parametrize('mode',['silent','matching','rejected'])
def test_all_inherited_prefix_checks_keep_original_actions(monkeypatch,mode):
    record=controller_record(monkeypatch,nominal_reply=mode=='matching',reject_clear=mode=='rejected')
    result=audit.audit_release_scheduling_prefix(record)
    assert result['passed'] and all(result[k]['passed'] for k in ('r12','r8','range','scheduling'))
    assert result['forecast_scheduled_macros']==(0 if mode=='rejected' else 1)


@pytest.mark.parametrize('child',['audit_joint_continuation_prefix','audit_clear_before_probe_prefix',
                                 'audit_range_prefix','audit_scheduling_prefix'])
def test_old_checker_false_never_silently_passes(monkeypatch,child):
    record=controller_record(monkeypatch)
    monkeypatch.setattr(audit.base,child,lambda record:{'passed':False})
    with pytest.raises(ValueError,match='Inherited release'): audit.audit_release_scheduling_prefix(record)


def test_reused_replay_identity_pinned_before_compatibility_view(monkeypatch):
    audit.verify_source_contract()
    record=controller_record(monkeypatch)
    monkeypatch.setattr(audit,'BASE_AUDIT_SHA256','0'*64)
    with pytest.raises(ValueError,match='R34 replay'): audit.audit_release_scheduling_prefix(record)


def test_spec_cannot_borrow_parent_safe_behavior_for_different_algorithm(monkeypatch):
    record=controller_record(monkeypatch)
    record['spec']['kwargs']['config']='all_known_matrix'
    with pytest.raises(ValueError,match='spec'): audit.audit_release_scheduling_prefix(record)


def test_ready_nonready_mix_after_last_cover_is_not_forecast_exposure(monkeypatch):
    from experiments.q4_release_scheduling_release import actual_service_count
    record=controller_record(monkeypatch,tail_ready=True)
    result=audit.audit_release_scheduling_prefix(record)
    events=record['summary']['strategy_parameters']['release_scheduling_plan_log']
    mixed=[e for e in events if e['end_actual_action_count']>e['after_actual_action_count']
           and any(s['ready'] for s in e['source_evidence']) and not all(s['ready'] for s in e['source_evidence'])]
    tail=[e for e in mixed if not e['remaining_covers']]
    assert tail and all(all(r==0 for r in e['release_indices']) for e in tail)
    assert result['forecast_scheduled_macros']==len(mixed)-len(tail)==actual_service_count(record)
