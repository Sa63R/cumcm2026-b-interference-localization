"""Actual inherited controller paths with explicit scripted replies, no scenes."""
import copy
import math
from types import SimpleNamespace

import pytest

from simulator_client.state import Position
from strategies.q4_clear_region import Q4ClearRegion,run_q4_clear_region
from strategies.q4_range_scheduling import Q4RangeScheduling
from strategies.q4_r2_scheduling import _ServiceSliceExpired
from strategies.search import _StopSearch


class Replies:
    def __init__(self):
        self.state=SimpleNamespace(position=Position(0,0),current_channel=1,virtual_time_s=0.,
                                   max_virtual_duration_s=360000.,sources={})
        self.calls=[];self.measure_replies=[];self.accept=True;self.clear_result='success'
    def measure(self,p,c):
        result,bearing=self.measure_replies.pop(0)
        reply=self._call('measure',p,c,result)
        if bearing is not None:reply['svd_deg']=bearing
        return reply
    def clear(self,p,c):return self._call('clear',p,c,self.clear_result)
    def _call(self,kind,p,c,result):
        self.calls.append((kind,p,c))
        if not self.accept:return {'accepted':False}
        movement=round(self.state.position.distance_to(p)/5*1e6)/1e6
        cost=movement+(5 if kind=='measure' or result=='success' else 3)
        if kind=='measure':
            cost+=c!=self.state.current_channel;self.state.current_channel=c
        self.state.position=p;self.state.virtual_time_s+=cost
        return {'accepted':True,'virtual_time_s':self.state.virtual_time_s,
                'measure_result' if kind=='measure' else 'clear_result':result}


def make(monkeypatch,config='incoming'):
    import planning.q4_directional_cover as geometry
    points=(Position(0,0),Position(100,0),Position(0,100))
    monkeypatch.setattr(geometry,'certified_cover_points',lambda profile:(points,{'passed':True,'test_only':True}))
    client=Replies()
    return Q4ClearRegion(client,20000,6,config=config,max_expansions=0),client


def prime_region(policy,client,scale=300.):
    client.measure_replies=[('direction',0.),('direction',90.)]
    policy._perform('measure',Position(-scale,0),1,'active_localization')
    policy._perform('measure',Position(0,-scale),1,'active_localization')
    return Position.coerce(policy.regions[1].enclosing_disk().center)


def route(policy,channel=1,anchor_kind='cover',anchor_position=(0,100),prefix=None):
    n=len(policy.report.action_history) if prefix is None else prefix
    original=policy._target(channel)
    source_channels=[channel];positions=[[original.x,original.y]]
    if anchor_kind=='source':
        policy.detected.add(2);policy.near_points[2]=Position(*anchor_position)
        source_channels.append(2);positions.append(list(anchor_position))
    second=[anchor_kind,1 if anchor_kind=='source' else 0]
    event={'after_actual_action_count':n,'selected_kind':'source','selected_channel':channel,
           'source_channels':source_channels,'source_positions':positions,'remaining_covers':[list(anchor_position)],
           'result':{'order':[['source',0],second]}}
    policy.route_log.append(event)
    return event


def test_only_clear_request_position_changes_and_positive_prefix_is_preserved(monkeypatch):
    policy,client=make(monkeypatch)
    original=prime_region(policy,client)
    before=copy.deepcopy(policy.report.action_history)
    assert policy._resolve(1)
    event=policy.clear_region_log[-1]
    assert policy.report.action_history[:-1]==before
    assert client.calls[-1][0]=='clear' and client.calls[-1][1]!=original
    assert policy.cleared=={1} and event['executed'] and event['geometry']['certificate']['passed']
    assert event['after_actual_action_count']==2 and event['end_action_count']==3
    assert policy._resolve_anchor is None


def test_actual_near_prefix_gets_15m_service_disk(monkeypatch):
    policy,client=make(monkeypatch)
    client.measure_replies=[('near',None),('no_signal',None)]
    policy._perform('measure',Position(0,0),1,'coverage')
    policy._perform('measure',Position(100,0),2,'coverage')
    assert policy._resolve(1)
    e=policy.clear_region_log[-1]
    assert e['phase']=='near_clear' and e['geometry']['certificate']['kind']=='near_disk'
    assert client.calls[-1][1].x==pytest.approx(14.99998)
    assert client.state.current_channel==2  # Clear retains real receiver tuning.


@pytest.mark.parametrize('kind',['cover','source'])
def test_anchor_matches_second_task_of_current_selected_source_macro(monkeypatch,kind):
    policy,client=make(monkeypatch,'anchored');prime_region(policy,client)
    route(policy,anchor_kind=kind,anchor_position=(100,50))
    policy._resolve(1)
    e=policy.clear_region_log[-1]
    assert e['anchor']==[100,50] and e['anchor_status']=='valid'
    assert e['anchor_source']['kind']=='route_second_task' and e['anchor_source']['prefix']==2
    assert e['geometry']['method']=='finite_rays' and policy._resolve_anchor is None


@pytest.mark.parametrize('change',['old_prefix','wrong_channel','wrong_first_task','empty_order'])
def test_stale_or_unrelated_route_cannot_supply_anchor(monkeypatch,change):
    policy,client=make(monkeypatch,'anchored');prime_region(policy,client)
    e=route(policy)
    if change=='old_prefix':e['after_actual_action_count']=1
    elif change=='wrong_channel':e['selected_channel']=2
    elif change=='wrong_first_task':e['result']['order'][0]=['cover',0]
    else:e['result']['order']=[]
    policy._resolve(1)
    log=policy.clear_region_log[-1]
    assert log['anchor'] is None and log['geometry']['method']=='boundary_candidates'


def test_route_anchor_expires_after_real_probe_in_same_resolve(monkeypatch):
    policy,client=make(monkeypatch,'anchored');prime_region(policy,client,scale=1000.)
    assert not policy._ready(1)
    route(policy)
    client.measure_replies=[('near',None)]
    assert policy._resolve(1)
    e=policy.clear_region_log[-1]
    assert e['anchor'] is None and e['anchor_status']=='route_prefix_expired'
    assert e['phase']=='near_clear'


def test_early_anchor_remains_valid_for_its_fixed_next_cover_through_probe(monkeypatch):
    policy,client=make(monkeypatch,'anchored');prime_region(policy,client,scale=1000.)
    # An additional real silent scan changes only the current point.
    client.measure_replies=[('no_signal',None),('near',None)]
    policy._perform('measure',Position(0,-50),2,'coverage')
    candidate=policy._early_candidate(Position(200,0))
    assert candidate is not None
    policy._early_service(candidate)
    e=policy.clear_region_log[-1]
    assert e['anchor']==[200,0] and e['anchor_source']['kind']=='early_service_next_cover'
    assert e['anchor_source']['candidate_prefix']==3 and e['after_actual_action_count']==4
    assert policy.cleared=={1} and policy.service_deadline is None
    assert policy._early_anchor is None and policy._resolve_anchor is None


def test_original_service_slice_checks_selected_request_before_sending(monkeypatch):
    policy,client=make(monkeypatch);original=prime_region(policy,client)
    before=len(client.calls)
    policy.service_deadline=client.state.virtual_time_s+1.
    with pytest.raises(_ServiceSliceExpired):policy._resolve(1)
    e=policy.clear_region_log[-1]
    assert len(client.calls)==before and e['executed'] is False and e['end_action_count']==2
    assert policy._resolve_anchor is None


@pytest.mark.parametrize('accepted,result',[(False,'success'),(True,'no_target_in_range')])
def test_executed_means_accepted_action_not_planned_or_successful(monkeypatch,accepted,result):
    policy,client=make(monkeypatch);original=prime_region(policy,client)
    client.accept=accepted;client.clear_result=result
    if not accepted:
        with pytest.raises(_StopSearch):policy._clear(original,1,'certified_clear')
    else:
        assert policy._clear(original,1,'certified_clear') is False
    e=policy.clear_region_log[-1]
    assert e['executed'] is accepted
    assert e['end_action_count']==2+accepted and not policy.cleared


def test_noncertified_optical_clear_is_unchanged_and_unlogged(monkeypatch):
    policy,client=make(monkeypatch)
    policy._clear(Position(23,41),1,'guaranteed_clearance')
    assert client.calls[-1]==('clear',Position(23,41),1) and not policy.clear_region_log


def test_fallback_without_matching_certificate_keeps_original_request(monkeypatch):
    policy,client=make(monkeypatch)
    policy._clear(Position(23,41),1,'certified_clear')
    assert client.calls[-1][1]==Position(23,41)
    assert policy.clear_region_log[-1]['geometry']['status']=='fallback'


def test_original_probe_scan_route_and_service_gate_are_inherited():
    for name in ('_next_probe','_scan','_execute_plan','_target','_ready','_perform'):
        assert getattr(Q4ClearRegion,name) is getattr(Q4RangeScheduling,name)


@pytest.mark.parametrize('kwargs',[{'problem':3},{'config':'bad'},{'max_expansions':True},{'max_active_probes':31}])
def test_invalid_configuration_rejected_before_client_access(kwargs):
    with pytest.raises(ValueError):run_q4_clear_region(object(),**kwargs)
