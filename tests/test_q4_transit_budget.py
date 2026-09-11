"""Scripted actual controller calls, no generated scenario or simulator."""
import copy
from types import SimpleNamespace

import pytest

from simulator_client.state import Position
from strategies.q4_joint_continuation import Q4JointContinuation
from strategies.q4_transit_budget import Q4TransitBudget, _TransitSliceExpired, run_q4_transit_budget
from strategies.search import _StopSearch
from tests.test_q4_clear_before_probe import Replies


DESTINATION = Position(0., 1000.)


class ScriptedReplies(Replies):
    def measure(self, point, channel):
        if not self.measure_replies:
            self.measure_replies = [("no_signal", None)]
        return super().measure(point, channel)


def make(monkeypatch, *, max_actions=20000, active=6, base=False):
    monkeypatch.setattr("planning.q4_directional_cover.certified_cover_points", lambda profile:
        ((Position(0,0), DESTINATION), {"passed":True,"test_only":True}))
    client = ScriptedReplies()
    cls = Q4JointContinuation if base else Q4TransitBudget
    return cls(client,max_actions,active,max_expansions=0), client


def prime(policy, client):
    # These are actual positive actions, not falsely completed cover stations.
    client.measure_replies = [("direction",0.),("direction",90.)]
    policy._perform("measure",Position(-1000,0),1,"constructed_observation")
    policy._perform("measure",Position(0,-1000),1,"constructed_observation")
    disk=policy.regions[1].enclosing_disk()
    assert 19.9 < disk.radius < 40.
    return Position.coerce(disk.center)


def test_actual_positive_prefix_selects_originally_too_long_service(monkeypatch):
    policy, client=make(monkeypatch);center=prime(policy,client)
    assert policy._early_candidate(DESTINATION) is None
    candidates=policy._transit_candidates(DESTINATION)
    assert len(candidates)==1 and candidates[0]['channel']==1
    assert candidates[0]['center']==[center.x,center.y]
    assert candidates[0]['original_first_cost_s'] > 60.
    assert .1 <= candidates[0]['projection_fraction'] <= .9
    assert candidates[0]['detour_m'] <= 100.


def test_real_r8_success_closes_resolver_then_complete_parent_scan(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client)
    previous=copy.deepcopy(policy.regions[1].vertices)
    policy._scan(DESTINATION)
    event=policy.transit_service_log[0]
    assert event['selected']['channel']==1 and event['service_status']=='cleared'
    assert event['resolver_start_action_count']==2 and event['service_end_action_count']==3
    assert event['scan_start_action_count']==3 and event['scan_end_action_count']==22
    assert event['scan_status']=='completed' and policy.report.coverage_points_visited==1
    assert event['service_end_virtual_us']-event['start_virtual_us'] > 60_000_000
    assert event['actual_incremental_through_first_measure_us'] <= 60_000_000
    assert [a['channel'] for a in policy.report.action_history[3:]]==list(range(2,21))
    assert all(a['phase']=='coverage' and a['position']==[0.,1000.] for a in policy.report.action_history[3:])
    assert [g['kind'] for g in event['gates']]==['atomic_r8','action']
    assert all(g['executed_action_count']==1 for g in event['gates'])
    assert policy.transit_attempted=={1} and policy.early_attempted==set()
    assert not policy.early_service_log and policy.service_deadline is None
    assert policy._transit_context is policy._joint_context is policy._probe_resolving_channel is None
    assert policy.pending_pair is None and policy.regions[1].vertices==previous
    assert policy.joint_resolvers[event['resolver_id']]['end_actual_action_count']==3


def test_real_r8_miss_and_same_point_measure_are_atomically_reserved(monkeypatch):
    policy,client=make(monkeypatch);center=prime(policy,client)
    client.clear_replies=['no_target_in_range','success']
    client.measure_replies=[('near',None)]
    policy._scan(DESTINATION)
    event=policy.transit_service_log[0]
    service=policy.report.action_history[2:event['service_end_action_count']]
    assert [a['phase'] for a in service]==['speculative_clear_before_probe','active_localization','near_clear']
    assert service[0]['position']==service[1]['position']==[center.x,center.y]
    assert policy.clear_before_probe_log[0]['status']=='failed_then_measured'
    assert policy.clear_before_probe_log[0]['budget']['skip_reasons']==[]
    assert policy.clear_before_probe_log[0]['budget']['service_deadline'] is None
    assert event['gates'][0]['kind']=='atomic_r8' and event['gates'][0]['predicted_actions']==2
    assert event['gates'][0]['executed_action_count']==2
    assert [g['kind'] for g in event['gates'][1:]]==['action']*3
    assert all(g['admitted'] for g in event['gates'])
    assert event['actual_incremental_through_first_measure_us'] <= 60_000_000


def test_atomic_action_reserve_can_stop_before_clear_while_original_scan_fits(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client)
    policy.max_actions=policy.actions+22  # one action+20 scans+exit fits; two do not
    policy._scan(DESTINATION)
    event=policy.transit_service_log[0]
    assert event['service_status']=='slice_expired' and event['scan_status']=='completed'
    assert event['service_end_action_count']==event['resolver_start_action_count']==2
    assert len(event['gates'])==1 and event['gates'][0]['kind']=='atomic_r8'
    assert event['gates'][0]['reasons']==['return_scan_action_reserve']
    assert event['gates'][0]['executed_action_count']==0
    assert not policy.clear_before_probe_log and not policy.speculative_attempted
    assert policy.transit_attempted=={1} and policy.report.coverage_points_visited==1
    assert all(a['action']=='measure' for a in policy.report.action_history)
    assert len(policy.report.action_history)==22


def test_atomic_virtual_reserve_does_not_spend_baseline_scan_budget(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client)
    client.state.max_virtual_duration_s=client.state.virtual_time_s+522.
    policy._scan(DESTINATION)
    event=policy.transit_service_log[0]
    assert event['service_status']=='slice_expired' and event['scan_status']=='completed'
    assert event['gates'][0]['reasons']==['return_scan_virtual_reserve']
    assert not policy.clear_before_probe_log and event['service_end_action_count']==2
    assert client.state.virtual_time_s < client.state.max_virtual_duration_s


@pytest.mark.parametrize('stop',['real_deadline','reject_clear'])
def test_terminal_service_error_never_forces_scan_and_parent_finally_runs(monkeypatch,stop):
    policy,client=make(monkeypatch);prime(policy,client)
    if stop=='real_deadline':client.remaining_real_time_s=1.
    else:client.reject_kind='clear'
    with pytest.raises(_StopSearch):policy._scan(DESTINATION)
    event=policy.transit_service_log[0]
    assert event['service_status']=='interrupted' and event['scan_status']=='not_started'
    assert event['scan_start_action_count'] is None and event['end_actual_action_count']==2
    assert policy.report.coverage_points_visited==0
    assert policy._transit_context is policy._joint_context is policy._probe_resolving_channel is None
    assert policy.pending_pair is None
    assert not any(a['phase']=='coverage' for a in policy.report.action_history)
    assert all(g['executed_action_count']==0 for g in event['gates'])


def test_failed_clear_then_real_rejection_keeps_one_actual_action_without_scan(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client)
    client.clear_replies=['no_target_in_range'];client.reject_kind='measure'
    with pytest.raises(_StopSearch):policy._scan(DESTINATION)
    event=policy.transit_service_log[0]
    assert event['service_status']=='interrupted' and event['scan_status']=='not_started'
    assert event['service_end_action_count']==3 and event['gates'][0]['executed_action_count']==1
    assert policy.report.action_history[-1]['action']=='clear'
    assert not policy.cleared and policy.report.coverage_points_visited==0
    assert policy.clear_before_probe_log[0]['status']=='interrupted'


def test_scan_rejection_after_success_is_not_counted_as_completed_cover(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client);client.reject_kind='measure'
    with pytest.raises(_StopSearch):policy._scan(DESTINATION)
    event=policy.transit_service_log[0]
    assert event['service_status']=='cleared' and event['scan_status']=='interrupted'
    assert event['scan_start_action_count']==event['scan_end_action_count']==3
    assert event['first_coverage_action_index'] is None and policy.report.coverage_points_visited==0


def test_actual_action_gate_intercepts_far_later_action_and_returns_to_cover(monkeypatch):
    policy,client=make(monkeypatch);center=prime(policy,client)
    def scripted_resolve(channel):
        policy._perform('measure',center,channel,'active_localization')
        policy._perform('measure',Position(2000.,0.),channel,'active_localization')
    # This isolated budget test supplies planned actions; the tests above run
    # the complete unchanged resolver, including R8 and cleanup paths.
    monkeypatch.setattr(policy,'_resolve',scripted_resolve)
    policy._scan(DESTINATION)
    event=policy.transit_service_log[0]
    assert event['service_status']=='slice_expired' and event['service_end_action_count']==3
    assert event['gates'][0]['admitted'] and event['gates'][0]['executed_action_count']==1
    assert event['gates'][1]['reasons']==['incremental_budget']
    assert event['gates'][1]['executed_action_count']==0
    assert event['actual_incremental_through_first_measure_us'] <= 60_000_000
    assert not any(a['position']==[2000.,0.] for a in policy.report.action_history)


def test_same_position_scan_matches_inherited_trace_exactly(monkeypatch):
    actual,client=make(monkeypatch);baseline,base_client=make(monkeypatch,base=True)
    for p,c in ((actual,client),(baseline,base_client)):
        prime(p,c);p._scan(c.state.position)
    assert actual.report.action_history==baseline.report.action_history
    assert client.calls==base_client.calls
    assert actual.transit_service_log[0]['skip_reason']=='same_position'
    assert actual.transit_attempted==set()


@pytest.mark.parametrize('kind',['known16','macro_cap','old_candidate'])
def test_existing_discovery_and_early_service_take_precedence(monkeypatch,kind):
    policy,client=make(monkeypatch);prime(policy,client)
    if kind=='known16':policy.detected.update(range(1,17))
    elif kind=='macro_cap':policy.transit_attempted={2,3,4,5}
    else:monkeypatch.setattr(policy,'_early_candidate',lambda q:(0.,50.,2,25.))
    policy._scan(DESTINATION)
    event=policy.transit_service_log[0]
    assert event['selected'] is None and not event['gates']
    assert event['skip_reason']=={'known16':'discovery_count_cap','macro_cap':'macro_limit','old_candidate':'old_early_candidate'}[kind]
    assert event['scan_status']=='completed'


@pytest.mark.parametrize('excluded',['cleared','blocked','early_attempted','transit_attempted','near'])
def test_live_candidate_exclusions(excluded,monkeypatch):
    policy,client=make(monkeypatch);center=prime(policy,client)
    if excluded=='near':policy.near_points[1]=center
    else:getattr(policy,excluded).add(1)
    assert not policy._transit_candidates(DESTINATION)


def test_pure_geometry_boundaries_and_stable_channel_tie(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client)
    policy.detected.add(2);policy.regions[2]=policy.regions[1].copy()
    assert [c['channel'] for c in policy._transit_candidates(DESTINATION)]==[1,2]
    center=policy.regions[1].enclosing_disk().center
    for radius,allowed in ((19.9,False),(19.900001,True),(40.,True),(40.000001,False)):
        monkeypatch.setattr(policy.regions[1],'enclosing_disk',lambda r=radius:SimpleNamespace(center=center,radius=r))
        assert any(c['channel']==1 for c in policy._transit_candidates(DESTINATION))==allowed
    monkeypatch.setattr(policy.regions[1],'enclosing_disk',lambda:SimpleNamespace(center=(1000.,0.),radius=25.))
    assert not any(c['channel']==1 for c in policy._transit_candidates(DESTINATION))  # detour >100


def test_unmodified_parent_decision_methods_and_independent_old_budget(monkeypatch):
    policy,_=make(monkeypatch)
    for name in ('_execute_plan','_resolve','_next_probe','_perform','_clear','_early_candidate','_early_service'):
        assert getattr(Q4TransitBudget,name) is getattr(Q4JointContinuation,name)
    assert policy.scheduling_config['early_services']==4 and policy.service_deadline is None
    assert policy.report.strategy_parameters['transit_budget_limits']['macros']==4


@pytest.mark.parametrize('kwargs',[{'problem':3},{'config':'return_credit_60'},
    {'max_actions':True},{'max_active_probes':-1},{'max_expansions':10001}])
def test_invalid_entrypoint_rejected_before_client(kwargs):
    with pytest.raises(ValueError):run_q4_transit_budget(object(),**kwargs)
