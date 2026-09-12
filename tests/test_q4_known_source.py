"""Actual controller calls with scripted replies, no scene or network."""
import copy

import pytest

from simulator_client.state import Position
from strategies.q4_joint_continuation import Q4JointContinuation
from strategies.q4_known_source import Q4KnownSource, run_q4_known_source
from strategies.search import _StopSearch
from tests.test_q4_clear_before_probe import Replies


class ScriptedReplies(Replies):
    def measure(self,point,channel):
        if not self.measure_replies:self.measure_replies=[('no_signal',None)]
        return super().measure(point,channel)


def make(monkeypatch,*,points=((1000.,0.),(1000.,1000.)),active=6):
    # Controller fixture only; independent cover audit uses the real22 stations.
    monkeypatch.setattr('planning.q4_directional_cover.certified_cover_points',lambda profile:
                        (tuple(Position(*p) for p in points),{'passed':True,'test_only':True}))
    client=ScriptedReplies()
    return Q4KnownSource(client,20000,active,max_expansions=0),client


def prime(policy,client,channel=1,point=(-1000.,0.),bearing=0.):
    client.measure_replies=[('direction',bearing)]
    policy._perform('measure',Position(*point),channel,'constructed_observation')
    assert policy.regions[channel].enclosing_disk().radius>40


def test_first_positive_broad_source_enters_plan_and_full_real_resolver(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client)
    before=copy.deepcopy(policy.regions[1].vertices)
    client.measure_replies=[('near',None)]
    policy._execute_plan()
    e=policy.route_log[0];s=policy.known_source_service_log[0]
    assert e['source_channels']==[1] and e['selected_kind']=='source'
    assert e['source_services_s']==[0.] and e['source_evidence'][0]['ready'] is False
    assert e['source_evidence'][0]['vertices']==[list(v) for v in before]
    assert s['selected']['radius_m']>40 and s['selected']['ready_before'] is False
    assert s['remaining_covers_before']==[[1000.,0.],[1000.,1000.]]
    assert s['resolver_start_action_count']==1 and s['service_end_action_count']==3
    assert [a['phase'] for a in policy.report.action_history[1:3]]==['active_localization','near_clear']
    assert s['actual_cost_s']>5 and s['cleared'] and s['status']=='resolved'
    assert policy.report.coverage_points_visited==2 and policy.report.coverage_complete
    assert policy.joint_resolvers[s['resolver_id']]['end_actual_action_count']==3
    assert policy.early_attempted==set() and policy.service_deadline is None
    assert policy._joint_context is policy._probe_resolving_channel is None


def test_matrix_accounts_ready_far_blocked_and_unknown_without_mutating_prefix(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client)
    prime(policy,client,channel=2);policy.blocked.add(2)
    client.measure_replies=[('near',None)]
    policy._perform('measure',Position(0,0),3,'constructed_observation')
    policy.cleared.add(4)
    history=copy.deepcopy(policy.report.action_history)
    vertices=copy.deepcopy(policy.regions[1].vertices)
    covers=[Position(0,0),Position(-3000,0)]
    b,w,bg,evidence,cells=policy._scan_matrix(covers,[1,3])
    assert bg==[2]+list(range(5,21))
    assert w==[[6.,0.],[0.,0.]] and b==[102.,96.]
    assert cells[1][0]['reason']=='positive_region_beyond_max_reception_radius'
    assert cells[1][1]['fee_s']==0. and evidence[2]['blocked']
    for k,row in enumerate(cells):
        assert b[k]+sum(w[k])==sum(item['fee_s'] for item in row)
        assert [item['channel'] for item in row]==[1,2,3]+list(range(5,21))
    assert policy.report.action_history==history and policy.regions[1].vertices==vertices
    assert not policy.range_skips and not policy.skipped_scans  # predictions create no evidence


def test_parent_early_service_still_precedes_matrix(monkeypatch):
    policy,client=make(monkeypatch,points=((100.,0.),))
    client.measure_replies=[('direction',0.),('direction',90.)]
    policy._perform('measure',Position(-1000,0),1,'constructed_observation')
    policy._perform('measure',Position(0,-1000),1,'constructed_observation')
    client.measure_replies=[('no_signal',None)]
    policy._perform('measure',Position(0,0),20,'constructed_observation')
    assert policy._early_candidate(policy.points[0]) is not None
    policy._execute_plan()
    assert len(policy.early_service_log)==1 and policy.cleared=={1}
    assert not policy.known_source_service_log and not policy.route_log


def test_zero_accepted_service_rejection_logs_real_empty_interval_and_no_scan(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client);client.reject_kind='measure'
    with pytest.raises(_StopSearch):policy._execute_plan()
    s=policy.known_source_service_log[0];e=policy.route_log[0]
    assert s['status']==e['status']=='interrupted'
    assert s['resolver_start_action_count']==s['service_end_action_count']==1
    assert e['end_actual_action_count']==1 and s['actual_cost_s']==0
    assert policy.report.coverage_points_visited==0 and not policy.cleared
    assert policy._joint_context is policy._probe_resolving_channel is None


def test_cover_rejection_does_not_pop_or_certify_station(monkeypatch):
    policy,client=make(monkeypatch);client.reject_kind='measure'
    with pytest.raises(_StopSearch):policy._execute_plan()
    assert policy.report.coverage_points_visited==0 and not policy.report.coverage_complete
    assert not policy.known_source_service_log and not policy.route_log


def test_known16_stops_discovery_but_still_performs_all_real_clears(monkeypatch):
    policy,client=make(monkeypatch)
    for ch in range(1,17):
        client.measure_replies=[('near',None)]
        policy._perform('measure',Position(0,0),ch,'constructed_observation')
    client.clear_replies=['success']*16
    with pytest.raises(_StopSearch,match='source_count_upper_bound'):policy._execute_plan()
    assert len(policy.cleared)==16 and len(policy.known_source_service_log)==16
    assert policy.report.coverage_points_visited==0 and policy.report.completion_certified_under_model
    assert all(not e['remaining_covers'] and e['source_scan_s']==[] for e in policy.route_log)
    assert all(not s['remaining_covers_before'] for s in policy.known_source_service_log)
    assert len([a for a in policy.report.action_history if a['action']=='clear'])==16


def test_old_runtime_actions_and_probe_algorithms_are_inherited():
    for name in ('_perform','_scan','_resolve','_next_probe','_clear','_check_budget','_insertion_budget',
                 '_early_candidate','_early_service','_ready','_target','run'):
        assert getattr(Q4KnownSource,name) is getattr(Q4JointContinuation,name)


@pytest.mark.parametrize('kwargs',[{'problem':3},{'config':'all_known'}, {'max_actions':True},
                                    {'max_active_probes':-1},{'max_expansions':10001}])
def test_invalid_entrypoint_rejected_before_client(kwargs):
    with pytest.raises(ValueError):run_q4_known_source(object(),**kwargs)
