"""Constructed accepted-wire prefixes; no engine, scenario or network access."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from simulator_client.state import Position
from strategies.q4_cover_feedback import Q4CoverFeedback, run_q4_cover_feedback
from strategies.q4_joint_continuation import Q4JointContinuation
from strategies.q4_r2_scheduling import _ServiceSliceExpired
from strategies.search import _StopSearch
from tests.test_q4_clear_before_probe import Replies
import strategies.q4_cover_feedback as controller


def make(monkeypatch, cls=Q4CoverFeedback, *, max_actions=20000):
    import planning.q4_directional_cover as cover
    monkeypatch.setattr(cover,'certified_cover_points',lambda profile:
        ((Position(100.,100.),Position(300.,0.)),{'passed':True,'test_only':True}))
    client=Replies()
    policy=cls(client,max_actions,6,max_expansions=200)
    return policy,client


def prime(policy,client):
    """Two real directional replies for unready 2, real near for ready 1."""
    client.measure_replies=[('direction',0.),('direction',90.),('near',None)]
    policy._perform('measure',Position(-1000.,0.),2,'coverage')
    policy._perform('measure',Position(0.,-1000.),2,'coverage')
    policy._perform('measure',Position(0.,-900.),1,'coverage')
    assert policy._ready(1) and not policy._ready(2)
    assert policy._early_candidate(policy.points[0]) is None
    return len(policy.report.action_history)


def one_macro(monkeypatch,policy,before):
    original=policy._early_candidate
    def early(q):
        if len(policy.report.action_history)>before:
            raise _StopSearch('constructed_macro_boundary')
        return original(q)
    monkeypatch.setattr(policy,'_early_candidate',early)
    with pytest.raises(_StopSearch,match='constructed_macro_boundary'):
        policy._execute_plan()


def score(veto,expanded=7):
    # A deliberate score stub tests physical routing ownership, not inference.
    return dict(status='scored',information_changed=True,recommend_veto=veto,expanded=expanded)


def test_no_change_original_actions_and_r12_resolver_are_inherited(monkeypatch):
    policy,client=make(monkeypatch); n=prime(policy,client)
    old,old_client=make(monkeypatch,Q4JointContinuation); old_n=prime(old,old_client)
    monkeypatch.setattr(policy,'_prediction',lambda *a:dict(status='fallback',information_changed=False,
        recommend_veto=False,expanded=0,fallback_reason='no_information_change'))
    one_macro(monkeypatch,policy,n); one_macro(monkeypatch,old,old_n)
    assert client.calls==old_client.calls
    assert policy.report.action_history==old.report.action_history
    assert policy.cleared=={1} and policy.cover_feedback_vetoes==0
    assert policy.pending_pair is None and policy._joint_context is None
    event=policy.cover_feedback_log[0]
    assert event['executed_kind']=='source' and not event['executed_cover']
    assert event['after_actual_action_count']==3 and event['end_actual_action_count']==4


def test_real_model_controller_prefix_without_stub_keeps_canonical(monkeypatch):
    policy,client=make(monkeypatch); n=prime(policy,client)
    before=deepcopy(policy.regions[2].__dict__)
    # Real scoring is deliberately chosen at zero-distance ready clear: veto is
    # too costly, but all 11 real model calls execute and are JSON-compatible.
    one_macro(monkeypatch,policy,n)
    event=policy.cover_feedback_log[0]
    assert event['prediction']['status']=='scored'
    assert len(event['prediction']['solve_log'])==11
    assert not event['veto_selected']
    assert policy.regions[2].__dict__==before


def test_veto_owns_complete_original_cover_and_replans_real_feedback(monkeypatch):
    policy,client=make(monkeypatch); n=prime(policy,client)
    before=deepcopy(policy.regions[2].vertices)
    client.measure_replies=[('direction',225.)]+[('no_signal',None)]*18
    monkeypatch.setattr(policy,'_prediction',lambda *args:score(True))
    one_macro(monkeypatch,policy,n)
    event=policy.cover_feedback_log[0]; route=policy.route_log[0]
    assert event['veto_selected'] and event['executed_cover']
    assert event['actual_vetoes_before']==0 and event['actual_vetoes_after']==1
    assert event['coverage_visited_after']==1 and policy.vetoed_channels=={1}
    assert event['end_actual_action_count']==n+19
    assert route['selected_kind']=='source' and route['selected_channel']==1
    assert route['execution_role']=='incumbent_not_executed'
    assert route['result']['order'][0][0]=='source'
    assert all(a['phase']=='coverage' and a['position']==[100.,100.]
               for a in policy.report.action_history[n:])
    assert not policy.cleared and policy.regions[2].vertices!=before
    assert policy.prediction_expanded==7
    assert policy.total_expansions==route['result']['expanded']


@pytest.mark.parametrize('accepted_before_stop',[0,1,5])
def test_incomplete_scan_never_pops_or_counts_real_veto(monkeypatch,accepted_before_stop):
    policy,client=make(monkeypatch); n=prime(policy,client)
    client.measure_replies=[('no_signal',None)]*19
    monkeypatch.setattr(policy,'_prediction',lambda *a:score(True))
    def stop():
        if len(policy.report.action_history)>=n+accepted_before_stop:
            raise _StopSearch('constructed_global_deadline')
    client.before_measure=stop
    with pytest.raises(_StopSearch,match='constructed_global_deadline'):
        policy._execute_plan()
    event=policy.cover_feedback_log[0]
    assert event['status']=='interrupted' and event['veto_selected']
    assert not event['executed_cover'] and event['actual_vetoes_after']==0
    assert policy.report.coverage_points_visited==0 and policy.vetoed_channels==set()
    assert event['end_actual_action_count']==n+accepted_before_stop


def test_rejected_real_wire_is_not_accepted_veto(monkeypatch):
    policy,client=make(monkeypatch); n=prime(policy,client)
    monkeypatch.setattr(policy,'_prediction',lambda *a:score(True))
    client.measure_replies=[('no_signal',None)]; client.reject_kind='measure'
    with pytest.raises(_StopSearch): policy._execute_plan()
    event=policy.cover_feedback_log[0]
    assert event['end_actual_action_count']==n and not event['executed_cover']
    assert policy.cover_feedback_vetoes==0


@pytest.mark.parametrize('attribute,value,reason',[
    ('vetoed_channels',{1},'channel_veto_limit'),('cover_feedback_vetoes',8,'session_veto_limit'),
    ('prediction_calls',32,'prediction_call_limit'),('prediction_expanded',70400,'prediction_expansion_limit')])
def test_caps_retain_original_without_running_model(monkeypatch,attribute,value,reason):
    policy,client=make(monkeypatch); n=prime(policy,client)
    setattr(policy,attribute,value)
    def forbidden(*args): raise AssertionError('Model called beyond cap')
    monkeypatch.setattr(policy,'_prediction',forbidden)
    one_macro(monkeypatch,policy,n)
    event=policy.cover_feedback_log[0]
    assert event['eligibility_reason']==reason and event['prediction'] is None
    assert client.calls[-1][0]=='clear' and policy.cleared=={1}


def test_normal_non_source_cover_route_never_predicts(monkeypatch):
    policy,client=make(monkeypatch); n=prime(policy,client)
    client.state.position=policy.points[0]
    client.measure_replies=[('no_signal',None)]*19
    monkeypatch.setattr(policy,'_prediction',lambda *a:pytest.fail('cover-first model call'))
    one_macro(monkeypatch,policy,n)
    event=policy.cover_feedback_log[0]
    assert event['eligibility_reason']=='not_source_then_next_cover'
    assert event['executed_cover'] and not event['veto_selected']
    assert event['actual_vetoes_after']==0


def test_original_early_service_priority_before_any_route(monkeypatch):
    policy,client=make(monkeypatch); prime(policy,client)
    monkeypatch.setattr(policy,'_early_candidate',lambda q:(0.,0.,2,25.))
    def stop(candidate): raise _ServiceSliceExpired('constructed_early_owner')
    monkeypatch.setattr(policy,'_early_service',stop)
    with pytest.raises(_ServiceSliceExpired):policy._execute_plan()
    assert not policy.route_log and not policy.cover_feedback_log and policy.prediction_calls==0


def test_no_ready_task_keeps_original_scan_branch(monkeypatch):
    policy,client=make(monkeypatch); prime(policy,client)
    policy.blocked.add(1)
    n=len(policy.report.action_history)
    client.measure_replies=[('no_signal',None)]*19
    monkeypatch.setattr(policy,'_prediction',lambda *a:pytest.fail('no ready task model'))
    one_macro(monkeypatch,policy,n)
    assert not policy.route_log and not policy.cover_feedback_log
    assert policy.report.coverage_points_visited==1 and not policy.blocked


def test_known_sixteen_keeps_original_discovery_stop_and_no_model(monkeypatch):
    policy,client=make(monkeypatch); prime(policy,client)
    policy.detected=set(range(1,17)); policy.cleared=set(range(2,17))
    monkeypatch.setattr(policy,'_prediction',lambda *a:pytest.fail('cap16 model'))
    with pytest.raises(_StopSearch,match='source_count_upper_bound_reached'):
        policy._execute_plan()
    assert policy.cleared==set(range(1,17))
    assert policy.discovery_stop_log[0]['omitted_cover_stations']==2
    assert policy.report.coverage_points_visited==0
    assert policy.cover_feedback_log[0]['eligibility_reason']=='not_source_then_next_cover'


def test_sixteen_actual_clears_stop_immediately(monkeypatch):
    policy,_=make(monkeypatch);policy.cleared=set(range(1,17))
    with pytest.raises(_StopSearch,match='source_count_upper_bound_reached'):policy._execute_plan()
    assert policy.report.completion_certified_under_model and not policy.cover_feedback_log


@pytest.mark.parametrize('kwargs',[{'problem':3},{'problem':True},{'config':'other'},
    {'max_expansions':True},{'max_actions':1},{'max_active_probes':31}])
def test_entry_rejects_invalid_before_client(kwargs):
    with pytest.raises(ValueError):run_q4_cover_feedback(None,**kwargs)


def test_original_budget_stays_separate_from_prediction(monkeypatch):
    policy,client=make(monkeypatch);n=prime(policy,client)
    policy.total_expansions=60000
    monkeypatch.setattr(policy,'_prediction',lambda *a:score(False,200))
    original=controller.solve_chain_route;budgets=[]
    def solve(*a,**kw):
        budgets.append(kw['max_expansions']);return original(*a,**kw)
    monkeypatch.setattr(controller,'solve_chain_route',solve)
    one_macro(monkeypatch,policy,n)
    assert budgets==[0] and policy.total_expansions==60000
    assert policy.prediction_expanded==200
