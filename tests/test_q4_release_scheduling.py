"""Constructed controller calls and forecasts, no scenario/network execution."""
import copy
from dataclasses import replace
import math

import pytest

from geometry import Circle
from localization import BearingObservation, CandidateRegion
from simulator_client.state import Position
from strategies.q4_known_source import Q4KnownSource
from strategies.q4_release_scheduling import Q4ReleaseScheduling, run_q4_release_scheduling
from strategies.search import _StopSearch
from tests.test_q4_known_source import ScriptedReplies, prime


def make(monkeypatch, *, points=(), active=6):
    monkeypatch.setattr('planning.q4_directional_cover.certified_cover_points', lambda profile:
        (tuple(Position(*p) for p in points), {'passed': True, 'test_only': True}))
    client = ScriptedReplies()
    return Q4ReleaseScheduling(client,20000,active,max_expansions=0), client


def set_points(policy, values):
    # Public constructed controller fixture, not a cover-certificate test.
    policy.points = tuple(Position.coerce(p) for p in values)
    policy.report.coverage_points = [[p.x,p.y] for p in policy.points]
    policy.report.coverage_points_total = len(policy.points)


def near(policy, client, channel, point):
    client.measure_replies = [('near', None)]
    policy._perform('measure',Position.coerce(point),channel,'constructed_observation')


def test_forecast_uses_copy_fixed_target_and_correct_completed_cover_index(monkeypatch):
    policy, client = make(monkeypatch); prime(policy, client)
    target = policy._target(1)
    region_before = copy.deepcopy(policy.regions[1].__dict__)
    history_before = copy.deepcopy(policy.report.action_history)
    observed_before = copy.deepcopy(policy.observed_positions)
    q = Position(target.x,100.)
    event = policy._release_forecast(1,[target,q,Position(1000,1000)],target)
    assert event['release_index'] == 2 and event['status'] == 'nominal_ready'
    assert [e['status'] for e in event['steps']] == ['skip_near_distance','predicted_ready']
    assert event['steps'][0]['bearing_deg'] is None
    assert event['steps'][1]['bearing_deg'] == 270. and event['steps'][1]['predicted_radius_m'] < 19.9
    assert event['steps'][1]['stopped'] and not event['steps'][0]['stopped']
    assert policy.regions[1].__dict__ == region_before
    assert policy.report.action_history == history_before and policy.observed_positions == observed_before
    assert not policy._ready(1) and not policy.cleared
    assert not policy.skipped_scans and not policy.range_skips


def test_nominal_target_does_not_follow_predicted_mec_drift(monkeypatch):
    policy, client = make(monkeypatch); prime(policy, client)
    target = policy._target(1)
    covers = [Position(0,1000), Position(target.x,100.)]
    event = policy._release_forecast(1,covers,target)
    assert len(event['steps']) == 2 and event['release_index'] == 2
    predicted = policy.regions[1].copy()
    for q, step in zip(covers,event['steps']):
        angle = round(math.degrees(math.atan2(target.y-q.y,target.x-q.x))%360,2)%360
        assert step['bearing_deg'] == angle
        predicted.observe(q,angle)
        assert step['predicted_radius_m'] == predicted.enclosing_disk().radius
    assert event['target'] == [target.x,target.y]


@pytest.mark.parametrize('distance,expected',[ (5.,'skip_near_distance'),(5.000001,None),
    (20.,None),(1500.,None),(1500.000001,'skip_beyond_reception_radius')])
def test_only_distance_five_to_1500_gets_nominal_direction(monkeypatch,distance,expected):
    policy, client = make(monkeypatch)
    region = CandidateRegion()
    region.vertices = ((-30.,-1.),(30.,-1.),(30.,1.),(-30.,1.))
    region.observations = [BearingObservation((-1000.,0.),0.)]
    region._circle = Circle((0.,0.), math.hypot(30.,1.))
    policy.regions[1]=region; policy.detected.add(1)
    event=policy._release_forecast(1,[Position(distance,0)],Position(0,0))
    step=event['steps'][0]
    if expected:
        assert step['status']==expected and step['bearing_deg'] is None
        assert event['status']=='cover_tail'
    else:
        assert step['bearing_deg']==180. and not step['status'].startswith('skip')
    assert event['release_index']==1  # Nonready never opens at k=0.


def test_actual_ready_and_no_remaining_covers_release_zero_without_predictions(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client)
    event=policy._release_forecast(1,[],policy._target(1))
    assert event['release_index']==0 and event['status']=='no_remaining_covers' and not event['steps']
    near(policy,client,2,(0,0))
    event=policy._release_forecast(2,[Position(100,100)],policy._target(2))
    assert event['release_index']==0 and event['status']=='actual_ready' and not event['steps']


@pytest.mark.parametrize('invalid',['empty','nonfinite','no_positive'])
def test_invalid_canonical_forecast_falls_back_to_full_cover_tail(monkeypatch,invalid):
    policy,client=make(monkeypatch);prime(policy,client)
    target=policy._target(1);region=policy.regions[1]
    if invalid=='empty':region.vertices=()
    elif invalid=='nonfinite':region.vertices=((math.nan,0.),(1.,0.),(0.,1.))
    else:region.observations=[]
    event=policy._release_forecast(1,[Position(0,100),Position(0,200)],target)
    assert event['release_index']==2 and event['status']=='fallback_invalid_canonical'
    assert not event['steps']


@pytest.mark.parametrize('invalid',['empty','exception'])
def test_failed_prediction_resets_entire_source_to_tail_without_live_mutation(monkeypatch,invalid):
    policy,client=make(monkeypatch);prime(policy,client);target=policy._target(1)
    before=copy.deepcopy(policy.regions[1].__dict__)
    def fake_observe(copy_region, point, bearing):
        assert copy_region is not policy.regions[1]
        if invalid=='exception':raise ValueError('constructed geometry failure')
        copy_region.vertices=();copy_region._circle=None;return copy_region
    monkeypatch.setattr(CandidateRegion,'observe',fake_observe)
    event=policy._release_forecast(1,[Position(target.x,100),Position(0,1000)],target)
    assert event['release_index']==2 and event['status'].startswith('fallback_')
    assert len(event['steps'])==1 and event['steps'][0]['stopped']
    assert policy.regions[1].__dict__==before


def test_optimistic_ready_prediction_cannot_start_real_nonready_service(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client);target=policy._target(1)
    set_points(policy,[Position(target.x,100.)])
    original=copy.deepcopy(policy.regions[1].vertices)
    client.measure_replies=[('no_signal',None)]*20+[('near',None)]
    client.clear_replies=['success']
    policy._execute_plan()
    first=policy.route_log[0]
    assert first['selected_kind']=='cover' and first['release_indices']==[1]
    assert first['release_forecasts'][0]['status']=='nominal_ready'
    assert first['source_evidence'][0]['vertices']==[list(v) for v in original]
    assert first['source_services_s']==[5.] and first['config']=='nominal_release'
    assert [a['phase'] for a in policy.report.action_history[1:21]]==['coverage']*20
    service=policy.known_source_service_log[0]
    assert not service['remaining_covers_before'] and service['resolver_start_action_count']==21
    assert service['selected']['ready_before'] is False and service['selected']['radius_m']>40
    assert service['cleared'] and policy.cleared=={1}
    assert policy.report.coverage_points_visited==1 and policy.report.coverage_complete
    assert policy.report.strategy_parameters['release_scheduling_plan_log'] is policy.route_log
    assert policy.report.strategy_parameters['known_source_plan_log'] is policy.route_log
    assert policy.report.strategy_parameters['known_source_proxy']['source_service_s']==5.


def test_actual_ready_first_macro_with_future_nonready_source_is_logged(monkeypatch):
    policy,client=make(monkeypatch);prime(policy,client);target=policy._target(1)
    set_points(policy,[Position(target.x,100.)])
    near(policy,client,2,(-1000,0))
    client.clear_replies=['success','success']
    # Ready2 omitted after true removal; cover must still really measure1 and all unknowns.
    client.measure_replies=[('no_signal',None)]*19+[('near',None)]
    policy._execute_plan()
    first=policy.route_log[0]
    assert first['source_channels']==[1,2] and first['release_indices']==[1,0]
    assert first['selected_kind']=='source' and first['selected_channel']==2
    assert [e['ready'] for e in first['source_evidence']]==[False,True]
    assert first['end_actual_action_count']-first['after_actual_action_count']==1
    assert first['source_positions']==[[target.x,target.y],[-1000.,0.]]
    assert policy.known_source_service_log[0]['selected']['ready_before'] is True


def test_ready_first_guard_rejects_malformed_solver_without_real_actions(monkeypatch):
    import strategies.q4_release_scheduling as module
    policy,client=make(monkeypatch);prime(policy,client);target=policy._target(1)
    set_points(policy,[Position(target.x,100.)])
    original_solver=module.solve_release_chain_route
    def corrupt(*args,**kwargs):
        result=original_solver(*args,**kwargs)
        return replace(result,order=(('source',0),('cover',0)))
    monkeypatch.setattr(module,'solve_release_chain_route',corrupt)
    calls=len(client.calls)
    with pytest.raises(RuntimeError,match='nonready first source'):policy._execute_plan()
    assert len(client.calls)==calls and not policy.joint_resolvers
    event=policy.route_log[0];service=policy.known_source_service_log[0]
    assert event['status']=='interrupted' and event['after_actual_action_count']==event['end_actual_action_count']
    assert service['service_end_action_count']==service['resolver_start_action_count']


@pytest.mark.parametrize('gate',['request','actions','virtual','real'])
def test_first_cover_budget_rejection_retains_obligation_and_no_service(monkeypatch,gate):
    policy,client=make(monkeypatch);prime(policy,client);target=policy._target(1)
    set_points(policy,[Position(target.x,100.)])
    if gate=='request':client.reject_kind='measure'
    elif gate=='actions':policy.max_actions=policy.actions+1
    elif gate=='virtual':client.state.max_virtual_duration_s=client.state.virtual_time_s+1
    else:client.remaining_real_time_s=2.
    with pytest.raises(_StopSearch):policy._execute_plan()
    event=policy.route_log[0]
    assert event['status']=='interrupted' and event['after_actual_action_count']==event['end_actual_action_count']
    assert policy.report.coverage_points_visited==0 and not policy.report.coverage_complete
    assert not policy.known_source_service_log and not policy.cleared


def test_existing_early_service_precedes_any_release_prediction(monkeypatch):
    policy,client=make(monkeypatch,points=((100.,0.),))
    for point,bearing in [((-1000,0),0.),((0,-1000),90.)]:
        client.measure_replies=[('direction',bearing)]
        policy._perform('measure',Position(*point),1,'constructed_observation')
    client.measure_replies=[('no_signal',None)]
    policy._perform('measure',Position(0,0),20,'constructed_observation')
    assert policy._early_candidate(policy.points[0]) is not None
    policy._execute_plan()
    assert len(policy.early_service_log)==1 and policy.cleared=={1}
    assert not policy.route_log and not policy.known_source_service_log
    assert policy.service_deadline is None and policy._joint_context is None


def test_actual_known16_cancels_covers_and_opens_even_nonready_sources(monkeypatch):
    policy,client=make(monkeypatch,points=((100.,0.),));prime(policy,client)
    for channel in range(2,17):near(policy,client,channel,(-1000,0))
    client.clear_replies=['success']*16;client.measure_replies=[('near',None)]
    with pytest.raises(_StopSearch,match='source_count_upper_bound'):policy._execute_plan()
    assert len(policy.cleared)==16 and policy.report.coverage_points_visited==0
    assert all(not e['remaining_covers'] and not any(e['release_indices']) for e in policy.route_log)
    assert any(f['status']=='no_remaining_covers' for e in policy.route_log for f in e['release_forecasts'])
    assert policy.report.completion_certified_under_model


def test_all_physical_safety_and_scanning_methods_are_inherited():
    for name in ('_perform','_scan','_resolve','_next_probe','_clear','_ready','_target',
                 '_early_candidate','_early_service','_check_budget','_insertion_budget','_scan_matrix','run'):
        assert getattr(Q4ReleaseScheduling,name) is getattr(Q4KnownSource,name)


@pytest.mark.parametrize('kwargs',[{'problem':3},{'problem':True},{'config':'all_known_matrix'},
    {'max_actions':True},{'max_active_probes':31},{'max_expansions':-1}])
def test_invalid_entrypoint_before_client(kwargs):
    with pytest.raises(ValueError):run_q4_release_scheduling(object(),**kwargs)
