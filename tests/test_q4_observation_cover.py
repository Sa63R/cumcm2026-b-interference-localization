"""Constructed feedback tests of constructor-only R12 inheritance."""
import copy

import pytest

from planning.q4_directional_cover import certified_cover_points
from strategies.q4_joint_continuation import Q4JointContinuation
from strategies.q4_observation_cover import Q4ObservationCover,run_q4_observation_cover
from strategies.search import _StopSearch
from tests.test_q4_clear_before_probe import Replies,prime


def make(config='ring_28'):
    client=Replies()
    return Q4ObservationCover(client,20000,6,max_expansions=200,config=config),client


@pytest.mark.parametrize('config,count',[('ring_28',28),('ring_31',31)])
def test_only_points_and_genuine_certificate_change(config,count):
    policy,client=make(config)
    assert not client.calls
    assert len(policy.points)==policy.report.coverage_points_total==count
    assert policy.report.coverage_points==[[p.x,p.y] for p in policy.points]
    p=policy.report.strategy_parameters
    assert p['observation_cover_config']==config
    assert p['q4_compact_profile']=='observation_'+config
    assert p['directional_cover_certificate']['station_count']==count
    assert p['directional_cover_certificate']['passed'] is True
    assert p['directional_cover_certificate']['station_sha256']!=certified_cover_points('compact_22')[1]['station_sha256']
    assert p['compact_schedule']=='joint'
    assert p['joint_visibility_continuation_config']=='after_active_miss_optical'
    assert policy.max_expansions==200 and policy.max_active_probes==6
    for name in ('run','_perform','_scan','_execute_plan','_resolve','_next_probe','_clear','_check_budget','_early_candidate','_early_service'):
        assert getattr(Q4ObservationCover,name) is getattr(Q4JointContinuation,name)


@pytest.mark.parametrize('config,count',[('ring_28',28),('ring_31',31)])
def test_unknown_scanning_consumes_exact_new_fixed_route(config,count):
    policy,client=make(config)
    client.measure_replies=[('no_signal',None)]*(count*20)
    points=policy.points
    policy._execute_plan()
    assert len(client.calls)==count*20 and policy.report.coverage_points_visited==count
    assert policy.report.coverage_complete and not policy.cleared and not policy.detected
    assert [client.calls[i*20][1] for i in range(count)]==list(points)
    assert all(call[0]=='measure' for call in client.calls)


@pytest.mark.parametrize('config',['ring_28','ring_31'])
def test_original_r8_true_success_and_finally_unchanged(config):
    policy,client=make(config)
    center=prime(policy,client)
    vertices=copy.deepcopy(policy.regions[1].vertices)
    policy.pending_pair=('test',)
    assert policy._resolve(1)
    assert client.calls[-1]==('clear',center,1)
    assert policy.cleared=={1} and policy.regions[1].vertices==vertices
    assert policy.pending_pair is None and policy._probe_resolving_channel is None
    assert policy.clear_before_probe_log[-1]['status']=='cleared'


def test_actual_sixteen_clear_still_ends_without_scanning_extra_stations():
    policy,client=make()
    policy.cleared.update(range(1,17))
    with pytest.raises(_StopSearch,match='source_count_upper_bound_reached'):
        policy._execute_plan()
    assert not client.calls and policy.report.completion_certified_under_model


@pytest.mark.parametrize('kwargs',[{'problem':3},{'problem':True},{'config':'ring_25'},{'config':[]},
    {'max_actions':1},{'max_actions':True},{'max_active_probes':31},{'max_expansions':10001}])
def test_invalid_inputs_do_not_use_client(kwargs):
    with pytest.raises(ValueError):run_q4_observation_cover(object(),**kwargs)


def test_public_entry_dispatches_fixed_config(monkeypatch):
    monkeypatch.setattr(Q4ObservationCover,'run',lambda self:self.report.strategy_parameters['observation_cover_config'])
    assert run_q4_observation_cover(Replies(),config='ring_31')=='ring_31'
