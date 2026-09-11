"""Constructed actual prefixes and source/geometry/adapter integrity tests."""
import copy

import pytest

from experiments import audit_q4_observation_cover as audit
from strategies.q4_observation_cover import Q4ObservationCover
from strategies.search import _StopSearch
from tests.test_q4_clear_before_probe import Replies
from tests.test_audit_q4_clear_before_probe import wrap


def example(config='ring_28',sixteen=False):
    client=Replies()
    n=28 if config=='ring_28' else 31
    client.measure_replies=([('near',None)]*16+[('no_signal',None)]*4 if sixteen
                            else [('no_signal',None)]*(20*n))
    if sixteen:client.clear_replies=['success']*16
    policy=Q4ObservationCover(client,20000,6,max_expansions=200,config=config)
    policy.actions=1
    reason=None
    try:policy._execute_plan()
    except _StopSearch as stopped:reason=str(stopped)
    record=wrap(policy,entered=True,reason=reason)
    record['spec']={'entrypoint':audit.ENTRY,'kwargs':{'config':config,'max_expansions':200}}
    record['row']['strategy']='compact_'+config
    return record


@pytest.mark.parametrize('config',['ring_28','ring_31'])
def test_full_real_unknown_scans_keep_new_points_and_original_safety(config):
    record=example(config);before=copy.deepcopy(record)
    result=audit.audit_observation_cover_prefix(record)
    assert result['passed'] and result['geometry']['stations']==(28 if config=='ring_28' else 31)
    assert result['geometry']['leaf_verification']['passed']
    assert result['r12']['passed'] and result['r8']['passed'] and result['scheduling']['passed']
    assert record==before and len(result['source_contract'])==51


@pytest.mark.parametrize('config',['ring_28','ring_31'])
def test_sixteen_actual_clears_omit_only_unvisited_new_stations(config):
    record=example(config,True)
    result=audit.audit_observation_cover_prefix(record)
    assert result['passed'] and result['scheduling']['discovery_stops']==1
    assert record['summary']['coverage_points_visited']==1
    assert sum(a['action']=='clear' and a['result']=='success' for a in record['summary']['action_history'])==16
    stop=record['summary']['strategy_parameters']['discovery_stop_log'][0]
    assert stop['omitted_cover_stations']==(27 if config=='ring_28' else 30)


@pytest.mark.parametrize('change',['coords','order','count22','origin','profile22','label',
    'config','spec','extra_kwargs','missing_kwargs','old_proof','leaf_hash','station_hash',
    'cert_radius','cert_arena','cert_margin','cert_depth','cert_passed','cert_verification',
    'cert_proposal','cert_stations','cert_route','negative_runtime','nan_runtime',
    'indices','inner_count','outer_radius','phase','length','formula','seconds','min_receivers',
    'angle_fraction','bearing_scope','route_scope','metadata_leaf','metadata_fields',
    'false_wire','false_time','actual_coverage_point'])
def test_tampered_new_geometry_or_physical_prefix_is_rejected(change):
    record=example();summary=record['summary'];p=summary['strategy_parameters']
    c=p['directional_cover_certificate'];m=p['observation_cover_route']
    if change=='coords':summary['coverage_points'][1][0]+=.001
    elif change=='order':summary['coverage_points'][1:3]=summary['coverage_points'][2:0:-1]
    elif change=='count22':summary['coverage_points_total']=22
    elif change=='origin':summary['coverage_points'][0]=[1.,0.]
    elif change=='profile22':p['q4_compact_profile']='compact_22'
    elif change=='label':record['row']['strategy']='baseline'
    elif change=='config':p['observation_cover_config']='ring_31'
    elif change=='spec':record['spec']['entrypoint']='unreviewed:run'
    elif change=='extra_kwargs':record['spec']['kwargs']['max_actions']=20001
    elif change=='missing_kwargs':del record['spec']['kwargs']['max_expansions']
    elif change=='old_proof':
        from planning.q4_directional_cover import certified_cover_points
        p['directional_cover_certificate']=certified_cover_points('compact_22')[1]
    elif change=='leaf_hash':c['full_leaf_certificate_sha256']='0'*64
    elif change=='station_hash':c['station_sha256']='0'*64
    elif change=='cert_radius':c['reception_radius']=1001.
    elif change=='cert_arena':c['arena_radius']=1799.
    elif change=='cert_margin':c['range_margin_m']=1e-8
    elif change=='cert_depth':c['max_depth']=24
    elif change=='cert_passed':c['passed']=False
    elif change=='cert_verification':c['independent_leaf_verification']['passed']=False
    elif change=='cert_proposal':c['proposal']['inner_radius_m']=970.
    elif change=='cert_stations':c['stations'][0][0]+=.001
    elif change=='cert_route':c['route_length_m']=17612.4194
    elif change=='negative_runtime':p['observation_cover_setup_runtime_s']=-1
    elif change=='nan_runtime':c['runtime_s']=float('nan')
    elif change=='indices':m['native_station_route_ids'][1]=0
    elif change=='inner_count':m['inner_count']=10
    elif change=='outer_radius':m['outer_radius_m']=1900.
    elif change=='phase':m['inner_phase_deg']=1.
    elif change=='length':m['route_length_m']-=1.
    elif change=='formula':m['route_closed_form_m']-=1.
    elif change=='seconds':m['pure_movement_s']*=5
    elif change=='min_receivers':m['boundary_radial_outward']['minimum_receiving_outer_stations']=3
    elif change=='angle_fraction':m['boundary_radial_outward']['angle_fraction_at_least_two']=.5
    elif change=='bearing_scope':m['boundary_scope']='All sources always have two receivers'
    elif change=='route_scope':m['route_scope']='Globally minimum task time'
    elif change=='metadata_leaf':m['full_leaf_certificate_sha256']='0'*64
    elif change=='metadata_fields':m['unreviewed_gain']=100
    elif change=='false_wire':record['history'][1]['response']['measure_result']='near'
    elif change=='false_time':summary['action_history'][21]['virtual_time_s']-=5
    elif change=='actual_coverage_point':summary['action_history'][20]['position'][0]+=.01
    with pytest.raises((ValueError,KeyError,AssertionError)):
        audit.audit_observation_cover_prefix(record)


def test_geometry_cache_cannot_be_corrupted_by_callers():
    first=audit.public_geometry('ring_28')
    first['points'].clear();first['certificate']['passed']=False
    first['metadata']['native_station_route_ids'].clear()
    again=audit.public_geometry('ring_28')
    assert len(again['points'])==28 and again['certificate']['passed']
    assert len(again['metadata']['native_station_route_ids'])==28


@pytest.mark.parametrize('config',[True,28,'compact_22','ring_25',None,[]])
def test_geometry_config_does_not_use_coercion(config):
    with pytest.raises(ValueError):audit.public_geometry(config)


def test_independent_verifier_false_cannot_become_passed(monkeypatch):
    audit._geometry_bytes.cache_clear()
    monkeypatch.setattr(audit,'verify_directional_cover_certificate',lambda *args:{'passed':False})
    with pytest.raises(ValueError,match='full-leaf verification'):
        audit.public_geometry('ring_28')
    audit._geometry_bytes.cache_clear()


def test_spec_adapter_keeps_true_new_geometry_and_actual_history(monkeypatch):
    record=example();before=copy.deepcopy(record)
    real=audit.audit_joint_continuation_prefix
    def inspect(view):
        assert view['summary'] is record['summary'] and view['history'] is record['history']
        assert view['summary']['coverage_points_total']==28
        assert view['summary']['strategy_parameters']['q4_compact_profile']=='observation_ring_28'
        assert view['spec']['entrypoint']=='strategies.q4_joint_continuation:run_q4_joint_continuation'
        return real(view)
    monkeypatch.setattr(audit,'audit_joint_continuation_prefix',inspect)
    assert audit.audit_observation_cover_prefix(record)['passed']
    assert record==before


@pytest.mark.parametrize('name',['audit_joint_continuation_prefix','audit_clear_before_probe_prefix',
                               'audit_range_prefix','audit_scheduling_prefix'])
def test_inherited_false_must_fail(monkeypatch,name):
    record=example()
    if name=='audit_range_prefix':record['summary']['strategy_parameters']['range_skipped_scans']=[{'mocked':'false audit'}]
    monkeypatch.setattr(audit,name,lambda record:{'passed':False})
    with pytest.raises(ValueError,match='Inherited audit failed'):
        audit.audit_observation_cover_prefix(record)


def test_generic_wrapper_receives_new_geometry_and_propagates_failure(monkeypatch):
    record=example('ring_31')
    def generic(view):
        assert view is record and view['summary']['coverage_points_total']==31
        assert view['row']['strategy']=='compact_ring_31'
        return {'passed':False,'errors':['constructed physical failure']}
    monkeypatch.setattr(audit,'audit_record',generic)
    with pytest.raises(ValueError,match='constructed physical failure'):
        audit.audit_full(record)


def test_changed_source_rejected_before_spec_adapter(tmp_path):
    name=next(iter(audit.SOURCE_CONTRACT));path=tmp_path/name;path.parent.mkdir(parents=True)
    path.write_bytes(b'not reviewed source')
    with pytest.raises(ValueError,match='source differs'):
        audit.verify_source_contract(tmp_path)


def test_prefix_never_reads_truth():
    record=example()
    class NoTruth(dict):
        def __getitem__(self,key):raise AssertionError('Truth access')
        def get(self,*args):raise AssertionError('Truth access')
    record['evaluation']=NoTruth()
    assert audit.audit_observation_cover_prefix(record)['passed']
