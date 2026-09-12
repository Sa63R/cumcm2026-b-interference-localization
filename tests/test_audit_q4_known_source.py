"""Independent matrix arithmetic and observation-prefix tamper tests."""
import copy
import itertools
import math

import pytest

from experiments import audit_q4_known_source as audit


def observed_record(actions=()):
    wire = [{'action': '/enter', 'response': {'accepted': True, 'max_virtual_duration_s': 360000.}}]
    history, current, tuned, elapsed = [], (0., 0.), 1, 0.
    for action, p, c, result, bearing in actions:
        elapsed += round(math.dist(current, p)/5.*1e6)/1e6
        if action == 'measure':
            elapsed += 5.+int(c != tuned); tuned = c
        else:
            elapsed += 3.+2.*(result == 'success')
        item = dict(action=action, position=list(p), channel=c, result=result,
                    phase='constructed_observation', virtual_time_s=elapsed)
        response = dict(accepted=True, virtual_time_s=elapsed,
                        **{('measure_result' if action == 'measure' else 'clear_result'): result})
        if bearing is not None:
            item['bearing_deg'] = response['svd_deg'] = bearing
        history.append(item)
        wire.append(dict(action='/'+action, position=list(p), channel=c, response=response))
        current = p
    return dict(summary={'action_history': history}, history=wire,
                spec={'entrypoint': audit.ENTRY, 'kwargs': {}}, row={'strategy': audit.LABEL})


def observed_prefix():
    return audit.Prefix(observed_record([
        ('measure', (-1000., 0.), 1, 'direction', 0.),
        ('measure', (0., -1000.), 1, 'direction', 90.),
        ('measure', (0., 0.), 2, 'near', None),
        ('clear', (0., 0.), 3, 'success', None)]))


def test_matrix_zero_ready_far_and_cleared_do_not_drop_unknown_channels():
    p = observed_prefix()
    assert not p.ready(4, 1) and p.ready(4, 2)
    others, background, matrix = audit.frozen_matrix(p, 4, [(0., 0.), (4000., 4000.)], [1, 2])
    assert 3 not in others and 1 not in others and 2 not in others
    assert len(others) == 17
    assert matrix == [[6., 0.], [0., 0.]]
    assert background == [102., 102.]
    assert matrix[0][0]+background[0] == 108.


def test_background_uses_same_frozen_known_rule_not_uniform_count():
    p = observed_prefix()
    others, background, matrix = audit.frozen_matrix(p, 4, [(0., 0.), (4000., 4000.)], [2])
    assert 1 in others and len(others) == 18
    assert background == [108., 102.]
    assert matrix == [[0.], [0.]]


def test_proxy_mask_identity_not_number_cleared_changes_scan_cost():
    covers, sources = [(0., 0.)], [(0., 0.), (0., 0.)]
    a_first = [('source', 0), ('cover', 0), ('source', 1)]
    b_first = [('source', 1), ('cover', 0), ('source', 0)]
    assert audit.matrix_route_cost(covers, sources, (0., 0.), a_first, [10.], [[6., 0.]]) == 10.
    assert audit.matrix_route_cost(covers, sources, (0., 0.), b_first, [10.], [[6., 0.]]) == 16.


def test_per_station_matrix_and_source_service_zero_are_applied_once():
    order = [('cover', 0), ('source', 0), ('cover', 1), ('source', 1)]
    assert audit.matrix_route_cost([(0., 0.)]*2, [(0., 0.)]*2, (0., 0.), order,
                                   [10., 20.], [[6., 0.], [0., 6.]]) == 42.


def test_deleted_chain_and_source_mst_bound_against_complete_small_enumeration():
    covers, sources, start = [(50., 40.), (100., 0.)], [(20., -40.), (60., 10.)], (0., 0.)
    bg, weights = [12., 6.], [[6., 0.], [0., 6.]]
    tasks = [('cover', 0), ('cover', 1), ('source', 0), ('source', 1)]
    costs = [audit.matrix_route_cost(covers, sources, start, order, bg, weights)
             for order in itertools.permutations(tasks) if order.index(('cover', 0)) < order.index(('cover', 1))]
    assert len(costs) == 12
    assert audit.root_relaxation(covers, sources, start, bg) <= min(costs)+1e-10
    assert audit.small_optimum(covers,sources,start,bg,weights) == pytest.approx(min(costs))


@pytest.mark.parametrize('order', [
    [('cover', 1), ('cover', 0), ('source', 0)],
    [('cover', 0), ('cover', 1), ('source', 1)],
    [('cover', 0), ('source', 0), ('source', 0)],
    [('cover', 0), ('source', 0)]])
def test_route_cannot_duplicate_omit_or_reorder_task_classes(order):
    with pytest.raises(ValueError):
        audit.matrix_route_cost([(0., 0.)]*2, [(0., 0.)], (0., 0.), order, [6., 6.], [[6.], [6.]])


@pytest.mark.parametrize('matrix', [[[float('nan')]], [[-6.]], [[]]])
def test_nonfinite_negative_and_wrong_dimension_matrix_rejected(matrix):
    with pytest.raises(ValueError):
        audit.matrix_route_cost([(0., 0.)], [(0., 0.)], (0., 0.), [('cover', 0), ('source', 0)], [6.], matrix)


def test_bearing_and_clear_evidence_must_match_actual_wire():
    record = observed_record([('measure', (-1000., 0.), 1, 'direction', 0.)])
    record['summary']['action_history'][0]['bearing_deg'] = 90.
    with pytest.raises(ValueError, match='bearing'):
        audit.Prefix(record)


def full_controller(monkeypatch, *, reject=False):
    """Real inherited calls with scripted radio; two-station control fixture.

    No source truths/scenes. This exercises prefix safety, not the generic
    full-disk coverage theorem (which remains mandatory for real records).
    """
    from tests.test_q4_known_source import make
    from strategies.search import _StopSearch
    policy,client=make(monkeypatch,points=((-1000.,0.),(1000.,0.)))
    policy.max_expansions=200; policy.actions=1
    wire=[dict(action='/enter',response=dict(accepted=True,max_virtual_duration_s=360000.))]
    count=0
    original_measure,original_clear=client.measure,client.clear
    def measure(p,c):
        nonlocal count
        if c==1:
            count+=1
            client.measure_replies=[('direction',0.) if count==1 else ('near',None)]
        else: client.measure_replies=[('no_signal',None)]
        if reject and count>1: client.reject_kind='measure'
        response=original_measure(p,c)
        wire.append(dict(action='/measure',position=[p.x,p.y],channel=c,response=copy.deepcopy(response)))
        return response
    def clear(p,c):
        response=original_clear(p,c)
        wire.append(dict(action='/clear',position=[p.x,p.y],channel=c,response=copy.deepcopy(response)))
        return response
    monkeypatch.setattr(client,'measure',measure); monkeypatch.setattr(client,'clear',clear)
    try: policy._execute_plan()
    except _StopSearch as e: policy.report.completion_reason=str(e)
    return dict(summary=policy.report.as_dict(),history=wire,row={'strategy':audit.LABEL},
        spec=dict(entrypoint=audit.ENTRY,kwargs=dict(config=audit.CONFIG,max_expansions=200)))


def test_real_complete_parent_prefix_and_broad_actions(monkeypatch):
    record=full_controller(monkeypatch)
    result=audit.audit_known_source_prefix(record)
    assert result['passed'] and result['broad_service_actions']==2
    assert result['known_source_macros']==1 and result['completed_scans']==2
    assert all(result[k]['passed'] for k in ('r12','r8','range','scheduling'))


def test_rejected_zero_action_macro_not_counted_and_not_fake_cover(monkeypatch):
    record=full_controller(monkeypatch,reject=True)
    result=audit.audit_known_source_prefix(record)
    assert result['passed'] and result['broad_service_actions']==0
    assert result['known_source_macros']==1 and result['completed_scans']==1


@pytest.mark.parametrize('field',['source_channels','source_positions','source_services_s','source_evidence',
    'background_channels','background_scan_s','source_scan_s','matrix_evidence','blocked_channels',
    'current_position','current_channel','max_expansions','total_expansions_before','total_expansions_after',
    'end_actual_action_count','selected_kind','route_cost','route_duplicate','lower_bound'])
def test_changed_frozen_plan_inputs_cost_budget_or_execution_refused(monkeypatch,field):
    record=full_controller(monkeypatch)
    e=record['summary']['strategy_parameters']['known_source_plan_log'][0]
    if field=='source_channels': e[field]=[2]
    elif field=='source_positions': e[field][0][0]+=1
    elif field=='source_services_s': e[field][0]=5.
    elif field=='source_evidence': e[field][0]['ready']=True
    elif field=='background_channels': e[field].pop()
    elif field in {'background_scan_s','source_scan_s'}:
        if field=='background_scan_s': e[field][0]-=6
        else: e[field][0][0]=0
    elif field=='matrix_evidence': e[field][0][0]['reason']='ready'
    elif field=='blocked_channels': e[field]=[1]
    elif field=='current_position': e[field][0]+=1
    elif field=='selected_kind': e[field]='cover'
    elif field=='route_cost': e['result']['cost_s']-=5
    elif field=='route_duplicate': e['result']['order']=[['cover',0],['cover',0]]
    elif field=='lower_bound': e['result']['lower_bound_s']=e['result']['cost_s']+10
    else: e[field]+=1
    with pytest.raises(ValueError): audit.audit_known_source_prefix(record)


@pytest.mark.parametrize('change',['missing','duplicate','wrong_ready','wrong_radius','no_covers','wrong_resolver',
    'wrong_start','include_scan','wrong_cost','false_clear','orphan_plan','missing_resolver','phase_fake'])
def test_real_service_identity_and_trigger_cannot_be_fabricated(monkeypatch,change):
    record=full_controller(monkeypatch)
    p=record['summary']['strategy_parameters']; e=p['known_source_service_log'][0]
    if change=='missing': p['known_source_service_log']=[]
    elif change=='duplicate': p['known_source_service_log'].append(copy.deepcopy(e))
    elif change=='wrong_ready': e['selected']['ready_before']=True
    elif change=='wrong_radius': e['selected']['radius_m']=39.
    elif change=='no_covers': e['remaining_covers_before']=[]
    elif change=='wrong_resolver': e['resolver_id']+=1
    elif change=='wrong_start': e['resolver_start_action_count']-=1
    elif change=='include_scan': e['service_end_action_count']+=1
    elif change=='wrong_cost': e['actual_cost_s']=0.
    elif change=='false_clear': e['cleared']=False
    elif change=='orphan_plan': p['known_source_plan_log'].append(copy.deepcopy(p['known_source_plan_log'][0]))
    elif change=='missing_resolver': p['joint_visibility_resolver_log']=[]
    else: record['summary']['action_history'][e['resolver_start_action_count']]['phase']='coverage'
    with pytest.raises(ValueError): audit.audit_known_source_prefix(record)


@pytest.mark.parametrize('child',['audit_joint_continuation_prefix','audit_clear_before_probe_prefix',
                                 'audit_range_prefix','audit_scheduling_prefix'])
def test_inherited_checker_false_never_silently_passes(monkeypatch,child):
    record=full_controller(monkeypatch)
    monkeypatch.setattr(audit,child,lambda record: {'passed':False})
    with pytest.raises(ValueError,match='Inherited audit'): audit.audit_known_source_prefix(record)


def test_source_contract_and_spec_bound_before_parent_normalization(monkeypatch,tmp_path):
    audit.verify_source_contract()
    with pytest.raises((ValueError,FileNotFoundError)): audit.verify_source_contract(tmp_path)
    record=full_controller(monkeypatch)
    record['spec']['kwargs']['config']='all_ready'
    with pytest.raises(ValueError,match='spec'): audit.audit_known_source_prefix(record)
