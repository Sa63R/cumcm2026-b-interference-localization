"""Real scripted feedback and corrupt-record checks; no benchmark scenarios."""
import copy
import gzip
import json
import math

import pytest

from experiments.audit_q4_feedback_symmetry import (
    cycle_permutation, original_geometry, route_length, audit_feedback_symmetry_prefix, SOURCE_CONTRACT)
from strategies.q4_feedback_symmetry import Q4FeedbackSymmetry
from strategies.search import _StopSearch
from tests.test_q4_clear_before_probe import Replies
from tests.test_audit_q4_clear_before_probe import wrap


def test_independent_angular_cycles_are_fourteen_same_station_isometries():
    base, _ = original_geometry()
    all_orders = []
    for reflected in (False, True):
        for k in range(7):
            order = cycle_permutation(base, k, reflected)
            all_orders.append(tuple(order))
            mapped = [base[i] for i in order]
            assert order[0] == 0 and set(mapped) == set(base)
            assert abs(route_length(mapped)-route_length(base)) < 1e-8
            for i in range(22):
                for j in range(i):
                    assert abs(math.dist(mapped[i], mapped[j])-math.dist(base[i], base[j])) < 1e-8
    assert len(set(all_orders)) == 14
    assert all_orders[0] == tuple(range(22))


def scripted(config='bearing_mean', directions=((1,240.),(2,250.)), nears=(), limit=40, entered=False):
    client = Replies()
    responses = dict(directions)
    client.measure_replies = [('near',None) if c in nears else ('direction',responses[c])
                              if c in responses else ('no_signal',None) for c in range(1,21)]
    client.measure_replies += [('no_signal',None)]*500
    client.clear_replies = ['success']*20
    policy = Q4FeedbackSymmetry(client,20000,6,max_expansions=0,config=config)
    policy.actions += int(entered)
    def stop():
        if len(policy.report.action_history) >= limit:
            raise _StopSearch('real_deadline')
    client.before_measure = stop
    with pytest.raises(_StopSearch) as raised:
        policy._execute_plan()
    record = wrap(policy,entered=entered,reason=raised.value.reason)
    record['spec'] = dict(entrypoint='strategies.q4_feedback_symmetry:run_q4_feedback_symmetry',
                          kwargs=dict(config=config,max_actions=20000,max_active_probes=6,max_expansions=0))
    return policy,record


@pytest.mark.parametrize('config',['bearing_mean','early_centers'])
@pytest.mark.parametrize('entered',[False,True])
def test_actual_first_scan_and_subsequent_locked_station(config,entered):
    policy, record = scripted(config,entered=entered)
    original = copy.deepcopy(record)
    result = audit_feedback_symmetry_prefix(record)
    assert result['passed'] and result['initial_measurements'] == 20 and result['events'] == 1
    assert result['changed_routes'] == 1
    assert result['inherited_r8']['passed'] and result['inherited_r12']['passed'] and result['inherited_scheduling']['passed']
    assert policy.report.coverage_points_visited == 2 and len(policy.points) == 22
    assert record == original


@pytest.mark.parametrize('config,directions,status',[
    ('bearing_mean',(),'insufficient_directions'),
    ('bearing_mean',((1,120.),),'insufficient_directions'),
    ('bearing_mean',((1,0.),(2,180.)),'low_concentration'),
    ('bearing_mean',((1,204.),(2,206.)),'identity_best'),
    ('early_centers',(),'no_positive_regions'),
    ('early_centers',((1,205.),),'identity_best'),
])
def test_gate_and_identity_tie_rebuilt_from_real_feedback(config,directions,status):
    _, record = scripted(config,directions)
    result = audit_feedback_symmetry_prefix(record)
    assert result['status'] == status and result['changed_routes'] == 0


def test_circular_mean_across_zero_uses_actual_original_first_station():
    _, record = scripted(directions=((1,359.),(2,1.)))
    result = audit_feedback_symmetry_prefix(record)
    event = record['summary']['strategy_parameters']['feedback_symmetry_log'][0]
    assert result['changed_routes'] == 1
    assert abs(math.remainder(event['mean_bearing_deg'],360.)) < 1e-10


def test_sixteen_actual_near_channels_keep_identity_and_clear_without_outer_scan():
    policy, record = scripted(directions=(),nears=tuple(range(1,17)),limit=100)
    result = audit_feedback_symmetry_prefix(record)
    assert result['status'] == 'source_count_cap' and result['changed_routes'] == 0
    assert len(policy.cleared) == 16 and policy.report.coverage_points_visited == 1
    assert all(a['phase'] != 'coverage' for a in record['summary']['action_history'][20:])


@pytest.mark.parametrize('limit',[0,1,19])
def test_incomplete_origin_scan_has_no_decision_or_credit(limit):
    _, record = scripted(limit=limit)
    result = audit_feedback_symmetry_prefix(record)
    assert result['events'] == 0 and result['status'] == 'origin_interrupted'
    assert result['initial_measurements'] == limit


@pytest.mark.parametrize('change',['missing','duplicate','start','stop','visited','known','bearing','centers',
    'rho','mean','gate','candidate_missing','permutation','reflected','rotation','score','units','selected',
    'selected_order','old_points','new_points','report_points','total','certificate','budget','fee',
    'wire','unpaid','wrong_chain','repeated_origin','limits','spec','source_contract'])
def test_tampered_feedback_score_identity_or_execution_rejected(monkeypatch,change):
    _, record = scripted()
    summary = record['summary']; params = summary['strategy_parameters']; event = params['feedback_symmetry_log'][0]
    if change == 'missing': params['feedback_symmetry_log'].clear()
    elif change == 'duplicate': params['feedback_symmetry_log'].append(copy.deepcopy(event))
    elif change == 'start': event['origin_scan_start'] = 1
    elif change == 'stop': event['end_actual_action_count'] = 21
    elif change == 'visited': event['coverage_points_visited'] = 0
    elif change == 'known': event['known_channels'].append(3)
    elif change == 'bearing': event['bearings'][0]['bearing_deg'] += 1.
    elif change == 'centers': event['positive_centers'][0]['center'][0] += .1
    elif change == 'rho': event['concentration'] -= .1
    elif change == 'mean': event['mean_bearing_deg'] += 1.
    elif change == 'gate': event['status'] = 'source_count_cap'
    elif change == 'candidate_missing': event['candidates'].pop()
    elif change == 'permutation': event['candidates'][1]['permutation'][1:3] = reversed(event['candidates'][1]['permutation'][1:3])
    elif change == 'reflected': event['candidates'][0]['reflected'] = True
    elif change == 'rotation': event['candidates'][1]['rotation_index'] = 2
    elif change == 'score': event['candidates'][0]['score'] += .1
    elif change == 'units': event['score_units'] = 'meters'
    elif change == 'selected': event['selected_id'] = 0
    elif change == 'selected_order': event['selected_permutation'] = list(range(22))
    elif change == 'old_points': event['original_full_points'][1][0] += .1
    elif change == 'new_points': event['selected_full_points'][1][0] += .1
    elif change == 'report_points': summary['coverage_points'][1][0] += .1
    elif change == 'total': summary['coverage_points_total'] = 21
    elif change == 'certificate': params['directional_cover_certificate']['station_sha256'] = '0'*64
    elif change == 'budget': event['budget']['policy_actions_after_scan'] -= 1
    elif change == 'fee': event['budget']['virtual_after_scan'] -= 1
    elif change == 'wire': record['history'][0]['response']['svd_deg'] += 1.
    elif change == 'unpaid': summary['action_history'][0]['phase'] = 'constructed_observation'
    elif change == 'wrong_chain':
        p = summary['coverage_points'][2]
        for a,w in zip(summary['action_history'][20:],record['history'][20:]):
            a['position'] = list(p); w['position'] = list(p)
    elif change == 'repeated_origin': summary['action_history'][20]['position'] = [0.,0.]
    elif change == 'limits': params['feedback_symmetry_limits']['early_station_count'] = 4
    elif change == 'spec': record['spec']['kwargs']['config'] = 'early_centers'
    else: monkeypatch.setitem(SOURCE_CONTRACT,'src/strategies/q4_feedback_symmetry.py','0'*64)
    with pytest.raises((ValueError,AssertionError,KeyError)):
        audit_feedback_symmetry_prefix(record)


def test_batch_identity_and_no_overwrite(tmp_path):
    from experiments.audit_q4_feedback_symmetry_batch import audit
    _, record = scripted(); label = 'compact_feedback_mean'
    record['row'].update(strategy=label,seed=9)
    manifest = dict(seeds=[9],specs={label:record['spec']},source_sha256=SOURCE_CONTRACT)
    for name,value in [('manifest.json',manifest),('freeze.json',{}),('independent_audit.json',dict(all_passed=True))]:
        (tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'records').mkdir()
    (tmp_path/'records'/f'{label}-9.json.gz').write_bytes(gzip.compress(json.dumps(record).encode()))
    assert audit(tmp_path,label) == 0
    result = json.loads((tmp_path/'feedback_symmetry_mean_audit.json').read_bytes())
    assert result['records'] == result['passed_records'] == 1 and result['all_passed']
    with pytest.raises(ValueError,match='Preserve'): audit(tmp_path,label)


@pytest.mark.parametrize('change',['duplicate_seeds','no_seeds','spec','source','row','missing_record'])
def test_batch_refuses_missing_duplicate_or_unbound_input(tmp_path,change):
    from experiments.audit_q4_feedback_symmetry_batch import audit
    _, record = scripted(); label = 'compact_feedback_mean'
    record['row'].update(strategy=label,seed=9)
    manifest = dict(seeds=[9],specs={label:copy.deepcopy(record['spec'])},source_sha256=dict(SOURCE_CONTRACT))
    if change == 'duplicate_seeds': manifest['seeds'] = [9,9]
    elif change == 'no_seeds': manifest['seeds'] = []
    elif change == 'spec': manifest['specs'][label]['kwargs']['config'] = 'early_centers'
    elif change == 'source': manifest['source_sha256'].clear()
    elif change == 'row': record['row']['seed'] = 10
    for name,value in [('manifest.json',manifest),('freeze.json',{}),('independent_audit.json',dict(all_passed=True))]:
        (tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'records').mkdir()
    if change != 'missing_record':
        (tmp_path/'records'/f'{label}-9.json.gz').write_bytes(gzip.compress(json.dumps(record).encode()))
    with pytest.raises((ValueError,FileNotFoundError)): audit(tmp_path,label)
    assert not (tmp_path/'feedback_symmetry_mean_audit.json').exists()
