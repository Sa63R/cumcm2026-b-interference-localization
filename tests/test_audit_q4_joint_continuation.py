"""Actual scripted prefixes and corrupt evidence; no generated benchmark scenes."""
import copy

import pytest

from experiments.audit_q4_joint_continuation import audit_joint_continuation_prefix
from experiments.audit_q4_clear_before_probe import audit_clear_before_probe_prefix
from tests.test_q4_joint_continuation import make
from tests.test_q4_joint_visibility_strategy import prime
from tests.test_audit_q4_clear_before_probe import wrap


def example(monkeypatch, misses=1):
    policy, client = make(monkeypatch)
    prime(policy, client, negative=False)
    client.clear_replies = ['no_target_in_range', 'success']
    client.measure_replies = [('no_signal', None)]*misses + [('near', None)]
    assert policy._resolve(1)
    record = wrap(policy)
    record['spec'] = {'entrypoint': 'strategies.q4_joint_continuation:run_q4_joint_continuation',
        'kwargs': {'config': 'after_active_miss_optical', 'max_actions': 20000,
                   'max_active_probes': 6, 'max_expansions': 0}}
    return record


def test_recursive_two_misses_reuse_only_previously_certified_outer(monkeypatch):
    record = example(monkeypatch, misses=2)
    result = audit_joint_continuation_prefix(record)
    assert result['continuation_calls'] == 2
    events = record['summary']['strategy_parameters']['joint_visibility_continuation_log']
    assert events[1]['input_source'] == 'auxiliary'
    assert events[1]['input_vertices'] == events[0]['output_aux_vertices']
    assert events[1]['basis']['previous_applied_event_id'] == 0
    assert audit_clear_before_probe_prefix(record)['passed']


def test_real_helper_new_aux_after_canonical_miss_keeps_actual_index(monkeypatch):
    record = example(monkeypatch)
    original = copy.deepcopy(record)
    result = audit_joint_continuation_prefix(record)
    assert result['passed'] and result['continuation_events'] == result['continuation_calls'] == 1
    assert result['continuation_applied'] == 1
    assert audit_clear_before_probe_prefix(record)['passed']
    params = record['summary']['strategy_parameters']
    assert params['joint_visibility_resolver_log'][0]['initial_aux_vertices'] is None
    assert params['joint_visibility_probe_log'][0]['index'] == 1
    assert record == original


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'future_prefix', 'wrong_trigger',
    'wrong_channel', 'wrong_source', 'fake_positive', 'fake_negative', 'canonical_change',
    'source_polygon', 'earlier_basis', 'bearing_basis', 'spent_budget', 'budget_limit',
    'skipped_helper', 'output', 'helper_input', 'helper_cell', 'fake_aux_index', 'contradiction'])
def test_corrupt_recursive_evidence_is_rejected(monkeypatch, change):
    record = example(monkeypatch)
    p = record['summary']['strategy_parameters']
    event = p['joint_visibility_continuation_log'][0]
    if change == 'missing': p['joint_visibility_continuation_log'].clear()
    elif change == 'duplicate': p['joint_visibility_continuation_log'].append(copy.deepcopy(event))
    elif change == 'future_prefix': event['after_actual_action_count'] += 1
    elif change == 'wrong_trigger': event['trigger_actual_action_index'] -= 1
    elif change == 'wrong_channel': event['channel'] = 2
    elif change == 'wrong_source': event['input_source'] = 'auxiliary'
    elif change == 'fake_positive': event['positive_positions'].append([1., 2.])
    elif change == 'fake_negative': event['negative_positions'].clear()
    elif change == 'canonical_change': event['canonical_vertices'][0][0] += .1
    elif change == 'source_polygon': event['input_vertices'][0][0] += .1
    elif change == 'earlier_basis': event['basis']['previous_applied_event_id'] = 0
    elif change == 'bearing_basis': event['basis']['bearing_update_prefixes'] = [1]
    elif change == 'spent_budget': event['budget']['session_calls_after'] += 1
    elif change == 'budget_limit': p['joint_visibility_continuation_limits']['per_session'] = 121
    elif change == 'skipped_helper': event['helper_called'] = False
    elif change == 'output': event['output_aux_vertices'][0][0] += .1
    elif change == 'helper_input': event['helper_evidence']['canonical_vertices'][0][0] += .1
    elif change == 'helper_cell': event['helper_evidence']['old_vertices_excluded'] = []
    elif change == 'fake_aux_index': p['joint_visibility_probe_log'][0]['index'] = 0
    elif change == 'contradiction': p['joint_visibility_model_contradictions'].append({'reason': 'constructed'})
    with pytest.raises((ValueError, AssertionError, KeyError)):
        audit_joint_continuation_prefix(record)
