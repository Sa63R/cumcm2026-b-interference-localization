"""Constructed prefix and tamper tests, never performance scenarios."""
import copy
from types import SimpleNamespace

import pytest

from experiments import audit_q4_transit_budget as audit


def fake_prefix(position=(0., 0.), tuned=1, now=0.):
    return SimpleNamespace(h=[None], before=[((0., 0.), 1, 0.), (position, tuned, now)],
                           enter_count=1, virtual_limit=360000., max_actions=20000)


def test_shared_long_move_counts_only_increment_above_direct_edge():
    p = fake_prefix()
    x = audit.budget_numbers(p, 0, 0, (1000., 0.), 'measure', (500., 0.), 2)
    assert x['baseline_us'] == 200_000_000
    assert x['action_us'] == 106_000_000
    assert x['return_first_us'] == 106_000_000
    assert x['incremental_us'] == 12_000_000
    assert x['global_projected_actions'] == 23
    assert x['global_projected_us'] == 326_000_000


def test_r8_atomic_reserve_includes_failed_clear_original_measure_and_return():
    p = fake_prefix()
    x = audit.budget_numbers(p, 0, 0, (1000., 0.), 'measure', (500., 0.), 2, atomic=True)
    assert x['action_us'] == 109_000_000 and x['action_count'] == 2
    assert x['incremental_us'] == 15_000_000
    assert x['global_projected_actions'] == 24
    with pytest.raises(ValueError, match='original measure'):
        audit.budget_numbers(p, 0, 0, (1000., 0.), 'clear', (500., 0.), 2, atomic=True)


def test_spent_not_reset_and_clear_does_not_charge_switch():
    p = fake_prefix(position=(500., 0.), tuned=7, now=151.)
    x = audit.budget_numbers(p, 0, 1, (1000., 0.), 'clear', (500., 0.), 2)
    assert x['spent_us'] == 151_000_000
    assert x['action_us'] == 5_000_000
    assert x['incremental_us'] == 62_000_000


def test_microsecond_bounds_are_outward_not_nearest_rounding():
    a, b = (0., 0.), (0.0000021, 0.)
    assert audit.movement_lower(a, b) == 0
    assert audit.movement_upper(a, b) == 1
    with pytest.raises(ValueError):
        audit.micros(float('nan'))


def live_service(monkeypatch, mode='success'):
    """Replay actual scripted calls; only the cover geometry is a tiny fixture.

    This tests wire/gate/resolver boundaries, not a full-scene cover claim.
    """
    from tests.test_q4_transit_budget import make, prime, DESTINATION
    policy, client = make(monkeypatch)
    wire = [dict(action='/enter', response={'accepted': True, 'max_virtual_duration_s': 360000.})]
    for name in ('measure', 'clear'):
        original = getattr(client, name)
        def logged(p, c, action=name, original=original):
            response = original(p, c)
            wire.append(dict(action='/'+action, position=[p.x, p.y], channel=c, response=copy.deepcopy(response)))
            return response
        monkeypatch.setattr(client, name, logged)
    policy.actions = 1
    prime(policy, client)
    if mode == 'failed_near':
        client.clear_replies = ['no_target_in_range', 'success']
        client.measure_replies = [('near', None)]
    elif mode == 'atomic_stop':
        policy.max_actions = policy.actions+22
    policy._scan(DESTINATION)
    record = dict(summary=policy.report.as_dict(), history=wire,
        row={'strategy': audit.LABEL}, spec={'entrypoint': audit.ENTRY,
            'kwargs': {'config': audit.CONFIG, 'max_actions': policy.max_actions, 'max_expansions': 200}})
    return record


@pytest.mark.parametrize('mode,count', [('success', 1), ('failed_near', 3), ('atomic_stop', 0)])
def test_actual_scripted_service_gate_replay(monkeypatch, mode, count):
    record = live_service(monkeypatch, mode)
    p = audit.Prefix(record)
    event = record['summary']['strategy_parameters']['transit_service_log'][0]
    resolver = record['summary']['strategy_parameters']['joint_visibility_resolver_log'][0]
    stops = set()
    assert audit.audit_gates(event, p, resolver, stops) == count
    assert bool(stops) == (mode == 'atomic_stop')
    assert len(stops) <= 1


@pytest.mark.parametrize('field', ['spent_us', 'movement_upper_us', 'action_fee_upper_us',
    'action_upper_us', 'return_upper_us', 'baseline_floor_us', 'incremental_upper_us',
    'predicted_actions', 'policy_action_count', 'full_scan_reserve_actions',
    'exit_reserve_actions', 'return_first_measure_upper_us', 'virtual_limit_us'])
def test_tampered_integer_budget_refused(monkeypatch, field):
    record = live_service(monkeypatch)
    event = record['summary']['strategy_parameters']['transit_service_log'][0]
    event['gates'][0][field] += 1
    with pytest.raises(ValueError):
        audit.audit_gates(event, audit.Prefix(record), record['summary']['strategy_parameters']['joint_visibility_resolver_log'][0], set())


@pytest.mark.parametrize('change', ['missing_atomic', 'missing_action', 'duplicate_action',
    'atomic_after_action', 'false_executed', 'wrong_point', 'wrong_channel', 'future_prefix'])
def test_tampered_causal_gate_binding_refused(monkeypatch, change):
    record = live_service(monkeypatch)
    event = record['summary']['strategy_parameters']['transit_service_log'][0]
    gates = event['gates']
    if change == 'missing_atomic':
        del gates[0]; gates[0]['id'] = 0
    elif change == 'missing_action':
        del gates[1]
    elif change == 'duplicate_action':
        gates.append(copy.deepcopy(gates[-1])); gates[-1]['id'] = 2
    elif change == 'atomic_after_action':
        gates.reverse()
        for i,g in enumerate(gates): g['id'] = i
    elif change == 'false_executed':
        gates[0]['executed_action_count'] = 0
    elif change == 'wrong_point':
        gates[0]['point'][0] += 1
    elif change == 'wrong_channel':
        gates[0]['channel'] = 2
    else:
        gates[0]['after_actual_action_count'] += 1
    with pytest.raises(ValueError):
        audit.audit_gates(event, audit.Prefix(record), record['summary']['strategy_parameters']['joint_visibility_resolver_log'][0], set())


@pytest.mark.parametrize('change', ['admitted', 'reason', 'id', 'exception', 'status'])
def test_zero_action_atomic_stop_needs_its_own_true_rejection(monkeypatch, change):
    record = live_service(monkeypatch, 'atomic_stop')
    event = record['summary']['strategy_parameters']['transit_service_log'][0]
    resolver = record['summary']['strategy_parameters']['joint_visibility_resolver_log'][0]
    if change == 'admitted': event['gates'][0]['admitted'] = True
    elif change == 'reason': event['stopped_reasons'] = ['incremental_budget']
    elif change == 'id': event['stopped_gate_id'] = 1
    elif change == 'exception': resolver['interruption_type'] = '_StopSearch'
    else: event['service_status'] = 'cleared'
    with pytest.raises(ValueError):
        audit.audit_gates(event, audit.Prefix(record), resolver, set())


def test_actual_wire_bearing_tamper_cannot_prime_a_fake_region(monkeypatch):
    record = live_service(monkeypatch)
    record['summary']['action_history'][0]['bearing_deg'] += 20
    with pytest.raises(ValueError, match='bearing'):
        audit.Prefix(record)


def test_candidate_geometry_uses_real_prefix_without_claimed_center(monkeypatch):
    record = live_service(monkeypatch)
    p = audit.Prefix(record)
    result = audit.candidates(p, 2, (0., 1000.), set(), set(), set())
    assert len(result) == 1 and result[0]['channel'] == 1
    assert result[0]['original_first_cost_s'] > 60
    assert audit.candidates(p, 2, (0., 1000.), {1}, set(), set()) == []
    assert audit.candidates(p, 2, (0., 1000.), set(), {1}, set()) == []
    assert audit.candidates(p, 2, (0., 1000.), set(), set(), {1}) == []
    assert audit.candidates(p, 2, p.before[2][0], set(), set(), set()) == []


@pytest.mark.parametrize('child', ['r12', 'r8', 'range', 'scheduling'])
def test_wrapper_rejects_any_explicit_false_child(monkeypatch, child):
    """Wrapper integrity only; physical/geometry replay is not mocked in QA."""
    record = live_service(monkeypatch)
    monkeypatch.setattr(audit, 'verify_source_contract', lambda: None)
    monkeypatch.setattr(audit, 'replay_macros', lambda r: (None, set(), {}))
    def replay(record):
        return {'passed': False}
    if child == 'r12':
        monkeypatch.setattr(audit, 'audit_joint_continuation_prefix', replay)
    else:
        def okay(record): return {'passed': True}
        monkeypatch.setattr(audit, 'audit_joint_continuation_prefix', okay)
    for name, kind in [('audit_clear_before_probe_prefix', 'r8'), ('audit_range_prefix', 'range'),
                       ('audit_scheduling_prefix', 'scheduling')]:
        monkeypatch.setattr(audit, name, lambda r, k=kind: {'passed': k != child})
    with pytest.raises(ValueError, match='Inherited'):
        audit.audit_transit_budget_prefix(record)


def test_private_stopped_extension_does_not_change_modules_or_actual_history(monkeypatch):
    record = live_service(monkeypatch, 'atomic_stop')
    original = copy.deepcopy(record)
    event = record['summary']['strategy_parameters']['transit_service_log'][0]
    gate = event['gates'][0]
    key = (gate['after_actual_action_count'], 1, 'measure', audit.point(gate['point']))
    monkeypatch.setattr(audit, 'verify_source_contract', lambda: None)
    monkeypatch.setattr(audit, 'replay_macros', lambda r: (None, {key}, {}))
    def inherited(r):
        # `stopped` is deliberately absent from this test module's globals.
        h, before = audit.wire_prefix(r)
        e = r['summary']['strategy_parameters']['transit_service_log'][0]
        g = e['gates'][0]
        return {'passed': stopped(r['summary'], h, before, g['after_actual_action_count'], 1, 'measure', g['point'])}
    assert 'stopped' not in inherited.__globals__
    monkeypatch.setattr(audit, 'audit_joint_continuation_prefix', inherited)
    for name in ('audit_clear_before_probe_prefix', 'audit_range_prefix', 'audit_scheduling_prefix'):
        monkeypatch.setattr(audit, name, lambda r: {'passed': True})
    assert audit.audit_transit_budget_prefix(record)['passed']
    assert 'stopped' not in inherited.__globals__
    assert record == original


def test_source_contract_rejects_changed_reviewed_bytes(tmp_path, monkeypatch):
    p = tmp_path/'src.py'; p.write_bytes(b'original')
    import hashlib
    expected = hashlib.sha256(b'original').hexdigest()
    monkeypatch.setattr(audit, 'SOURCE_CONTRACT', {})
    monkeypatch.setattr(audit, 'NEW_SOURCE_CONTRACT', {'src.py': expected})
    audit.verify_source_contract(tmp_path)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        audit.verify_source_contract(tmp_path)


@pytest.fixture(scope='module')
def opened_record():
    """One already-authorized old QA archive; no policy rerun or truth use."""
    import gzip
    import json
    p = audit.ROOT/'research/q4_transit_budget/old-smoke/compact_transit_budget-621003.json.gz'
    if not p.exists():
        pytest.skip('Optional archived old-development QA not present')
    raw = json.loads(gzip.decompress(p.read_bytes()))
    return {k: raw[k] for k in ('summary', 'history', 'spec', 'row')}


def test_complete_old_actual_macro_replay_without_truth(opened_record):
    record = copy.deepcopy(opened_record)
    result = audit.audit_transit_budget_prefix(record)
    assert result['passed'] and result['transit_services'] == result['transit_service_actions'] == 1
    assert all(result[k]['passed'] for k in ('r12', 'r8', 'range', 'scheduling'))
    assert record == opened_record


@pytest.mark.parametrize('change', ['missing_skip_event', 'missing_service_event', 'extra_service_event',
    'known_future', 'wrong_destination', 'wrong_origin', 'candidate_radius', 'candidate_projection',
    'candidate_approach', 'fake_old_early', 'wrong_resolver', 'scan_delayed', 'skip_first_scan',
    'premature_visited', 'first_measure_false', 'extra_bound_false', 'source_selected_in_chain',
    'cover_reordered'])
def test_complete_macro_tampering_refused(opened_record, change):
    r = copy.deepcopy(opened_record)
    params = r['summary']['strategy_parameters']
    events = params['transit_service_log']
    e = next(x for x in events if x['selected'])
    if change == 'missing_skip_event': del events[0]
    elif change == 'missing_service_event': events.remove(e)
    elif change == 'extra_service_event': events.insert(e['id'], copy.deepcopy(e))
    elif change == 'known_future': e['known_channels'].append(19)
    elif change == 'wrong_destination': e['destination'][0] += 1
    elif change == 'wrong_origin': e['origin'][0] += 1
    elif change.startswith('candidate_'):
        field = {'candidate_radius':'radius_m', 'candidate_projection':'projection_fraction',
                 'candidate_approach':'approach_m'}[change]
        e['candidates'][0][field] += 1
        e['selected'][field] += 1
    elif change == 'fake_old_early': e['early_attempted_before'] = [e['selected']['channel']]
    elif change == 'wrong_resolver': e['resolver_id'] += 1
    elif change == 'scan_delayed': e['scan_start_action_count'] += 1
    elif change == 'skip_first_scan':
        first = e['scan_start_action_count']
        r['summary']['action_history'][first]['phase'] = 'active_localization'
    elif change == 'premature_visited': e['coverage_visited_before'] += 1
    elif change == 'first_measure_false': e['first_coverage_action_index'] += 1
    elif change == 'extra_bound_false': e['actual_incremental_through_first_measure_us'] += 1
    elif change == 'source_selected_in_chain':
        first_chain = params['chain_route_log'][0]
        first_chain['selected_kind'] = 'cover' if first_chain['selected_kind'] == 'source' else 'source'
    else:
        r['summary']['coverage_points'][2:4] = r['summary']['coverage_points'][2:4][::-1]
    with pytest.raises((ValueError, AssertionError, KeyError)):
        audit.replay_macros(r)


def test_no_aux_rejected_gate_must_name_actual_canonical_next_probe(monkeypatch):
    r = live_service(monkeypatch, 'atomic_stop')
    prefix = audit.Prefix(r)
    resolver = r['summary']['strategy_parameters']['joint_visibility_resolver_log'][0]
    action, point = audit.canonical_next_action(prefix, resolver)
    gate = r['summary']['strategy_parameters']['transit_service_log'][0]['gates'][0]
    assert action == 'measure' and point == tuple(gate['point'])
    false_key = (resolver['end_actual_action_count'], 1, 'measure', (10000., 10000.))
    monkeypatch.setattr(audit, 'replay_macros', lambda r: (prefix, {false_key}, {}))
    # Real R12 checker sees no auxiliary branch here. The new canonical guard
    # must still reject a fake far-away action that would trivially exceed 60s.
    with pytest.raises(ValueError, match='canonical resolver next action'):
        audit.audit_transit_budget_prefix(r)


def test_canonical_next_ready_near_and_optical_tail(monkeypatch):
    r = live_service(monkeypatch, 'failed_near')
    prefix = audit.Prefix(r)
    resolver = copy.deepcopy(r['summary']['strategy_parameters']['joint_visibility_resolver_log'][0])
    resolver['end_actual_action_count'] = 4  # real failed clear then real near measurement
    action, p = audit.canonical_next_action(prefix, resolver)
    assert action == 'clear' and p == tuple(r['summary']['action_history'][3]['position'])
    # With zero remaining active allowance, the same true entry region leads
    # to the original full optical lattice, not an arbitrarily distant point.
    resolver['end_actual_action_count'] = 2
    r['spec']['kwargs']['max_active_probes'] = 0
    action, p = audit.canonical_next_action(audit.Prefix(r), resolver)
    assert action == 'clear' and max(abs(x) for x in p) < 100
