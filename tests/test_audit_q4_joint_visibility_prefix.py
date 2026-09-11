"""Scripted accepted replies and certificate tampering; no scenes or network."""
import copy
import gzip
import importlib.util
import json
import math
from pathlib import Path
import socket

import pytest

from experiments.audit_q4_joint_visibility_prefix import (verify_grid, wire_prefix,
    audit_joint_visibility_prefix, summarize_joint_visibility_audits, boundary_exclusions)
from planning.coverage import clearance_grid
from simulator_client import SimulatorClient


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No simulator/network in pure prefix tests')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(SimulatorClient, '_exchange', forbidden)


@pytest.mark.parametrize('vertices', [
    [(-30., -1.), (30., -1.), (30., 1.), (-30., 1.)],
    [(-28., -28.), (56., -28.), (56., 56.), (-28., 56.)],
    [(3., 4.)], [(-50., 0.), (50., 0.)]])
@pytest.mark.parametrize('bearing', [0., 17., 90., 359.999])
def test_full_original_grid_verified_by_whole_strips(vertices, bearing):
    grid = [(p.x, p.y) for p in clearance_grid(vertices, bearing_deg=bearing, start=(100., 20.))]
    assert verify_grid(vertices, bearing, (100., 20.), grid) == len(grid)


def test_interior_grid_hole_is_rejected_even_if_all_polygon_vertices_are_covered():
    vertices = [(-56., -56.), (56., -56.), (56., 56.), (-56., 56.)]
    grid = [(p.x, p.y) for p in clearance_grid(vertices, start=(0., 0.))]
    bad = [p for p in grid if p != (14., 14.)]
    assert all(any(math.dist(v, p) < 20 for p in bad) for v in vertices)
    with pytest.raises(ValueError, match='interior column'):
        verify_grid(vertices, 0., (0., 0.), bad)


@pytest.mark.parametrize('change', ['spacing', 'duplicate', 'order'])
def test_optical_lattice_and_complete_common_order_are_checked(change):
    vertices = [(-30., -1.), (30., -1.), (30., 1.), (-30., 1.)]
    grid = [(p.x, p.y) for p in clearance_grid(vertices)]
    if change == 'spacing': grid[0] = (grid[0][0]+1, grid[0][1])
    if change == 'duplicate': grid.append(grid[0])
    if change == 'order': grid.reverse()
    with pytest.raises(ValueError): verify_grid(vertices, 0., (0., 0.), grid)


def test_wire_binding_rejects_fictitious_negative_or_bearing():
    a = dict(action='measure', position=[0., 0.], channel=2, result='direction',
             bearing_deg=90., virtual_time_s=6.)
    w = dict(action='/measure', position=[0., 0.], channel=2,
             response=dict(accepted=True, measure_result='direction', svd_deg=90., virtual_time_s=6.))
    record = {'summary': {'action_history': [a]}, 'history': [w]}
    assert wire_prefix(record)[1][-1] == ((0., 0.), 2, 6.)
    for key, value in [('result', 'no_signal'), ('bearing_deg', 91.), ('virtual_time_s', 5.), ('channel', 3)]:
        bad = copy.deepcopy(record)
        bad['summary']['action_history'][0][key] = value
        with pytest.raises(ValueError): wire_prefix(bad)


def fixture_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example(monkeypatch, config='probe', outcome='r8_success'):
    from strategies.q4_joint_visibility import Q4JointVisibility
    from simulator_client.state import Position
    import planning.q4_directional_cover as cover
    monkeypatch.setattr(cover, 'certified_cover_points', lambda profile:
        ((Position(0, 0), Position(100, 0)), {'passed': True, 'test_only': True}))
    base = fixture_module('test_q4_clear_before_probe')
    client = base.Replies()
    policy = Q4JointVisibility(client, 20000, 0 if config == 'probe_optical' or outcome == 'terminal_ready' else 6,
                              config=config, max_expansions=0)
    base.prime(policy, client)
    negatives = ([(-30., 0.), (0., -30.)] if outcome in {'aux_ready', 'terminal_ready'}
        else [(1400., 1400.)] if outcome == 'r8_success' and config == 'probe' else [(-10., -10.)])
    for negative in negatives:
        client.measure_replies = [('no_signal', None)]
        policy._perform('measure', Position.coerce(negative), 1, 'coverage')
    if outcome == 'near':
        client.clear_replies = ['success']
        client.measure_replies = [('near', None)]
    elif outcome == 'direction':
        client.clear_replies = ['success']
        client.measure_replies = [('direction', 225.)]
    elif outcome == 'service':
        # Real inherited 60s slice at a distant negative observer cannot pay
        # the selected centre's incoming leg; no request is sent.
        client.measure_replies = [('no_signal', None)]
        policy._perform('measure', Position(1400, 1400), 2, 'coverage')
        service_start = len(policy.report.action_history)
        policy.service_deadline = client.state.virtual_time_s+60.
    try:
        policy._resolve(1)
    except base._ServiceSliceExpired:
        assert outcome == 'service'
        n = len(policy.report.action_history)
        policy.early_service_log.append(dict(channel=1, after_actual_action_count=service_start,
            end_actual_action_count=n, interrupted=True, budget_s=60.))
    result = fixture_module('test_audit_q4_clear_before_probe').wrap(policy)
    result['spec']['kwargs'].update(config=config, max_active_probes=policy.max_active_probes)
    return result


@pytest.mark.parametrize('outcome', ['r8_success', 'near', 'direction', 'service'])
def test_real_inherited_controller_prefixes(monkeypatch, outcome):
    record = example(monkeypatch, outcome=outcome)
    saved = copy.deepcopy(record)
    audited = audit_joint_visibility_prefix(record)
    assert audited['epochs'] == 1 and audited['probe_decisions'] == (outcome != 'r8_success')
    assert audited['executed_probes'] == (outcome in {'near', 'direction'})
    assert record == saved
    assert summarize_joint_visibility_audits([audited, audited])['epochs'] == 2


def test_real_auxiliary_optical_grid_and_success_prefix(monkeypatch):
    record = example(monkeypatch, config='probe_optical')
    audited = audit_joint_visibility_prefix(record)
    assert audited['optical_epochs'] == audited['optical_attempts'] == 1
    assert audited['probe_decisions'] == 0


def test_actual_aux_ready_clear_has_real_geometric_certificate_and_no_fictitious_probe(monkeypatch):
    # Feasible physical example: source(-16,-16), R1000, orientation225deg;
    # the two scripted bearings need <1.005deg error, both negatives are outside
    # its emitting half-plane. This test uses only their accepted responses.
    record = example(monkeypatch, outcome='aux_ready')
    audited = audit_joint_visibility_prefix(record)
    assert audited['refined_epochs'] == audited['auxiliary_clear_attempts'] == audited['auxiliary_clear_successes'] == 1
    assert audited['executed_probes'] == 0
    p = record['summary']['strategy_parameters']
    assert not p['clear_before_probe_log']
    assert record['summary']['action_history'][-1]['phase'] == 'joint_visibility_clear'
    bad = copy.deepcopy(record)
    bad['summary']['strategy_parameters']['joint_visibility_probe_log'][0]['point'][0] += 1.
    with pytest.raises(ValueError, match='certificate'): audit_joint_visibility_prefix(bad)


@pytest.mark.parametrize('config', ['probe', 'probe_optical'])
def test_zero_remaining_probes_still_certifies_real_aux_terminal_clear(monkeypatch, config):
    record = example(monkeypatch, config=config, outcome='terminal_ready')
    audited = audit_joint_visibility_prefix(record)
    assert audited['terminal_clear_events'] == audited['auxiliary_clear_successes'] == 1
    assert audited['probe_decisions'] == audited['executed_probes'] == audited['optical_attempts'] == 0
    terminal_log = record['summary']['strategy_parameters']['joint_visibility_terminal_clear_log']
    bad = copy.deepcopy(record)
    bad['summary']['strategy_parameters']['joint_visibility_terminal_clear_log'][0]['canonical_first_point'][0] += 28
    with pytest.raises(ValueError, match='original canonical'): audit_joint_visibility_prefix(bad)
    terminal_log.clear()
    with pytest.raises(ValueError, match='Unlogged'): audit_joint_visibility_prefix(record)


def test_unchanged_helper_preserves_r8_without_auxiliary_decisions(monkeypatch):
    record = example(monkeypatch)
    p = record['summary']['strategy_parameters']
    event = p['joint_visibility_resolver_log'][0]
    assert event['skip_reason'] == 'no_boundary_reduction' and event['initial_aux_vertices'] is None
    assert not p['joint_visibility_probe_log'] and not p['joint_visibility_grid_log']
    assert not event['aux_updates']
    assert audit_joint_visibility_prefix(record)['refined_epochs'] == 0
    assert record['summary']['action_history'][-1]['phase'] == 'speculative_clear_before_probe'
    event['skip_reason'] = None
    event['initial_aux_vertices'] = copy.deepcopy(event['helper_evidence']['output_vertices'])
    with pytest.raises(ValueError, match='No-boundary-reduction'): audit_joint_visibility_prefix(record)


def test_deleted_interior_cells_do_not_prove_convex_boundary_reduction():
    from planning.joint_visibility_region import joint_visibility_outer
    from experiments.audit_q4_joint_visibility import audit_joint_visibility_certificate
    original = ((990., -1.), (1015., -1.), (1015., 1.), (990., 1.))
    outer, evidence = joint_visibility_outer(original, [(0., -100.), (0., 100.)], [(1005., 0.)])
    assert evidence['status'] == 'outer_refined' and evidence['deleted_intersecting_cells'] > 0
    assert audit_joint_visibility_certificate(evidence)['passed']
    assert boundary_exclusions(original, outer) == []


def test_boundary_reduction_field_cannot_hide_or_invent_actual_exclusion(monkeypatch):
    for outcome, fake in [('r8_success', [0]), ('near', [])]:
        record = example(monkeypatch, outcome=outcome)
        e = record['summary']['strategy_parameters']['joint_visibility_resolver_log'][0]['helper_evidence']
        e['old_vertices_excluded'] = fake
        with pytest.raises(ValueError, match='exact old-vertex'): audit_joint_visibility_prefix(record)


@pytest.mark.parametrize('change', ['future_negative', 'fake_positive', 'canonical', 'initial_aux',
    'probe_aux', 'point', 'fresh', 'canonical_point', 'unlogged_probe', 'wire', 'model_contradiction'])
def test_corrupted_actual_prefix_or_probe_rejected(monkeypatch, change):
    record = example(monkeypatch, outcome='near')
    p = record['summary']['strategy_parameters']
    resolver, probe = p['joint_visibility_resolver_log'][0], p['joint_visibility_probe_log'][0]
    if change == 'future_negative': resolver['helper_evidence']['negative_positions'].append([0, 0])
    elif change == 'fake_positive': resolver['helper_evidence']['positive_positions'].pop()
    elif change == 'canonical': resolver['helper_evidence']['canonical_vertices'][0][0] += 1
    elif change == 'initial_aux': resolver['initial_aux_vertices'][0][0] += 1
    elif change == 'probe_aux': probe['aux_vertices'][0][0] += 1
    elif change == 'point': probe['point'][0] += 1
    elif change == 'fresh': probe['fresh_indices'] = []
    elif change == 'canonical_point': probe['canonical_point'][0] += 1
    elif change == 'unlogged_probe': p['joint_visibility_probe_log'] = []
    elif change == 'wire': record['history'][0]['response']['svd_deg'] += 1
    else: p['joint_visibility_model_contradictions'].append({'reason': 'test'})
    with pytest.raises(ValueError): audit_joint_visibility_prefix(record)


@pytest.mark.parametrize('change', ['bearing', 'prefix', 'vertices', 'missing'])
def test_auxiliary_updates_must_be_actual_direction_only(monkeypatch, change):
    record = example(monkeypatch, outcome='direction')
    updates = record['summary']['strategy_parameters']['joint_visibility_resolver_log'][0]['aux_updates']
    assert len(updates) == 1
    if change == 'bearing': updates[0]['bearing_deg'] += 1
    elif change == 'prefix': updates[0]['after_actual_action_count'] -= 1
    elif change == 'vertices': updates[0]['aux_vertices'][0][0] += 1
    else: updates.clear()
    with pytest.raises(ValueError, match='Auxiliary update'): audit_joint_visibility_prefix(record)


@pytest.mark.parametrize('change', ['grid', 'start', 'rotation', 'count', 'unlogged', 'fake_interrupt', 'canonical_first'])
def test_optical_evidence_cannot_omit_or_fabricate_actual_search(monkeypatch, change):
    record = example(monkeypatch, config='probe_optical')
    p = record['summary']['strategy_parameters']
    g = p['joint_visibility_grid_log'][0]
    if change == 'grid': g['grid'].pop()
    elif change == 'start': g['start'][0] += 1
    elif change == 'rotation': g['bearing_deg'] += 1
    elif change == 'count': g['actual_grid_actions'] += 1
    elif change == 'unlogged': p['joint_visibility_grid_log'].clear()
    elif change == 'canonical_first': g['canonical_first_point'][0] += 28.
    else: g['status'] = 'interrupted'
    with pytest.raises(ValueError): audit_joint_visibility_prefix(record)


def test_duplicate_or_overbudget_probe_cannot_inflate_decisions(monkeypatch):
    record = example(monkeypatch, outcome='near')
    probes = record['summary']['strategy_parameters']['joint_visibility_probe_log']
    probes.append(copy.deepcopy(probes[0]))
    probes[-1]['index'] += 1
    with pytest.raises(ValueError, match='order'): audit_joint_visibility_prefix(record)
    probes.pop()
    record['spec']['kwargs']['max_active_probes'] = 0
    with pytest.raises(ValueError, match='budget'): audit_joint_visibility_prefix(record)


def test_partial_grid_is_accepted_only_for_actual_service_gate(monkeypatch):
    record = example(monkeypatch, config='probe_optical', outcome='service')
    assert audit_joint_visibility_prefix(record)['optical_attempts'] == 0
    record['summary']['strategy_parameters']['early_service_log'][0]['budget_s'] = 10000.
    with pytest.raises(ValueError, match='gate'): audit_joint_visibility_prefix(record)


def test_batch_runs_inherited_r8_audit_and_preserves_output(monkeypatch, tmp_path):
    from experiments.audit_q4_joint_visibility_prefix_batch import audit
    record = example(monkeypatch)
    record['row'].update(seed=0, strategy='fixture')
    record['spec']['entrypoint'] = 'strategies.q4_joint_visibility:run_q4_joint_visibility'
    (tmp_path / 'records').mkdir()
    path = tmp_path / 'records/fixture-0.json.gz'
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        json.dump(record, stream)
    for name, value in [('manifest.json', {'seeds': [0], 'specs': {'fixture': record['spec']}}),
                        ('freeze.json', {'fixture': True}), ('independent_audit.json', {'all_passed': True})]:
        (tmp_path / name).write_text(json.dumps(value), encoding='utf-8')
    assert audit(tmp_path) == 0
    result = json.loads((tmp_path / 'joint_visibility_prefix_audit.json').read_bytes())
    assert result['passed_records'] == 1 and result['audits'][0]['inherited_r8_audit']['successes'] == 1
    with pytest.raises(ValueError, match='Preserve'): audit(tmp_path)
