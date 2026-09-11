"""R12 launcher: real qualification/read_set/comparison and post-exit bounds, offline."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import zipfile

import pytest

from experiments import run_q4_joint_continuation_practice as entry
from experiments import evaluate_q4_joint_continuation as frozen
from practice_control import BridgeError
from simulator_client import SimulatorClient


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('No real network in practice-entry tests')
    monkeypatch.setattr(socket, 'create_connection', fail)
    monkeypatch.setattr(socket.socket, 'connect', fail)
    monkeypatch.setattr(SimulatorClient, '_exchange', fail)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    root = tmp_path / 'workspace'
    root.mkdir()
    research = root / 'research/q4_joint_continuation'
    results = root / 'results/q4_joint_continuation'
    rl_root = tmp_path/'separate-r9-worker'
    rl_research = rl_root/'research/q4_joint_visibility'
    for path in (root/'src/frozen.py', root/'experiments/evaluate_q4_round2.py',
                 rl_root/'experiments/q4_frozen_rl_worker.py', rl_root/'experiments/run_q4_frozen_rl_compare.py',
                 research/'PROTOCOL.md',
                 *(rl_research/'rl_reference'/p for p in ('reference.json', 'source.zip', 'checkpoint.pt'))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'inert offline fixture')
    source = lambda: {'src/frozen.py': entry.digest(root/'src/frozen.py')}
    monkeypatch.setattr(entry, 'ROOT', root)
    monkeypatch.setattr(entry, 'hashes', source)
    monkeypatch.setattr(frozen, 'ROOT', root)
    monkeypatch.setattr(frozen, 'RL_ROOT', rl_root)
    monkeypatch.setattr(frozen, 'RESEARCH', research)
    monkeypatch.setattr(frozen, 'RESULTS', results)
    monkeypatch.setattr(frozen, 'hashes', source)
    specs = {'compact_baseline': {'entrypoint': 'strategies.q4_cover_search:run_q4_cover_search', 'kwargs': {}},
             'compact_clear_before_probe': {'entrypoint': 'strategies.q4_clear_before_probe:run_q4_clear_before_probe', 'kwargs': {}},
             frozen.REFERENCE: {'entrypoint': 'strategies.q4_joint_visibility:run_q4_joint_visibility', 'kwargs': {'config': 'probe'}},
             entry.SELECTED: copy.deepcopy(entry.SPEC)}
    identity = dict(source_sha256=source(), rl_identity=frozen.rl_identity(),
        evaluator_sha256=entry.digest(Path(frozen.__file__)),
        evaluator_dependencies_sha256=frozen.evaluator_dependencies(), protocol_sha256=entry.digest(research/'PROTOCOL.md'))
    selection = dict(selected=entry.SELECTED, specs=specs, reserved_seeds=copy.deepcopy(entry.STAGES), **identity)
    selection_path, qualification_path = research/'selection.json', research/'qualification.json'
    dump(selection_path, selection)
    reports = {}
    for stage, seeds in entry.STAGES.items():
        directory, rl_directory = results/stage, results/(stage+'-rl')
        manifest = dict(stage=stage, seeds=seeds, specs=specs, source_sha256=source(),
                        selection_sha256=entry.digest(selection_path))
        dump(directory/'manifest.json', manifest)
        dump(directory/'freeze.json', dict(manifest_sha256=entry.manifest_digest(manifest), git_commit='b'*40))
        with zipfile.ZipFile(directory/'source.zip', 'w') as archive:
            archive.writestr('src/frozen.py', (root/'src/frozen.py').read_bytes())
        rows, summaries = [], {}
        for label, value in zip(specs, (1200., 1100., 1000., 900.)):
            summaries[label] = dict(mean_time_s=value, p95_time_s=value)
            for seed in seeds:
                rows.append(dict(seed=seed, strategy=label, case_sha256=f'{seed:064x}', case_id=f'fixture-{seed}',
                    successful=True, penalized_time_s=value, virtual_time_s=value,
                    common_lower_bound_s=500., time_over_lower_bound=value/500.))
        rows.sort(key=lambda r: (r['seed'], r['strategy']))
        reports[stage] = dict(rows=rows, summaries=summaries)
        dump(directory/'summary.json', reports[stage])
        for name, count in [('independent_audit.json', len(seeds)*len(specs)),
                            ('clear_before_probe_audit.json', len(seeds)),
                            ('joint_visibility_prefix_audit.json', len(seeds)),
                            ('joint_continuation_prefix_audit.json', len(seeds))]:
            dump(directory/name, dict(all_passed=True, records=count, passed_records=count))
        freeze_rl = dict(reference_sha256=identity['rl_identity']['research/q4_joint_visibility/rl_reference/reference.json'],
            worker_sha256=identity['rl_identity']['experiments/q4_frozen_rl_worker.py'],
            runner_sha256=identity['rl_identity']['experiments/run_q4_frozen_rl_compare.py'],
            state_manifest_sha256=entry.digest(directory/'manifest.json'), state_summary_sha256=entry.digest(directory/'summary.json'),
            reference_arm=False)
        dump(rl_directory/'freeze.json', freeze_rl)
        rl_rows = [dict(r, strategy='compact_macro_ppo512') for r in rows if r['strategy'] == frozen.REFERENCE]
        dump(rl_directory/'comparison.json', dict(rows=rows+rl_rows, rl_audits_all_passed=True,
            freeze_sha256=entry.digest(rl_directory/'freeze.json')))
    qualification = dict(phase='confirmation', selected=entry.SELECTED, passed=True,
                         selection_sha256=entry.digest(selection_path), **identity)
    host = SimpleNamespace(root=root, research=research, results=results, rl_root=rl_root, rl_research=rl_research, selection=selection,
        qualification=qualification, selection_path=selection_path, qualification_path=qualification_path,
        reports=reports, output=root/'output', events=[], calls=[], solves=[], before_solver=None,
        after_solver=None, alter_registered=None, completed=True, cleared=10,
        state={'active': False, 'mode': '', 'case_code': '', 'phase': ''})

    def refresh_qualification():
        evidence, comparisons = {}, {}
        for stage in entry.STAGES:
            directory, rl_directory = results/stage, results/(stage+'-rl')
            dump(directory/'summary.json', reports[stage])
            freeze_rl = json.loads((rl_directory/'freeze.json').read_bytes())
            freeze_rl['state_summary_sha256'] = entry.digest(directory/'summary.json')
            dump(rl_directory/'freeze.json', freeze_rl)
            combined = json.loads((rl_directory/'comparison.json').read_bytes())
            combined['rows'] = reports[stage]['rows']+[r for r in combined['rows'] if r['strategy']=='compact_macro_ppo512']
            combined['freeze_sha256'] = entry.digest(rl_directory/'freeze.json')
            dump(rl_directory/'comparison.json', combined)
            report, _, _ = frozen.read_set(stage, evidence)
            comparisons[stage] = frozen.comparison(report, entry.SELECTED, frozen.REFERENCE)
        qualification.update(evidence_sha256=evidence, comparisons_vs_incumbent={entry.SELECTED: comparisons})
        dump(qualification_path, qualification)
    host.refresh = refresh_qualification
    refresh_qualification()
    sim_dir = root/'simulator'
    sim_dir.mkdir()
    (sim_dir/'jammers-simulator-full.exe').write_bytes(b'inert executable fixture')
    host.sim_dir = sim_dir

    class Bridge:
        def __init__(self, port): host.events.append('bridge')
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def current_test(self): return host.state

    @contextmanager
    def lock(path):
        assert path == sim_dir/'.practice-control/controller.lock'
        host.events.append('lock')
        yield

    def solver(client, **kwargs):
        host.solves.append(kwargs)
        return {'mock_solver': True}

    def run_once(bridge, **kwargs):
        host.calls.append(kwargs)
        run = kwargs['output']
        run.mkdir()
        if host.before_solver: host.before_solver()
        kwargs['solver'](object(), problem=kwargs['problem'], variant=kwargs['variant'], max_actions=kwargs['max_actions'])
        summary = dict(completed=host.completed, problem=4, data_origin='simulator_http_session',
            declared_mode='practice', case_code=f'CASE-{len(host.calls)}', pending_request=None,
            state=dict(session='exited', cleared_count=host.cleared, virtual_time_s=3000.),
            search=dict(time_breakdown={}, action_history=[dict(action='clear', result='success', channel=i+1,
                        position=[float(i*20), 0.]) for i in range(host.cleared)]))
        dump(run/'summary.json', summary)
        registered = dict(data_origin='registered_official_practice', problem=4, case_code=summary['case_code'],
            source_total=10, source_total_source='official_simulator_result_file', official_result_session_time_matched=True,
            summary_sha256=entry.digest(run/'summary.json'), cleared_count=host.cleared,
            virtual_time_s=3000., search_completed=host.completed)
        if host.alter_registered: host.alter_registered(registered)
        record_path = run.parent/'registered'/f"{summary['case_code']}.json"
        dump(record_path, registered)
        dump(run/'registration.json', dict(record=str(record_path), case_code=summary['case_code'], source_total=10))
        host.events.append('registered_mock')
        if host.after_solver: host.after_solver()
        return dict(completed=host.completed, problem=4, case_code=summary['case_code'], summary=str(run/'summary.json'),
                    source_total=10, cleared_count=host.cleared, virtual_time_s=3000.)

    monkeypatch.setattr(entry, 'PracticeBridge', Bridge)
    monkeypatch.setattr(entry, 'controller_lock', lock)
    monkeypatch.setattr(entry, 'run_q4_joint_continuation', solver)
    monkeypatch.setattr(entry, 'run_once', run_once)
    # Deliberately do not replace preflight/read_set/comparison/rl_identity or
    # post_registration_bounds/session_lower_bounds.analyze.
    def main(*extra):
        monkeypatch.setattr('sys.argv', ['practice.py', '--selection', str(selection_path),
            '--qualification', str(qualification_path), '--robot-id', 'TEST-TEAM', '--simulator-dir', str(sim_dir),
            '--output', str(host.output), *extra])
        return entry.main()
    host.main = main
    return host


def test_full_twenty_evidence_preflight_is_offline(rig, capsys):
    assert rig.main('--preflight-only') == 0
    value = json.loads(capsys.readouterr().out)
    assert value['selected'] == entry.SELECTED and value['config'] == 'after_active_miss_optical' and value['source_commit'] == 'b'*40
    assert len(rig.qualification['evidence_sha256']) == 20
    assert len(value['source_archive_sha256']) == 2 and not rig.events and not rig.output.exists()


@pytest.mark.parametrize('key,value', [('passed', False), ('passed', 1), ('phase', 'development'),
    ('selected', 'compact_combo'), ('selection_sha256', '0'*64), ('source_sha256', {}), ('rl_identity', {}),
    ('evaluator_sha256', '0'*64), ('evaluator_dependencies_sha256', {}), ('protocol_sha256', '0'*64), ('evidence_sha256', {})])
def test_invalid_qualification_refused_before_bridge(rig, key, value):
    rig.qualification[key] = value
    dump(rig.qualification_path, rig.qualification)
    with pytest.raises(ValueError): rig.main()
    assert not rig.events


@pytest.mark.parametrize('change', ['source', 'checkpoint', 'protocol', 'archive', 'audit', 'commit', 'comparison', 'rl_audit'])
def test_independent_artifact_drift_is_not_hidden_by_passed_flag(rig, change):
    directory = rig.results/'confirmation'
    if change in {'source', 'checkpoint', 'protocol'}:
        path = {'source': rig.root/'src/frozen.py', 'checkpoint': rig.rl_research/'rl_reference/checkpoint.pt',
                'protocol': rig.research/'PROTOCOL.md'}[change]
        path.write_bytes(b'changed')
    elif change == 'archive':
        with zipfile.ZipFile(directory/'source.zip', 'w') as archive: archive.writestr('extra', b'changed')
    elif change == 'audit':
        dump(directory/'joint_visibility_prefix_audit.json', dict(all_passed=False, records=128, passed_records=128))
    elif change == 'commit':
        data = json.loads((directory/'freeze.json').read_bytes()); data['git_commit'] = 'not-a-sha'
        dump(directory/'freeze.json', data)
    else:
        path = rig.results/'confirmation-rl/comparison.json'
        data = json.loads(path.read_bytes())
        if change == 'comparison': data['rows'][-1]['case_sha256'] = '0'*64
        else: data['rl_audits_all_passed'] = False
        dump(path, data)
    with pytest.raises(ValueError): rig.main()
    assert not rig.events


@pytest.mark.parametrize('change', ['duplicate', 'wrong_case', 'wrong_bound'])
def test_complete_pair_matrix_is_rechecked_even_if_qualification_hashes_are_rebuilt(rig, change):
    rows = rig.reports['confirmation']['rows']
    candidate = next(r for r in rows if r['strategy'] == entry.SELECTED)
    if change == 'duplicate': rows.append(copy.deepcopy(candidate))
    elif change == 'wrong_case': candidate['case_sha256'] = '0'*64
    else: candidate['common_lower_bound_s'] += 1.
    if change == 'duplicate':
        # Keep candidate count valid for the evaluator by duplicating another arm.
        rows.pop(); rows.append(copy.deepcopy(next(r for r in rows if r['strategy']=='compact_clear_before_probe')))
    rig.refresh()
    with pytest.raises(ValueError, match='pair matrix|mismatched'): rig.main()
    assert not rig.events


def test_true_recomputed_paired_gate_overrides_forged_passed_flag(rig):
    report = rig.reports['confirmation']
    for row in report['rows']:
        if row['strategy'] == entry.SELECTED:
            row['virtual_time_s'] = row['penalized_time_s'] = 1000.
            row['time_over_lower_bound'] = 2.
    report['summaries'][entry.SELECTED] = dict(mean_time_s=1000., p95_time_s=1000.)
    rig.refresh()
    assert rig.qualification['passed'] is True
    with pytest.raises(ValueError, match='gate failed'): rig.main()
    assert not rig.events


@pytest.mark.parametrize('state', [None, {}, {'active': 0}, {'active': True, 'mode': 'practice'},
    {'active': False, 'mode': 'formal'}, {'active': False, 'mode': 'unknown'},
    {'active': False, 'case_code': 'OLD'}, {'active': False, 'phase': 'ended'}])
def test_only_explicit_idle_session_may_be_started(rig, state):
    rig.state = state
    with pytest.raises(ValueError, match='occupied'): rig.main()
    assert not rig.calls and not rig.solves


def test_shared_busy_lock_precedes_any_bridge(rig, monkeypatch):
    @contextmanager
    def busy(path):
        raise BridgeError('held')
        yield
    monkeypatch.setattr(entry, 'controller_lock', busy)
    with pytest.raises(BridgeError): rig.main()
    assert not rig.events


def test_registered_success_uses_fixed_solver_and_real_postexit_lower_bound(rig):
    assert rig.main() == 0
    assert rig.events[:2] == ['lock', 'bridge']
    assert rig.solves == [dict(problem=4, max_actions=20000, **entry.SPEC['kwargs'])]
    assert rig.calls[0]['method_label'] == entry.SELECTED
    bound = json.loads((rig.output/'run-001/lower_bounds.json').read_bytes())
    assert bound['official_all_clear_verified'] is True
    assert bound['time_to_conditional_lower_bound_ratio'] == pytest.approx(3000./bound['conditional_guaranteed_all_clear_lower_s'])
    assert entry.post_registration_bounds is entry.shared_practice.post_registration_bounds
    assert entry.shared_practice.SELECTED == 'compact_joint'
    for invalid in ({'problem': 3}, {'variant': 'adaptive'}, {'max_actions': 19999}):
        kwargs = dict(problem=4, variant='triangular', max_actions=20000)
        kwargs.update(invalid)
        with pytest.raises(ValueError): rig.calls[0]['solver'](object(), **kwargs)
    assert len(rig.solves) == 1


def test_missing_qualification_never_opens_bridge(rig):
    rig.qualification_path.unlink()
    with pytest.raises(FileNotFoundError): rig.main()
    assert not rig.events and not rig.output.exists()


def test_comparison_dependency_drift_is_rejected_before_bridge(rig):
    (rig.root/'experiments/evaluate_q4_round2.py').write_bytes(b'changed comparison implementation')
    with pytest.raises(ValueError, match='identity changed'):
        rig.main()
    assert not rig.events


@pytest.mark.parametrize('name', ['joint_visibility_prefix_audit.json', 'joint_continuation_prefix_audit.json'])
@pytest.mark.parametrize('change', ['missing', 'failed', 'short'])
def test_both_static_and_recursive_audit_layers_are_mandatory(rig, name, change):
    path = rig.results/'confirmation'/name
    if change == 'missing':
        path.unlink()
    else:
        dump(path, dict(all_passed=change != 'failed', records=128, passed_records=127 if change == 'short' else 128))
    if change == 'short':
        # Rebuilding file digests must not turn a short audit into full evidence.
        rig.refresh()
    with pytest.raises((ValueError, FileNotFoundError)):
        rig.main()
    assert not rig.events


def test_rl_identity_uses_separate_frozen_r9_worker_root(rig):
    assert frozen.RL_ROOT == rig.rl_root and frozen.RL_ROOT != rig.root
    identities = frozen.rl_identity()
    assert identities['research/q4_joint_visibility/rl_reference/checkpoint.pt'] == entry.digest(
        rig.rl_research/'rl_reference/checkpoint.pt')
    # An unrelated local namesake is not the selected source or model.
    local = rig.research/'rl_reference/checkpoint.pt'
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(b'not the selected checkpoint')
    assert rig.main('--preflight-only') == 0 and not rig.events


def test_mismatched_registration_never_creates_lower_bound_result(rig):
    rig.alter_registered = lambda r: r.update(case_code='ANOTHER')
    with pytest.raises(ValueError, match='registration evidence'): rig.main()
    assert not (rig.output/'run-001/lower_bounds.json').exists()
    assert (rig.output/'results.json').is_file()


def test_failed_run_stops_repeat_and_preserves_output(rig):
    rig.completed, rig.cleared = False, 9
    assert rig.main('--repeat', '2') == 1 and len(rig.calls) == 1
    bound = json.loads((rig.output/'run-001/lower_bounds.json').read_bytes())
    assert bound['time_to_conditional_lower_bound_ratio'] is None
    saved = (rig.output/'results.json').read_bytes()
    with pytest.raises(FileExistsError): rig.main()
    assert saved == (rig.output/'results.json').read_bytes()


@pytest.mark.parametrize('when', ['before_solver', 'after_solver'])
def test_source_rechecked_around_solver_and_registration(rig, when):
    setattr(rig, when, lambda: (rig.root/'src/frozen.py').write_bytes(b'drift'))
    with pytest.raises(ValueError): rig.main('--repeat', '2')
    assert len(rig.calls) == 1 and len(rig.solves) == (when == 'after_solver')
    assert not (rig.output/'run-001/lower_bounds.json').exists()


@pytest.mark.parametrize('args', [('--repeat', '0'), ('--repeat', '21'), ('--robot-id', ' '), ('--robot-id', 'bad\nteam')])
def test_invalid_controls_never_open_bridge(rig, args):
    with pytest.raises(ValueError): rig.main(*args)
    assert not rig.events
