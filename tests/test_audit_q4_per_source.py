"""Batch-envelope integrity tests, not physics/model or performance tests.

Only the expensive per-record physics and prefix checkers are mocked. Plan
reconstruction, public count quotas, archive/file hashes, release validation,
and the published-summary recomputation use the real implementations. The
records below are synthetic dictionaries; no simulator/scenario is constructed.
"""
import copy
import gzip
import hashlib
import json
import shutil
import warnings
import zipfile

import pytest

from experiments import audit_q4_per_source as audit_module
from experiments import run_q4_per_source as runner


from experiments import q4_reference_release as reference
from tests.test_q4_per_source import synthetic_historical_root
LABEL = reference.LABEL
SPEC = copy.deepcopy(reference.SPEC)
SOURCE = {'src/synthetic.py': b'# synthetic envelope fixture, never executed\n'}
IDENTITY = {name: hashlib.sha256(data).hexdigest() for name, data in SOURCE.items()}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_record(path, record):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        json.dump(record, stream, allow_nan=False)


class Batch:
    def __init__(self, directory, split):
        self.directory = directory
        directory.mkdir()
        (directory / 'records').mkdir()
        self.plan = runner.make_plan(split, {LABEL: copy.deepcopy(SPEC)})
        write_json(directory / 'plan.json', self.plan)
        self.manifest = {'plan': self.plan, 'plan_sha256': runner.sha(directory / 'plan.json'),
                         'release_sha256': None}
        self.release=reference.build_release(self.plan,self.manifest['plan_sha256'],directory.parent)
        write_json(directory/'release.json',self.release)
        self.manifest['release_sha256']=runner.sha(directory/'release.json')
        self.save_manifest()
        with zipfile.ZipFile(directory / 'source.zip', 'w') as archive:
            for name, data in SOURCE.items():
                archive.writestr(name, data)
        self.records = []
        for seed in self.plan['seed_selection']['seeds']:
            n = runner.count_from_seed(seed)
            truth = {'synthetic_identity_only': seed}
            row = dict(seed=seed, strategy=LABEL, stage=runner.DESIGNS[split]['stage'],
                source_total=n, cleared_total=n, successful=True,
                virtual_time_s=100. * n, penalized_time_s=100. * n,
                common_lower_bound_s=1000., case_sha256=runner.digest(truth))
            self.records.append(dict(row=row, spec=copy.deepcopy(SPEC),
                evaluation={'ground_truth': truth}, evaluation_phase='after_policy_termination',
                summary={'strategy_parameters': {}}, history=[]))
        self.save_records()
        self.save_summary()

    def path(self, record):
        return self.directory / 'records' / f"{record['row']['strategy']}-{record['row']['seed']}.json.gz"

    def save_records(self):
        for record in self.records:
            write_record(self.path(record), record)

    def save_manifest(self):
        write_json(self.directory / 'manifest.json', self.manifest)
        write_json(self.directory / 'freeze.json', {'manifest_sha256': runner.digest(self.manifest),
                                                   'git_commit': 'synthetic-not-a-run'})

    def save_summary(self):
        value = runner.summarize([record['row'] for record in self.records], self.plan['split'],
                                 self.plan['seed_selection']['seeds'])
        value.update(complete=True, source_unchanged=True, infrastructure_errors=[])
        write_json(self.directory / 'summary.json', value)

    def output(self):
        return runner.read(self.directory / 'independent_audit.json')


@pytest.fixture
def make_batch(tmp_path, monkeypatch):
    # Seven tiny count strata instead of any frozen 630 performance split.
    for name, start in [('development', 1001), ('confirmation', 2001)]:
        monkeypatch.setitem(runner.DESIGNS, name, dict(start=start, scan_limit=100,
            quota=1, stage='pilot' if name == 'development' else 'confirmation', stress=False))
    monkeypatch.setattr(runner, 'BOOTSTRAP_SAMPLES', 64)
    monkeypatch.setattr(runner, 'source_hashes', lambda: dict(IDENTITY))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(audit_module, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'one', lambda *a, **k: pytest.fail('No scenario or policy may run'))
    synthetic_historical_root(tmp_path,monkeypatch,SOURCE)
    calls = {'generic': [], 'r12': [], 'r8': [], 'scheduling': [], 'range': []}

    def generic(record):
        calls['generic'].append(record)
        return {'passed': True, 'errors': []}

    def prefix(record, kind):
        assert set(record) == {'summary', 'history', 'row', 'spec'}
        assert 'evaluation' not in record
        calls[kind].append(record)
        return {'passed': True}

    monkeypatch.setattr(audit_module, 'audit_record', generic)
    monkeypatch.setattr(audit_module, 'audit_joint_continuation_prefix', lambda r: prefix(r, 'r12'))
    monkeypatch.setattr(audit_module, 'audit_clear_before_probe_prefix', lambda r: prefix(r, 'r8'))
    monkeypatch.setattr(audit_module, 'audit_scheduling_prefix', lambda r: prefix(r, 'scheduling'))
    monkeypatch.setattr(audit_module, 'audit_range_prefix', lambda r: prefix(r, 'range'))

    def make(split='confirmation'):
        batch = Batch(tmp_path / 'batch', split)
        batch.calls = calls
        return batch
    return make


def assert_rejected(batch):
    # Missing/malformed metadata may raise before an audit artifact exists;
    # either failure mode is closed and must never publish all_passed=True.
    try:
        status = audit_module.audit(batch.directory)
    except (ValueError, KeyError, TypeError, FileNotFoundError, zipfile.BadZipFile):
        if (batch.directory / 'independent_audit.json').exists():
            assert not batch.output()['all_passed']
    else:
        assert status == 1
        assert batch.output()['all_passed'] is False


def test_complete_envelope_recomputes_summary_and_hashes_all_inputs(make_batch):
    batch = make_batch()
    assert audit_module.audit(batch.directory) == 0
    output = batch.output()
    assert output['all_passed'] and output['records'] == output['passed_records'] == 7
    assert len(batch.calls['generic']) == len(batch.calls['r12']) == 7
    assert not batch.calls['range']
    assert output['plan_sha256'] == runner.sha(batch.directory / 'plan.json')
    expected = {'manifest.json', 'freeze.json', 'plan.json', 'source.zip', 'summary.json', 'release.json'}
    expected |= {batch.path(record).relative_to(batch.directory).as_posix() for record in batch.records}
    assert set(output['input_sha256']) == expected
    assert all(value == runner.sha(batch.directory / name) for name, value in output['input_sha256'].items())


def test_independent_release_and_historical_qualification_is_actually_bound(make_batch):
    batch = make_batch('confirmation')
    assert audit_module.audit(batch.directory) == 0
    assert batch.output()['input_sha256']['release.json'] == runner.sha(batch.directory / 'release.json')


@pytest.mark.parametrize('name', ['plan.json', 'manifest.json', 'freeze.json', 'source.zip', 'summary.json'])
def test_missing_required_file_fails_closed(make_batch, name):
    batch = make_batch()
    (batch.directory / name).unlink()
    assert_rejected(batch)


@pytest.mark.parametrize('name', ['plan_sha256', 'manifest_digest', 'source_hash'])
def test_identity_tampering_rejected(make_batch, name):
    batch = make_batch()
    if name == 'plan_sha256':
        batch.manifest['plan_sha256'] = '0' * 64
        batch.save_manifest()  # internally re-hashed manifest cannot forge plan bytes
    elif name == 'manifest_digest':
        batch.manifest['unbound_change'] = True
        write_json(batch.directory / 'manifest.json', batch.manifest)
    else:
        batch.plan['source_sha256']['src/synthetic.py'] = '0' * 64
        write_json(batch.directory / 'plan.json', batch.plan)
        batch.manifest['plan_sha256'] = runner.sha(batch.directory / 'plan.json')
        batch.save_manifest()
    assert_rejected(batch)


def test_equivalent_plan_json_with_changed_bytes_does_not_match_original_hash(make_batch):
    batch = make_batch()
    (batch.directory / 'plan.json').write_text(json.dumps(batch.plan, indent=4), encoding='utf-8')
    assert_rejected(batch)


@pytest.mark.parametrize('kind', ['changed', 'extra', 'duplicate', 'missing'])
def test_source_archive_exact_membership_and_bytes(make_batch, kind):
    batch = make_batch()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(batch.directory / 'source.zip', 'w') as archive:
            if kind != 'missing':
                for name, data in SOURCE.items():
                    if kind == 'duplicate':
                        archive.writestr(name, b'# hidden earlier duplicate')
                    archive.writestr(name, b'changed' if kind == 'changed' else data)
            if kind == 'extra':
                archive.writestr('unlisted.py', b'extra')
    assert_rejected(batch)


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'filename', 'unexpected', 'wrong_label'])
def test_fixed_record_matrix_rejects_missing_extra_duplicate_or_renamed_case(make_batch, kind):
    batch = make_batch()
    first = batch.path(batch.records[0])
    if kind == 'missing':
        first.unlink()
    elif kind == 'duplicate':
        shutil.copyfile(first, first.with_name('duplicate.json.gz'))
    elif kind == 'filename':
        first.rename(first.with_name('renamed.json.gz'))
    elif kind == 'unexpected':
        record = copy.deepcopy(batch.records[0])
        record['row']['seed'] = 99999
        write_record(batch.path(record), record)
    else:
        batch.records[0]['row']['strategy'] = 'compact_impostor'
        write_record(first, batch.records[0])
    assert_rejected(batch)


@pytest.mark.parametrize('kind', ['quota', 'trace'])
def test_rehashed_plan_cannot_change_fixed_quota_or_selection_trace(make_batch, kind):
    batch = make_batch()
    if kind == 'quota':
        batch.plan['seed_selection']['design']['quota'] = 2
    else:
        batch.plan['seed_selection']['trace'][0]['accepted'] = not batch.plan['seed_selection']['trace'][0]['accepted']
    write_json(batch.directory / 'plan.json', batch.plan)
    batch.manifest['plan_sha256'] = runner.sha(batch.directory / 'plan.json')
    batch.save_manifest()
    assert_rejected(batch)


@pytest.mark.parametrize('kind', ['spec', 'stage', 'stratum', 'truth_hash', 'duplicate_truth'])
def test_record_identity_not_accepted_from_row_claims(make_batch, kind):
    batch = make_batch()
    record = batch.records[0]
    if kind == 'spec':
        record['spec']['kwargs']['config'] = 'impostor'
    elif kind == 'stage':
        record['row']['stage'] = 'stress'
    elif kind == 'stratum':
        record['row']['source_total'] = 10 if record['row']['source_total'] != 10 else 11
    elif kind == 'truth_hash':
        record['evaluation']['ground_truth']['synthetic_identity_only'] += 1
    else:
        record['evaluation'] = copy.deepcopy(batch.records[1]['evaluation'])
        record['row']['case_sha256'] = batch.records[1]['row']['case_sha256']
    batch.save_records()
    assert_rejected(batch)


@pytest.mark.parametrize('field,value', [('mean_time_per_source_s', 0.), ('complete', False),
    ('source_unchanged', False), ('infrastructure_errors', ['lost worker']), ('all_clear', False)])
def test_summary_cannot_override_actual_complete_rows(make_batch, field, value):
    batch = make_batch()
    saved = runner.read(batch.directory / 'summary.json')
    saved[field] = value
    write_json(batch.directory / 'summary.json', saved)
    assert_rejected(batch)


def test_legal_failure_remains_auditable_with_full_penalty_and_no_success_claim(make_batch):
    batch = make_batch()
    batch.records[0]['row'].update(successful=False, cleared_total=0, virtual_time_s=1., penalized_time_s=360000.)
    batch.save_records()
    batch.save_summary()
    assert audit_module.audit(batch.directory) == 0
    output = batch.output()
    assert output['all_passed'] and not output['all_clear']
    saved = runner.read(batch.directory / 'summary.json')
    assert len(saved['failure_rows']) == 1
    assert saved['mean_time_per_source_s'] > 1000


def test_failure_short_trajectory_cannot_get_speed_credit(make_batch):
    batch = make_batch()
    batch.records[0]['row'].update(successful=False, cleared_total=0, virtual_time_s=1., penalized_time_s=1.)
    batch.save_records()
    assert_rejected(batch)


def test_infrastructure_failure_is_retained_and_fails_physical_audit(make_batch):
    batch = make_batch()
    batch.records[0]['record_kind'] = 'infrastructure_failure_without_completed_case'
    batch.save_records()
    assert_rejected(batch)
    assert batch.output()['records'] == 7 and batch.output()['passed_records'] == 6


def test_inherited_audit_failure_propagates_instead_of_using_summary_claim(make_batch, monkeypatch):
    batch = make_batch()
    monkeypatch.setattr(audit_module, 'audit_record', lambda r: {'passed': False, 'errors': ['synthetic wire failure']})
    assert_rejected(batch)
    assert batch.output()['passed_records'] == 0
    assert batch.calls['r12'] == []


def test_range_and_r12_only_receive_truth_free_prefix(make_batch):
    batch = make_batch()
    batch.records[0]['summary']['strategy_parameters']['range_skipped_scans'] = [{'synthetic': True}]
    batch.save_records()
    assert audit_module.audit(batch.directory) == 0
    assert len(batch.calls['range']) == 1
    assert len(batch.calls['r12']) == 7


def test_r12_prefix_rejection_propagates(make_batch, monkeypatch):
    batch = make_batch()
    def reject(record):
        raise AssertionError('synthetic conditional-prefix mismatch')
    monkeypatch.setattr(audit_module, 'audit_joint_continuation_prefix', reject)
    assert_rejected(batch)
    assert batch.output()['passed_records'] == 0


@pytest.mark.parametrize('kind', ['missing', 'bytes', 'unauthorized', 'selection_evidence'])
def test_independent_release_tampering_cannot_pass_batch(make_batch, kind):
    batch = make_batch('confirmation')
    release_path = batch.directory / 'release.json'
    if kind == 'missing':
        release_path.unlink()
    elif kind == 'bytes':
        release_path.write_text(json.dumps(batch.release, indent=4), encoding='utf-8')
    elif kind == 'unauthorized':
        batch.release['authorized'] = False
        write_json(release_path, batch.release)
        batch.manifest['release_sha256'] = runner.sha(release_path)
        batch.save_manifest()
    else:
        write_json(batch.directory.parent / 'selection.json', {'passed': False})
    assert_rejected(batch)


def test_preserve_existing_audit_without_rewrite(make_batch):
    batch = make_batch()
    path = batch.directory / 'independent_audit.json'
    original = b'{"sentinel":"preserve existing audit"}\n'
    path.write_bytes(original)
    with pytest.raises(ValueError, match='Preserve existing audit'):
        audit_module.audit(batch.directory)
    assert path.read_bytes() == original


@pytest.mark.parametrize('name',['audit_joint_continuation_prefix',
    'audit_clear_before_probe_prefix','audit_scheduling_prefix','audit_range_prefix'])
def test_each_inherited_prefix_false_flag_cannot_be_published_as_pass(make_batch,monkeypatch,name):
    batch=make_batch()
    if name=='audit_range_prefix':
        for record in batch.records:
            record['summary']['strategy_parameters']['range_skipped_scans']=[{'synthetic':True}]
        batch.save_records()
    monkeypatch.setattr(audit_module,name,lambda record:{'passed':False})
    assert_rejected(batch)
    assert batch.output()['passed_records']==0


def test_actual_scripted_r12_prefix_runs_all_inherited_checkers_without_models_or_truth(monkeypatch):
    from tests.test_q4_joint_continuation import make
    from tests.test_audit_q4_clear_before_probe import wrap
    from simulator_client.state import Position
    policy,client=make(monkeypatch)
    client.measure_replies=[('direction',0.),('direction',90.)]
    policy._perform('measure',Position(-1000,0),1,'constructed_observation')
    policy._perform('measure',Position(0,-1000),1,'constructed_observation')
    client.clear_replies=['no_target_in_range','success']
    client.measure_replies=[('no_signal',None),('no_signal',None),('near',None)]
    assert policy._resolve(1)
    record=wrap(policy)
    record['spec']=dict(entrypoint=SPEC['entrypoint'],kwargs=dict(SPEC['kwargs'],max_expansions=0))
    assert audit_module.audit_joint_continuation_prefix(record)['continuation_calls']==2
    assert audit_module.audit_clear_before_probe_prefix(record)['failures']==1
    assert audit_module.audit_scheduling_prefix(record)['passed']
