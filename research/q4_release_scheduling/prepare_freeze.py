"""Create development metadata after stable tests; no simulator/case generation."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from experiments import run_q4_per_source as runner
from experiments.q4_release_scheduling_release import actual_service_count
from experiments.audit_q4_release_scheduling import audit_full, verify_source_contract

BASE = '81aa6e1a1231a2358cf098fbba3fdd0557b540ef'
RESEARCH = ROOT / 'research/q4_release_scheduling'


def git(*args, env=None):
    return subprocess.check_output(['git', *args], cwd=ROOT, env=env)


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def prepare_metadata():
    assert not (RESEARCH / 'source-freeze.json').exists(), 'Refuse replacing freeze'
    for name in ('pre-qa-counter-contract-tests-console.txt', 'final-contract-tests-console.txt'):
        text = (RESEARCH / name).read_text(encoding='utf-8-sig')
        assert 'passed' in text and 'FAILED' not in text and 'ERROR ' not in text, 'Required test run did not pass'
    identity = runner.source_hashes()
    verify_source_contract()
    base_paths = git('ls-tree', '-r', '--name-only', BASE, 'src').decode().splitlines()
    assert len(base_paths) == 49 and sum(p.endswith('.py') for p in base_paths) == 44
    for path in base_paths:
        assert git('show', f'{BASE}:{path}') == (ROOT / path).read_bytes(), path
    spec = runner.read(RESEARCH / 'spec.json')
    runner.validate_spec(spec)
    qa = RESEARCH / 'old-smoke'
    manifest, summary = runner.read(qa / 'manifest.json'), runner.read(qa / 'summary.json')
    assert manifest['seeds'] == [621003, 621013] and manifest['stage'] == 'pilot'
    assert manifest['label'] == 'compact_release_scheduling' and {manifest['label']:manifest['spec']} == spec
    assert summary['source_unchanged'] is True
    with zipfile.ZipFile(qa / 'source.zip') as archive:
        assert set(archive.namelist()) == set(manifest['source_sha256'])
        assert len(archive.namelist()) == len(set(archive.namelist()))
        for name, expected in manifest['source_sha256'].items():
            assert hash_bytes(archive.read(name)) == expected, name
    prior_differences = {name:dict(qa=expected, final=runner.sha(ROOT/name))
        for name,expected in manifest['source_sha256'].items() if runner.sha(ROOT/name) != expected}
    assert set(prior_differences) <= {'experiments/audit_q4_release_scheduling.py'}, prior_differences
    qa_rows = []
    for seed in (621003, 621013):
        raw = qa / f'compact_release_scheduling-{seed}.json.gz'
        record = json.loads(gzip.decompress(raw.read_bytes()))
        first = runner.read(qa / f'{seed}-first-audit.json')
        assert first['input_sha256'] == runner.sha(raw)
        reconciled = runner.read(qa / f'{seed}-reconciled-audit.json')
        assert first['passed'] is False and first['error'] == ''
        assert "assert count == audited['prefix']['forecast_scheduled_macros']" in first['traceback']
        assert reconciled['passed'] is True and reconciled['input_sha256'] == runner.sha(raw)
        assert reconciled['first_audit_sha256'] == runner.sha(qa / f'{seed}-first-audit.json')
        assert record['row']['successful'] is True and record['spec'] == next(iter(spec.values()))
        final = audit_full(record)
        assert final['passed'] is True
        count = actual_service_count(record)
        assert final['prefix']['forecast_scheduled_macros'] == count
        assert reconciled['release_helper_count'] == reconciled['corrected_forecast_macros'] == count
        assert reconciled['legacy_forecast_macros'] > count
        path = qa / f'{seed}-final-contract-audit.json'
        runner.write_new(path, dict(**final, input_sha256=runner.sha(raw), final_source_sha256=identity))
        qa_rows.append(dict(seed=seed, first_audit_passed=False, final_audit_passed=True,
            first_failure='QA exposure counter included mixed-ready tail macros after discovery ended; physical audits had returned passed',
            executed_forecast_macros=count, legacy_forecast_macros=reconciled['legacy_forecast_macros'],
            reconciled_audit_sha256=runner.sha(qa / f'{seed}-reconciled-audit.json'), raw_sha256=runner.sha(raw)))
    plan_directory = RESEARCH / 'plans'
    plan_directory.mkdir(exist_ok=False)
    plans = {}
    for split in ('development', 'development-stress'):
        plan = runner.make_plan(split, spec, identity)
        runner.validate_plan(plan, identity)
        path = plan_directory / f'{split}.json'
        runner.write_new(path, plan)
        plans[path.relative_to(ROOT).as_posix()] = runner.sha(path)
    assert sum(len(runner.read(ROOT/path)['seed_selection']['seeds']) for path in plans) == 119
    assert identity == runner.source_hashes(), 'Source changed during preparation'
    freeze = dict(base_commit=BASE, source_sha256=identity, runtime_file_count=len(identity),
        all_49_base_src_byte_identical=True, base_src_file_count=49, base_src_python_count=44,
        plan_sha256=plans, new_cases_generated=False, independent_plans_written=False,
        old_qa_rows=qa_rows, old_qa_historical_source_differences=prior_differences,
        old_qa_history_scope='First audits and original source.zip retained. Final audit replays those same existing raw files without rerunning cases.',
        old_qa_files_sha256={p.relative_to(ROOT).as_posix():runner.sha(p) for p in sorted(qa.iterdir()) if p.is_file()},
        test_output_sha256=runner.sha(RESEARCH/'final-contract-tests-console.txt'),
        pre_qa_test_output_sha256=runner.sha(RESEARCH/'pre-qa-counter-contract-tests-console.txt'))
    runner.write_new(RESEARCH / 'source-freeze.json', freeze)
    return freeze


def verify_existing():
    """Use a closed prior console; redirect this verifier outside the worktree."""
    freeze = runner.read(RESEARCH / 'source-freeze.json')
    identity = runner.source_hashes()
    assert identity == freeze['source_sha256']
    verify_source_contract()
    for path in git('ls-tree', '-r', '--name-only', BASE, 'src').decode().splitlines():
        assert git('show', f'{BASE}:{path}') == (ROOT/path).read_bytes(), path
    plans = freeze['plan_sha256']
    for path, expected in plans.items():
        assert runner.sha(ROOT/path) == expected
        runner.validate_plan(runner.read(ROOT/path), identity)
    for path, expected in freeze['old_qa_files_sha256'].items():
        assert runner.sha(ROOT/path) == expected
    assert runner.sha(RESEARCH/'final-contract-tests-console.txt') == freeze['test_output_sha256']
    assert runner.sha(RESEARCH/'pre-qa-counter-contract-tests-console.txt') == freeze['pre_qa_test_output_sha256']
    qa_rows = freeze['old_qa_rows']
    scratch = ROOT.parent / 'q3-v1-artifacts/freeze-checks-20260912/r36'
    scratch.mkdir(parents=True, exist_ok=True)
    serial = 0
    while (scratch / f'index-verify-{serial:02d}').exists():
        serial += 1
    index = scratch / f'index-verify-{serial:02d}'
    real_index = Path(git('rev-parse', '--git-path', 'index').decode().strip())
    real_index = real_index if real_index.is_absolute() else ROOT / real_index
    before = real_index.read_bytes()
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    git('read-tree', 'HEAD', env=env)
    git('add', '--', '.gitattributes', 'src/planning/matrix_chain_route.py', 'tests/test_matrix_chain_route.py', 'src/strategies/q4_release_scheduling.py', 'src/planning/release_chain_route.py', 'tests/test_release_chain_route.py', 'src/strategies/q4_known_source.py', 'experiments/audit_q4_known_source.py', 'tests/test_q4_known_source.py',
        'experiments/run_q4_per_source.py', 'experiments/audit_q4_per_source.py',
        'experiments/audit_q4_release_scheduling.py', 'experiments/q4_release_scheduling_release.py',
        'tests/test_q4_release_scheduling.py', 'tests/test_audit_q4_release_scheduling.py',
        'tests/test_q4_per_source.py', 'tests/test_q4_release_scheduling_release.py',
        'research/q4_release_scheduling', env=env)
    git('diff', '--cached', '--check', env=env)
    changed = git('diff', '--cached', '--name-only', env=env).decode().splitlines()
    for name in set(changed) | set(identity):
        assert git('show', f':{name}', env=env) == (ROOT/name).read_bytes(), name
    assert real_index.read_bytes() == before, 'Actual Git index was modified'
    assert identity == runner.source_hashes()
    check = dict(passed=True, runtime_file_count=len(identity), base_files_exact=49,
        temporary_index_working_bytes_exact=True, checked_paths=len(set(changed) | set(identity)),
        actual_index_unchanged=True, source_freeze_sha256=runner.sha(RESEARCH/'source-freeze.json'),
        plan_sha256=plans, old_qa_rows=qa_rows, new_cases_generated=False,
        note='HEAD publication check remains mandatory after root commits; this preflight proves staged byte preservation using an isolated index.')
    runner.write_new(RESEARCH/'freeze-preflight.json', check)
    print(json.dumps(check))


if __name__ == '__main__':
    if sys.argv[1:] == ['--verify-existing']:
        verify_existing()
    elif not sys.argv[1:]:
        prepare_metadata()
        verify_existing()
    else:
        raise SystemExit('Only --verify-existing is supported')
