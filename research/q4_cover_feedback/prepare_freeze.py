"""Freeze complete tests and two old QA traces before any 639 scenario is run."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments import run_q4_per_source as runner
from experiments.audit_q4_cover_feedback import audit_full, verify_source_contract
from experiments.q4_cover_feedback_release import actual_service_count

BASE='81aa6e1a1231a2358cf098fbba3fdd0557b540ef'
RESEARCH=ROOT/'research/q4_cover_feedback'


def git(*args,env=None):
    return subprocess.check_output(['git',*args],cwd=ROOT,env=env)


def verify_base():
    paths=git('ls-tree','-r','--name-only',BASE,'src').decode().splitlines()
    assert len(paths)==49 and sum(p.endswith('.py') for p in paths)==44
    for path in paths:
        assert git('show',f'{BASE}:{path}')==(ROOT/path).read_bytes(),path
    for imported in runner.read(RESEARCH/'imports.json'):
        assert runner.sha(ROOT/imported['path'])==imported['sha256']
        assert git('show',imported['commit']+':'+imported['path'])==(ROOT/imported['path']).read_bytes()
    verify_source_contract()


def prepare():
    assert not (RESEARCH/'source-freeze.json').exists(),'Preserve previous freeze'
    verify_base()
    tests=(RESEARCH/'final-contract-tests-console.txt').read_text(encoding='utf-8-sig')
    assert 'passed' in tests and 'FAILED' not in tests and 'ERROR ' not in tests
    identity=runner.source_hashes()
    spec=runner.read(RESEARCH/'spec.json');runner.validate_spec(spec)
    qa=RESEARCH/'old-smoke'
    manifest=runner.read(qa/'manifest.json');summary=runner.read(qa/'summary.json')
    assert manifest['seeds']==[621003,621013] and manifest['stage']=='pilot'
    assert {manifest['label']:manifest['spec']}==spec and manifest['source_sha256']==identity
    assert summary['source_unchanged'] is True
    with zipfile.ZipFile(qa/'source.zip') as archive:
        assert set(archive.namelist())==set(identity) and len(archive.namelist())==len(identity)
        for path,expected in identity.items():
            assert hashlib.sha256(archive.read(path)).hexdigest()==expected,path
    rows=[]
    for seed in (621003,621013):
        raw=qa/f'compact_cover_feedback-{seed}.json.gz'
        record=json.loads(gzip.decompress(raw.read_bytes()))
        first=runner.read(qa/f'{seed}-first-audit.json')
        assert first['passed'] is True and first['input_sha256']==runner.sha(raw)
        assert record['row']['successful'] is True and record['spec']==next(iter(spec.values()))
        final=audit_full(record)
        assert final['passed'] is True
        count=actual_service_count(record)
        assert count==final['prefix']['cover_feedback_vetoes']==first['prefix']['cover_feedback_vetoes']
        runner.write_new(qa/f'{seed}-final-contract-audit.json',dict(**final,input_sha256=runner.sha(raw),final_source_sha256=identity))
        rows.append(dict(seed=seed,first_audit_passed=True,final_audit_passed=True,actual_cover_vetoes=count,raw_sha256=runner.sha(raw)))
    directory=RESEARCH/'plans';directory.mkdir(exist_ok=False)
    plans={}
    for split in ('development','development-stress'):
        plan=runner.make_plan(split,spec,identity);runner.validate_plan(plan,identity)
        path=directory/f'{split}.json';runner.write_new(path,plan)
        plans[path.relative_to(ROOT).as_posix()]=runner.sha(path)
    assert sum(len(runner.read(ROOT/path)['seed_selection']['seeds']) for path in plans)==119
    assert identity==runner.source_hashes()
    result=dict(base_commit=BASE,source_sha256=identity,runtime_file_count=len(identity),
        all_49_base_src_byte_identical=True,base_src_file_count=49,base_src_python_count=44,
        imported_sources=runner.read(RESEARCH/'imports.json'),plan_sha256=plans,
        new_cases_generated=False,independent_plans_written=False,old_qa_rows=rows,
        old_qa_files_sha256={p.relative_to(ROOT).as_posix():runner.sha(p) for p in sorted(qa.iterdir()) if p.is_file()},
        test_output_sha256=runner.sha(RESEARCH/'final-contract-tests-console.txt'))
    runner.write_new(RESEARCH/'source-freeze.json',result)


def verify(published=False):
    verify_base()
    freeze=runner.read(RESEARCH/'source-freeze.json');identity=runner.source_hashes()
    assert freeze['source_sha256']==identity
    for path,expected in freeze['plan_sha256'].items():
        assert runner.sha(ROOT/path)==expected
        runner.validate_plan(runner.read(ROOT/path),identity)
    for path,expected in freeze['old_qa_files_sha256'].items():
        assert runner.sha(ROOT/path)==expected
    assert runner.sha(RESEARCH/'final-contract-tests-console.txt')==freeze['test_output_sha256']
    if published:
        branch=git('branch','--show-current').decode().strip()
        assert git('rev-parse','HEAD')==git('rev-parse','origin/'+branch)
        for path in set(identity)|set(freeze['old_qa_files_sha256'])|set(freeze['plan_sha256'])|{
                'research/q4_cover_feedback/source-freeze.json','research/q4_cover_feedback/final-contract-tests-console.txt'}:
            assert git('show','HEAD:'+path)==(ROOT/path).read_bytes(),path
        print(json.dumps(dict(passed=True,git_commit=git('rev-parse','HEAD').decode().strip(),runtime_files=len(identity),base_files_exact=49,
            source_freeze_sha256=runner.sha(RESEARCH/'source-freeze.json'),source_unchanged=True,scope='Published HEAD/origin and exact runtime, QA and plan bytes')))
        return
    scratch=ROOT.parent/'q3-v1-artifacts/freeze-checks-20260912/r39';scratch.mkdir(parents=True,exist_ok=True)
    serial=0
    while (scratch/f'index-{serial:02d}').exists():serial+=1
    index=scratch/f'index-{serial:02d}'
    real_index=Path(git('rev-parse','--git-path','index').decode().strip())
    if not real_index.is_absolute():real_index=ROOT/real_index
    before=real_index.read_bytes();env=dict(os.environ,GIT_INDEX_FILE=str(index))
    git('read-tree','HEAD',env=env)
    git('add','--','.gitattributes','src/planning/q4_conditional_belief.py','src/planning/q4_cover_feedback.py',
        'src/strategies/q4_cover_feedback.py','src/planning/matrix_chain_route.py','src/planning/release_chain_route.py','experiments/run_q4_per_source.py','experiments/audit_q4_per_source.py',
        'experiments/audit_q4_cover_feedback.py','experiments/q4_cover_feedback_release.py',
        'tests/test_q4_cover_feedback.py','tests/test_q4_cover_feedback_model.py','tests/test_audit_q4_cover_feedback.py',
        'tests/test_q4_per_source.py','tests/test_q4_cover_feedback_release.py','research/q4_cover_feedback',env=env)
    git('diff','--cached','--check',env=env)
    changed=git('diff','--cached','--name-only',env=env).decode().splitlines()
    for path in set(changed)|set(identity):
        assert git('show',':'+path,env=env)==(ROOT/path).read_bytes(),path
    assert real_index.read_bytes()==before
    assert runner.source_hashes()==identity
    result=dict(passed=True,runtime_file_count=len(identity),base_files_exact=49,temporary_index_working_bytes_exact=True,
        checked_paths=len(set(changed)|set(identity)),actual_index_unchanged=True,new_cases_generated=False,
        source_freeze_sha256=runner.sha(RESEARCH/'source-freeze.json'),plan_sha256=freeze['plan_sha256'],old_qa_rows=freeze['old_qa_rows'])
    runner.write_new(RESEARCH/'freeze-preflight.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    if sys.argv[1:]==['--verify-existing']:verify()
    elif sys.argv[1:]==['--published']:verify(published=True)
    elif not sys.argv[1:]:prepare();verify()
    else:raise SystemExit('Expected no option, --verify-existing or --published')
