"""Metadata-only R29 freeze preparation; never constructs or runs a scenario."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.run_q4_per_source import make_plan,validate_plan,source_hashes,write_new,read,sha
from experiments.audit_q4_observation_cover import verify_source_contract,SOURCE_CONTRACT

BASE='81aa6e1a1231a2358cf098fbba3fdd0557b540ef'


def git(*args,env=None):
    return subprocess.run(['git','-C',str(ROOT),*args],capture_output=True,check=True,env=env)


def exact(cwd_ref,name):
    return git('cat-file','blob',f'{cwd_ref}:{name}').stdout


def main():
    plans=HERE/'plans'
    destinations=(plans,HERE/'source-freeze.json',HERE/'freeze-preflight.json')
    if any(p.exists() for p in destinations):raise ValueError('Refuse existing freeze/plan overwrite')
    if (ROOT/'results/q4_observation_cover').exists():raise ValueError('New performance output already exists')
    verify_source_contract()
    actual_index=Path(git('rev-parse','--git-path','index').stdout.decode().strip())
    if not actual_index.is_absolute():actual_index=ROOT/actual_index
    index_before=actual_index.read_bytes()
    head_before=git('rev-parse','HEAD').stdout.decode().strip()
    base_files=git('ls-tree','-r','--name-only',BASE,'src').stdout.decode().splitlines()
    assert len(base_files)==49 and sum(p.endswith('.py') for p in base_files)==44
    base_hashes={}
    for name in base_files:
        committed=exact(BASE,name)
        assert (ROOT/name).read_bytes()==committed==exact('HEAD',name),name
        base_hashes[name]=hashlib.sha256(committed).hexdigest()
    assert len(SOURCE_CONTRACT)==51
    identity=source_hashes()
    plans.mkdir()
    plan_hashes={}; selections={}; counts={}
    for config in ('ring_28','ring_31'):
        spec=read(HERE/f'spec-{config}.json')
        for split in ('development','development-stress','confirmation','stress'):
            plan=make_plan(split,spec)
            assert plan['source_sha256']==identity
            validate_plan(plan)
            path=plans/f'{config}-{split}.json'
            write_new(path,plan)
            plan_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
            selected=plan['seed_selection']
            assert len(selected['trace'])==1000
            if split in selections:assert selections[split]==selected
            else:selections[split]=selected
            counts[f'{config}/{split}']=len(selected['seeds'])
    expected={'development':70,'development-stress':49,'confirmation':140,'stress':98}
    for split,n in expected.items():assert len(selections[split]['seeds'])==n
    for i,split in enumerate(expected):
        for other in list(expected)[i+1:]:
            assert not set(selections[split]['seeds']) & set(selections[other]['seeds'])
    qa_files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted((HERE/'old-smoke').rglob('*')) if p.is_file()}
    qa=read(HERE/'old-smoke/summary.json')
    assert len(qa['rows'])==2
    for row in qa['rows']:
        assert row['seed']==621001 and row['successful'] is True and row['audit_passed'] is True
        assert read(HERE/'old-smoke'/f"{row['strategy']}-audit.json")['passed'] is True
    freeze=dict(base_commit=BASE,all_49_base_src_byte_identical=True,source_sha256=identity,
        base_src_file_count=49,base_src_python_count=44,new_src_contract_files=51,
        test_groups=dict(strategy_planner_inherited=95,geometry_prefix_audit=60,runner_selection_release=85),
        old_smoke_seeds=[621001],old_smoke_configs=['ring_28','ring_31'],
        old_smoke_full_clear_and_audited=True,old_smoke_input_sha256=qa_files,
        plan_sha256=plan_hashes,new_cases_generated=False)
    write_new(HERE/'source-freeze.json',freeze)

    # An isolated index exercises the actual checkout-to-Git clean filters
    # without staging anything in the shared real index or changing refs.
    scratch=ROOT.parent/'q3-v1-artifacts/freeze-checks-20260912/r29'
    scratch.mkdir(parents=True,exist_ok=False)
    temporary_index=scratch/'index'
    env=dict(os.environ,GIT_INDEX_FILE=str(temporary_index))
    git('read-tree','HEAD',env=env)
    owned=['.gitattributes','src/planning/observation_cover_route.py','src/strategies/q4_observation_cover.py',
        'experiments/audit_q4_observation_cover.py','experiments/audit_q4_per_source.py',
        'experiments/q4_observation_cover_release.py','experiments/run_q4_per_source.py',
        'tests/test_audit_q4_observation_cover.py','tests/test_audit_q4_per_source.py',
        'tests/test_observation_cover_route.py','tests/test_q4_observation_cover.py',
        'tests/test_q4_observation_cover_release.py','tests/test_q4_per_source.py',
        'research/q4_observation_cover']
    git('add','--',*owned,env=env)
    git('diff','--check')
    git('diff','--cached','--check',env=env)
    check_paths=set(identity)|set(plan_hashes)|set(qa_files)|set(base_hashes)
    check_paths.update(p.relative_to(ROOT).as_posix() for p in HERE.rglob('*') if p.is_file())
    index_hashes={}
    for name in sorted(check_paths):
        value=git('show',':'+name,env=env).stdout
        assert value==(ROOT/name).read_bytes(),('temporary_index_changed_bytes',name)
        index_hashes[name]=hashlib.sha256(value).hexdigest()
    assert source_hashes()==identity
    verify_source_contract()
    assert actual_index.read_bytes()==index_before and git('rev-parse','HEAD').stdout.decode().strip()==head_before
    preflight=dict(metadata_only=True,new_cases_generated=False,base_commit=BASE,head_at_preparation=head_before,
        runtime_hash_entries=len(identity),runtime_contract_exact=True,base_src_files=49,base_src_python_files=44,
        all_base_working_and_HEAD_blobs_identical=True,all_51_auditor_source_contract_bytes_identical=True,
        real_index_unchanged=True,temporary_index=str(temporary_index),
        git_diff_check_passed=True,temporary_index_diff_check_passed=True,
        temporary_index_checked_files=len(index_hashes),temporary_index_file_sha256=index_hashes,
        source_freeze_sha256=sha(HERE/'source-freeze.json'),plan_sha256=plan_hashes,plan_case_counts=counts,
        same_split_candidates_have_identical_seed_selection=True,four_split_seed_sets_disjoint=True,
        source_or_plan_committed=False,independent_releases_written=False)
    write_new(HERE/'freeze-preflight.json',preflight)
    # The preflight does not hash itself; nevertheless verify its clean filter.
    git('add','--','research/q4_observation_cover/freeze-preflight.json',env=env)
    assert git('show',':research/q4_observation_cover/freeze-preflight.json',env=env).stdout==(HERE/'freeze-preflight.json').read_bytes()
    git('diff','--cached','--check',env=env)
    assert actual_index.read_bytes()==index_before
    print(json.dumps(dict(plans=len(plan_hashes),runtime_hash_entries=len(identity),
        source_freeze_sha256=preflight['source_freeze_sha256'],plan_sha256=plan_hashes,
        case_counts=counts,checked_index_files=len(index_hashes)+1,metadata_only=True,real_index_unchanged=True)))


if __name__=='__main__':main()
