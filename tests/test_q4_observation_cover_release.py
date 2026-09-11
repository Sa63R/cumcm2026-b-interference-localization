"""Synthetic envelope tests: no policy, simulator or 633 scene is constructed."""
import copy
import gzip
import hashlib
import json
import zipfile
import pytest
from experiments import run_q4_per_source as r
from experiments import q4_observation_cover_release as q


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def record_write(path, value):
    with gzip.open(path, 'wt', encoding='utf-8') as stream: json.dump(value, stream)


def record_read(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream: return json.load(stream)


def observation_cover_pass():
    return dict(passed=True, generic=dict(passed=True), prefix=dict(passed=True,
        **{k:dict(passed=True) for k in ('r12','r8','range','scheduling')}))


def rebind(folder):
    audit = q.read(folder/'independent_audit.json')
    inputs = list((folder/'records').glob('*.json.gz')) + [folder/name for name in
        ('manifest.json','freeze.json','source.zip','summary.json','plan.json')]
    audit['input_sha256'] = {p.relative_to(folder).as_posix():q.sha(p) for p in inputs}
    write(folder/'independent_audit.json', audit)


def summary_from_records(folder, infrastructure_errors=None):
    plan = q.read(folder/'plan.json')
    rows = [record_read(folder/'records'/f"{plan['label']}-{s}.json.gz")['row'] for s in plan['seed_selection']['seeds']]
    summary = r.summarize(rows, plan['split'], plan['seed_selection']['seeds'], check_generator_counts=True)
    infrastructure_errors = [] if infrastructure_errors is None else infrastructure_errors
    summary.update(complete=not infrastructure_errors, source_unchanged=True, infrastructure_errors=infrastructure_errors)
    write(folder/'summary.json', summary)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    # Trusted physical-audit envelopes are synthetic; the real auditor has its
    # own constructed physics tests. Here we test bindings, failure retention,
    # selection arithmetic and the actual release code, with short CI loops.
    monkeypatch.setattr(r, 'BOOTSTRAP_SAMPLES', 20)
    source = b'synthetic frozen metadata fixture\n'
    identity = {'fixture.py':hashlib.sha256(source).hexdigest()}
    monkeypatch.setattr(r, 'source_hashes', lambda:identity)
    (tmp_path/'fixture.py').write_bytes(source)
    directories = {}
    for label, spec in q.FIXED_SPECS.items():
        directories[label] = {}
        for split in q.SPLITS:
            folder = tmp_path/label/split
            (folder/'records').mkdir(parents=True)
            directories[label][split] = folder
            plan = r.make_plan(split, {label:spec}, identity)
            write(folder/'plan.json', plan)
            manifest = dict(plan=plan, plan_sha256=q.sha(folder/'plan.json'), release_sha256=None)
            write(folder/'manifest.json', manifest)
            write(folder/'freeze.json', dict(manifest_sha256=r.digest(manifest), git_commit='a'*40))
            with zipfile.ZipFile(folder/'source.zip', 'w') as archive: archive.writestr('fixture.py', source)
            items = []
            for seed in plan['seed_selection']['seeds']:
                n = r.count_from_seed(seed)
                truth = dict(synthetic_identity_only=seed, split=split)
                row = dict(seed=seed,strategy=label,stage=r.DESIGNS[split]['stage'],source_total=n,
                    successful=True,all_cleared=True,completion_certified=True,accepted_exit=True,
                    cleared_total=n,virtual_time_s=490.*n,penalized_time_s=490.*n,
                    common_lower_bound_s=1000.,case_sha256=r.digest(truth),errors=[])
                record = dict(row=row,spec=spec,evaluation_phase='after_policy_termination',
                    evaluation=dict(ground_truth=truth),history=[],summary=dict(action_history=[],strategy_parameters={}))
                record_write(folder/'records'/f'{label}-{seed}.json.gz', record)
                items.append(dict(seed=seed,strategy=label,passed=True,errors=[],observation_cover=observation_cover_pass()))
            summary_from_records(folder)
            write(folder/'independent_audit.json', dict(all_passed=True,all_clear=True,records=len(items),
                passed_records=len(items),errors=[],plan_sha256=manifest['plan_sha256'],audits=items,input_sha256={}))
            rebind(folder)
    return tmp_path, directories, identity


def selection(evidence):
    root, directories, _ = evidence
    return q.build_selection(directories, root)


def released(evidence, label=None, split='confirmation'):
    root, _, identity = evidence
    chosen = selection(evidence)
    write(root/'selection.json', chosen)
    label = chosen['selected'] if label is None else label
    plan = r.make_plan(split, {label:q.FIXED_SPECS[label]}, identity)
    write(root/'independent-plan.json', plan)
    digest = q.sha(root/'independent-plan.json')
    return plan, digest, q.build_release(plan,digest,'selection.json',root)


def set_mean(folder, value):
    for path in (folder/'records').glob('*.json.gz'):
        record = record_read(path); row=record['row']
        row['virtual_time_s']=row['penalized_time_s']=value*row['source_total']
        record_write(path,record)
    summary_from_records(folder); rebind(folder)


@pytest.mark.parametrize('split', ['confirmation','stress'])
def test_all_four_batches_bound_and_tie_chooses_label(evidence, split):
    root, _, _ = evidence
    s=selection(evidence)
    assert s['passed'] and s['selected']=='compact_ring_28'
    assert len(s['read_set_sha256'])==238+24
    assert sum(b['runs'] for c in s['candidates'].values() for b in c['batches'].values())==238
    assert 'triggered_cases' not in s  # R25's action delta is not present here.
    plan,digest,release=released(evidence,split=split)
    assert len(release['development_evidence_sha256'])==263
    r.validate_release(plan,digest,release,root)


@pytest.mark.parametrize('a,b,winner', [((480.,480.),(481.,400.),'compact_ring_28'),
    ((490.,480.),(490.,470.),'compact_ring_31'), ((490.,480.),(490.,480.),'compact_ring_28')])
def test_exact_max_then_average_then_label_rule(evidence,a,b,winner):
    _,dirs,_=evidence
    for label,means in zip(q.FIXED_SPECS,(a,b)):
        for split,mean in zip(q.SPLITS,means):set_mean(dirs[label][split],mean)
    assert selection(evidence)['selected']==winner


def test_500_is_inclusive_but_both_groups_required(evidence):
    _,dirs,_=evidence
    for label in q.FIXED_SPECS:
        for split in q.SPLITS:set_mean(dirs[label][split],500.)
    assert selection(evidence)['selected']=='compact_ring_28'
    set_mean(dirs['compact_ring_28']['development-stress'],500.000001)
    assert selection(evidence)['selected']=='compact_ring_31'
    set_mean(dirs['compact_ring_31']['development'],500.000001)
    result=selection(evidence)
    assert not result['passed'] and result['selected'] is None and result['selected_spec']=={}
    with pytest.raises(ValueError,match='recomputed passed'):released(evidence,label='compact_ring_28')


@pytest.mark.parametrize('kind',['case_failure','audit_failure','infrastructure_failure'])
def test_failed_candidate_retained_while_other_can_win(evidence,kind):
    _,dirs,_=evidence;folder=dirs['compact_ring_28']['development']
    plan=q.read(folder/'plan.json');seed=plan['seed_selection']['seeds'][0]
    path=folder/'records'/f'compact_ring_28-{seed}.json.gz'
    record=record_read(path);audit=q.read(folder/'independent_audit.json')
    if kind in ('case_failure','infrastructure_failure'):
        record['row'].update(successful=False,completion_certified=False,penalized_time_s=360000.)
        audit['all_clear']=False
    if kind=='infrastructure_failure':
        record=dict(row=dict(record['row'],virtual_time_s=None,infrastructure_error=True),record_kind='infrastructure_failure_without_completed_case')
    if kind in ('audit_failure','infrastructure_failure'):
        audit['audits'][0].update(passed=False,errors=['physical audit rejected'])
        audit['audits'][0].pop('observation_cover')
        audit.update(all_passed=False,passed_records=len(audit['audits'])-1)
    record_write(path,record);write(folder/'independent_audit.json',audit)
    summary_from_records(folder, [{'seed':seed,'error':'missing completed case'}] if kind=='infrastructure_failure' else None)
    rebind(folder)
    result=selection(evidence)
    assert result['selected']=='compact_ring_31' and not result['candidates']['compact_ring_28']['eligible']
    bad=result['candidates']['compact_ring_28']['batches']['development']
    assert bad['runs']==70 and len(result['read_set_sha256'])==262
    if kind!='audit_failure': assert seed in bad['failed_seeds']
    if kind!='case_failure': assert seed in bad['audit_failed_seeds']


@pytest.mark.parametrize('tamper',['candidate_missing','split_missing','directory_wrong_label','record_missing',
    'record_extra','audit_missing','audit_item_missing','audit_item_duplicate','nested_missing','input_missing',
    'input_sha','summary_mean','row_seed','case_pair','manifest','freeze','zip_bytes','zip_duplicate','source_plan'])
def test_complete_matrix_and_evidence_tampering_rejected(evidence,tamper):
    root,dirs,_=evidence
    folder=dirs['compact_ring_28']['development'];plan=q.read(folder/'plan.json')
    path=folder/'records'/f"compact_ring_28-{plan['seed_selection']['seeds'][0]}.json.gz"
    audit=q.read(folder/'independent_audit.json');rehash=True
    if tamper=='candidate_missing':dirs.pop('compact_ring_31')
    elif tamper=='split_missing':dirs['compact_ring_31'].pop('development-stress')
    elif tamper=='directory_wrong_label':dirs['compact_ring_31']['development']=folder
    elif tamper=='record_missing':path.unlink()
    elif tamper=='record_extra':(folder/'records'/'extra.json.gz').write_bytes(path.read_bytes())
    elif tamper=='audit_missing':(folder/'independent_audit.json').unlink();rehash=False
    elif tamper=='audit_item_missing':audit['audits'].pop()
    elif tamper=='audit_item_duplicate':audit['audits'][0]=audit['audits'][1]
    elif tamper=='nested_missing':audit['audits'][0]['observation_cover']['prefix'].pop('r8')
    elif tamper=='input_missing':audit['input_sha256'].pop('source.zip');rehash=False
    elif tamper=='input_sha':audit['input_sha256']['source.zip']='bad';rehash=False
    elif tamper=='summary_mean':
        v=q.read(folder/'summary.json');v['mean_time_per_source_s']=1.;write(folder/'summary.json',v)
    elif tamper in ('row_seed','case_pair'):
        v=record_read(path)
        if tamper=='row_seed':v['row']['seed']+=10000
        else:
            v['evaluation']['ground_truth']['changed_world']=True
            v['row']['case_sha256']=r.digest(v['evaluation']['ground_truth'])
        record_write(path,v)
        if tamper=='case_pair':summary_from_records(folder)
    elif tamper=='manifest':
        v=q.read(folder/'manifest.json');v['release_sha256']='fake';write(folder/'manifest.json',v)
    elif tamper=='freeze':write(folder/'freeze.json',dict(manifest_sha256='bad',git_commit='a'*40))
    elif tamper in ('zip_bytes','zip_duplicate'):
        with zipfile.ZipFile(folder/'source.zip','w' if tamper=='zip_bytes' else 'a') as z:
            if tamper=='zip_duplicate':
                with pytest.warns(UserWarning):z.writestr('fixture.py','duplicate')
            else:z.writestr('fixture.py','changed')
    elif tamper=='source_plan':plan['source_sha256']={};write(folder/'plan.json',plan)
    if tamper!='audit_missing':write(folder/'independent_audit.json',audit)
    if rehash:rebind(folder)
    with pytest.raises((ValueError,FileNotFoundError)):q.build_selection(dirs,root)


def test_release_refuses_loser_even_though_it_is_eligible(evidence):
    with pytest.raises(ValueError,match='selected winner'):released(evidence,label='compact_ring_31')


@pytest.mark.parametrize('field',['authorized','source_sha256','reserved_seeds','evidence','stored_winner','stored_criteria'])
def test_release_recomputes_choice_and_binds_losing_arm_too(evidence,field):
    root,_,_=evidence;plan,digest,release=released(evidence)
    if field=='authorized':release[field]=False
    elif field in ('source_sha256','reserved_seeds'):release[field]={} if field=='source_sha256' else []
    elif field=='evidence':
        release['development_evidence_sha256']={k:v for k,v in release['development_evidence_sha256'].items() if 'compact_ring_31' not in k}
    else:
        selected=q.read(root/'selection.json')
        if field=='stored_winner':selected['selected']='compact_ring_31'
        else:selected['candidates']['compact_ring_28']['eligible']=False
        write(root/'selection.json',selected)
        release['development_evidence_sha256']['selection.json']=q.sha(root/'selection.json')
    with pytest.raises(ValueError):r.validate_release(plan,digest,release,root)


@pytest.mark.parametrize('change',['label','entrypoint','config','max_expansions','multiarms'])
def test_fixed_spec_family_only(change):
    spec={'compact_ring_28':copy.deepcopy(q.FIXED_SPECS['compact_ring_28'])}
    if change=='label':spec['ring_28']=spec.pop('compact_ring_28')
    elif change=='multiarms':spec=copy.deepcopy(q.FIXED_SPECS)
    elif change=='entrypoint':spec['compact_ring_28']['entrypoint']='strategies.q4_joint_continuation:run_q4_joint_continuation'
    else:spec['compact_ring_28']['kwargs'][change]='wrong'
    with pytest.raises(ValueError):r.validate_spec(spec)


def test_fixed_633_matrices_are_disjoint_and_shared_between_candidates():
    matrices=[r.seed_selection(split)['seeds'] for split in r.DESIGNS]
    assert list(map(len,matrices))==[70,49,140,98]
    assert len(set(sum(matrices,[])))==sum(map(len,matrices))
    assert all(6330001<=s<=6335000 for seeds in matrices for s in seeds)
    for split in r.DESIGNS:
        plans=[r.make_plan(split,{label:spec},{}) for label,spec in q.FIXED_SPECS.items()]
        assert plans[0]['seed_selection']==plans[1]['seed_selection']
