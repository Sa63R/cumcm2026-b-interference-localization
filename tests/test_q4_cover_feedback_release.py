"""Synthetic envelope tests: no policy, simulator or 639 scene is constructed."""
import copy
import gzip
import hashlib
import json
import zipfile
import pytest
from experiments import run_q4_per_source as r
from experiments import q4_cover_feedback_release as q


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def record_write(path, value):
    with gzip.open(path, 'wt', encoding='utf-8') as stream: json.dump(value, stream)


def record_read(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream: return json.load(stream)


def cover_feedback_pass(count=0):
    return dict(passed=True, generic=dict(passed=True), prefix=dict(passed=True, cover_feedback_vetoes=count,
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


LABEL = 'compact_cover_feedback'


def action_fixture(count):
    actions=[dict(action='measure',channel=20,position=[3.,4.],result='no_signal',virtual_time_s=float(6*(i+1)),phase='coverage') for i in range(count)]
    routes=[dict(after_actual_action_count=i,selected_kind='source',selected_channel=i+1,remaining_covers=[[3.,4.]],
        source_channels=[i+1],result=dict(order=[['source',0],['cover',0]])) for i in range(max(count,1))]
    events=[dict(id=i,route_index=i,channel=i+1,next_cover=[3.,4.],veto_selected=True,executed_cover=True,status='completed',
        actual_vetoes_before=i,actual_vetoes_after=i+1,after_actual_action_count=i,end_actual_action_count=i+1) for i in range(count)]
    if not events:
        events=[dict(id=0,route_index=0,channel=1,next_cover=[3.,4.],veto_selected=True,executed_cover=False,
            actual_vetoes_before=0,actual_vetoes_after=0,after_actual_action_count=0,end_actual_action_count=0)]
    actions.append(dict(action='measure',channel=1,position=[9.,9.],result='no_signal',virtual_time_s=float(6*(count+1)),phase='coverage'))
    wire=[dict(action='/measure',channel=a['channel'],position=dict(x=a['position'][0],y=a['position'][1]),
        response=dict(accepted=True,measure_result=a['result'],virtual_time_s=a['virtual_time_s'])) for a in actions]
    return dict(history=wire,summary=dict(action_history=actions,strategy_parameters=dict(cover_feedback_log=events,chain_route_log=routes)))


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
                triggered = split == 'development' and seed in plan['seed_selection']['seeds'][:5]
                record = dict(row=row,spec=spec,evaluation_phase='after_policy_termination',
                    evaluation=dict(ground_truth=truth), **action_fixture(int(triggered)))
                record_write(folder/'records'/f'{label}-{seed}.json.gz', record)
                items.append(dict(seed=seed,strategy=label,passed=True,errors=[],cover_feedback=cover_feedback_pass(int(triggered))))
            summary_from_records(folder)
            write(folder/'independent_audit.json', dict(all_passed=True,all_clear=True,records=len(items),
                passed_records=len(items),errors=[],plan_sha256=manifest['plan_sha256'],audits=items,input_sha256={}))
            rebind(folder)
    return tmp_path, directories, identity


def selection(evidence):
    root, directories, _ = evidence
    return q.build_selection(directories[LABEL]['development'], directories[LABEL]['development-stress'], root)


def released(evidence, label=None, split='confirmation'):
    root, _, identity = evidence
    chosen = selection(evidence)
    write(root/'selection.json', chosen)
    label = LABEL if label is None else label
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


@pytest.mark.parametrize('split',['confirmation','stress'])
def test_complete_single_candidate_development_and_actual_release(evidence,split):
    root,dirs,_=evidence;s=selection(evidence)
    assert s['passed'] and s['selected']==LABEL and s['triggered_cases']==5
    assert s['development']['runs']==70 and s['development_stress']['runs']==49
    assert len(s['read_set_sha256'])==119+12
    plan,digest,release=released(evidence,split=split)
    assert len(release['development_evidence_sha256'])==132
    r.validate_release(plan,digest,release,root)


def test_zero_action_attempt_and_return_scan_are_not_triggers():
    assert q.actual_service_count(action_fixture(0))==0
    assert q.actual_service_count(action_fixture(2))==2


@pytest.mark.parametrize('change',['overlap','scan','rejected','point','response','clock','range','bool'])
def test_actual_service_count_must_be_wire_bound(change):
    x=action_fixture(1);event=x['summary']['strategy_parameters']['cover_feedback_log'][0]
    if change=='overlap':x['summary']['strategy_parameters']['cover_feedback_log'].append(copy.deepcopy(event))
    elif change=='scan':event['end_actual_action_count']=2;event['scan_start_action_count']=2
    elif change=='rejected':x['history'][0]['response']['accepted']=False
    elif change=='point':x['history'][0]['position']['x']=5.
    elif change=='response':x['history'][0]['response']['measure_result']='near'
    elif change=='clock':x['history'][0]['response']['virtual_time_s']=999.
    elif change=='range':event['end_actual_action_count']=100
    else:event['after_actual_action_count']=False
    with pytest.raises(ValueError):q.actual_service_count(x)


def test_both_means_required_and_500_inclusive(evidence):
    _,dirs,_=evidence
    for split in q.SPLITS:set_mean(dirs[LABEL][split],500.)
    assert selection(evidence)['passed']
    set_mean(dirs[LABEL]['development-stress'],500.000001)
    s=selection(evidence);assert not s['passed'] and s['selected'] is None and s['selected_spec']=={}
    with pytest.raises(ValueError,match='recomputed passed'):released(evidence)


def test_fewer_than_five_distinct_real_cases_fails(evidence):
    _,dirs,_=evidence;folder=dirs[LABEL]['development'];plan=q.read(folder/'plan.json')
    seed=plan['seed_selection']['seeds'][0];path=folder/'records'/f'{LABEL}-{seed}.json.gz'
    record=record_read(path);record.update(action_fixture(0));record_write(path,record)
    audit=q.read(folder/'independent_audit.json');audit['audits'][0]['cover_feedback']['prefix']['cover_feedback_vetoes']=0
    write(folder/'independent_audit.json',audit);rebind(folder)
    assert selection(evidence)['triggered_cases']==4 and not selection(evidence)['passed']


@pytest.mark.parametrize('kind',['case_failure','audit_failure','infrastructure_failure'])
def test_failures_retained_not_replaced_by_smaller_complete_set(evidence,kind):
    _,dirs,_=evidence;folder=dirs[LABEL]['development'];plan=q.read(folder/'plan.json');seed=plan['seed_selection']['seeds'][0]
    path=folder/'records'/f'{LABEL}-{seed}.json.gz';record=record_read(path);audit=q.read(folder/'independent_audit.json')
    if kind in ('case_failure','infrastructure_failure'):
        record['row'].update(successful=False,completion_certified=False,penalized_time_s=360000.)
        audit['all_clear']=False
    if kind=='infrastructure_failure':
        record=dict(row=dict(record['row'],virtual_time_s=None,infrastructure_error=True),record_kind='infrastructure_failure_without_completed_case')
    if kind in ('audit_failure','infrastructure_failure'):
        audit['audits'][0].update(passed=False,errors=['physical audit rejected'])
        audit['audits'][0].pop('cover_feedback');audit.update(all_passed=False,passed_records=len(audit['audits'])-1)
    record_write(path,record);write(folder/'independent_audit.json',audit)
    summary_from_records(folder,[{'seed':seed,'error':'missing completed case'}] if kind=='infrastructure_failure' else None);rebind(folder)
    s=selection(evidence);assert not s['passed'] and s['development']['runs']==70 and len(s['read_set_sha256'])==131
    if kind!='audit_failure':assert seed in s['development']['failed_seeds']
    if kind!='case_failure':assert seed in s['development']['audit_failed_seeds']


@pytest.mark.parametrize('tamper',['record_missing','audit_item_missing','audit_item_duplicate','nested_missing','trigger_counter','trigger_bool',
    'input_sha','summary_mean','row_seed','case_sha','manifest','freeze','zip_bytes','zip_duplicate','source_plan'])
def test_complete_evidence_tampering_rejected(evidence,tamper):
    root,dirs,_=evidence;folder=dirs[LABEL]['development'];plan=q.read(folder/'plan.json');path=folder/'records'/f"{LABEL}-{plan['seed_selection']['seeds'][0]}.json.gz"
    audit=q.read(folder/'independent_audit.json');rehash=True
    if tamper=='record_missing':path.unlink()
    elif tamper=='audit_item_missing':audit['audits'].pop()
    elif tamper=='audit_item_duplicate':audit['audits'][0]=audit['audits'][1]
    elif tamper=='nested_missing':audit['audits'][0]['cover_feedback']['prefix'].pop('r8')
    elif tamper=='trigger_counter':audit['audits'][0]['cover_feedback']['prefix']['cover_feedback_vetoes']=999
    elif tamper=='trigger_bool':audit['audits'][0]['cover_feedback']['prefix']['cover_feedback_vetoes']=True
    elif tamper=='input_sha':audit['input_sha256']['source.zip']='bad';rehash=False
    elif tamper=='summary_mean':
        value=q.read(folder/'summary.json');value['mean_time_per_source_s']=1.;write(folder/'summary.json',value)
    elif tamper in ('row_seed','case_sha'):
        value=record_read(path)
        if tamper=='row_seed':value['row']['seed']+=10000
        else:value['row']['case_sha256']='invalid'
        record_write(path,value)
    elif tamper=='manifest':
        value=q.read(folder/'manifest.json');value['release_sha256']='fake';write(folder/'manifest.json',value)
    elif tamper=='freeze':write(folder/'freeze.json',dict(manifest_sha256='bad',git_commit='a'*40))
    elif tamper in ('zip_bytes','zip_duplicate'):
        with zipfile.ZipFile(folder/'source.zip','w' if tamper=='zip_bytes' else 'a') as z:
            if tamper=='zip_duplicate':
                with pytest.warns(UserWarning):z.writestr('fixture.py','duplicate')
            else:z.writestr('fixture.py','changed')
    else:plan['source_sha256']={};write(folder/'plan.json',plan)
    write(folder/'independent_audit.json',audit)
    if rehash:rebind(folder)
    with pytest.raises((ValueError,FileNotFoundError)):selection(evidence)


@pytest.mark.parametrize('field',['authorized','source_sha256','reserved_seeds','evidence','stored_criteria'])
def test_release_recomputes_complete_development_not_claimed_pass(evidence,field):
    root,_,_=evidence;plan,digest,release=released(evidence)
    if field=='authorized':release[field]=False
    elif field in ('source_sha256','reserved_seeds'):release[field]={} if field=='source_sha256' else []
    elif field=='evidence':release['development_evidence_sha256'].pop(next(iter(release['development_evidence_sha256'])))
    else:
        s=q.read(root/'selection.json');s['criteria']['complete_and_audited']=False;write(root/'selection.json',s)
        release['development_evidence_sha256']['selection.json']=q.sha(root/'selection.json')
    with pytest.raises(ValueError):r.validate_release(plan,digest,release,root)


@pytest.mark.parametrize('change',['label','entrypoint','config','max_expansions','multiarms'])
def test_only_fixed_single_method(change):
    spec=copy.deepcopy(q.FIXED_SPECS)
    if change=='label':spec['wrong']=spec.pop(LABEL)
    elif change=='multiarms':spec['other']=copy.deepcopy(spec[LABEL])
    elif change=='entrypoint':spec[LABEL]['entrypoint']='strategies.q4_joint_continuation:run_q4_joint_continuation'
    else:spec[LABEL]['kwargs'][change]='wrong'
    with pytest.raises(ValueError):r.validate_spec(spec)


def test_fixed_639_matrix_sizes_and_disjoint_ranges_metadata_only():
    matrices=[r.seed_selection(split)['seeds'] for split in r.DESIGNS]
    assert list(map(len,matrices))==[70,49,140,98]
    assert len(set(sum(matrices,[])))==sum(map(len,matrices))
    assert all(6390001<=seed<=6395000 for seeds in matrices for seed in seeds)


@pytest.mark.parametrize('change',['bool_selected','bool_counter','negative_counter','wrong_before','too_many','wrong_station','wrong_order','wrong_channel','missing_route','false_counter'])
def test_owned_cover_veto_fail_closed(change):
    record=action_fixture(1);event=record['summary']['strategy_parameters']['cover_feedback_log'][0]
    if change=='bool_selected':event['veto_selected']=1
    elif change=='bool_counter':event['actual_vetoes_after']=True
    elif change=='negative_counter':event['actual_vetoes_after']=-1
    elif change=='wrong_before':event['actual_vetoes_before']=1
    elif change=='too_many':event['actual_vetoes_after']=9
    elif change=='wrong_station':event['next_cover']=[4.,4.]
    elif change=='wrong_order':record['summary']['strategy_parameters']['chain_route_log'][0]['result']['order'].reverse()
    elif change=='wrong_channel':event['channel']=4
    elif change=='missing_route':event['route_index']=100
    else:event['actual_vetoes_after']=0
    with pytest.raises(ValueError):q.actual_service_count(record)


def test_partial_scan_is_retained_but_not_counted_as_completed_veto():
    record=action_fixture(1);event=record['summary']['strategy_parameters']['cover_feedback_log'][0]
    event.update(executed_cover=False,actual_vetoes_after=0)
    assert q.actual_service_count(record)==0
    event.update(veto_selected=False,executed_cover=True)
    with pytest.raises(ValueError):q.actual_service_count(record)


def test_same_source_cannot_be_vetoed_twice_even_with_different_original_routes():
    record=action_fixture(2);p=record['summary']['strategy_parameters']
    p['cover_feedback_log'][1]['channel']=1
    p['chain_route_log'][1].update(selected_channel=1,source_channels=[1])
    with pytest.raises(ValueError):q.actual_service_count(record)


def test_original_cover_completion_does_not_count_as_a_veto():
    record=action_fixture(1);params=record['summary']['strategy_parameters']
    params['cover_feedback_log'][0].update(veto_selected=False,actual_vetoes_after=0)
    params['chain_route_log'][0].update(selected_kind='cover',selected_channel=None)
    assert q.actual_service_count(record)==0


def test_interrupted_cover_cannot_claim_completion():
    record=action_fixture(1)
    record['summary']['strategy_parameters']['cover_feedback_log'][0]['status']='interrupted'
    with pytest.raises(ValueError):q.actual_service_count(record)
