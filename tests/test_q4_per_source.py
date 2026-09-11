import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from experiments import run_q4_per_source as r

from experiments import q4_reference_release as reference
SPEC={reference.LABEL:copy.deepcopy(reference.SPEC)}
IDENTITY={'fake-source.py':'abc'}

@pytest.mark.parametrize('split,count', [('development',70),('development-stress',49),('confirmation',140),('stress',98)])
def test_first_public_count_draw_and_earliest_strata(split,count):
    with patch.object(r,'one',side_effect=AssertionError('no case construction in metadata')):
        s=r.seed_selection(split)
    assert len(s['seeds'])==count and len(s['trace'])==1000
    assert s['seeds']==sorted(s['seeds']) and all(v==r.DESIGNS[split]['quota'] for v in s['counts'].values())
    for key in s['counts']:
        eligible=[x['seed'] for x in s['trace'] if x['stratum']==key]
        chosen=[x['seed'] for x in s['trace'] if x['stratum']==key and x['accepted']]
        assert chosen==eligible[:r.DESIGNS[split]['quota']]
    assert all(x['reason']=='stratum_quota_full' for x in s['trace'] if not x['accepted'])

@pytest.mark.parametrize('seed,stage',[(621001,'pilot'),(621004,'pilot'),(621042,'stress')])
def test_only_previously_opened_generator_count_contract(seed,stage):
    from experiments.run_q4_state_study import make_case
    assert len(make_case(seed,stage).sources)==r.count_from_seed(seed)

@pytest.mark.parametrize('tamper',['seed','trace','quota','source','label'])
def test_plan_reconstructs_more_than_its_claimed_digest(tamper):
    p=r.make_plan('development',SPEC,IDENTITY);q=copy.deepcopy(p)
    if tamper=='seed':q['seed_selection']['seeds'][0]+=1
    elif tamper=='trace':q['seed_selection']['trace'][0]['reason']='invented'
    elif tamper=='quota':q['seed_selection']['design']['quota']=9
    elif tamper=='source':q['source_sha256']={'fake-source.py':'changed'}
    else:q['label']='../escape'
    with pytest.raises(ValueError):r.validate_plan(q,IDENTITY)

def row(seed,n,t,ok=True):
    return dict(seed=seed,source_total=n,successful=ok,virtual_time_s=t,penalized_time_s=t if ok else 360000.,common_lower_bound_s=1000.)

def test_equal_run_main_is_not_pooled_and_failures_are_not_dropped():
    rows=[row(1,10,1000),row(2,16,3200)]
    s=r.summarize(rows,'development',with_interval=False)
    assert s['mean_time_per_source_s']==150 and s['pooled_time_per_source_s']==pytest.approx(4200/26)
    f=r.summarize([rows[0],row(2,16,12,False)],'development',with_interval=False)
    assert f['mean_time_per_source_s']==11300 and not f['all_clear'] and len(f['failure_rows'])==1
    assert f['by_source_count']['16']['mean_time_per_source_s']==22500
    assert f['mean_actual_time_per_source_s']==50.375

def test_duplicates_missing_failure_discount_and_nonfinite_rejected():
    with pytest.raises(ValueError):r.summarize([row(1,10,10),row(1,10,20)],'development',with_interval=False)
    with pytest.raises(ValueError):r.summarize([row(1,10,10)],'development',[1,2],False)
    bad=row(1,10,10,False);bad['penalized_time_s']=10
    with pytest.raises(ValueError):r.summarize([bad],'development',with_interval=False)
    with pytest.raises(ValueError):r.summarize([row(1,10,float('nan'))],'development',with_interval=False)

def test_infrastructure_failure_has_unknown_actual_time_but_full_penalty():
    f=dict(row(1,10,0,False),virtual_time_s=None,infrastructure_error=True,common_lower_bound_s=None)
    s=r.summarize([f],'development',with_interval=False)
    assert not s['all_clear'] and s['mean_time_per_source_s']==36000
    assert s['failure_rows'][0]['virtual_time_s'] is None and s['mean_lower_bound_s'] is None
    assert s['mean_actual_time_per_source_s'] is None and s['mean_program_runtime_s'] is None

def test_observed_components_and_counts_keep_every_case():
    rows=[dict(row(1,10,1000),cleared_total=10,program_runtime_s=1.,movement_s=700.),
          dict(row(2,16,3200,False),cleared_total=16,program_runtime_s=3.,movement_s=2100.)]
    # A failed exit/certificate can fail the run even when all sources cleared.
    s=r.summarize(rows,'development',with_interval=False)
    assert s['mean_program_runtime_s']==2 and s['mean_components_s']['movement_s']==1400
    assert s['mean_components_s']['detection_s'] is None
    bad=copy.deepcopy(rows);bad[0]['cleared_total']=9
    with pytest.raises(ValueError):r.summarize(bad,'development',with_interval=False)
    wrong=dict(row(621001,10,1000),source_total=10 if r.count_from_seed(621001)!=10 else 11)
    with pytest.raises(ValueError):r.summarize([wrong],'development',with_interval=False,check_generator_counts=True)

def test_bootstrap_is_fixed_stratified_and_singleton_not_claimed_certain():
    a=r.stratified_interval([row(1,10,1000),row(2,10,2000),row(3,16,3200),row(4,16,4800)],'development')
    assert a==r.stratified_interval([row(1,10,1000),row(2,10,2000),row(3,16,3200),row(4,16,4800)],'development')
    assert a==r.stratified_interval([row(4,16,4800),row(2,10,2000),row(1,10,1000),row(3,16,3200)],'development')
    assert a['ci_informative'] and a['samples']==10000
    b=r.stratified_interval([row(621042,10,2000)],'development-stress')
    assert b['ci95_s']==[200.,200.] and not b['ci_informative'] and b['singleton_strata']==1

def synthetic_historical_root(root,monkeypatch,source=None):
    source=source or {'src/synthetic.py':b'# synthetic reference; never executed\n'}
    for name,data in source.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    identity={name:hashlib.sha256(data).hexdigest() for name,data in source.items()}
    def write(name,obj):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(obj,sort_keys=True)+'\n',encoding='utf-8');return r.sha(path)
    selection='selection.json';qualification='qualification.json'
    sh=write(selection,dict(role='Frozen one candidate before independent cases',selected=reference.LABEL,specs=SPEC))
    proof=write('proof.json',{'all_passed':True,'synthetic':True})
    qh=write(qualification,dict(passed=True,selected=reference.LABEL,source_sha256=identity,
        selection_sha256=sh,evidence_sha256={'proof.json':proof}))
    anchor=dict(base_commit=reference.BASE_COMMIT,source_sha256=identity,qualification_path=qualification,
        qualification_sha256=qh,selection_path=selection,selection_sha256=sh)
    ah=write(reference.ANCHOR_PATH,anchor)
    monkeypatch.setattr(reference,'ANCHOR_SHA256',ah)
    return identity


def release_fixture(tmp_path,monkeypatch):
    identity=synthetic_historical_root(tmp_path,monkeypatch)
    plan=r.make_plan('confirmation',SPEC,identity)
    release=reference.build_release(plan,'a'*64,tmp_path)
    return plan,release


def test_reference_release_binds_historical_qualification_without_fake_new_development(tmp_path,monkeypatch):
    p,x=release_fixture(tmp_path,monkeypatch);r.validate_release(p,'a'*64,x,tmp_path)
    assert x['schema']=='q4-qualified-reference-release-v1'
    assert 'development_evidence_sha256' not in x and 'development_selection_path' not in x
    for key,value in [('authorized',False),('plan_sha256','wrong'),('reserved_seeds',[]),('spec',{})]:
        bad=copy.deepcopy(x);bad[key]=value
        with pytest.raises(ValueError):r.validate_release(p,'a'*64,bad,tmp_path)
    with pytest.raises(ValueError):r.validate_release(p,'a'*64,None,tmp_path)
    (tmp_path/'selection.json').write_text('{}')
    with pytest.raises(ValueError):r.validate_release(p,'a'*64,x,tmp_path)


@pytest.mark.parametrize('change',['runtime','extra_runtime','qualification','proof','anchor'])
def test_historical_reference_byte_tampering_rejected(tmp_path,monkeypatch,change):
    plan,release=release_fixture(tmp_path,monkeypatch)
    paths={'runtime':'src/synthetic.py','extra_runtime':'src/extra.py','qualification':'qualification.json',
           'proof':'proof.json','anchor':reference.ANCHOR_PATH}
    (tmp_path/paths[change]).write_text('changed',encoding='utf-8')
    with pytest.raises((ValueError,KeyError)):r.validate_release(plan,'a'*64,release,tmp_path)


def test_historical_reference_never_releases_reserved_development(tmp_path,monkeypatch):
    plan,_=release_fixture(tmp_path,monkeypatch);plan['split']='development'
    with pytest.raises(ValueError,match='confirmation/stress'):reference.build_release(plan,'a'*64,tmp_path)


def test_no_overwrite_and_preflight_rejects_without_worker(tmp_path):
    p=tmp_path/'plan.json';r.write_new(p,{'a':1})
    with pytest.raises(FileExistsError):r.write_new(p,{'a':2})
    with patch.object(r,'worker',side_effect=AssertionError('no policy may run')):
        with pytest.raises(ValueError):r.run(p,'bad-sha',tmp_path/'out')
    assert not (tmp_path/'out').exists()

def test_multiarms_and_unsafe_labels_rejected():
    with pytest.raises(ValueError):r.validate_spec({**SPEC,'other':next(iter(SPEC.values()))})
    with pytest.raises(ValueError):r.validate_spec({'../../escape':next(iter(SPEC.values()))})


@pytest.mark.parametrize('name',['problem','max_actions','max_active_probes'])
def test_reserved_run_one_arguments_rejected_before_cases(name):
    spec=copy.deepcopy(SPEC)
    next(iter(spec.values()))['kwargs'][name]=4
    with pytest.raises(ValueError,match='reserved arguments'):r.validate_spec(spec)


@pytest.mark.parametrize('has_release',[False,True])
def test_run_preserves_control_file_exact_bytes_before_any_worker(tmp_path,has_release):
    plan=r.make_plan('confirmation' if has_release else 'development',SPEC,{})
    p=tmp_path/'input.json';p.write_bytes(json.dumps(plan,indent=3).encode()+b'\r\n')
    release=tmp_path/'authorization.json'
    release.write_bytes(b'{ "authorized" : true }\r\n')
    output=tmp_path/'out'
    with patch.object(r,'validate_plan',return_value=plan),patch.object(r,'validate_release'),patch.object(r.subprocess,'check_output',return_value='test-head'),patch.object(r,'ProcessPoolExecutor',side_effect=RuntimeError('stop before cases')):
        with pytest.raises(RuntimeError,match='stop before cases'):
            r.run(p,r.sha(p),output,release if has_release else None)
    assert (output/'plan.json').read_bytes()==p.read_bytes()
    manifest=r.read(output/'manifest.json')
    assert manifest['plan_sha256']==r.sha(output/'plan.json')
    if has_release:
        assert (output/'release.json').read_bytes()==release.read_bytes()
        assert manifest['release_sha256']==r.sha(output/'release.json')
    else:
        assert not (output/'release.json').exists() and manifest['release_sha256'] is None
    assert not list((output/'records').iterdir())
