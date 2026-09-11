"""Batch wrapper tests mock only expensive per-record physics/prefix checks."""
import copy
import pytest
from experiments import audit_q4_per_source as a
from experiments import run_q4_per_source as r
from tests.test_q4_observation_cover_release import evidence, q, write, record_read, record_write, observation_cover_pass


@pytest.fixture
def batch(evidence,monkeypatch):
    root,dirs,_=evidence
    folder=dirs['compact_ring_28']['development']
    (folder/'independent_audit.json').unlink()
    monkeypatch.setattr(a,'ROOT',root)
    calls=[]
    def full(record):
        calls.append(record)
        return observation_cover_pass()
    monkeypatch.setattr(a,'audit_full',full)
    monkeypatch.setattr(r,'one',lambda *args,**kwargs:pytest.fail('No scenario may run'))
    return folder,calls


def assert_rejected(folder):
    try:status=a.audit(folder)
    except (ValueError,KeyError,TypeError,FileNotFoundError):
        if (folder/'independent_audit.json').exists():assert not q.read(folder/'independent_audit.json')['all_passed']
    else:assert status==1 and not q.read(folder/'independent_audit.json')['all_passed']


def test_complete_real_wrapper_hashes_every_input_and_checks_all_cases(batch):
    folder,calls=batch
    assert a.audit(folder)==0
    result=q.read(folder/'independent_audit.json')
    assert result['records']==result['passed_records']==len(calls)==70
    assert len(result['input_sha256'])==75
    assert all(q.sha(folder/k)==v for k,v in result['input_sha256'].items())
    assert all(i['observation_cover']['prefix']['r12']['passed'] for i in result['audits'])


@pytest.mark.parametrize('field',['passed','generic','prefix','r12','r8','range','scheduling'])
def test_cannot_claim_pass_if_any_nested_checker_did_not_pass(batch,monkeypatch,field):
    folder,_=batch
    value=observation_cover_pass()
    if field=='passed':value['passed']=False
    elif field in ('generic','prefix'):value[field]['passed']=False
    else:value['prefix'][field]['passed']=False
    monkeypatch.setattr(a,'audit_full',lambda record:value)
    assert_rejected(folder)


@pytest.mark.parametrize('kind',['missing_record','duplicate_record','wrong_label','case_sha','source_count','wrong_spec','wrong_stage','summary','manifest','plan_bytes','no_release'])
def test_actual_batch_metadata_rejects_corruption(batch,kind):
    folder,_=batch;plan=q.read(folder/'plan.json')
    path=folder/'records'/f"{plan['label']}-{plan['seed_selection']['seeds'][0]}.json.gz"
    if kind=='missing_record':path.unlink()
    elif kind=='duplicate_record':(folder/'records'/'duplicate.json.gz').write_bytes(path.read_bytes())
    elif kind in ('wrong_label','case_sha','source_count','wrong_spec','wrong_stage'):
        v=record_read(path)
        if kind=='wrong_label':v['row']['strategy']='compact_ring_31'
        elif kind=='case_sha':v['row']['case_sha256']='changed'
        elif kind=='source_count':v['row']['source_total']=10 if v['row']['source_total']!=10 else 11
        elif kind=='wrong_spec':v['spec']=copy.deepcopy(q.FIXED_SPECS['compact_ring_31'])
        else:v['row']['stage']='stress'
        record_write(path,v)
    elif kind=='summary':
        v=q.read(folder/'summary.json');v['mean_time_per_source_s']=1.;write(folder/'summary.json',v)
    elif kind=='manifest':
        v=q.read(folder/'manifest.json');v['plan_sha256']='changed';write(folder/'manifest.json',v)
    elif kind=='plan_bytes':
        (folder/'plan.json').write_text(__import__('json').dumps(plan,indent=4),encoding='utf-8')
    else:
        # Switch to an independently fixed plan with correctly rehashed controls;
        # missing release must still prevent any claimed audit success.
        plan=r.make_plan('confirmation',{plan['label']:plan['spec']})
        write(folder/'plan.json',plan)
        manifest=dict(plan=plan,plan_sha256=q.sha(folder/'plan.json'),release_sha256=None)
        write(folder/'manifest.json',manifest)
        write(folder/'freeze.json',dict(manifest_sha256=r.digest(manifest),git_commit='a'*40))
    assert_rejected(folder)
