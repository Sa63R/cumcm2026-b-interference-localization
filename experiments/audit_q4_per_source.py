"""Audit a fixed single-arm batch, its complete quota and actual physical prefix."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.run_q4_per_source import (
    validate_plan, validate_release, summarize, count_from_seed, sha, read, write_new, digest, DESIGNS)
from experiments.audit_q4_observation_cover import audit_full


def audit(directory):
    directory=Path(directory).resolve()
    destination=directory/'independent_audit.json'
    if destination.exists(): raise ValueError('Preserve existing audit')
    manifest=read(directory/'manifest.json');plan=validate_plan(manifest['plan'])
    freeze=read(directory/'freeze.json');saved=read(directory/'summary.json')
    errors=[];items=[];rows=[];inputs={}
    if freeze['manifest_sha256']!=digest(manifest): errors.append('Manifest changed')
    if sha(directory/'plan.json')!=manifest['plan_sha256'] or read(directory/'plan.json')!=plan:
        errors.append('Original plan bytes or plan body differ')
    release_path=directory/'release.json'
    release=read(release_path) if release_path.exists() else None
    if manifest['release_sha256']!=(sha(release_path) if release is not None else None):
        errors.append('Original release bytes differ')
    try:validate_release(plan,manifest['plan_sha256'],release,root=ROOT)
    except (ValueError,KeyError,TypeError) as error:errors.append(str(error))
    with zipfile.ZipFile(directory/'source.zip') as z:
        if len(z.namelist())!=len(set(z.namelist())):errors.append('Duplicated archived source entry')
        if set(z.namelist())!=set(plan['source_sha256']):errors.append('Archived source file set differs')
        for p,h in plan['source_sha256'].items():
            if hashlib.sha256(z.read(p)).hexdigest()!=h:errors.append('Archived source differs: '+p)
    expected={(s,plan['label']) for s in plan['seed_selection']['seeds']};seen=set();case_hashes=set()
    for path in sorted((directory/'records').glob('*.json.gz')):
        inputs[path.relative_to(directory).as_posix()]=sha(path)
        with gzip.open(path,'rt',encoding='utf-8') as stream:record=json.load(stream)
        row=record['row'];rows.append(row);key=(row['seed'],row['strategy'])
        item=dict(seed=row['seed'],strategy=row['strategy'],passed=False,errors=[])
        try:
            if key not in expected or key in seen:raise ValueError('Unexpected or duplicate case')
            seen.add(key)
            if path.name!=f"{plan['label']}-{row['seed']}.json.gz":raise ValueError('Record filename differs')
            if record.get('record_kind')=='infrastructure_failure_without_completed_case':raise ValueError('Infrastructure failure lacks auditable physical case')
            if record['spec']!=plan['spec'] or row['stage']!=DESIGNS[plan['split']]['stage']:raise ValueError('Spec/stage differs')
            signature=digest(record['evaluation']['ground_truth'])
            if signature!=row['case_sha256'] or signature in case_hashes:raise ValueError('Changed or duplicated scenario')
            case_hashes.add(signature)
            if row['source_total']!=count_from_seed(row['seed']):raise ValueError('Source-count stratum differs')
            item['observation_cover']=audit_full(record)
            observation_cover=item['observation_cover'];prefix=observation_cover.get('prefix',{})
            if (observation_cover.get('passed') is not True or observation_cover.get('generic',{}).get('passed') is not True or
                prefix.get('passed') is not True or any(prefix.get(k,{}).get('passed') is not True
                    for k in ('r12','r8','range','scheduling'))):
                raise ValueError('Physical/observation_cover/inherited audit did not pass in full')
            item['passed']=True
        except (ValueError,AssertionError,KeyError,TypeError) as error:item['errors'].append(str(error))
        items.append(item)
    if seen!=expected:errors.append('Missing fixed-quota cases')
    try:
        recomputed=summarize(rows,plan['split'],plan['seed_selection']['seeds'])
        if any(saved.get(k)!=v for k,v in recomputed.items()):errors.append('Summary differs from complete audited rows')
        if saved.get('complete') is not True or saved.get('source_unchanged') is not True or saved.get('infrastructure_errors')!=[]:
            errors.append('Execution was incomplete, changed source, or failed infrastructure')
    except (ValueError,KeyError,TypeError) as error:errors.append(str(error))
    for name in ('manifest.json','freeze.json','source.zip','summary.json','plan.json'):inputs[name]=sha(directory/name)
    if release_path.exists():inputs['release.json']=sha(release_path)
    result=dict(all_passed=not errors and len(items)==len(expected) and all(x['passed'] for x in items),
        records=len(items),passed_records=sum(x['passed'] for x in items),errors=errors,
        all_clear=bool(rows) and all(r['successful'] for r in rows),input_sha256=inputs,
        plan_sha256=manifest['plan_sha256'],audits=items,
        scope='Complete count-stratified single-arm matrix, actual costs, full cover and bound, certified observation-cover geometry and independent R12/R8/range/scheduling prefix safety')
    write_new(destination,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('audits','input_sha256')}),flush=True)
    return int(not result['all_passed'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    raise SystemExit(audit(parser.parse_args().input))
