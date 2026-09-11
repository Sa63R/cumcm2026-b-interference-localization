"""One-arm, count-stratified Q4 evaluation. Planning reads only public RNG count.

plan never constructs scenarios. run requires an externally supplied plan SHA;
independent splits also require a separate release bound to development evidence.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.run_q4_round2 import hashes as inherited_hashes, one, digest
from experiments.run_q4_state_study import percentile, PROTOCOL as GENERATOR_PROTOCOL

DESIGNS={
    'development':dict(start=6260001,scan_limit=1000,quota=10,stage='pilot',stress=False),
    'development-stress':dict(start=6261001,scan_limit=1000,quota=1,stage='stress',stress=True),
    'confirmation':dict(start=6262001,scan_limit=1000,quota=20,stage='confirmation',stress=False),
    'stress':dict(start=6264001,scan_limit=1000,quota=2,stage='stress',stress=True),
}
BOOTSTRAP_SEED=626941
BOOTSTRAP_SAMPLES=10000
FAILURE_SECONDS=360000.

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def source_hashes():
    result=inherited_hashes()
    result[Path(__file__).relative_to(ROOT).as_posix()]=sha(__file__)
    return result

def count_from_seed(seed):
    if type(seed) is not int: raise ValueError('Seed must be integer')
    return random.Random(seed+4*1000003).randint(10,16)

def family_from_seed(seed):
    return GENERATOR_PROTOCOL['stress_families'][(seed-294201)%7]

def seed_selection(split):
    """Exactly the first eligible seeds for each predetermined stratum quota."""
    if split not in DESIGNS: raise ValueError('Unknown frozen split')
    d=DESIGNS[split]
    keys=[f'{n}/{f}' for n in range(10,17) for f in GENERATOR_PROTOCOL['stress_families']] if d['stress'] else [str(n) for n in range(10,17)]
    counts={k:0 for k in keys};trace=[];seeds=[]
    for seed in range(d['start'],d['start']+d['scan_limit']):
        n=count_from_seed(seed);family=family_from_seed(seed) if d['stress'] else None
        key=f'{n}/{family}' if d['stress'] else str(n)
        accepted=counts[key]<d['quota']
        if accepted: counts[key]+=1;seeds.append(seed)
        trace.append(dict(seed=seed,source_count=n,family=family,stratum=key,accepted=accepted,
            reason='earliest_unfilled_stratum' if accepted else 'stratum_quota_full'))
    if any(v!=d['quota'] for v in counts.values()): raise ValueError('Predetermined 1000-seed interval cannot fill quotas')
    return dict(seeds=seeds,counts=counts,trace=trace,trace_sha256=digest(trace),
        design=dict(d),stratification='N x family' if d['stress'] else 'N')

def validate_spec(mapping):
    if not isinstance(mapping,dict) or len(mapping)!=1: raise ValueError('Exactly one fixed method required')
    label,spec=next(iter(mapping.items()))
    if not isinstance(label,str) or re.fullmatch(r'[a-z][a-z0-9_]{0,79}',label) is None: raise ValueError('Invalid output label')
    if not isinstance(spec,dict) or set(spec)!={'entrypoint','kwargs'} or not isinstance(spec['kwargs'],dict): raise ValueError('Need entrypoint and kwargs')
    if not isinstance(spec['entrypoint'],str) or re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*',spec['entrypoint']) is None: raise ValueError('Invalid import entrypoint')
    reserved={'problem','max_actions','max_active_probes'} & set(spec['kwargs'])
    if reserved: raise ValueError('The frozen runner supplies reserved arguments: '+','.join(sorted(reserved)))
    json.dumps(mapping,allow_nan=False)
    return label,spec

def make_plan(split,spec_mapping,identity=None):
    label,spec=validate_spec(spec_mapping)
    return dict(schema='q4-per-source-plan-v1',metadata_only=True,split=split,label=label,spec=spec,
        seed_selection=seed_selection(split),source_sha256=source_hashes() if identity is None else identity,
        metric='equal-run mean(penalized_time_s/source_total)',failure_penalty_s=FAILURE_SECONDS,
        bootstrap=dict(seed=BOOTSTRAP_SEED,samples=BOOTSTRAP_SAMPLES,strata='N x family' if DESIGNS[split]['stress'] else 'N'))

def validate_plan(plan,identity=None):
    expected=make_plan(plan['split'],{plan['label']:plan['spec']},identity)
    if plan!=expected: raise ValueError('Plan source/spec/fixed quota/earliest seed trace differs from reconstruction')
    return plan

def validate_release(plan,plan_sha256,release,root=ROOT):
    if plan['split'] in ('development','development-stress'):
        if release is not None: raise ValueError('Development does not accept an independent release')
        return
    if not isinstance(release,dict) or release.get('authorized') is not True: raise ValueError('Explicit independent release required')
    for key,value in dict(plan_sha256=plan_sha256,split=plan['split'],label=plan['label'],spec=plan['spec'],source_sha256=plan['source_sha256'],reserved_seeds=plan['seed_selection']['seeds']).items():
        if release.get(key)!=value: raise ValueError('Independent release differs: '+key)
    evidence=release.get('development_evidence_sha256')
    selection=release.get('development_selection_path')
    if not isinstance(evidence,dict) or not evidence or selection not in evidence: raise ValueError('Release must bind actual development selection and evidence')
    root=Path(root).resolve()
    for name,value in evidence.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p)!=value: raise ValueError('Development evidence changed or escapes root')
    chosen=read(root/selection)
    if chosen.get('passed') is not True or chosen.get('selected_spec')!={plan['label']:plan['spec']} or chosen.get('source_sha256')!=plan['source_sha256']:
        raise ValueError('Released method was not the actual source-bound development selection')

def stratified_interval(rows,split):
    groups={}
    for r in sorted(rows,key=lambda item:item['seed']):
        key=(r['source_total'],family_from_seed(r['seed'])) if DESIGNS[split]['stress'] else (r['source_total'],)
        groups.setdefault(key,[]).append((r['virtual_time_s'] if r['successful'] else FAILURE_SECONDS)/r['source_total'])
    strata=[groups[k] for k in sorted(groups)];rng=random.Random(BOOTSTRAP_SEED);values=[]
    for _ in range(BOOTSTRAP_SAMPLES):
        values.append(sum(sum(g[rng.randrange(len(g))] for _ in g) for g in strata)/len(rows))
    return dict(ci95_s=[percentile(values,.025),percentile(values,.975)],samples=BOOTSTRAP_SAMPLES,seed=BOOTSTRAP_SEED,
        strata={str(k):len(v) for k,v in sorted(groups.items())},ci_informative=any(len(g)>1 for g in strata),
        singleton_strata=sum(len(g)==1 for g in strata),
        limitation='Within-stratum empirical resampling only. Singleton strata have no estimable within-stratum variation; zero interval width is not certainty.')

def summarize(rows,split,expected_seeds=None,with_interval=True,check_generator_counts=False):
    if split not in DESIGNS or not rows: raise ValueError('Need a nonempty fixed split')
    keys=[r['seed'] for r in rows]
    if len(keys)!=len(set(keys)) or (expected_seeds is not None and (len(expected_seeds)!=len(set(expected_seeds)) or set(keys)!=set(expected_seeds))): raise ValueError('Missing or duplicated cases')
    for r in rows:
        if type(r['source_total']) is not int or not 10<=r['source_total']<=16 or type(r['successful']) is not bool: raise ValueError('Invalid source count or outcome')
        if check_generator_counts and r['source_total']!=count_from_seed(r['seed']): raise ValueError('Source count differs from public generator first draw')
        for name in ('cleared_total','cleared_sources','cleared_count'):
            if name in r:
                cleared=r[name]
                if type(cleared) is not int or not 0<=cleared<=r['source_total'] or (r['successful'] and cleared!=r['source_total']): raise ValueError('Successful outcome conflicts with actual cleared count')
        unknown_time=r['virtual_time_s'] is None and not r['successful'] and r.get('infrastructure_error') is True
        if not unknown_time and (not isinstance(r['virtual_time_s'],(int,float)) or not math.isfinite(r['virtual_time_s']) or r['virtual_time_s']<0): raise ValueError('Invalid observed time')
        if r['penalized_time_s']!=(r['virtual_time_s'] if r['successful'] else FAILURE_SECONDS): raise ValueError('Failure was omitted or under-penalized')
    def group(rs):
        t=[r['penalized_time_s'] for r in rs];v=[r['penalized_time_s']/r['source_total'] for r in rs];lower=[r.get('common_lower_bound_s') for r in rs]
        valid_lower=all(isinstance(x,(int,float)) and math.isfinite(x) and x>0 for x in lower)
        def mean_observed(key):
            values=[r.get(key) for r in rs]
            return statistics.mean(values) if all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in values) else None
        actual_tn=[r['virtual_time_s']/r['source_total'] for r in rs if r['virtual_time_s'] is not None]
        return dict(runs=len(rs),successful=sum(r['successful'] for r in rs),all_clear=all(r['successful'] for r in rs),
            mean_time_s=statistics.mean(t),mean_time_per_source_s=statistics.mean(v),pooled_time_per_source_s=sum(t)/sum(r['source_total'] for r in rs),
            p95_time_per_source_s=percentile(v,.95),total_time_s=sum(t),total_sources=sum(r['source_total'] for r in rs),
            mean_actual_time_per_source_s=statistics.mean(actual_tn) if len(actual_tn)==len(rs) else None,
            mean_components_s={key:mean_observed(key) for key in ('movement_s','detection_s','switching_s','optical_s','removal_s')},
            mean_program_runtime_s=mean_observed('program_runtime_s'),
            mean_lower_bound_s=statistics.mean(lower) if valid_lower else None,
            mean_time_over_mean_lower_bound=statistics.mean(t)/statistics.mean(lower) if valid_lower else None)
    result=group(rows);result.update(primary_metric='equal-run mean(T_i/N_i), failed T_i=360000',failure_penalty_s=FAILURE_SECONDS,
        failure_rows=[r for r in rows if not r['successful']],rows=sorted(rows,key=lambda r:r['seed']),
        by_source_count={str(n):group([r for r in rows if r['source_total']==n]) for n in sorted({r['source_total'] for r in rows})})
    if DESIGNS[split]['stress']:
        result['by_source_count_family']={f'{n}/{family}':group(rs) for n in range(10,17) for family in GENERATOR_PROTOCOL['stress_families'] if (rs:=[r for r in rows if r['source_total']==n and family_from_seed(r['seed'])==family])}
    if with_interval: result['stratified_bootstrap']=stratified_interval(rows,split)
    return result

def worker(seed,plan):
    if source_hashes()!=plan['source_sha256']: raise ValueError('Frozen source changed before case')
    record=one(seed,DESIGNS[plan['split']]['stage'],plan['label'],plan['spec'],inherited_hashes())
    if record['row']['source_total']!=count_from_seed(seed): raise ValueError('Generator first-draw count contract changed')
    return record

def run(plan_path,expected_sha,output,release_path=None,workers=3):
    if type(workers) is not int or not 1<=workers<=3: raise ValueError('Workers must be 1..3')
    plan_bytes=Path(plan_path).read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest()!=expected_sha: raise ValueError('Plan file differs from explicit expected SHA')
    release_bytes=Path(release_path).read_bytes() if release_path else None
    plan=validate_plan(json.loads(plan_bytes));validate_release(plan,expected_sha,json.loads(release_bytes) if release_bytes is not None else None)
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'records').mkdir()
    with (output/'plan.json').open('xb') as f:f.write(plan_bytes)
    if release_bytes is not None:
        with (output/'release.json').open('xb') as f:f.write(release_bytes)
    manifest=dict(plan=plan,plan_sha256=expected_sha,release_sha256=hashlib.sha256(release_bytes).hexdigest() if release_bytes is not None else None)
    write_new(output/'manifest.json',manifest)
    write_new(output/'freeze.json',dict(manifest_sha256=digest(manifest),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    with zipfile.ZipFile(output/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in plan['source_sha256']:z.write(ROOT/p,p)
    rows=[];errors=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks={pool.submit(worker,seed,plan):seed for seed in plan['seed_selection']['seeds']}
        for future in as_completed(tasks):
            seed=tasks[future]
            try:record=future.result()
            except Exception:
                error=dict(seed=seed,error=traceback.format_exc());errors.append(error)
                row=dict(seed=seed,strategy=plan['label'],stage=DESIGNS[plan['split']]['stage'],source_total=count_from_seed(seed),successful=False,virtual_time_s=None,penalized_time_s=FAILURE_SECONDS,common_lower_bound_s=None,infrastructure_error=True,errors=[error['error']])
                record=dict(row=row,record_kind='infrastructure_failure_without_completed_case',error=error)
            row=record['row'];rows.append(row)
            with gzip.open(output/'records'/f"{plan['label']}-{seed}.json.gz",'wt',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,allow_nan=False)
            print(json.dumps({k:row.get(k) for k in ('seed','successful','virtual_time_s','source_total')}),flush=True)
    result=summarize(rows,plan['split'],plan['seed_selection']['seeds'],check_generator_counts=True);result.update(source_unchanged=source_hashes()==plan['source_sha256'],infrastructure_errors=errors)
    result['complete']=not errors and result['source_unchanged']
    write_new(output/'summary.json',result)
    print(json.dumps({k:result[k] for k in ('runs','successful','all_clear','complete','mean_time_per_source_s','pooled_time_per_source_s','p95_time_per_source_s','mean_time_over_mean_lower_bound')}),flush=True)
    return int(not result['complete'] or not result['all_clear'])

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan');p.add_argument('--split',choices=DESIGNS,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--release',type=Path);p.add_argument('--workers',type=int,default=3);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='plan':
        plan=make_plan(args.split,read(args.spec));write_new(args.output,plan)
        print(json.dumps(dict(split=args.split,cases=len(plan['seed_selection']['seeds']),plan_sha256=sha(args.output),metadata_only=True)));return 0
    return run(args.plan,args.plan_sha256,args.output,args.release,args.workers)

if __name__=='__main__':raise SystemExit(main())
