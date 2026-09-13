#!/usr/bin/env python3
"""Recompute all reported comparisons from the included raw self-built data."""
from pathlib import Path
import csv,json,math
import numpy as np
ROOT=Path(__file__).resolve().parent

def read(name):
    with (ROOT/'results'/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def stats(rows):
    out={}
    for m in sorted(set(r['mode'] for r in rows)):
        rr=[r for r in rows if r['mode']==m]
        out[m]={'cases':len(rr),'cleared_sources':sum(int(r['cleared']) for r in rr),'all_cleared_cases':sum(int(r['cleared'])==int(r['true_sources']) for r in rr),'failed_clears':sum(int(r['failed_clears']) for r in rr)}
        for k in ['virtual_seconds','seconds_per_source','movement_metres','detects','switches']:
            out[m]['mean_'+k]=float(np.mean([float(r[k]) for r in rr]))
        out[m]['p95_virtual_seconds']=float(np.quantile([float(r['virtual_seconds']) for r in rr],.95))
    return out

def paired(rows):
    modes={m:{(r['seed'],r['noise'],r.get('kind','random')):r for r in rows if r['mode']==m} for m in ['v2','v3']}
    keys=sorted(set(modes['v2'])&set(modes['v3']))
    a=np.array([float(modes['v2'][k]['virtual_seconds']) for k in keys]);z=np.array([float(modes['v3'][k]['virtual_seconds']) for k in keys]);d=a-z
    rng=np.random.default_rng(119971)
    boot=np.mean(d[rng.integers(0,len(d),(10000,len(d)))],axis=1) if not any('kind' in r for r in rows) else None
    return {'pairs':len(d),'mean_saved_seconds':float(d.mean()),'relative_mean_time_reduction':float(d.mean()/a.mean()),'faster_cases':int((d>1e-7).sum()),'slower_cases':int((d< -1e-7).sum()),'ties':int((abs(d)<=1e-7).sum()),'median_saved_seconds':float(np.median(d)),'max_saving_seconds':float(d.max()),'max_slowdown_seconds':float(-d.min()),'bootstrap95_mean_saving_seconds':([float(x) for x in np.quantile(boot,[.025,.975])] if boot is not None else None),'worst_case_key':list(keys[int(d.argmin())])}

if __name__=='__main__':
    summary={'warning':'SELF-BUILT SIMULATION ONLY; no official practice or formal runs. Frozen default: initial_channels=0, expanded_probes=True, route_trials=40, through_clear=True, range_prior=False. Development seeds 0-79; regression 2000-2199; untouched validation 5000-5399.'}
    for name in ['holdout400','regression_ablation200','stress350']:
        rows=read(name+'.csv');summary[name]={'means':stats(rows),'paired_v2_v3':paired(rows)}
    rows=read('stress350.csv')
    summary['stress_groups']={g:{'means':stats([r for r in rows if r['kind']==g]),'paired':paired([r for r in rows if r['kind']==g])} for g in ['boundary','special']}
    dev=[]
    for p in sorted((ROOT/'results').glob('screen_*.csv')):dev+=read(p.name)
    summary['development']={'distinct_configurations':len(set(r['mode'] for r in dev)),'case_configuration_evaluations':len(dev),'all_cleared':all(int(r['cleared'])==int(r['true_sources']) for r in dev)}
    runtime=ROOT/'results'/'runtime_summary.json'
    if runtime.exists():summary['serial_runtime']=json.loads(runtime.read_text())
    (ROOT/'results'/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    for name in ['holdout400','regression_ablation200','stress350']:
        print(name,json.dumps(summary[name]['paired_v2_v3'],ensure_ascii=False))
    print('development',summary['development'])
