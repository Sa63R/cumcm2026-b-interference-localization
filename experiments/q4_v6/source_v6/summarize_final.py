"""Summarize the second frozen test. Never silently drop failing cases."""
from __future__ import annotations
import argparse,csv,hashlib,json,pathlib,statistics
from summarize_v6 import comparison
from validate_final import hashes

def compare(rows,alt,reference='v5',seeds=None,boot=False):
    rr=[]
    for r in rows:
        if r['version'] not in [reference,alt]:continue
        x=dict(r);x['version']='v5'if r['version']==reference else 'candidate';rr.append(x)
    c=comparison(rr,'candidate',seeds,boot);c['reference']=reference;c['variant']=alt
    return c

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results_final');a=p.parse_args();root=pathlib.Path(a.out)
    frozen=json.loads((root/'frozen_config.json').read_text());ev=frozen['evaluation']
    rows=[json.loads(x)for x in (root/'records.jsonl').read_text().splitlines()]
    expected=3*(6*ev['main_cases']+3*ev['stress_cases'])
    assert len(rows)==expected,('Incomplete',len(rows),expected)
    assert len({(r['seed'],r['version'])for r in rows})==expected
    errors=[r for r in rows if'error'in r]
    if errors:
        (root/'errors.json').write_text(json.dumps(errors,indent=2));raise SystemExit(f'{len(errors)} errors, no all-success claim valid')
    assert frozen['source_hashes']==hashes(),'Frozen code changed'
    base={r['seed']:r for r in rows if r['version']=='v5'}
    main={s for s,r in base.items()if r['scenario']not in ['cluster','all_radius_1000','smooth_error']};extra=set(base)-main
    result={'local_only':True,'distinct_cases':len(base),'strategy_runs':len(rows),'source_total':sum(r['targets']for r in base.values()),
            'errors':0,'all_cleared':all(r['cleared']==r['targets']for r in rows),'records_sha256':hashlib.sha256((root/'records.jsonl').read_bytes()).hexdigest(),
            'main':compare(rows,'v6',seeds=main,boot=True),'additional_distributions':compare(rows,'v6',seeds=extra,boot=True),'all':compare(rows,'v6',boot=True),
            'unprotected_vs_v5':compare(rows,'unprotected',boot=True),'guarded_vs_unprotected':compare(rows,'v6','unprotected',boot=True),
            'by_scenario':{g:compare(rows,'v6',seeds={s for s,r in base.items()if r['scenario']==g})for g in sorted({r['scenario']for r in rows})},
            'count_subgroups':{str(n):compare(rows,'v6',seeds={s for s,r in base.items()if r['targets']==n})for n in range(10,17)}}
    rr=[];lookup={(r['seed'],r['version']):r for r in rows}
    for s in sorted(base):
        r=dict(seed=s,scenario=base[s]['scenario'],targets=base[s]['targets'])
        for v in ['v5','unprotected','v6']:
            for k in ['seconds_per_cleared','virtual_seconds','distance_m','detections','switches','optical_attempts','failed_clears','tail_after_last_clear','cpu_seconds','wall_seconds','transit_stops','route_accepts']:
                r[v+'_'+k]=lookup[(s,v)].get(k,0.)
        r['saving_seconds_per_source']=r['v5_seconds_per_cleared']-r['v6_seconds_per_cleared'];r['improvement_pct']=100*r['saving_seconds_per_source']/r['v5_seconds_per_cleared'];rr.append(r)
    with(root/'paired_records.csv').open('w',encoding='utf-8-sig',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:result[k]for k in ['distinct_cases','strategy_runs','source_total','all_cleared']},indent=2))
    for key in ['main','additional_distributions','all','unprotected_vs_v5','guarded_vs_unprotected']:
        x=result[key];print(key,x['baseline_seconds_per_source'],x['variant_seconds_per_source'],x['improvement_pct'],x['stratified_bootstrap_95_CI'],'worst',x['worst_relative_slowdown_pct'])
