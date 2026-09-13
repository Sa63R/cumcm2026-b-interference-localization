"""Summarize all frozen paired runs, preserving failures rather than hiding them."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,pathlib,random,statistics,collections

def quantile(xs,p):
    xs=sorted(xs);t=(len(xs)-1)*p;j=int(t);k=min(j+1,len(xs)-1)
    return xs[j]+(t-j)*(xs[k]-xs[j])

def comparison(rows,version,seeds=None,bootstrap=False):
    base={r['seed']:r for r in rows if r['version']=='v5'}
    alt={r['seed']:r for r in rows if r['version']==version}
    ids=sorted(set(base)&set(alt));ids=[s for s in ids if seeds is None or s in seeds]
    x=[base[s]['seconds_per_cleared']for s in ids];y=[alt[s]['seconds_per_cleared']for s in ids];d=[a-b for a,b in zip(x,y)]
    rel=[100*(b/a-1)for a,b in zip(x,y)];mean=statistics.mean(d);se=statistics.stdev(d)/math.sqrt(len(d))if len(d)>1 else 0.
    result={'cases':len(ids),'baseline_seconds_per_source':statistics.mean(x),'variant_seconds_per_source':statistics.mean(y),'improvement_pct':100*(1-statistics.mean(y)/statistics.mean(x)),
        'mean_paired_saving_seconds':mean,'normal_95_CI':[mean-1.96*se,mean+1.96*se],
        'faster':sum(t>1e-7 for t in d),'slower':sum(t<-1e-7 for t in d),'tied':sum(abs(t)<=1e-7 for t in d),
        'worst_relative_slowdown_pct':max(rel),'best_relative_improvement_pct':-min(rel),'p95_baseline':quantile(x,.95),'p95_variant':quantile(y,.95)}
    if bootstrap:
        groups=collections.defaultdict(list)
        for s,t in zip(ids,d):groups[base[s]['scenario']].append(t)
        rng=random.Random(715931);vals=[]
        for _ in range(3000):vals.append(sum(sum(rng.choices(v,k=len(v)))for v in groups.values())/len(ids))
        result['stratified_bootstrap_95_CI']=[quantile(vals,.025),quantile(vals,.975)]
    metrics=['virtual_seconds','distance_m','detections','switches','optical_attempts','failed_clears','tail_after_last_clear','wall_seconds','cpu_seconds','transit_stops','transit_detections','route_accepts']
    result['mean_metrics']={k:{'v5':statistics.mean(base[s].get(k,0.)for s in ids),version:statistics.mean(alt[s].get(k,0.)for s in ids)}for k in metrics}
    result['mean_time_components_per_source']={}
    for label,lookup in [('v5',base),(version,alt)]:
        rr=[lookup[s]for s in ids]
        result['mean_time_components_per_source'][label]={'movement':statistics.mean(r['distance_m']/5/r['targets']for r in rr),
            'rf_and_switching':statistics.mean((5*r['detections']+r['switches'])/r['targets']for r in rr),
            'optical_and_clear':statistics.mean((3*r['optical_attempts']+2*r['cleared'])/r['targets']for r in rr)}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results_v6');a=p.parse_args();root=pathlib.Path(a.out)
    frozen=json.loads((root/'frozen_config.json').read_text());rows=[json.loads(s)for s in (root/'records.jsonl').read_text().splitlines()]
    ev=frozen['evaluation'];expected=2*(6*ev['main_cases']+3*ev['stress_cases'])+3*(6*min(ev['main_cases'],ev['ablation_cases'])+3*min(ev['stress_cases'],ev['ablation_cases']))
    assert len(rows)==expected,('Incomplete evaluation',len(rows),expected)
    assert len({(r['seed'],r['version'])for r in rows})==len(rows),'Duplicate runs'
    errors=[r for r in rows if'error'in r]
    if errors:
        (root/'errors.json').write_text(json.dumps(errors,indent=2));raise SystemExit(f'{len(errors)} failures are present; no all-success performance claim is valid.')
    main={r['seed']for r in rows if r['scenario']not in ['cluster','all_radius_1000','smooth_error']}
    stress={r['seed']for r in rows if r['scenario']in ['cluster','all_radius_1000','smooth_error']}
    seeds={r['seed']for r in rows};base={r['seed']:r for r in rows if r['version']=='v5'};alt={r['seed']:r for r in rows if r['version']=='v6'}
    result={'local_only':True,'errors':0,'distinct_cases':len(seeds),'source_total':sum(r['targets']for r in base.values()),'strategy_runs':len(rows),'all_sources_cleared':all(r['targets']==r['cleared']for r in rows),
        'records_sha256':hashlib.sha256((root/'records.jsonl').read_bytes()).hexdigest(),
        'main':comparison(rows,'v6',main,True),'additional_distributions':comparison(rows,'v6',stress,True),'all':comparison(rows,'v6',None,True),
        'by_scenario':{g:comparison(rows,'v6',{r['seed']for r in rows if r['scenario']==g})for g in sorted({r['scenario']for r in rows})}}
    ab={r['seed']for r in rows if r['version']=='transit_only'}
    result['predeclared_ablations']={v:comparison(rows,v,ab)for v in ['transit_only','route_only','cheap','v6']}
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    records=[]
    for s in sorted(seeds):
        r={'seed':s,'scenario':base[s]['scenario'],'targets':base[s]['targets'],'v5_seconds_per_source':base[s]['seconds_per_cleared'],'v6_seconds_per_source':alt[s]['seconds_per_cleared'],
           'saving_seconds_per_source':base[s]['seconds_per_cleared']-alt[s]['seconds_per_cleared'],'relative_improvement_pct':100*(1-alt[s]['seconds_per_cleared']/base[s]['seconds_per_cleared'])}
        for k in ['distance_m','detections','switches','failed_clears','virtual_seconds','wall_seconds','cpu_seconds','tail_after_last_clear','transit_stops','route_accepts']:
            r['v5_'+k]=base[s].get(k,0.);r['v6_'+k]=alt[s].get(k,0.)
        records.append(r)
    with (root/'paired_records.csv').open('w',newline='',encoding='utf-8-sig')as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(json.dumps({k:result[k]for k in ['distinct_cases','source_total','strategy_runs','all_sources_cleared']},indent=2))
    for k in ['main','additional_distributions','all']:
        t=result[k];print(k,t['baseline_seconds_per_source'],t['variant_seconds_per_source'],t['improvement_pct'],t['stratified_bootstrap_95_CI'])
