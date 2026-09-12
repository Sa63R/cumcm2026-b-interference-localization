"""Paired descriptive and bootstrap statistics; no source/model tuning."""
import csv
import json
from pathlib import Path
import random
import statistics as st
import sys

ROOT=Path(__file__).resolve().parent


def percentile(values,p):
    values=sorted(values); x=(len(values)-1)*p; lo=int(x); hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(x-lo)


def compare(rows,baseline,candidate,replicates=10000):
    index={(r['case_key'],r['method']):r for r in rows}
    keys=sorted({r['case_key'] for r in rows})
    aa=[index[k,baseline]['seconds_per_source'] for k in keys]
    bb=[index[k,candidate]['seconds_per_source'] for k in keys]
    differences=[a-b for a,b in zip(aa,bb)]
    relative=[(b/a-1)*100 for a,b in zip(aa,bb)]
    rng=random.Random(12912026);boot=[];boot_pct=[];n=len(keys)
    for _ in range(replicates):
        sample=rng.choices(range(n),k=n)
        delta=sum(differences[j] for j in sample)/n
        base=sum(aa[j] for j in sample)/n
        boot.append(delta);boot_pct.append(100*delta/base)
    worst=max(range(n),key=lambda j:relative[j]); best=min(range(n),key=lambda j:relative[j])
    return dict(baseline=baseline,candidate=candidate,paired_cases=n,
        baseline_mean=st.mean(aa),candidate_mean=st.mean(bb),saved_seconds_per_source=st.mean(differences),
        saving_percent=100*st.mean(differences)/st.mean(aa),
        saved_seconds_ci95=[percentile(boot,.025),percentile(boot,.975)],
        saving_percent_ci95=[percentile(boot_pct,.025),percentile(boot_pct,.975)],
        faster=sum(v>1e-6 for v in differences),slower=sum(v< -1e-6 for v in differences),
        equal=sum(abs(v)<=1e-6 for v in differences),
        worst=dict(case_key=keys[worst],slower_percent=relative[worst],baseline=aa[worst],candidate=bb[worst]),
        best=dict(case_key=keys[best],saving_percent=-relative[best],baseline=aa[best],candidate=bb[best]))


def main():
    rows=[json.loads(x) for x in (ROOT/'main/records.jsonl').read_text().splitlines()]
    plan=json.loads((ROOT/'plan.json').read_text())
    expected={(c['key'],m) for c in plan['cases'] for m in plan['methods']}
    assert len(rows)==len(expected) and {(r['case_key'],r['method']) for r in rows}==expected
    errors=[r for r in rows if r['error'] or not r['all_cleared']]
    if errors:
        (ROOT/'errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
        raise RuntimeError(f'{len(errors)} unsuccessful runs; do not publish success-only means')
    results=dict(local_only=True,unique_cases=len(plan['cases']),complete_runs=len(rows),
                 source_clear_events=sum(r['source_total'] for r in rows),
                 unique_source_instances=sum(r['source_total'] for r in rows if r['method']=='v4'),
                 all_cleared=True,errors=0,groups={})
    for group in dict.fromkeys(c['group'] for c in plan['cases']):
        rr=[r for r in rows if r['group']==group]
        methods={}
        for m in plan['methods']:
            mm=[r for r in rr if r['method']==m]
            fields=['seconds_per_source','virtual_time_s','distance_m','measurement_count','failed_clear_count',
                    'switch_count','wall_seconds','cpu_seconds','tail_after_last_clear_s','transit_stops','route_accepts','planner_seconds']
            methods[m]={field:st.mean(r[field] for r in mm) for field in fields}
            methods[m]['cases']=len(mm)
            methods[m]['time_per_source_breakdown']={key:st.mean(r['time_breakdown_s'][key]/r['source_total'] for r in mm)
                for key in mm[0]['time_breakdown_s']}
        results['groups'][group]=dict(methods=methods,comparisons=[compare(rr,a,b) for a,b in [('v4','v5'),('v4','v6'),('v5','v6')]])
    main_rows=[r for r in rows if r['group']=='practice_generator']
    results['main_by_source_count']={str(n):dict(cases=sum(r['source_total']==n and r['method']=='v4' for r in main_rows),
        methods={m:st.mean(r['seconds_per_source'] for r in main_rows if r['source_total']==n and r['method']==m) for m in plan['methods']}) for n in range(10,17)}
    (ROOT/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    simple=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]
    with (ROOT/'paired_records.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(simple[0]));writer.writeheader();writer.writerows(simple)
    for group,g in results['groups'].items():
        print(group)
        for comparison in g['comparisons']:print(json.dumps(comparison,ensure_ascii=False))


if __name__=='__main__': main()
