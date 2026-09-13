"""Paired means, frozen component contrasts and exploratory interactions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
REPLICATES=50000
PRIMARY=['no_route','no_transit','no_guard','no_rollout','no_rb']


def interval(samples,coverage=.95):
    tail=(1-coverage)/2
    return np.quantile(samples,[tail,1-tail]).tolist()


def summarize_group(rows,methods):
    keys=sorted({r['case_key'] for r in rows});n=len(keys)
    index={(r['case_key'],r['method']):r for r in rows}
    values=np.array([[index[key,m]['seconds_per_source'] for m in methods] for key in keys])
    assert len(index)==n*len(methods)==len(rows)
    means=values.mean(axis=0);rng=np.random.default_rng(912202609)
    bootstrap=np.empty((REPLICATES,len(methods)))
    for start in range(0,REPLICATES,500):
        size=min(500,REPLICATES-start)
        chosen=rng.integers(0,n,size=(size,n))
        bootstrap[start:start+size]=values[chosen].mean(axis=1)
    full=methods.index('full');v4=methods.index('v4');summary={}
    for j,method in enumerate(methods):
        rr=[index[key,method] for key in keys]
        difference=values[:,j]-values[:,full]
        ratio=100*(values[:,j]/values[:,full]-1)
        worst=int(np.argmax(ratio));best=int(np.argmin(ratio))
        data=dict(cases=n,mean_seconds_per_source=float(means[j]),
            change_vs_full_s=float(difference.mean()),
            change_vs_full_ci95=interval(bootstrap[:,j]-bootstrap[:,full]),
            saving_vs_full_percent=float(100*(1-means[j]/means[full])),
            saving_vs_v4_percent=float(100*(1-means[j]/means[v4])),
            faster_than_full=int(np.sum(difference < -1e-6)),
            slower_than_full=int(np.sum(difference > 1e-6)),
            equal_to_full=int(np.sum(np.abs(difference)<=1e-6)),
            worst_vs_full=dict(case_key=keys[worst],slower_percent=float(ratio[worst])),
            best_vs_full=dict(case_key=keys[best],saving_percent=float(-ratio[best])),
            p95_slowdown_vs_v4_percent=float(np.quantile(100*(values[:,j]/values[:,v4]-1),.95)),
            worst_slowdown_vs_v4_percent=float(np.max(100*(values[:,j]/values[:,v4]-1))))
        if method in PRIMARY:
            data['component_benefit_ci99_bonferroni5']=interval(bootstrap[:,j]-bootstrap[:,full],.99)
        for field in ['virtual_time_s','cpu_seconds','wall_seconds','distance_m','measurement_count',
                      'failed_clear_count','switch_count','transit_stops','transit_detections',
                      'route_calls','route_accepts','planner_calls','planner_accepts','planner_seconds',
                      'rb_share_checks','rb_share_failures','shared_detections','visited_station_count',
                      'last_discovery_s','tail_after_last_clear_s']:
            data['mean_'+field]=float(np.mean([r[field] for r in rr]))
        data['mean_time_per_source_breakdown']={field:float(np.mean([r['time_breakdown_s'][field]/r['source_total'] for r in rr]))
            for field in rr[0]['time_breakdown_s']}
        summary[method]=data
    def contrast(weights):
        vector=np.array([weights.get(name,0) for name in methods])
        return dict(weights=weights,mean=float(means@vector),ci95=interval(bootstrap@vector))
    interactions={
        'route_transit_guard4':contrast(dict(full=1,no_transit=-1,no_route=-1,v5=1)),
        'route_transit_guard0':contrast(dict(no_guard=1,route_only_unguarded=-1,transit_only_unguarded=-1,v5=1)),
        'rollout_rb_full_learned':contrast(dict(full=1,no_rb=-1,no_rollout=-1,no_rollout_rb=1)),
        'route_average_marginal_benefit':contrast(dict(v5=.5,no_transit=-.5,no_route=.5,full=-.5)),
        'transit_average_marginal_benefit':contrast(dict(v5=.5,no_route=-.5,no_transit=.5,full=-.5)),
        'guard_with_route_only':contrast(dict(route_only_unguarded=1,no_transit=-1)),
        'guard_with_transit_only':contrast(dict(transit_only_unguarded=1,no_route=-1)),
        'learned_transit_vs_geometry':contrast(dict(transit_geometry=1,full=-1)),
        'rb_without_rollout':contrast(dict(no_rollout_rb=1,no_rollout=-1)),
        'rollout_without_rb':contrast(dict(no_rollout_rb=1,no_rb=-1)),
    }
    by_n={}
    for total in range(10,17):
        chosen=[key for key in keys if index[key,'full']['source_total']==total]
        if not chosen:continue
        by_n[str(total)]=dict(cases=len(chosen),methods={name:dict(
            mean_seconds_per_source=float(np.mean([index[key,name]['seconds_per_source'] for key in chosen])),
            mean_last_discovery_s=float(np.mean([index[key,name]['last_discovery_s'] for key in chosen])),
            mean_visited_stations=float(np.mean([index[key,name]['visited_station_count'] for key in chosen]))) for name in methods})
    return dict(case_count=n,methods=summary,exploratory_contrasts=interactions,by_source_count=by_n)


def main():
    plan=json.loads((ROOT/'plan.json').read_text());methods=list(plan['methods'])
    rows=[json.loads(x) for x in (ROOT/'main/records.jsonl').read_text().splitlines()]
    expected={(case['key'],method) for case in plan['cases'] for method in methods}
    assert len(rows)==len(expected) and {(r['case_key'],r['method']) for r in rows}==expected
    failed=[r for r in rows if r['error'] or not r['all_cleared']]
    if failed:
        (ROOT/'failures.json').write_text(json.dumps(failed,ensure_ascii=False,indent=2))
        raise RuntimeError('Unsuccessful runs present; no success-only mean published')
    summary=dict(local_reconstructed_only=True,runs=len(rows),all_cleared=True,
        unique_new_cases=370,known_diagnostic_cases=3,
        new_source_instances=sum(r['source_total'] for r in rows if r['method']=='full' and r['group']!='known_regression_diagnostic'),
        new_source_clear_events=sum(r['source_total'] for r in rows if r['group']!='known_regression_diagnostic'),
        planner_wall_budget_reached=sum(bool(r['planner_budget_reached']) for r in rows),
        bootstrap_replicates=REPLICATES,primary_family=PRIMARY,groups={})
    for group in dict.fromkeys(case['group'] for case in plan['cases']):
        rr=[r for r in rows if r['group']==group]
        summary['groups'][group]=summarize_group(rr,methods)
        if group=='known_regression_diagnostic':
            diagnostic=summary['groups'][group]
            diagnostic['inference_note']='Hand-picked prior regression cases; descriptive diagnostics only, no population confidence intervals.'
            for item in diagnostic['methods'].values():
                for key in list(item):
                    if 'ci95' in key or 'ci99' in key:del item[key]
            for item in diagnostic['exploratory_contrasts'].values():item.pop('ci95',None)
        print('summarized',group,flush=True)
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    simple=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]
    with (ROOT/'paired_records.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(simple[0]));writer.writeheader();writer.writerows(simple)
    print(json.dumps({m:{k:d[k] for k in ['mean_seconds_per_source','change_vs_full_s','change_vs_full_ci95']}
                     for m,d in summary['groups']['new_practice']['methods'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
