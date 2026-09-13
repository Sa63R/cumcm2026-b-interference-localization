"""Paired V2/V3 validation on frozen historical and fresh synthetic scenarios.

Python 3.10+, standard library only. No official HTTP session or test log is
created. Defaults: 3 suites * 6 scenarios * 100 cases, each run on both solvers.
"""
from __future__ import annotations
import argparse,concurrent.futures,csv,dataclasses,hashlib,json,math,random,statistics,time
from pathlib import Path
import q4_baseline as b
from q4_fast_solver import solve_fast,FastConfig
from q4_v3_solver import solve_v3,V3Config

SCENARIOS=[('mixed_25pct',.25,'uniform','hash'),('mixed_50pct',.5,'uniform','hash'),('mixed_75pct',.75,'uniform','hash'),('outward_boundary',.75,'outward_boundary','hash'),('constant_plus1deg',.5,'uniform','plus'),('constant_minus1deg',.5,'uniform','minus')]
SUITES=[('original_cases',20260910),('previous_holdout',40160910),('fresh_holdout_v3',70160910)]

def single(task):
    si,gi,i,version,trace=task
    suite,base=SUITES[si];scenario,frac,placement,error=SCENARIOS[gi]
    seed=base+gi*100000+i
    sim=b.LocalSimulator(b.make_case(seed,frac,placement),seed,error,trace=trace)
    begin=time.perf_counter()
    report=solve_fast(sim,FastConfig()) if version=='v2' else solve_v3(sim,V3Config())
    row=dict(suite=suite,scenario=scenario,seed=seed,version=version,**sim.summary(),wall_seconds=time.perf_counter()-begin,
             visited_stations=len(report['visited_stations']),
             optical_cover_sources=sum(t['certificate']=='optical_cover_confirmed' for t in report['localizations']),
             negative_updates=sum(t.get('negative_updates',0) for t in report['localizations']),
             fallbacks=sum('fallback' in t['certificate'] for t in report['localizations']),
             shared_detections=report.get('shared_detections',0),stop_certificate=report['stop_certificate'])
    assert row['cleared']==row['targets'],row
    assert row['virtual_seconds']<360000,row
    formula=sim.distance_m/5+5*sim.detect_count+sim.switch_count+3*sim.optical_count+2*sim.clear_count
    assert abs(formula-sim.virtual_seconds)<1e-6,row
    return row,dict(warning='LOCAL ONLY: NOT official practice or a formal competition log',row=row,report=report,actions=sim.trace) if trace else None

def aggregate(rows):
    result={}
    for v in ('v2','v3'):
        data=[r for r in rows if r['version']==v]
        if not data:continue
        stats=dict(cases=len(data),all_cleared_cases=sum(r['targets']==r['cleared'] for r in data),sources=sum(r['targets'] for r in data),cleared=sum(r['cleared'] for r in data),optical_misses=sum(r['failed_clears'] for r in data),fallbacks=sum(r['fallbacks'] for r in data))
        for k in ('seconds_per_cleared','virtual_seconds','distance_m','detections','switches','wall_seconds','visited_stations','optical_attempts','failed_clears','optical_cover_sources','negative_updates','shared_detections'):
            stats['mean_'+k]=statistics.mean(r[k] for r in data)
        stats['max_wall_seconds']=max(r['wall_seconds'] for r in data)
        result[v]=stats
    if len(result)==2:
        old={r['seed']:r for r in rows if r['version']=='v2'};new={r['seed']:r for r in rows if r['version']=='v3'}
        diffs=[(old[s]['virtual_seconds']-new[s]['virtual_seconds'])/old[s]['targets'] for s in old]
        ratios=[1-new[s]['virtual_seconds']/old[s]['virtual_seconds'] for s in old]
        loss=[-d for d in diffs if d<0]
        miss_rate=statistics.mean(r['failed_clears']/r['targets'] for r in new.values())
        result['paired']=dict(mean_time_reduction=1-result['v3']['mean_seconds_per_cleared']/result['v2']['mean_seconds_per_cleared'],mean_saved_seconds_per_source=statistics.mean(diffs),faster_cases=sum(d>1e-7 for d in diffs),slower_cases=sum(d<-1e-7 for d in diffs),tied_cases=sum(abs(d)<=1e-7 for d in diffs),median_case_reduction=statistics.median(ratios),worst_case_reduction=min(ratios),best_case_reduction=max(ratios),mean_misses_per_source=miss_rate,
                              failure_cost_break_even_seconds=3+statistics.mean(diffs)/miss_rate if miss_rate else None)
        if len(diffs)>1:
            se=statistics.stdev(diffs)/math.sqrt(len(diffs))
            result['paired']['paired_normal95_saved_seconds_per_source']=[statistics.mean(diffs)-1.96*se,statistics.mean(diffs)+1.96*se]
        result['paired']['sensitivity_fixed_actions']={str(cost):1-(result['v3']['mean_seconds_per_cleared']+(cost-3)*miss_rate)/result['v2']['mean_seconds_per_cleared'] for cost in [3,5,10]}
    return result

def run(count,out,workers,suites):
    out.mkdir(parents=True,exist_ok=True)
    tasks=[(si,gi,i,v,si==2 and gi==1 and i<3) for si in range(suites) for gi in range(6) for i in range(count) for v in ['v2','v3']]
    rows=[];groups={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for j,(row,trace) in enumerate(ex.map(single,tasks,chunksize=5)):
            rows.append(row)
            if trace:(out/f"LOCAL_ONLY_{row['version']}_{row['seed']}.json").write_text(json.dumps(trace,indent=2),encoding='utf-8')
            if (j+1)%(2*count)==0:
                data=rows[-2*count:];key=row['suite']+':'+row['scenario'];groups[key]=aggregate(data)
                print(key,json.dumps(groups[key]),flush=True)
    overall=aggregate(rows)
    bysuite={name:aggregate([r for r in rows if r['suite']==name]) for name,base in SUITES[:suites]}
    result=dict(warning='SELF-BUILT SIMULATION; NOT official practice, formal test, or encrypted log.',case_count=count*suites*6,configuration=dataclasses.asdict(V3Config()),development_seeds='58110000 + g*100000 + [0,19]; 59110000 + g*100000 + [0,49], g=0..2; plus separate geometry unit tests',optical_miss_cost_assumption_seconds=3,groups=groups,by_suite=bysuite,overall=overall,rows=rows,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('q4_*.py')})
    (out/'paired_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'paired_cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('OVERALL',json.dumps(overall),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=int,default=100)
    p.add_argument('--suites',type=int,choices=[1,2,3],default=3)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--out',type=Path,default=Path('results_v3'))
    a=p.parse_args()
    if a.cases<1 or a.workers<1:p.error('cases and workers must be positive')
    run(a.cases,a.out,a.workers,a.suites)
