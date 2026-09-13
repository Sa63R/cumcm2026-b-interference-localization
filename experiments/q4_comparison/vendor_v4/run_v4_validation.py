"""Frozen paired validation. Local synthetic cases, not official test sessions.

python run_v4_validation.py --cases 100 --workers 4
Historical: three previous 600-case suites. New holdout: 600 untouched seeds.
Default versions: V3/V4 everywhere, conservative V4 and optical-only ablation
on new holdout. No policy parameter is selected with holdout outcomes.
"""
from __future__ import annotations
import argparse,concurrent.futures,csv,dataclasses,hashlib,json,math,statistics,time,traceback
from pathlib import Path
import q4_baseline as b
from q4_v3_solver import solve_v3,V3Config
from q4_v4_solver import solve_v4,V4Config
from q4_v4_local import LocalConfig
from run_v3_validation import SCENARIOS
SUITES=[('original',20260910),('v2_holdout',40160910),('v3_holdout',70160910),('fresh_v4_holdout',90110000)]

def config_for(version):
    if version=='v3':return V3Config()
    if version=='v4':return V4Config()
    if version=='v4_conservative':return V4Config(local=LocalConfig(optical_cover_limit=6))
    if version=='optical_only':return V4Config(local=LocalConfig(optical_cover_limit=12,pause_after_pair=False))
    raise ValueError(version)

def single(task):
    si,gi,i,version,trace=task
    suite,base=SUITES[si];scenario,fraction,placement,error=SCENARIOS[gi]
    seed=base+gi*100000+i
    sim=b.LocalSimulator(b.make_case(seed,fraction,placement),seed,error,trace)
    begin=time.perf_counter()
    config=config_for(version)
    rep=solve_v3(sim,config) if version=='v3' else solve_v4(sim,config)
    wall=time.perf_counter()-begin
    row=dict(suite=suite,scenario=scenario,seed=seed,version=version,**sim.summary(),wall_seconds=wall,
             visited_stations=len(rep['visited_stations']),shared_detections=rep['shared_detections'],
             rf_replans=rep.get('replans_after_pairs',0),
             optical_cover_sources=sum(t['certificate']=='optical_cover_confirmed' for t in rep['localizations']),
             fallbacks=sum('fallback' in t['certificate'] for t in rep['localizations']),
             stop_certificate=rep['stop_certificate'])
    assert row['cleared']==row['targets'],row
    assert row['virtual_seconds']<360000,row
    expected=sim.distance_m/5+5*sim.detect_count+sim.switch_count+3*sim.optical_count+2*sim.clear_count
    assert abs(expected-sim.virtual_seconds)<1e-6,row
    traceout=dict(warning='LOCAL_ONLY: NOT official practice or a formal test log',row=row,report=rep,actions=sim.trace) if trace else None
    return row,traceout

def summary(rows):
    result={}
    for v in sorted({r['version'] for r in rows}):
        data=[r for r in rows if r['version']==v]
        result[v]=dict(cases=len(data),all_cleared_cases=sum(r['cleared']==r['targets'] for r in data),sources=sum(r['targets'] for r in data),cleared=sum(r['cleared'] for r in data),fallbacks=sum(r['fallbacks'] for r in data))
        for k in ['seconds_per_cleared','virtual_seconds','distance_m','detections','switches','optical_attempts','failed_clears','wall_seconds','visited_stations','shared_detections','rf_replans']:
            result[v]['mean_'+k]=statistics.mean(r[k] for r in data)
        result[v]['max_wall_seconds']=max(r['wall_seconds'] for r in data)
        if v=='v3':continue
        old={r['seed']:r for r in rows if r['version']=='v3'}
        dd=[(old[r['seed']]['virtual_seconds']-r['virtual_seconds'])/r['targets'] for r in data]
        relative=[1-r['virtual_seconds']/old[r['seed']]['virtual_seconds'] for r in data]
        mo=statistics.mean(old[r['seed']]['seconds_per_cleared'] for r in data)
        se=statistics.stdev(dd)/math.sqrt(len(dd)) if len(dd)>1 else 0
        dmiss=statistics.mean((r['failed_clears']-old[r['seed']]['failed_clears'])/r['targets'] for r in data)
        result[v]['paired']=dict(mean_time_reduction=statistics.mean(dd)/mo,mean_saved_seconds_per_source=statistics.mean(dd),normal95_saved_seconds_per_source=[statistics.mean(dd)-1.96*se,statistics.mean(dd)+1.96*se],faster=sum(d>1e-7 for d in dd),slower=sum(d<-1e-7 for d in dd),tied=sum(abs(d)<=1e-7 for d in dd),worst_relative=min(relative),best_relative=max(relative),median_relative=statistics.median(relative),additional_misses_per_source=dmiss,
            failure_cost_sensitivity_fixed_policy={str(cost):(statistics.mean(dd)-(cost-3)*dmiss)/(mo+(cost-3)*statistics.mean(old[r['seed']]['failed_clears']/r['targets'] for r in data)) for cost in [3,5,10,15]})
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=int,default=100);p.add_argument('--workers',type=int,default=4);p.add_argument('--out',type=Path,default=Path('results_v4'));p.add_argument('--fresh-only',action='store_true')
    args=p.parse_args()
    if args.cases<1 or args.workers<1:p.error('positive cases and workers required')
    args.out.mkdir(parents=True,exist_ok=True)
    suites=[3] if args.fresh_only else range(4)
    tasks=[(si,gi,i,v,si==3 and gi==1 and i<3 and v in ['v3','v4']) for si in suites for gi in range(6) for i in range(args.cases) for v in (['v3','v4','v4_conservative','optical_only'] if si==3 else ['v3','v4'])]
    rows=[];tic=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for row,trace in ex.map(single,tasks,chunksize=5):
            rows.append(row)
            if trace:(args.out/f"LOCAL_ONLY_{row['version']}_{row['seed']}.json").write_text(json.dumps(trace,indent=2),encoding='utf-8')
            if len(rows)%200==0:print('completed',len(rows),'elapsed',time.perf_counter()-tic,flush=True)
    bysuite={name:summary([r for r in rows if r['suite']==name]) for name,base in SUITES if any(r['suite']==name for r in rows)}
    bygroup={f'{name}:{scenario}':summary([r for r in rows if r['suite']==name and r['scenario']==scenario]) for name,base in SUITES for scenario,*_ in SCENARIOS if any(r['suite']==name and r['scenario']==scenario for r in rows)}
    result=dict(warning='Self-built simulation only. NOT official practice, formal tests or encrypted logs.',configuration=dataclasses.asdict(V4Config()),development_seeds=['81110000+g*100000+i, g=0..5, i=0..29','82110000+g*100000+i, g=0..5, i=0..49'],optical_miss_seconds=3,by_suite=bysuite,groups=bygroup,overall=summary(rows),policy_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('q4_*.py')},wall_seconds=time.perf_counter()-tic)
    (args.out/'validation_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (args.out/'paired_cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('FINAL',json.dumps(bysuite),flush=True)
if __name__=='__main__':main()
