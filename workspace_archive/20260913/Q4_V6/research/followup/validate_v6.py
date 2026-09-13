"""Independent paired LOCAL_ONLY evaluation; no policy may read ground truth.
Predeclared: 600 main + 180 additional-distribution cases. Ablations use the
first 20 cases of each of the nine groups, regardless of observed performance.
"""
from __future__ import annotations
import argparse,concurrent.futures as cf,dataclasses,hashlib,json,pathlib,time,traceback
from validate_v5 import SCENARIOS,DeviceOnly,AuditSimulator,make_targets
from q4_v5 import solve_v5
from q4_v6 import solve_v6,default_config


def run(task):
    g,i,version,base,out=task;name,fr,placement,error=SCENARIOS[g];seed=base+g*100000+i
    trace=i==0 and g in [1,3,6]
    sim=AuditSimulator(make_targets(seed,fr,placement),seed,error,trace=trace)
    tic=time.perf_counter();cpu=time.process_time()
    try:
        if version=='v5':report=solve_v5(DeviceOnly(sim))
        else:
            cfg=default_config()
            if version=='transit_only':cfg=dataclasses.replace(cfg,route_ranking=False)
            elif version=='route_only':cfg=dataclasses.replace(cfg,transit_sensing=False)
            elif version=='cheap':cfg=dataclasses.replace(cfg,use_v5_local=False)
            elif version!='v6':raise ValueError(version)
            report=solve_v6(DeviceOnly(sim),cfg)
        assert sim.clear_count==len(sim._targets),'Incomplete clear'
        assert report['stop_certificate']!='in_progress'
        assert sim.virtual_seconds<360000.,'Virtual limit exceeded'
        assert abs(sim.virtual_seconds-(sim.distance_m/5+5*sim.detect_count+sim.switch_count+3*sim.optical_count+2*sim.clear_count))<1e-5,'Incorrect accounting'
        row=dict(version=version,seed=seed,scenario=name,**sim.summary(),wall_seconds=time.perf_counter()-tic,cpu_seconds=time.process_time()-cpu,
            tail_after_last_clear=sim.virtual_seconds-sim.last_clear_time,certificate=report['stop_certificate'],
            transit_stops=report.get('transit_stops',0),transit_detections=report.get('transit_detections',0),
            route_calls=report.get('route_calls',0),route_accepts=report.get('route_accepts',0),
            planner_calls=report.get('planner_calls',0),planner_seconds=report.get('planner_seconds',0.))
        if trace:
            p=pathlib.Path(out)/'local_logs';p.mkdir(exist_ok=True,parents=True)
            (p/f'LOCAL_ONLY_{seed}_{version}.json').write_text(json.dumps({'local_only':True,'row':row,'actions':sim.trace,'algorithm_report':report},ensure_ascii=False,indent=2))
        return row
    except Exception:
        return dict(version=version,seed=seed,scenario=name,error=traceback.format_exc(),wall_seconds=time.perf_counter()-tic)


def hashes():
    root=pathlib.Path(__file__).parent
    files=list(root.glob('q4_*.py'))+[root/'validate_v5.py',root/'validate_v6.py',root/'radius_direction.py',root/'route_critic_extra.json',root/'transit_critic_big_extra.json']
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main-cases',type=int,default=100);p.add_argument('--stress-cases',type=int,default=60);p.add_argument('--ablation-cases',type=int,default=20);p.add_argument('--seed-base',type=int,default=191100000);p.add_argument('--workers',type=int,default=5);p.add_argument('--out',default='results_v6');a=p.parse_args()
    out=pathlib.Path(a.out);out.mkdir(exist_ok=True,parents=True)
    freeze=out/'frozen_config.json'
    if not freeze.exists():raise SystemExit('Create/review the frozen experiment manifest before launching validation.')
    frozen=json.loads(freeze.read_text());assert frozen['source_hashes']==hashes(),'Source/model changed after freeze'
    assert frozen['configuration']==dataclasses.asdict(default_config()),'Default configuration changed after freeze'
    args={k:getattr(a,k)for k in ['main_cases','stress_cases','ablation_cases','seed_base']}
    assert args==frozen['evaluation'],('The evaluation plan differs from the frozen manifest',args)
    tasks=[]
    for g in range(9):
        for i in range(a.main_cases if g<6 else a.stress_cases):
            modes=['v5','v6']+(['transit_only','route_only','cheap']if i<a.ablation_cases else [])
            for m in modes:tasks.append((g,i,m,a.seed_base,str(out)))
    start=time.perf_counter();n=0
    with cf.ProcessPoolExecutor(max_workers=a.workers)as ex,(out/'records.jsonl').open('w')as f:
        for r in ex.map(run,tasks):
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();n+=1
            if n%100==0:print(n,len(tasks),round(time.perf_counter()-start,2),flush=True)
    assert frozen['source_hashes']==hashes(),'Source/model changed during evaluation'
    print('DONE',n,time.perf_counter()-start,flush=True)
