"""Second, independently frozen test after diagnosing first-round regressions.
The 780-case first test is diagnostic/development for this follow-up, not reused
as validation. All three policies run every fresh case, including difficult ones.
"""
from __future__ import annotations
import argparse,concurrent.futures as cf,dataclasses,datetime,hashlib,json,pathlib,time,traceback
from validate_v5 import SCENARIOS,DeviceOnly,AuditSimulator,make_targets
from q4_v5 import solve_v5
from q4_v6 import solve_v6,default_config
from q4_v6_unprotected import solve_v6 as unprotected

def hashes():
    root=pathlib.Path(__file__).parent
    paths=list(root.glob('q4_*.py'))+[root/'validate_v5.py',root/'validate_final.py',root/'radius_direction.py',root/'route_critic_extra.json',root/'transit_critic_big_extra.json']
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}

def run(task):
    g,i,v,base,out=task;name,frac,place,err=SCENARIOS[g];seed=base+g*100000+i
    trace=i==0 and g in [1,3,6]
    sim=AuditSimulator(make_targets(seed,frac,place),seed,err,trace=trace)
    tic=time.perf_counter();cpu=time.process_time()
    try:
        if v=='v5':r=solve_v5(DeviceOnly(sim))
        elif v=='unprotected':r=unprotected(DeviceOnly(sim))
        elif v=='v6':r=solve_v6(DeviceOnly(sim))
        else:raise ValueError(v)
        assert sim.clear_count==len(sim._targets),'Sources not all cleared'
        assert r['stop_certificate']!='in_progress'
        assert sim.virtual_seconds<360000.,'Virtual limit'
        expected=sim.distance_m/5+5*sim.detect_count+sim.switch_count+3*sim.optical_count+2*sim.clear_count
        assert abs(sim.virtual_seconds-expected)<1e-5,'Accounting error'
        row=dict(version=v,seed=seed,scenario=name,**sim.summary(),wall_seconds=time.perf_counter()-tic,cpu_seconds=time.process_time()-cpu,
                 tail_after_last_clear=sim.virtual_seconds-sim.last_clear_time,certificate=r['stop_certificate'],
                 transit_stops=r.get('transit_stops',0),transit_detections=r.get('transit_detections',0),route_calls=r.get('route_calls',0),route_accepts=r.get('route_accepts',0),
                 guarded_steps=r.get('guarded_steps',0),activation=r.get('activation'),planner_calls=r.get('planner_calls',0),planner_seconds=r.get('planner_seconds',0.))
        if trace:
            p=pathlib.Path(out)/'local_logs';p.mkdir(exist_ok=True,parents=True)
            (p/f'LOCAL_ONLY_{seed}_{v}.json').write_text(json.dumps({'local_only':True,'row':row,'actions':sim.trace,'algorithm_report':r},ensure_ascii=False,indent=2))
        return row
    except Exception:
        return dict(version=v,seed=seed,scenario=name,error=traceback.format_exc(),wall_seconds=time.perf_counter()-tic)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--main-cases',type=int,default=80);p.add_argument('--stress-cases',type=int,default=40);p.add_argument('--seed-base',type=int,default=231100000);p.add_argument('--workers',type=int,default=5);p.add_argument('--out',default='results_final');a=p.parse_args()
    root=pathlib.Path(a.out);root.mkdir(exist_ok=True,parents=True);freeze=root/'frozen_config.json'
    plan={k:getattr(a,k)for k in ['main_cases','stress_cases','seed_base']}
    if a.freeze:
        if freeze.exists():raise SystemExit('Existing frozen plan not overwritten.')
        record={'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'configuration':dataclasses.asdict(default_config()),'source_hashes':hashes(),'evaluation':plan,
                'versions':['v5','unprotected','v6'],'all_versions_on_all_cases':True,'local_only':True,'failed_optical_seconds':3,
                'prior_780_case_test':'Used to diagnose the N=16 early-discovery failure mode. NOT independent validation of the guarded method.',
                'guard_selection':'Compare delays 2,3,4,5,6 and two model combinations on the prior 90-case development set. Select four completed scan stations to reduce development tail regressions while retaining most mean benefit (453.602 vs baseline460.401; three stations452.612 has larger tail). This is an explicit mean/risk tradeoff, not the minimum development mean.',
                'negative_experiment':'Count-triggered additional unknown-channel scans at 13,14,15 known sources failed all nine development configurations; DISABLED.',
                'training_unchanged_since_first_freeze':True,'selected_training_seed_bases':[171100000,181100000],
                'new_test_data_note':'Additional distributions occur in training; these are independent new cases, not unseen distributions.'}
        freeze.write_text(json.dumps(record,ensure_ascii=False,indent=2));print(freeze);raise SystemExit(0)
    frozen=json.loads(freeze.read_text());assert frozen['source_hashes']==hashes(),'Post-freeze source/model change';assert frozen['evaluation']==plan;assert frozen['configuration']==dataclasses.asdict(default_config())
    rows=root/'records.jsonl'
    if rows.exists():raise SystemExit('Refusing to overwrite prior evaluation records.')
    tasks=[(g,i,v,a.seed_base,str(root))for g in range(9)for i in range(a.main_cases if g<6 else a.stress_cases)for v in frozen['versions']]
    n=0;tic=time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=a.workers)as ex,rows.open('w')as f:
        for r in ex.map(run,tasks):
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();n+=1
            if n%100==0:print(n,len(tasks),round(time.perf_counter()-tic,1),flush=True)
    assert frozen['source_hashes']==hashes(),'Source/model changed during evaluation';print('DONE',n,time.perf_counter()-tic,flush=True)
