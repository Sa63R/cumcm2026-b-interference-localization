"""Demo is a synthetic LOCAL_ONLY session, not an official contest result."""
import argparse,json,dataclasses,time
from validate_v5 import AuditSimulator,DeviceOnly,make_targets,SCENARIOS
from q4_v5 import solve_v5
from q4_v6 import solve_v6,default_config
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['v5','v6','cheap','transit_only','route_only','unprotected'],default='v6');p.add_argument('--seed',type=int,default=191200000);p.add_argument('--scenario',type=int,choices=range(9),default=1);p.add_argument('--out',default='LOCAL_ONLY_demo.json');a=p.parse_args()
    name,fr,place,err=SCENARIOS[a.scenario];sim=AuditSimulator(make_targets(a.seed,fr,place),a.seed,err,trace=True);tic=time.perf_counter()
    if a.mode=='v5':report=solve_v5(DeviceOnly(sim))
    elif a.mode=='unprotected':
        from q4_v6_unprotected import solve_v6 as unprotected
        report=unprotected(DeviceOnly(sim))
    else:
        c=default_config()
        if a.mode=='cheap':c=dataclasses.replace(c,use_v5_local=False)
        elif a.mode=='transit_only':c=dataclasses.replace(c,route_ranking=False)
        elif a.mode=='route_only':c=dataclasses.replace(c,transit_sensing=False)
        report=solve_v6(DeviceOnly(sim),c)
    out={'local_only':True,'mode':a.mode,'seed':a.seed,'scenario':name,'wall_seconds':time.perf_counter()-tic,'summary':sim.summary(),'report':report,'actions':sim.trace}
    with open(a.out,'w',encoding='utf-8')as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in out.items()if k not in ['actions','report']},ensure_ascii=False,indent=2))
