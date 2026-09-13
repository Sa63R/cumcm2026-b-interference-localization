from __future__ import annotations
import copy,json,random,time,concurrent.futures as cf
from validate_v5 import SCENARIOS,AuditSimulator,DeviceOnly,make_targets
from q4_state import Engine
from q4_v4_solver import V4Config
from q4_route_critic import route_features

def generate(task):
    g,i=task;name,fraction,place,error=SCENARIOS[g];seed=171100000+g*100000+i
    sim=AuditSimulator(make_targets(seed,fraction,place),seed,error);engine=Engine(DeviceOnly(sim),V4Config())
    rng=random.Random(seed+886);rows=[];selected=0
    while not engine.done():
        order=engine.ordered_actions();base=order[0]
        if len(order)>1 and engine.state.visited and engine.state.pending and selected<6 and rng.random()<.22:
            selected+=1;aa=route_features(engine,order)
            env0=copy.deepcopy(sim);Engine(DeviceOnly(env0),engine.config,engine.state.clone()).run_base();cost=env0.virtual_seconds
            for a,x in aa:
                if a==base:saving=0.
                else:
                    sim1=copy.deepcopy(sim);e1=Engine(DeviceOnly(sim1),engine.config,engine.state.clone());e1.execute(a);e1.run_base();saving=cost-sim1.virtual_seconds
                    assert sim1.clear_count==len(sim1._targets)
                rows.append(dict(seed=seed,scenario=name,action=engine.state.actions,features=x,saving_seconds=saving,targets=len(sim._targets)))
        engine.execute(base)
    return rows
if __name__=='__main__':
    tasks=[(g,i)for g in range(9)for i in range(60 if g<6 else 40)]
    start=time.perf_counter();n=0
    with cf.ProcessPoolExecutor(max_workers=5)as ex,open('route_critic_training.jsonl','w')as f:
        for j,rows in enumerate(ex.map(generate,tasks)):
            for r in rows:f.write(json.dumps(r)+'\n')
            f.flush();n+=len(rows)
            if j%20==0:print(j+1,n,time.perf_counter()-start,flush=True)
    print('DONE',len(tasks),n,time.perf_counter()-start,flush=True)
