from __future__ import annotations
import copy,json,random,time,concurrent.futures as cf
from validate_v5 import SCENARIOS,AuditSimulator,DeviceOnly,make_targets
from q4_state import Engine
from q4_v4_solver import V4Config
from q4_transit_critic import proposals,apply_probe

def generate(task):
    g,i=task;name,fraction,place,error=SCENARIOS[g];seed=181100000+g*100000+i
    sim=AuditSimulator(make_targets(seed,fraction,place),seed,error)
    engine=Engine(DeviceOnly(sim),V4Config());rng=random.Random(seed+799);rows=[];selected=0
    while not engine.done():
        order=engine.ordered_actions();base=order[0]
        cand=proposals(engine,base)
        if cand and selected<6 and rng.random()<.3:
            selected+=1
            env0=copy.deepcopy(sim);state=engine.state.clone();e0=Engine(DeviceOnly(env0),engine.config,state);e0.run_base();basecost=env0.virtual_seconds
            for proposal in cand:
                env1=copy.deepcopy(sim);st=engine.state.clone();e1=Engine(DeviceOnly(env1),engine.config,st)
                apply_probe(e1,proposal);e1.run_base()
                assert env1.clear_count==len(env1._targets)
                rows.append(dict(seed=seed,scenario=name,action=engine.state.actions,features=proposal['features'],saving_seconds=basecost-env1.virtual_seconds,targets=len(sim._targets)))
        engine.execute(base)
    return rows

if __name__=='__main__':
    tasks=[(g,i)for g in range(9)for i in range(240 if g<6 else 160)]
    start=time.perf_counter();n=0
    with cf.ProcessPoolExecutor(max_workers=3) as ex,open('critic_training_extra.jsonl','w') as f:
        for j,rows in enumerate(ex.map(generate,tasks)):
            for r in rows:f.write(json.dumps(r)+'\n')
            f.flush();n+=len(rows)
            if j%20==0:print(j+1,n,time.perf_counter()-start,flush=True)
    print('DONE',len(tasks),n,time.perf_counter()-start,flush=True)
