from experiments_round import *
from q4_phase_rollout import PhaseEngine

def runphase(task):
    g,i,mode=task;name,fraction,place,error=SCENARIOS[g];seed=147100000+100000*g+i
    sim=AuditSimulator(make_targets(seed,fraction,place),seed,error);tic=time.perf_counter()
    try:
        engine=Engine(DeviceOnly(sim),V4Config()) if mode=='v4' else PhaseEngine(DeviceOnly(sim),V4Config(),angles=8,scenes=12,validation=12)
        r=engine.run_base();assert sim.clear_count==len(sim._targets)
        return dict(mode=mode,seed=seed,scenario=name,**sim.summary(),wall=time.perf_counter()-tic,phase=getattr(engine,'phase_record',{}))
    except Exception:
        import traceback
        return dict(mode=mode,seed=seed,error=traceback.format_exc())
if __name__=='__main__':
    modes=['v4','phase'];tasks=[(g,i,m)for g in range(6)for i in range(6)for m in modes]
    with cf.ProcessPoolExecutor(max_workers=6) as ex,open('phase_development.jsonl','w') as out:
        for r in ex.map(runphase,tasks):out.write(json.dumps(r)+'\n');out.flush()
    rows=[json.loads(s) for s in open('phase_development.jsonl')]
    for m in modes:
        rr=[r for r in rows if r['mode']==m];print(m,len(rr),sum('error' in r for r in rr),statistics.mean(r['seconds_per_cleared'] for r in rr if 'error' not in r),flush=True)
