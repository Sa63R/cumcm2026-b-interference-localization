from experiments_round import *
from q4_discovery_value import DiscoveryEngine

def run_d(task):
    g,i,mode=task;name,fraction,place,error=SCENARIOS[g];seed=147100000+100000*g+i
    sim=AuditSimulator(make_targets(seed,fraction,place),seed,error);t=time.perf_counter()
    try:
        if mode=='v4':engine=Engine(DeviceOnly(sim),V4Config())
        else:
            ratio,bonus=map(float,mode.split('_'));engine=DiscoveryEngine(DeviceOnly(sim),V4Config(),ratio,bonus)
        r=engine.run_base();assert sim.clear_count==len(sim._targets)
        return dict(mode=mode,seed=seed,scenario=name,**sim.summary(),wall=time.perf_counter()-t,extras=r.get('extra_scans',0))
    except Exception:
        import traceback
        return dict(mode=mode,seed=seed,error=traceback.format_exc())
if __name__=='__main__':
    modes=['v4','1_0','1.5_0','2_0','1.5_20','2_40','3_0']
    with cf.ProcessPoolExecutor(max_workers=6) as ex,open('discovery_development.jsonl','w') as out:
        for r in ex.map(run_d,[(g,i,m) for g in range(6) for i in range(6) for m in modes]):out.write(json.dumps(r)+'\n');out.flush()
    rows=[json.loads(s) for s in open('discovery_development.jsonl')]
    for m in modes:
        rr=[r for r in rows if r['mode']==m];print(m,len(rr),sum('error' in r for r in rr),statistics.mean(r['seconds_per_cleared'] for r in rr if 'error' not in r),sum(r.get('extras',0) for r in rr),flush=True)
