from experiments_round import *
from q4_transit import TransitEngine

def runtr(task):
    g,i,mode=task;name,fraction,place,error=SCENARIOS[g];seed=147100000+g*100000+i
    sim=AuditSimulator(make_targets(seed,fraction,place),seed,error);t=time.perf_counter()
    try:
        if mode=='v4':engine=Engine(DeviceOnly(sim),V4Config())
        else:
            th,frac=mode.split('_');fs=(.25,.5,.75) if frac=='3' else(.5,)
            engine=TransitEngine(DeviceOnly(sim),V4Config(),threshold=float(th),fractions=fs)
        r=engine.run_base();assert sim.clear_count==len(sim._targets)
        return dict(mode=mode,seed=seed,scenario=name,**sim.summary(),wall=time.perf_counter()-t,transits=r.get('transit_stops',0))
    except Exception:
        import traceback
        return dict(mode=mode,seed=seed,error=traceback.format_exc())
if __name__=='__main__':
    modes=['v4']+[f'{t}_{f}' for t in [20,50,100,200] for f in ['1','3']]
    with cf.ProcessPoolExecutor(max_workers=6) as ex,open('transit_development.jsonl','w') as out:
        for r in ex.map(runtr,[(g,i,m) for g in range(6) for i in range(6) for m in modes]):out.write(json.dumps(r)+'\n');out.flush()
    rows=[json.loads(s) for s in open('transit_development.jsonl')]
    for m in modes:
        rr=[r for r in rows if r['mode']==m];print(m,len(rr),sum('error' in r for r in rr),statistics.mean(r['seconds_per_cleared'] for r in rr if 'error' not in r),flush=True)
