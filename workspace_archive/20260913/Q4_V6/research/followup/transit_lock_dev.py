from transit_v5_develop import *

def runlock(task):
    g,i,mode,base=task;name,fraction,place,error=SCENARIOS[g];seed=base+g*100000+i
    sim=AuditSimulator(make_targets(seed,fraction,place),seed,error);t=time.perf_counter()
    try:
        if mode=='v5':r=solve_v5(DeviceOnly(sim))
        else:
            th,stop,lock=map(int,mode.split('_'));r=solve_transit_v5(DeviceOnly(sim),threshold=th,maxstops=stop,lock=bool(lock))
        assert sim.clear_count==len(sim._targets)
        return dict(mode=mode,seed=seed,scenario=name,**sim.summary(),wall=time.perf_counter()-t,transit_stops=r.get('transit_stops',0),transit_detections=r.get('transit_detections',0),tail=sim.virtual_seconds-sim.last_clear_time)
    except Exception:
        import traceback
        return dict(mode=mode,seed=seed,error=traceback.format_exc())
if __name__=='__main__':
    modes=['v5','50_16_0','50_8_0','50_16_1','100_16_1','20_16_1','50_32_1']
    tasks=[(g,i,m,155100000)for g in range(6)for i in range(10)for m in modes]
    with cf.ProcessPoolExecutor(max_workers=6) as ex,open('transit_lock_development.jsonl','w') as out:
        for r in ex.map(runlock,tasks):out.write(json.dumps(r)+'\n');out.flush()
    rows=[json.loads(s) for s in open('transit_lock_development.jsonl')]
    for m in modes:
        rr=[r for r in rows if r['mode']==m];print(m,len(rr),sum('error' in r for r in rr),statistics.mean(r['seconds_per_cleared'] for r in rr if 'error' not in r),statistics.mean(r.get('transit_stops',0)for r in rr),flush=True)
