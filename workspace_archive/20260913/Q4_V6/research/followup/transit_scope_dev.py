from transit_v5_develop import *

def run_scope(task):
    g,i,mode,base=task;name,fraction,place,error=SCENARIOS[g];seed=base+100000*g+i
    sim=AuditSimulator(make_targets(seed,fraction,place),seed,error);t=time.perf_counter()
    try:
        if mode=='v5':r=solve_v5(DeviceOnly(sim))
        else:
            th,scope=mode.split('_');r=solve_transit_v5(DeviceOnly(sim),threshold=float(th),scope=scope)
        assert sim.clear_count==len(sim._targets)
        return dict(mode=mode,seed=seed,scenario=name,**sim.summary(),wall=time.perf_counter()-t,transit_stops=r.get('transit_stops',0),transit_detections=r.get('transit_detections',0),tail=sim.virtual_seconds-sim.last_clear_time)
    except Exception:
        import traceback
        return dict(mode=mode,seed=seed,error=traceback.format_exc())
if __name__=='__main__':
    modes=['v5']+[f'{th}_{sc}' for th in [20,50,100] for sc in ['sites','others','targets']]
    tasks=[(g,i,m,151100000) for g in range(6) for i in range(6) for m in modes]
    import os
    done={(r['seed'],r['mode']) for r in map(json.loads,open('transit_scope_development.jsonl'))} if os.path.exists('transit_scope_development.jsonl') else set()
    tasks=[t for t in tasks if(t[3]+100000*t[0]+t[1],t[2]) not in done]
    with cf.ProcessPoolExecutor(max_workers=6) as ex,open('transit_scope_development.jsonl','a') as out:
        for r in ex.map(run_scope,tasks):out.write(json.dumps(r)+'\n');out.flush()
    rows=[json.loads(s) for s in open('transit_scope_development.jsonl')]
    for m in modes:
        rr=[r for r in rows if r['mode']==m];print(m,len(rr),sum('error' in r for r in rr),statistics.mean(r['seconds_per_cleared'] for r in rr if 'error' not in r),flush=True)
