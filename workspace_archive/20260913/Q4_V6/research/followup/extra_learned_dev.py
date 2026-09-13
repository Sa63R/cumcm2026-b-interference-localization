from experiments_round import *
from q4_learned_transit import solve_learned
from q4_learned_route import solve_learned_route

def runx(task):
    g,i,mode,base=task;name,fraction,place,error=SCENARIOS[g];seed=base+g*100000+i
    sim=AuditSimulator(make_targets(seed,fraction,place),seed,error);t=time.perf_counter()
    try:
        scope,model,th,tr=mode.split('_')
        if scope=='transit':r=solve_learned(DeviceOnly(sim),'transit_critic_'+model+'.json',float(th),v5=True)
        else:r=solve_learned_route(DeviceOnly(sim),'route_critic_'+model+'.json',float(th),v5=True,transit='transit_critic_huber.json'if tr=='t'else None)
        assert sim.clear_count==len(sim._targets)
        return dict(mode=mode,seed=seed,scenario=name,**sim.summary(),wall=time.perf_counter()-t,
                    transit_stops=r.get('transit_stops',0),route_accepts=r.get('route_accepts',0))
    except Exception:
        import traceback
        return dict(mode=mode,seed=seed,error=traceback.format_exc())
if __name__=='__main__':
    modes=[f'transit_{loss}_{th}_n'for loss in['extra','forest']for th in[0,10,30]]+[f'route_{loss}_{th}_{tr}'for loss in['extra','forest']for th in[0,20]for tr in['n','t']]
    tasks=[(g,i,m,155100000)for g in range(6)for i in range(10)for m in modes]
    with cf.ProcessPoolExecutor(max_workers=4) as ex,open('extra_learned_development.jsonl','w') as out:
        for r in ex.map(runx,tasks):out.write(json.dumps(r)+'\n');out.flush()
    rows=[json.loads(s)for s in open('extra_learned_development.jsonl')]
    for m in modes:
        rr=[r for r in rows if r['mode']==m];print(m,len(rr),sum('error' in r for r in rr),statistics.mean(r['seconds_per_cleared']for r in rr if 'error'not in r),statistics.mean(r.get('transit_stops',0)for r in rr),flush=True)
