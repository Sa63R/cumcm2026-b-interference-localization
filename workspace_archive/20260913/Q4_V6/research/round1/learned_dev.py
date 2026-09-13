from experiments_round import *
from q4_v5 import solve_v5
from q4_learned_transit import solve_learned

def runl(task):
    g,i,mode,base=task;name,fraction,place,error=SCENARIOS[g];seed=base+g*100000+i
    sim=AuditSimulator(make_targets(seed,fraction,place),seed,error);t=time.perf_counter()
    try:
        if mode=='v5':r=solve_v5(DeviceOnly(sim))
        else:
            loss,th,back=mode.split('_');model='transit_critic_'+('huber'if loss=='h'else'squared_error')+'.json'
            r=solve_learned(DeviceOnly(sim),model,float(th),v5=back=='v5')
        assert sim.clear_count==len(sim._targets)
        return dict(mode=mode,seed=seed,scenario=name,**sim.summary(),wall=time.perf_counter()-t,transit_stops=r.get('transit_stops',0),transit_detections=r.get('transit_detections',0))
    except Exception:
        import traceback
        return dict(mode=mode,seed=seed,error=traceback.format_exc())
if __name__=='__main__':
    modes=['v5']+[f'{loss}_{th}_{back}'for loss in['h','s']for th in[0,10,30]for back in['v4','v5']]
    tasks=[(g,i,m,155100000)for g in range(6)for i in range(10)for m in modes]
    with cf.ProcessPoolExecutor(max_workers=5) as ex,open('learned_development.jsonl','w') as out:
        for r in ex.map(runl,tasks):out.write(json.dumps(r)+'\n');out.flush()
    rows=[json.loads(s)for s in open('learned_development.jsonl')]
    for m in modes:
        rr=[r for r in rows if r['mode']==m];print(m,len(rr),sum('error' in r for r in rr),statistics.mean(r['seconds_per_cleared']for r in rr if 'error'not in r),statistics.mean(r.get('transit_stops',0)for r in rr),flush=True)
