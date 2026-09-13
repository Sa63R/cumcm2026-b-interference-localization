from experiments_round import *
from q4_transit_v5 import solve_transit_v5
from q4_v5 import solve_v5

def run_v(task):
    g,i,mode,seedbase=task;name,fraction,place,error=SCENARIOS[g];seed=seedbase+100000*g+i
    sim=AuditSimulator(make_targets(seed,fraction,place),seed,error);t=time.perf_counter()
    try:
        if mode=='v5':r=solve_v5(DeviceOnly(sim))
        else:
            parts=mode.split('_');th=float(parts[0]);project='p' in parts;refine='r' in parts
            r=solve_transit_v5(DeviceOnly(sim),threshold=th,project=project,refine=refine)
        assert sim.clear_count==len(sim._targets)
        assert abs(sim.virtual_seconds-(sim.distance_m/5+5*sim.detect_count+sim.switch_count+3*sim.optical_count+2*sim.clear_count))<1e-5
        return dict(mode=mode,seed=seed,scenario=name,**sim.summary(),wall=time.perf_counter()-t,transit_stops=r.get('transit_stops',0),transit_detections=r.get('transit_detections',0),tail=sim.virtual_seconds-sim.last_clear_time)
    except Exception:
        import traceback
        return dict(mode=mode,seed=seed,error=traceback.format_exc())
if __name__=='__main__':
    modes=sys.argv[1:] or ['v5','50','100','200','100_p','100_r','100_p_r']
    tasks=[(g,i,m,151100000) for g in range(6) for i in range(6) for m in modes]
    with cf.ProcessPoolExecutor(max_workers=6) as ex,open('transit_v5_development.jsonl','w') as out:
        for r in ex.map(run_v,tasks):out.write(json.dumps(r)+'\n');out.flush()
    rows=[json.loads(s) for s in open('transit_v5_development.jsonl')]
    for m in modes:
        rr=[r for r in rows if r['mode']==m];print(m,len(rr),sum('error' in r for r in rr),statistics.mean(r['seconds_per_cleared'] for r in rr if 'error' not in r),flush=True)
