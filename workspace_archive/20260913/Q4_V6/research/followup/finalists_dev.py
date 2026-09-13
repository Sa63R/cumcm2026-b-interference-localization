from experiments_round import *
from q4_v5 import solve_v5
from q4_learned_transit import solve_learned
from q4_learned_route import solve_learned_route
from q4_transit_v5 import solve_transit_v5

def runf(task):
 g,i,mode,base=task;name,fr,place,err=SCENARIOS[g];seed=base+100000*g+i
 sim=AuditSimulator(make_targets(seed,fr,place),seed,err);t=time.perf_counter()
 try:
  if mode=='v5':r=solve_v5(DeviceOnly(sim))
  elif mode=='heur':r=solve_transit_v5(DeviceOnly(sim),threshold=100.,lock=True)
  else:
   route,transit,threshold,back=mode.split(':');back=back=='v5';tm='transit_critic_'+transit+'.json'
   if route=='none':r=solve_learned(DeviceOnly(sim),tm,float(threshold),v5=back)
   else:r=solve_learned_route(DeviceOnly(sim),'route_critic_'+route+'.json',0.,v5=back,transit=tm,transit_threshold=float(threshold))
  assert sim.clear_count==len(sim._targets)
  assert abs(sim.virtual_seconds-(sim.distance_m/5+5*sim.detect_count+sim.switch_count+3*sim.optical_count+2*sim.clear_count))<1e-5
  return dict(mode=mode,seed=seed,scenario=name,**sim.summary(),wall=time.perf_counter()-t,tail_after_last_clear=sim.virtual_seconds-sim.last_clear_time,
             transit_stops=r.get('transit_stops',0),transit_detections=r.get('transit_detections',0),route_accepts=r.get('route_accepts',0))
 except Exception:
  import traceback
  return dict(mode=mode,seed=seed,error=traceback.format_exc())
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--modes',nargs='+',default=['v5','heur','none:huber:0:v5','none:extra:10:v5','extra:huber:0:v5','extra:extra:10:v5','extra:huber:0:v4','extra:extra:10:v4']);p.add_argument('--base',type=int,default=157100000);p.add_argument('--out',default='finalists_development.jsonl');p.add_argument('--cases',type=int,default=10);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
 tasks=[(g,i,m,a.base)for g in range(9)for i in range(a.cases)for m in a.modes]
 with cf.ProcessPoolExecutor(max_workers=a.workers)as ex,open(a.out,'w')as out:
  for r in ex.map(runf,tasks):out.write(json.dumps(r)+'\n');out.flush()
 rows=[json.loads(s)for s in open(a.out)]
 for m in a.modes:
  rr=[r for r in rows if r['mode']==m];good=[r for r in rr if'error'not in r]
  print(m,len(rr),len(rr)-len(good),statistics.mean(r['seconds_per_cleared']for r in good),statistics.mean(r.get('transit_stops',0)for r in good),flush=True)
