# Historical unprotected entry retained explicitly after the public default was guarded.
from experiments_round import *
from q4_count_search import solve_count_search
from q4_v5 import solve_v5
from q4_v6_unprotected import solve_v6

def rung(task):
 g,i,mode,base=task;name,fr,place,err=SCENARIOS[g];seed=base+100000*g+i;sim=AuditSimulator(make_targets(seed,fr,place),seed,err);tic=time.perf_counter()
 try:
  if mode=='v5':report=solve_v5(DeviceOnly(sim))
  elif mode=='v6':report=solve_v6(DeviceOnly(sim))
  else:
   visits,count,kind=mode.split(':')
   report=solve_count_search(DeviceOnly(sim),minimum_known=15,max_scans=8 if int(count) else 0,v5=True,transit=kind in ['all','transit'],route=kind in ['all','route'],min_visits=int(visits))
  assert sim.clear_count==len(sim._targets)
  return dict(mode=mode,seed=seed,scenario=name,**sim.summary(),wall=time.perf_counter()-tic,extra=report.get('count_search_scans',0),discovered=report.get('count_search_discoveries',0),trigger=report.get('count_search_triggered_upper',False),transit_stops=report.get('transit_stops',0))
 except Exception:
  import traceback
  return dict(mode=mode,seed=seed,scenario=name,error=traceback.format_exc())
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--modes',nargs='+',default=['v5','v6']+[f'{k}:0:{kind}'for k in [2,3,4,5,6]for kind in['all','transit']]);p.add_argument('--base',type=int,default=157100000);p.add_argument('--cases',type=int,default=10);p.add_argument('--out',default='guarded_development.jsonl');a=p.parse_args()
 tasks=[(g,i,m,a.base)for g in range(9)for i in range(a.cases)for m in a.modes]
 with cf.ProcessPoolExecutor(max_workers=5)as ex,open(a.out,'w')as out:
  for r in ex.map(rung,tasks):out.write(json.dumps(r)+'\n');out.flush()
 rows=[json.loads(s)for s in open(a.out)]
 for m in a.modes:
  rr=[r for r in rows if r['mode']==m];ok=[r for r in rr if'error'not in r]
  print(m,len(rr),len(rr)-len(ok),statistics.mean(r['seconds_per_cleared']for r in ok),statistics.mean(r.get('extra',0)for r in ok),sum(r.get('trigger',False)for r in ok),flush=True)
