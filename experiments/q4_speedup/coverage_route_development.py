"""Isolated route effort check, development cases only."""
from pathlib import Path
import sys,json,time,statistics,math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'q4_comparison'))
import common
import q4_baseline as b
from online import State,finish
from q4_v4_solver import V4Config
from benchmark import PublicDevice
from run_v3_validation import SCENARIOS
OUT=Path('output/q4_speedup/coverage_dev');OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for gi,(scenario,fraction,placement,error) in enumerate(SCENARIOS):
 for ci in range(10):
  seed=301000000+gi*100000+ci
  for n in [1,4,8,32]:
   sim=b.LocalSimulator(b.make_case(seed,fraction,placement),seed,error)
   start=time.perf_counter();state=State(config=V4Config(route_starts=n));report=finish(state,PublicDevice(sim));wall=time.perf_counter()-start
   row=dict(method=str(n),seed=seed,scenario=scenario,**sim.summary(),wall=wall)
   assert sim.clear_count==len(sim._targets),row
   rows.append(row)
  print('case',gi,ci,'complete',flush=True)
  (OUT/'route_runs.json').write_text(json.dumps(rows,indent=2))
summary={};base={r['seed']:r for r in rows if r['method']=='8'}
for name in ['1','4','8','32']:
 rr=[r for r in rows if r['method']==name];dd=[(base[r['seed']]['virtual_seconds']-r['virtual_seconds'])/r['targets'] for r in rr]
 summary[name]=dict(cases=len(rr),seconds_per_source=statistics.mean(r['virtual_seconds']/r['targets'] for r in rr),saved_per_source=statistics.mean(dd),ci95=[statistics.mean(dd)-1.96*statistics.stdev(dd)/math.sqrt(len(dd)),statistics.mean(dd)+1.96*statistics.stdev(dd)/math.sqrt(len(dd))],mean_wall=statistics.mean(r['wall'] for r in rr),faster=sum(d>1e-6 for d in dd),slower=sum(d<-1e-6 for d in dd))
print(json.dumps(summary,indent=2),flush=True);(OUT/'route_summary.json').write_text(json.dumps(summary,indent=2))
