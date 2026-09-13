"""Paired coverage development cases; not the held-out validation."""
from pathlib import Path
import sys,json,time,statistics,math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'q4_comparison'))
import common
import q4_baseline as b
from online import State,finish
from benchmark import PublicDevice
from q4_coverage import certify
from coverage_candidate import CandidateState, OpportunisticState
from run_v3_validation import SCENARIOS
OUT=Path('output/q4_speedup/coverage_dev');OUT.mkdir(parents=True,exist_ok=True)
rows=[]
methods={'baseline':lambda:State(),'bonus_3':lambda:OpportunisticState(bonus_scans=3),
'bonus_6':lambda:OpportunisticState(bonus_scans=6),
'bonus_12':lambda:OpportunisticState(bonus_scans=12),
'bonus_6_near':lambda:OpportunisticState(bonus_scans=6,bonus_distance=450.,bonus_unknown=10)}
for gi,(scenario,fraction,placement,error) in enumerate(SCENARIOS):
 for ci in range(10):
  seed=301000000+gi*100000+ci
  for name,make_state in methods.items():
   sim=b.LocalSimulator(b.make_case(seed,fraction,placement),seed,error)
   start=time.perf_counter();state=make_state();report=finish(state,PublicDevice(sim));wall=time.perf_counter()-start
   c=certify(report['actual_scanpoints']) if report['stop_certificate']=='coverage_complete' else {'ok':True}
   row=dict(method=name,seed=seed,scenario=scenario,**sim.summary(),wall=wall,replacements=len(report.get('coverage_pruned_sites',[])),coverage_checks=report.get('coverage_checks',0),rotation=report.get('rotation_angle',0),certified=c['ok'])
   assert sim.clear_count==len(sim._targets) and c['ok'],row
   rows.append(row)
  print('case',gi,ci,'complete',flush=True)
  (OUT/'bonus_runs.json').write_text(json.dumps(rows,indent=2))
summary={};base={r['seed']:r for r in rows if r['method']=='baseline'}
for name in methods:
 rr=[r for r in rows if r['method']==name];dd=[(base[r['seed']]['virtual_seconds']-r['virtual_seconds'])/r['targets'] for r in rr]
 summary[name]=dict(cases=len(rr),seconds_per_source=statistics.mean(r['virtual_seconds']/r['targets'] for r in rr),saved_per_source=statistics.mean(dd),ci95=[statistics.mean(dd)-1.96*statistics.stdev(dd)/math.sqrt(len(dd)),statistics.mean(dd)+1.96*statistics.stdev(dd)/math.sqrt(len(dd))],mean_wall=statistics.mean(r['wall'] for r in rr),replacements=sum(r['replacements'] for r in rr),checks=sum(r['coverage_checks'] for r in rr),faster=sum(d>1e-6 for d in dd),slower=sum(d<-1e-6 for d in dd))
print(json.dumps(summary,indent=2),flush=True);(OUT/'bonus_summary.json').write_text(json.dumps(summary,indent=2))
