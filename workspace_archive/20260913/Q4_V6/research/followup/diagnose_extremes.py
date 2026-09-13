# Historical unprotected entry retained explicitly after the public default was guarded.
"""Selected first-validation failures, used for diagnosis, NOT fresh test data."""
import json,time,pathlib
from validate_v5 import SCENARIOS,AuditSimulator,DeviceOnly,make_targets
from q4_v5 import solve_v5
from q4_v6_unprotected import solve_v6
from q4_count_search import solve_count_search
out=[]
for g,i in [(6,11),(6,28),(6,52),(5,79),(1,92)]:
 name,fr,place,err=SCENARIOS[g];seed=191100000+g*100000+i
 for mode in ['v5','unprotected','guard2','guard4']:
  sim=AuditSimulator(make_targets(seed,fr,place),seed,err,trace=True)
  if mode=='v5':r=solve_v5(DeviceOnly(sim))
  elif mode=='unprotected':r=solve_v6(DeviceOnly(sim))
  else:r=solve_count_search(DeviceOnly(sim),max_scans=0,transit=True,route=True,min_visits=int(mode[-1]))
  known=set();t16=None
  for a in sim.trace:
   if a['action']=='detect' and a['status']!='none':known.add(a['channel'])
   if len(known)==16:t16=a['virtual_seconds'];break
  row=dict(seed=seed,mode=mode,**sim.summary(),first_16_discovered_time=t16,visited_stations=r['visited_stations']);out.append(row)
  print(seed,mode,round(sim.virtual_seconds/len(sim._targets),2),t16,r['visited_stations'],flush=True)
pathlib.Path('extreme_case_diagnosis.json').write_text(json.dumps(out,indent=2))
