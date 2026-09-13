"""Read-only, ground-truth-enabled diagnostic of v3; NOT a policy."""
from pathlib import Path
import sys, json, time
import numpy as np
import pandas as pd
from audit200 import optimal_open_tour
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'B_Q3_optimized_v3'))
import q3_base as b
import q3_v3 as v

def one(seed):
    sources=b.make_case(seed);n=len(sources);positions={s.channel:s.xy for s in sources}
    env=b.ToySimulator(sources,seed,'hash',True)
    agent=v.make_agent(env,'v3');agent.run()
    assert not env._live and not env.failed
    route=[row for row in env.log if row['action']=='clear' and row['success']]
    gt=np.vstack([np.zeros(2)]+[positions[r['channel']] for r in route])
    clearpoints=np.vstack([np.zeros(2)]+[np.array(r['xy']) for r in route])
    centerorder=float(np.linalg.norm(np.diff(gt,axis=0),axis=1).sum())
    clearorder=float(np.linalg.norm(np.diff(clearpoints,axis=0),axis=1).sum())
    opt,_=optimal_open_tour(np.array([s.xy for s in sources]))
    empty=0;outside=0;first=0;repeat=0;seen=set()
    for row in env.log:
        if row['action']!='detect':continue
        c=row['channel']
        if c not in positions:empty+=1
        elif row['result']=='none':outside+=1
        elif c not in seen:first+=1;seen.add(c)
        else:repeat+=1
    tail=env.virtual_time-route[-1]['virtual_seconds']
    return dict(seed=seed,n=n,time=env.virtual_time,avg=env.virtual_time/n,
                movement=env.distance,clear_center_order_metres=centerorder,
                clearpoint_order_metres=clearorder,optimal_center_metres=opt,
                non_clear_waypoint_extra_seconds_per_source=(env.distance-clearorder)/5/n,
                center_order_regret_seconds_per_source=(centerorder-opt)/5/n,
                tail_seconds_per_source=tail/n,
                empty_channel_detects=empty,outside_range_detects=outside,
                first_positive_detects=first,extra_positive_detects=repeat)

def main():
    v.warmup();t=time.perf_counter();rr=[]
    for seed in range(5000,5100):
        rr.append(one(seed))
    data=pd.DataFrame(rr);data.to_csv(HERE/'trace_audit100.csv',index=False)
    result=data.mean(numeric_only=True).to_dict()
    for k in ['empty_channel_detects','outside_range_detects','first_positive_detects','extra_positive_detects']:
        result[k+'_per_source']=float((data[k]/data.n).mean())
    (HERE/'trace_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));print('wall',time.perf_counter()-t)
if __name__=='__main__':main()
