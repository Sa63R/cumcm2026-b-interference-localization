"""Offline diagnostic only: ground truth may be read by this auditor, NEVER by a policy.
Recomputes per-case metric; solves exact shortest open centre tour as a relaxation.
"""
from pathlib import Path
import sys, json, time, itertools
import numpy as np
import pandas as pd
from numba import njit
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'B_Q3_optimized_v3'))
import q3_base as b

@njit(cache=True)
def optimal_open_tour(points):
    n=len(points)
    d=np.empty((n,n))
    for i in range(n):
        for j in range(n):
            dx=points[i,0]-points[j,0];dy=points[i,1]-points[j,1]
            d[i,j]=(dx*dx+dy*dy)**.5
    count=1<<n
    dp=np.full((count,n),np.inf)
    parent=np.full((count,n),-1,np.int16)
    for j in range(n):dp[1<<j,j]=np.sqrt((points[j]*points[j]).sum())
    for mask in range(1,count):
        for j in range(n):
            if (mask & (1<<j))==0:continue
            prev=mask ^ (1<<j)
            if prev==0:continue
            best=np.inf;which=-1
            for i in range(n):
                if prev & (1<<i):
                    val=dp[prev,i]+d[i,j]
                    if val<best:best=val;which=i
            dp[mask,j]=best;parent[mask,j]=which
    last=np.argmin(dp[count-1]);cost=dp[count-1,last]
    order=np.empty(n,np.int64);mask=count-1
    for k in range(n-1,-1,-1):
        order[k]=last;new=parent[mask,last];mask^=(1<<last);last=new
    return cost,order

def test_dp():
    rng=np.random.default_rng(817)
    for n in range(1,8):
        p=rng.normal(size=(n,2))
        got,order=optimal_open_tour(p)
        def length(rt):
            q=np.vstack([np.zeros(2),p[list(rt)]])
            return np.linalg.norm(np.diff(q,axis=0),axis=1).sum()
        exact=min(length(rt) for rt in itertools.permutations(range(n)))
        assert np.isclose(got,exact,atol=1e-9)
        assert np.isclose(got,length(order),atol=1e-9)

def main():
    test_dp()
    data=pd.read_csv(HERE/'data'/'holdout400.csv')
    data=data[data['mode']=='v3'].copy()
    rows=[];t=time.perf_counter()
    for row in data.to_dict('records'):
        sources=b.make_case(int(row['seed']))
        n=len(sources);points=np.array([s.xy for s in sources])
        cost,order=optimal_open_tour(points)
        # Every true clear point is within 20 m of its source, giving
        # centre_tour <= physical_path + 20 + 40*(n-1).
        lower_length=max(0.,cost-40*n+20)
        row.update(centre_tour_metres=float(cost),
                   relaxed_movement_lower_bound=lower_length,
                   ideal_centres_move_and_clear_per_source=cost/(5*n)+5,
                   relaxed_time_lower_bound_per_source=lower_length/(5*n)+5)
        rows.append(row)
    out=pd.DataFrame(rows)
    out.to_csv(HERE/'offline_oracle400.csv',index=False)
    metric={k:float(out[k].mean()) for k in [
        'seconds_per_source','centre_tour_metres','movement_metres',
        'ideal_centres_move_and_clear_per_source',
        'relaxed_time_lower_bound_per_source']}
    metric['fraction_lower_bound_over_200']=float((out.relaxed_time_lower_bound_per_source>200).mean())
    metric['fraction_v3_under_200']=float((out.seconds_per_source<=200).mean())
    metric['mean_per_source_movement']=float((out.movement_metres/out.true_sources/5).mean())
    metric['mean_per_source_detection']=float((out.detects/out.true_sources*5).mean())
    metric['mean_per_source_switching']=float((out.switches/out.true_sources).mean())
    metric['per_n']=out.groupby('true_sources')[['seconds_per_source','ideal_centres_move_and_clear_per_source','relaxed_time_lower_bound_per_source']].mean().to_dict('index')
    metric['note']='OFFLINE relaxed diagnostics on the SELF-BUILT simulator; not a deployable policy or an official score. No new 200-second solver is asserted.'
    (HERE/'oracle_summary.json').write_text(json.dumps(metric,indent=2))
    print(json.dumps(metric,indent=2))
    print('wall_seconds',time.perf_counter()-t)

if __name__=='__main__':main()
