#!/usr/bin/env python3
"""Self-built edge, clustered, and extreme-noise tests. Not official results."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time,math
import numpy as np
import q3_base as b, q3_v3 as v

def special_sources(k):
    rng=np.random.default_rng(93000+k);n=10+k%7
    if k%5==0:
        xy=np.repeat([[0.,0.]],n,axis=0)
    elif k%5==1:
        xy=np.repeat([[1800.,0.]],n,axis=0)
    elif k%5==2:
        xy=np.column_stack([np.linspace(-1800,1800,n),np.zeros(n)])
    elif k%5==3:
        th=np.linspace(-.025,.025,n);xy=1800*np.column_stack([np.cos(th),np.sin(th)])
    else:
        xy=rng.normal(0,4,(n,2))+np.array([400.,400.])
    channels=rng.choice(np.arange(1,21),n,replace=False)
    return [b.Source(int(c),p,1000. if k%2 else 1500.) for c,p in zip(channels,xy)]

def work(job):
    kind,seed,mode,noise=job
    sources=b.make_case(seed,True) if kind=='boundary' else special_sources(seed-93000)
    env=b.ToySimulator(sources,seed,noise);agent=v.make_agent(env,mode)
    t=time.perf_counter();agent.run();elapsed=time.perf_counter()-t
    if env._live or env.failed:raise AssertionError((kind,seed,mode,noise,env._live,env.failed))
    return dict(kind=kind,seed=seed,mode=mode,noise=noise,true_sources=len(sources),cleared=env.clears,virtual_seconds=env.virtual_time,seconds_per_source=env.virtual_time/env.clears,movement_metres=env.distance,detects=env.detects,switches=env.switches,failed_clears=env.failed,local_runtime_seconds=elapsed)

if __name__=='__main__':
    jobs=[]
    for noise in ['hash','smooth','plus','minus','alternating']:
        for mode in ['v2','v3']:
            jobs.extend(('boundary',s,mode,noise) for s in range(91000,91050))
            jobs.extend(('special',s,mode,noise) for s in range(93000,93020))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        fs={ex.submit(work,j):j for j in jobs}
        for f in as_completed(fs):
            try:rows.append(f.result())
            except Exception:
                print('FAILED',fs[f],flush=True);raise
            if len(rows)%100==0:print('completed',len(rows),'/',len(jobs),flush=True)
    rows.sort(key=lambda r:(r['kind'],r['seed'],r['noise'],r['mode']))
    Path('results').mkdir(exist_ok=True);b.save_csv(Path('results/stress350.csv'),rows)
    for mode in ['v2','v3']:
        rr=[r for r in rows if r['mode']==mode]
        print(mode,'cases',len(rr),'cleared',sum(r['cleared'] for r in rr),'mean_time',np.mean([r['virtual_seconds'] for r in rr]),'failed',sum(r['failed_clears'] for r in rr),flush=True)
