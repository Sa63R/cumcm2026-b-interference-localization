#!/usr/bin/env python3
"""Self-built, paired comparison; no official results."""
import argparse, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import q3_base as b
import q3_optimized as o
import q3_experiments as v

def work(job):
    seed,mode,stress,noise=job
    sources=b.make_case(seed,stress)
    env=b.ToySimulator(sources,seed,noise)
    # The policy receives the restricted public interface only.
    agent=v.make_agent(env,mode)
    start=time.perf_counter();agent.run();elapsed=time.perf_counter()-start
    if env._live or env.failed:raise AssertionError((seed,mode,env._live,env.failed))
    return dict(seed=seed,mode=mode,noise=noise,stress=stress,true_sources=len(sources),cleared=env.clears,virtual_seconds=env.virtual_time,seconds_per_source=env.virtual_time/env.clears,movement_metres=env.distance,detects=env.detects,switches=env.switches,failed_clears=env.failed,local_runtime_seconds=elapsed,steps=agent.steps,full_scans=len(getattr(agent,'full_scans',[])))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=0);ap.add_argument('--cases',type=int,default=40);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--modes',nargs='+',default=['v2','v3']);ap.add_argument('--noises',nargs='+',default=['hash']);ap.add_argument('--stress',action='store_true');ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    jobs=[(s,m,a.stress,n) for s in range(a.start,a.start+a.cases) for m in a.modes for n in a.noises];rows=[];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fs={ex.submit(work,j):j for j in jobs}
        for f in as_completed(fs):
            try:rows.append(f.result())
            except Exception:
                print('FAILED',fs[f],flush=True);raise
            if len(rows)%100==0:print('completed',len(rows),'/',len(jobs),round(time.perf_counter()-t,1),flush=True)
    rows.sort(key=lambda x:(x['seed'],x['mode'],x['noise']));b.save_csv(a.out,rows)
    for m in a.modes:
        rr=[r for r in rows if r['mode']==m];print(m,len(rr),{k:round(float(np.mean([x[k] for x in rr])),3) for k in ['virtual_seconds','movement_metres','detects','switches','local_runtime_seconds','full_scans']},flush=True)
if __name__=='__main__':main()
