#!/usr/bin/env python3
"""Paired self-built benchmarks. Not official competition results."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import q3_optimized as q
import q3_base as b
import numpy as np

def work(item):
    seed,mode,stress,noise=item
    row,_=q.evaluate(seed,mode,noise,stress)
    return row

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start',type=int,default=0);ap.add_argument('--cases',type=int,default=40)
    ap.add_argument('--modes',nargs='+',choices=q.CONFIGS,default=['baseline','combined'])
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--stress',action='store_true');ap.add_argument('--noises',nargs='+',default=['hash'])
    args=ap.parse_args()
    if args.cases<1 or args.workers<1:ap.error('--cases and --workers must be positive')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    jobs=[(s,m,args.stress,n) for s in range(args.start,args.start+args.cases) for m in args.modes for n in args.noises]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(work,j):j for j in jobs}
        for f in as_completed(futs):
            try:rows.append(f.result())
            except Exception:
                print('FAILED:',futs[f],flush=True);raise
    rows.sort(key=lambda r:(r['seed'],r['mode'],r['noise']))
    b.save_csv(args.out,rows)
    for m in args.modes:
        rr=[r for r in rows if r['mode']==m]
        print(m,len(rr),{k:round(float(np.mean([r[k] for r in rr])),3) for k in ['virtual_seconds','movement_metres','detects','local_runtime_seconds']},flush=True)
if __name__=='__main__':main()
