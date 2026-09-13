#!/usr/bin/env python3
"""Serial, warm numerical timings; no HTTP or first JIT compilation included."""
from pathlib import Path
import platform,time,json
import numpy as np
import q3_base as b,q3_v3 as v

if __name__=='__main__':
    v.warmup()
    rows=[]
    for repeat in range(2):
        for seed in range(5000,5050):
            sources=b.make_case(seed)
            for mode in (['v2','v3'] if repeat==0 else ['v3','v2']):
                sim=b.ToySimulator(sources,seed)
                start=time.perf_counter();agent=v.make_agent(sim,mode);agent.run();elapsed=time.perf_counter()-start
                assert not sim._live and sim.failed==0
                rows.append(dict(seed=seed,repeat=repeat,mode=mode,seconds=elapsed,virtual_seconds=sim.virtual_time))
    b.save_csv(Path('results/serial_runtime.csv'),rows)
    summary={'scope':'Serial local computation, including policy construction; excluding environment creation, HTTP and first JIT compilation. Same 50 cases, two repeats, alternating mode order.','python':platform.python_version(),'numpy':np.__version__}
    try:
        import numba;summary['numba']=numba.__version__
    except ImportError:summary['numba']=None
    for m in ['v2','v3']:
        ts=np.array([x['seconds'] for x in rows if x['mode']==m])
        summary[m]={'measurements':len(ts),'mean_seconds':float(ts.mean()),'median_seconds':float(np.median(ts)),'p95_seconds':float(np.quantile(ts,.95))}
    summary['mean_speedup']=summary['v2']['mean_seconds']/summary['v3']['mean_seconds']
    Path('results/runtime_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
