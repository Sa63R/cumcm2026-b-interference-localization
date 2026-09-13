"""Compare route implementations on exactly the same point sets and starts."""
import json,random,time
from pathlib import Path
from q4_route import multi_route as original
from q4_route_cached import multi_route as cached

def main():
    rng=random.Random(912300);tests=[]
    for i in range(500):
        n=rng.randrange(3,35);pts=[(rng.uniform(-2000,2000),rng.uniform(-2000,2000)) for j in range(n)]
        tests.append(((rng.uniform(-2000,2000),rng.uniform(-2000,2000)),list(range(n)),pts,rng.randrange(1,12)))
    elapsed={'original':0.,'cached':0.}
    for i,args in enumerate(tests):
        outcomes={}
        versions=[('original',original),('cached',cached)]
        if i%2:versions.reverse()
        for label,fn in versions:
            start=time.process_time();outcomes[label]=fn(*args);elapsed[label]+=time.process_time()-start
        assert outcomes['original']==outcomes['cached']
    result=dict(identical_routes=len(tests),cpu_seconds=elapsed,speedup=elapsed['original']/elapsed['cached'],note='Routing routine only, local CPU timings, no effect on virtual action time')
    Path('results_v4/routing_benchmark.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=='__main__':main()
