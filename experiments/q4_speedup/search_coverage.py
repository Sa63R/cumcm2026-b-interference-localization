"""Offline ring search; candidates must pass continuous certification."""
import json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'q4_comparison'))
import common
from q4_coverage import certify
from q4_route_cached import multi_route
OUT=Path('output/q4_speedup/coverage_dev');OUT.mkdir(parents=True,exist_ok=True)
rows=[];tested=0;start=time.time()
for total in [19,20,21]:
 for n1 in range(6,10):
  n2=total-1-n1
  if n2<10 or n2>14:continue
  for r1 in [900,930,960,980,990,998]:
   for extra in [1,5,15,30,60]:
    r2=1800/math.cos(math.pi/n2)+extra
    for pi in range(8):
     phase=2*math.pi*pi/(n1*n2*8)
     sites=[(0.,0.)]+[(r*math.cos(2*math.pi*k/n+p),r*math.sin(2*math.pi*k/n+p)) for n,r,p in [(n1,r1,0),(n2,r2,phase)] for k in range(n)]
     cert=certify(sites,max_depth=15);tested+=1
     if cert['ok']:
      order=multi_route((0.,0.),list(range(1,len(sites))),sites,8)
      route=sum(math.dist(sites[a],sites[z]) for a,z in zip([0]+order[:-1],order))
      row=dict(total=total,n1=n1,n2=n2,r1=r1,r2=r2,phase=phase,route=route,cert=cert,sites=sites)
      rows.append(row)
      print('PASS',len(rows),total,n1,n2,r1,r2,phase,route,flush=True)
  print('DONE',total,n1,n2,'tested',tested,'secs',time.time()-start,flush=True)
rows.sort(key=lambda r:r['route']);(OUT/'rings.json').write_text(json.dumps(dict(tested=tested,elapsed=time.time()-start,passed=rows),indent=2))
