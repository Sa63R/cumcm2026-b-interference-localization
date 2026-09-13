import json,math,time
from q4_coverage import certify
pp=json.load(open('/mnt/data/lastsrc/Q4_optimization_experiments/development_v6/layout20_failed.json'))['sites']
best=None
with open('repair20_attempts.jsonl','w') as f:
 for ei in [-15,-10,-5,0,5,10,15]:
  for eo in [0,1,2,3,4,5,8,10,15,20]:
   points=[]
   for i,p in enumerate(pp):
    d=math.hypot(*p)
    dr=ei if 1<=i<=7 else eo
    points.append(tuple(v*(d+dr)/d for v in p) if i else (0.,0.))
   r=certify(points,max_depth=22);row={'ei':ei,'eo':eo,'cert':r}
   if r['ok']:print('OK',ei,eo,flush=True);row['sites']=points
   f.write(json.dumps(row)+'\n');f.flush()
