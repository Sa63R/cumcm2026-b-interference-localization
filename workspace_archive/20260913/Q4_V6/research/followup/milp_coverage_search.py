"""Finite-scenario set cover, followed by mandatory continuous verification.
A finite MILP solution is never authorized as a search layout by itself.
"""
import math,json,time,numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import csr_matrix,vstack
from q4_coverage import certify
from q4_route_cached import multi_route
import q4_baseline as b
old=[(0.,0.)]+[(r*math.cos(2*math.pi*k/n),r*math.sin(2*math.pi*k/n)) for n,r in [(8,998),(12,1865)]for k in range(n)]
points=old[:]
for r in [450,850,950,1050,1200,1650,1900,2000]:
 for k in range(32):
  p=(r*math.cos(2*math.pi*k/32),r*math.sin(2*math.pi*k/32))
  if min(math.dist(p,q)for q in points)>10:points.append(p)
P=np.asarray(points);states=[]
for r in [0,200,500,800,950,1050,1250,1450,1650,1799.99,1800]:
 for k in range(1 if r==0 else 64):
  a=2*math.pi*k/64;g=np.array([r*math.cos(a),r*math.sin(a)])
  for j in range(16):
   th=2*math.pi*j/16;states.append((g,np.array([math.cos(th),math.sin(th)])))
def makeA(st):
 return csr_matrix(np.array([((np.sum((P-g)**2,axis=1)<=1000000-1e-5)&((P-g)@u>=-1e-8)).astype(float)for g,u in st]))
A=makeA(states);c=np.ones(len(P));c[len(old):]+=.0001
lb=np.zeros(len(P));lb[0]=1
constraints=[LinearConstraint(A,1,np.inf),LinearConstraint(np.ones((1,len(P))),0,20),LinearConstraint(np.array([[float(i<len(old))for i in range(len(P))]]),10,np.inf)]
tic=time.perf_counter();logs=[]
for step in range(4):
 res=milp(c,integrality=np.ones(len(P)),bounds=Bounds(lb,np.ones(len(P))),constraints=constraints,options={'time_limit':35.,'mip_rel_gap':.001})
 row={'step':step,'status':int(res.status),'message':res.message,'seconds':time.perf_counter()-tic}
 if res.x is None:logs.append(row);break
 ids=np.where(res.x>.5)[0];sites=[points[i]for i in ids];proof=certify(sites,max_depth=19);row.update(sites=sites,continuous=proof,objective=float(c@res.x));logs.append(row)
 open('milp_coverage_attempts.json','w').write(json.dumps(logs,indent=2));print(step,len(sites),proof,flush=True)
 if proof['ok']:
  route=multi_route((0.,0.),list(range(1,len(sites))),sites,8);row['route_length']=sum(b.dist(x,y)for x,y in zip([(0.,0.)]+[sites[i]for i in route[:-1]],[sites[i]for i in route]));print('CERTIFIED',row['route_length'],flush=True);break
 if 'witness'not in proof:break
 g=np.array(proof['witness'][:2]);near=[q for q in sites if math.dist(q,g)<=1000.]
 angles=sorted([math.atan2(q[1]-g[1],q[0]-g[0])%(2*math.pi)for q in near]);gaps=[((angles[(j+1)%len(angles)]-angles[j])%(2*math.pi),angles[j])for j in range(len(angles))] if angles else [(2*math.pi,0.)]
 gap,a=max(gaps);th=a+gap/2;cutstates=[(g,np.array([math.cos(th),math.sin(th)]))]
 for j in range(64):th=2*math.pi*j/64;cutstates.append((g,np.array([math.cos(th),math.sin(th)])))
 constraints.append(LinearConstraint(makeA(cutstates),1,np.inf))
open('milp_coverage_attempts.json','w').write(json.dumps(logs,indent=2));print('DONE',time.perf_counter()-tic,flush=True)
