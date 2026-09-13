import math,json,time,sys,itertools
import numpy as np
from q4_coverage import certify

def pts(n0,r0,r1=1150.,r2=1960.,phase=0.,middlephase=math.pi/8):
    return [(0.,0.)]+[(r*math.cos(2*math.pi*k/n+p),r*math.sin(2*math.pi*k/n+p)) for n,r,p in [(n0,r0,phase),(8,r1,middlephase),(8,r2,0.)] for k in range(n)]
angs=np.linspace(0,2*math.pi,240,endpoint=False)
rs=np.r_[np.linspace(0,1800,43),1799.99]
grid=np.array([(r*math.cos(a),r*math.sin(a)) for r in rs for a in angs])

def score(pp):
    pp=np.array(pp);v=pp[None,:,:]-grid[:,None,:];d=np.linalg.norm(v,axis=-1)
    angle=np.where(d<=999.9,np.mod(np.arctan2(v[:,:,1],v[:,:,0]),2*np.pi),100.)
    a=np.sort(angle,axis=1);ct=(angle<99).sum(axis=1);dif=np.diff(a,axis=1)
    dif=np.where(np.arange(len(pp)-1)[None,:]<(ct-1)[:,None],dif,-1.)
    gap=np.maximum(dif.max(axis=1),a[:,0]+2*np.pi-a[np.arange(len(a)),np.maximum(0,ct-1)])
    bad=np.where(d.min(axis=1)<1e-6,0.,np.maximum(0.,gap-np.pi))
    return float(bad.max()+bad.mean())
if __name__=='__main__':
    best=1000;t=time.time();count=0
    with open('three_ring_search.jsonl','w') as f:
      for n0,r0,r1,r2 in itertools.product([3,4,5],[400,550,700,850],[1050,1100,1150,1200,1250],[1950,1960,2000,2050]):
        pp=pts(n0,r0,r1,r2);sc=score(pp);count+=1
        if sc<best:best=sc;print('best',n0,r0,r1,r2,sc,flush=True)
        if sc<1e-7:
          cert=certify(pp,max_depth=22);row=dict(n0=n0,r0=r0,r1=r1,r2=r2,score=sc,cert=cert,sites=pp)
          f.write(json.dumps(row)+'\n');f.flush();print('CERT',row,flush=True)
      print(count,time.time()-t)
