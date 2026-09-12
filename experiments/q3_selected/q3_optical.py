"""Selected optical policy; algorithm definitions copied unchanged from the archive.

The original standalone pandas benchmark is retained in vendor for provenance.
Official feedback mapping and Annex charging live in q3_runtime.py.
"""
import numpy as np
import q3_base as b
import q3_v3 as v

def area_points(poly, subdivision=10):
    if len(poly)<3:
        return poly.copy(),np.ones(len(poly))/max(1,len(poly))
    # Equal-area subtriangle centroids approximate uniform polygon-area mass.
    pts=[];weights=[]
    for i in range(1,len(poly)-1):
        a,c,d=poly[0],poly[i],poly[i+1]
        dc=c-a;dd=d-a
        w=abs(dc[0]*dd[1]-dc[1]*dd[0])
        for k in range(subdivision):
            for j in range(subdivision-k):
                pts.append(a+((k+1/3)/subdivision)*dc+((j+1/3)/subdivision)*dd);weights.append(w)
                if k+j<subdivision-1:
                    pts.append(a+((k+2/3)/subdivision)*dc+((j+2/3)/subdivision)*dd);weights.append(w)
    weights=np.array(weights)
    if weights.sum()<1e-12:
        return poly.copy(),np.ones(len(poly))/max(1,len(poly))
    return np.array(pts),weights/weights.sum()

class ExperimentalOpticalAgent(v.FastAgent):
    def __init__(self,env,threshold=.4,max_radius=80.,cap=2):
        super().__init__(env)
        self.threshold=threshold;self.max_radius=max_radius;self.cap=cap
        self.trials={c:0 for c in b.CHANNELS}
        self.attempt_points={c:set() for c in b.CHANNELS}
        self.speculative_success=0;self.speculative_failure=0
    def _allowed(self,c):
        return c in self.tracks and self.trials[c]<self.cap and self.tracks[c].radius<=self.max_radius
    def _mass(self,c,q):
        pts,w=area_points(self.tracks[c].poly)
        return float(w[np.linalg.norm(pts-q,axis=1)<=20].sum())
    def _attempt(self,c):
        if not self._allowed(c):return False
        key=b.coord_key(self.env.pos)
        if key in self.attempt_points[c] or self._mass(c,self.env.pos)<self.threshold:return False
        self.trials[c]+=1;self.attempt_points[c].add(key)
        # Reads only the public success/failure return, not hidden coordinates.
        ok=self.env.clear(c)
        if ok:
            self.speculative_success+=1
            self.tracks.pop(c,None);self.unseen.discard(c);self.cleared.add(c);self.count_certificate()
            return True
        self.speculative_failure+=1
        # Do NOT turn a failure into a success or drop the target.
        return False
    def select_probe(self,c):
        if self._allowed(c) and self.local_steps[c]<4:
            t=self.tracks[c];pts,w=area_points(t.poly);q=np.sum(pts*w[:,None],axis=0)
            if b.coord_key(q) not in self.attempt_points[c] and self._mass(c,q)>=self.threshold and np.max(np.linalg.norm(t.poly-q,axis=1))<999.9:
                return q,b.norm(q-t.center)<1e-6
        return super().select_probe(c)
    def observe(self,c,cover_id=None,center_step=False):
        if self._attempt(c):return
        super().observe(c,cover_id,center_step)
        self._attempt(c)
