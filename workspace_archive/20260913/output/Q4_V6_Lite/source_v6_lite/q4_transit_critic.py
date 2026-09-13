"""Observation-only features for learning the total-time value of in-transit RF.
Training labels come from separate synthetic counterfactual continuations.
Inference sees only legal recorded observations, never environment parameters.
"""
from __future__ import annotations
import math,json,struct,pathlib
from functools import lru_cache

@lru_cache(maxsize=16)
def _load_model(path):
    return json.loads(pathlib.Path(path).read_text())

import q4_baseline as b
from q4_transit import TransitEngine
from q4_information import expected_radius_gain,heuristic_visibility


def proposals(engine,base,threshold=20.):
    if not engine.state.pending:return []
    start=engine.device.position;end=TransitEngine.entry(engine,base);length=b.dist(start,end)
    if length<200:return []
    out=[]
    for f in [.25,.5,.75]:
        q=b.add(start,b.mul(f,b.sub(end,start)));chs=[];stats=[]
        for c,tr in engine.state.pending.items():
            if tr.obs.status=='strong':continue
            cen,rad=b.enclosing_circle(tr.poly);near=min(b.dist(q,p) for p in tr.measured)
            if rad<35 or b.dist(cen,q)>800+rad or near<60:continue
            gain=expected_radius_gain(tr.poly,tr.positives,tr.negatives,q)
            if gain<=threshold:continue
            v0=b.sub(cen,tr.anchor);v1=b.sub(cen,q);nn=max(1e-12,b.norm(v0)*b.norm(v1))
            sine=abs(v0[0]*v1[1]-v0[1]*v1[0])/nn
            row=[gain,rad,gain/rad,b.dist(cen,q),near,len(tr.positives),len(tr.negatives),sine,
                 heuristic_visibility(cen,tr.positives,tr.negatives,q),b.dist(tr.anchor,q),
                 max(b.dist(q,p)for p in tr.poly),min(b.dist(q,p)for p in tr.poly),float(base.kind=='target' and base.key==c)]
            chs.append(c);stats.append(row)
        if not chs:continue
        stats.sort(reverse=True,key=lambda r:r[0]);n=len(chs)
        st=engine.state
        x=[length,f,length*f,length*(1-f),float(base.kind=='site'),len(st.remaining),len(st.pending),len(st.cleared),len(st.unknown),n,
           sum(r[0] for r in stats),sum(r[0]/r[1] for r in stats),sum(r[3] for r in stats)/n,6*n-(engine.device.channel in chs),
           b.norm(start),b.norm(q),b.norm(end),sum(r[8]for r in stats)/n]
        for k in range(3):x.extend(stats[k] if k<len(stats) else [0.]*13)
        out.append(dict(point=q,channels=chs,features=x,geometry_gain=sum(r[0]for r in stats)))
    return out


def apply_probe(engine,proposal):
    q=tuple(proposal['point']);engine.device.move(q);count=0
    for c in sorted(proposal['channels'],key=lambda c:(c!=engine.device.channel,c)):
        if c not in engine.state.pending:continue
        tr=engine.state.pending[c];obs=engine.device.detect(c);tr.add(q,obs);engine.state.shared_count+=1;count+=1
        if obs.status=='strong':
            b.checked_clear(engine.device,c,q);engine.finish(c,dict(certificate='transit_strong',stages=tr.rf_rounds))
    return count

class Critic:
    def __init__(self,path=None):
        p=pathlib.Path(path) if path else pathlib.Path(__file__).with_name('transit_critic.json')
        if not p.is_absolute() and not p.exists(): p=pathlib.Path(__file__).parent/p
        self.model=_load_model(str(p.resolve()))
    def predict(self,x):
        if len(x)!=self.model['feature_count'] or not all(math.isfinite(float(z)) for z in x):
            raise ValueError('Invalid action-feature vector')
        x=struct.unpack('<'+'f'*len(x),struct.pack('<'+'f'*len(x),*x))
        value=self.model['base']
        for tree in self.model['trees']:
            i=0
            while tree['feature'][i]>=0:
                i=tree['left'][i] if x[tree['feature'][i]]<=tree['threshold'][i] else tree['right'][i]
            value+=self.model['learning_rate']*tree['value'][i]
        return value
