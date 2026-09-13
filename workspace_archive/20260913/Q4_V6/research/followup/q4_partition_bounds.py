"""Conservative position partitioning before directional negative contraction.
Every position slice is relaxed over angle sectors; infeasible slices may be
removed. Taking the hull of surviving slices is an OUTER approximation.
"""
import math
from dataclasses import dataclass
import q4_baseline as b
from q4_information import intersect_arcs
from q4_coverage import hull
from q4_v4_solver import Belief
from q4_state import Engine


def contract_slice(poly, positives, negatives, sectors):
    center,radius=b.enclosing_circle(poly)
    lower=max([1000.]+[max(0.,b.dist(p,center)-radius) for p in positives])
    near=[q for q in negatives if max(b.dist(q,v) for v in poly)<lower-1e-5]
    if not near:return poly
    arcs=[(0.,2*math.pi)]
    for p in positives:
        for q in near:
            d=b.sub(p,q)
            if b.norm(d)<1e-9:return poly # ambiguous identical recorded point: fail closed
            arcs=intersect_arcs(arcs,math.atan2(d[1],d[0]))
            if not arcs:return []
    cons=[(p,1.) for p in positives]+[(q,-1.) for q in near]
    bounds=[max(b.dist(p,z) for z in poly) for p,sgn in cons]
    total=sum(z-a for a,z in arcs);vertices=[]
    for a,z in arcs:
        n=max(1,math.ceil(sectors*(z-a)/max(total,1e-12)))
        for k in range(n):
            lo=a+(z-a)*k/n;hi=a+(z-a)*(k+1)/n
            u=b.unit((lo+hi)/2);err=2*math.sin((hi-lo)/4)+1e-9;sub=poly[:]
            for (p,sgn),bound in zip(cons,bounds):
                nn=b.mul(sgn,u);sub=b.clip_halfplane(sub,nn,b.dot(nn,p)+err*bound+1e-6)
                if not sub:break
            vertices.extend(sub)
    if not vertices:return []
    hh=hull(vertices)
    return hh if len(hh)>=3 else vertices

def partition_contract(poly,pos,neg,n=12,sectors=16):
    if not neg or len(pos)<2 or len(poly)<3:return poly
    c,r=b.enclosing_circle(poly)
    if r<30:return poly
    far=max(poly,key=lambda p:b.dist(c,p));other=max(poly,key=lambda p:b.dist(far,p))
    vec=b.sub(far,other);dd=b.norm(vec)
    if dd<1e-7:return poly
    e=b.mul(1/dd,vec);values=[b.dot(e,p) for p in poly];lo,hi=min(values),max(values)
    # Limit narrow cells; the partition is overlapping at boundaries via clip EPS.
    n=min(n,max(1,math.ceil((hi-lo)/20)))
    vertices=[]
    for j in range(n):
        p=b.clip_halfplane(poly,e,lo+(hi-lo)*(j+1)/n)
        p=b.clip_halfplane(p,b.mul(-1,e),-(lo+(hi-lo)*j/n))
        if not p:continue
        vertices.extend(contract_slice(p,pos,neg,sectors))
    if len(vertices)<3:return poly # never declare an actual source impossible
    out=hull(vertices)
    return out if len(out)>=3 else poly

@dataclass
class PartitionBelief(Belief):
    partitions:int=8
    def tighten(self):
        super().tighten()
        if self.negative_enabled and self.poly and self.obs.status!='strong':
            self.poly=partition_contract(self.poly,self.positives,self.negatives,self.partitions,self.direction_sectors)

class PartitionEngine(Engine):
    def __init__(self,device,cfg,n=8):
        super().__init__(device,cfg);self.partitions=n
    def discovery(self):
        s=self.state;dev=self.device;cfg=self.config
        for c in sorted(s.unknown,key=lambda x:(x!=dev.channel,x)):
            if len(s.cleared)+len(s.pending)==16:break
            obs=dev.detect(c)
            if obs.status=='none':s.negatives[c].append(dev.position);continue
            s.unknown.remove(c)
            poly=b.initial_polygon(dev.position,obs.theta,1500.) if obs.status=='bearing' else []
            tr=PartitionBelief(channel=c,anchor=dev.position,obs=obs,poly=poly,measured=[dev.position],positives=[dev.position],negatives=s.negatives[c][:],negative_enabled=cfg.directional_negatives,direction_sectors=cfg.direction_sectors,partitions=self.partitions)
            tr.tighten();s.pending[c]=tr
        s.scanpoints.append(dev.position)
