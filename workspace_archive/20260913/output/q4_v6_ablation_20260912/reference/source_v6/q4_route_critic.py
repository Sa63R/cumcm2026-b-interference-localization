"""Observation-derived action features for offline cost-to-go regression."""
import math
import q4_baseline as b
from q4_transit import TransitEngine
from q4_information import expected_radius_gain,heuristic_visibility
from q4_v3_local import optical_cover

def route_features(engine,order):
    if len(order)<2 or not engine.state.visited or not engine.state.pending:return []
    st=engine.state;p=engine.device.position
    centers=[st.sites[a.key]if a.kind=='site'else st.pending[a.key].center()for a in order]
    route_len=sum(b.dist(x,y)for x,y in zip([p]+centers[:-1],centers))
    radii=[b.enclosing_circle(t.poly)[1]for t in st.pending.values()if t.obs.status!='strong']
    ctx=[len(st.remaining),len(st.pending),len(st.cleared),len(st.unknown),len(st.visited),len(st.scanpoints),route_len,b.norm(p),sum(radii),max(radii,default=0.),min(radii,default=0.)]
    choices=list(range(min(4,len(order))))
    for kind in ['site','target']:
        ids=[j for j,a in enumerate(order)if a.kind==kind]
        if ids:
            j=min(ids,key=lambda k:b.dist(p,centers[k]))
            if j not in choices:choices.append(j)
    attrs=[]
    for j in choices:
        a=order[j];z=centers[j];entry=TransitEngine.entry(engine,a)
        seq=[z]+[q for k,q in enumerate(centers)if k!=j]
        force=sum(b.dist(x,y)for x,y in zip([p]+seq[:-1],seq))
        gains=[]
        for c,t in st.pending.items():
            if t.obs.status=='strong':continue
            cen,rad=b.enclosing_circle(t.poly)
            if rad<25 or min(b.dist(z,q)for q in t.measured)<50 or b.dist(cen,z)>800+rad:continue
            gains.append(expected_radius_gain(t.poly,t.positives,t.negatives,z))
        near=sorted(b.dist(z,q)for q in st.scanpoints)
        loc=[0.]*10
        if a.kind=='target':
            t=st.pending[a.key]
            if t.obs.status=='strong':loc[0]=5.;loc[-1]=1.
            else:
                cen,r=b.enclosing_circle(t.poly);e=b.unit(t.obs.theta);v=(-e[1],e[0]);xs=[b.dot(e,q)for q in t.poly];ys=[b.dot(v,q)for q in t.poly]
                cov=optical_cover(t.poly)
                loc=[r,max(xs)-min(xs),max(ys)-min(ys),len(t.positives),len(t.negatives),t.rf_rounds,b.dist(t.anchor,z),len(cov),heuristic_visibility(z,t.positives,t.negatives,p),0.]
        vals=[float(a.kind=='site'),j,b.dist(p,z),b.dist(p,entry),b.dist(entry,z),b.norm(z),force-route_len,
              sum(gains),max(gains,default=0.),len(gains),near[0]if near else 3600.,near[1]if len(near)>1 else 3600.,
              b.dist(z,centers[0]),b.dist(z,centers[-1])]+loc
        attrs.append((a,vals))
    base=attrs[0][1];out=[]
    for a,vec in attrs:
        out.append((a,ctx+base+vec+[x-y for x,y in zip(vec,base)]))
    return out
