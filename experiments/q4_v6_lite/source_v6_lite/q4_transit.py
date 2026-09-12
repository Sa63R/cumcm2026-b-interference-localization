"""Experimental feedback-only in-transit sensing; fixed coverage is unchanged."""
import math
import q4_baseline as b
from q4_state import Engine,Action
from q4_information import expected_radius_gain
from q4_fast_local import nearest_clear_point
from q4_v3_local import optical_cover

class TransitEngine(Engine):
    def __init__(self,device,cfg,threshold=50.,maxstops=16,fractions=(.33,.67),optical=False):
        super().__init__(device,cfg);self.threshold=threshold;self.maxstops=maxstops;self.fractions=fractions;self.optical=optical;self.transits=0;self.attempts=0
    def entry(self,a):
        if a.kind=='site':return self.state.sites[a.key]
        tr=self.state.pending[a.key]
        if tr.obs.status=='strong':return tr.anchor
        c,r=b.enclosing_circle(tr.poly)
        if r<=b.CLEAR_CERT_RADIUS:return nearest_clear_point(tr.poly,self.device.position)
        cover=optical_cover(tr.poly)
        if cover and len(cover)<=self.config.local.optical_cover_limit:return min([cover[0],cover[-1]],key=lambda q:b.dist(self.device.position,q))
        s=tr.anchor;e=b.unit(tr.obs.theta);v=(-e[1],e[0]);upper=min(1500.,max(b.dist(s,p) for p in tr.poly)+1e-7)
        cfg=self.config.local;t=cfg.axial*upper;h=cfg.lateral*upper
        if cfg.advance_lower:
            lo=max(0.,min(b.dot(e,b.sub(p,s)) for p in tr.poly));adv=cfg.lower_factor*lo
            if lo/upper>.25:adv=max(adv,lo+cfg.range_fraction*(upper-lo));h=max(h,cfg.adaptive_lateral*upper)
            t=min(.95*upper,max(t,adv));h=max(h,t*b.TAN_D*1.05)
        mid=b.add(s,b.mul(t,e))
        return min([b.add(mid,b.mul(h,v)),b.add(mid,b.mul(-h,v))],key=lambda q:b.dist(self.device.position,q))
    def intercept(self,a):
        if self.transits>=self.maxstops or not self.state.pending:return False
        p=self.device.position;end=self.entry(a);d=b.dist(p,end)
        if d<200:return False
        best=None
        for f in self.fractions:
            q=b.add(p,b.mul(f,b.sub(end,p)));chs=[];total=0
            for c,tr in self.state.pending.items():
                if tr.obs.status=='strong':continue
                cen,rad=b.enclosing_circle(tr.poly)
                if rad<35 or b.dist(q,cen)>800+rad or min(b.dist(q,z) for z in tr.measured)<60:continue
                gain=expected_radius_gain(tr.poly,tr.positives,tr.negatives,q)
                if gain>self.threshold:
                    chs.append(c);total+=gain-self.threshold
            if chs and (best is None or total>best[0]):best=(total,q,chs)
        if best is None:return False
        _,q,chs=best;self.transits+=1
        self.device.move(q)
        for c in sorted(chs,key=lambda c:(c!=self.device.channel,c)):
            if c not in self.state.pending:continue
            obs=self.device.detect(c);tr=self.state.pending[c];tr.add(q,obs);self.state.shared_count+=1
            if obs.status=='strong':
                b.checked_clear(self.device,c,q)
                self.finish(c,dict(certificate='transit_strong',stages=tr.rf_rounds))
        return True
    def run_base(self,max_actions=None):
        while not self.done():
            a=self.ordered_actions()[0]
            if not self.intercept(a):self.execute(a)
        return dict(**self.report(),transit_stops=self.transits)
