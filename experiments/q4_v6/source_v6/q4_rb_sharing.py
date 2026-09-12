"""RB-posterior replacement for the legacy three-point shared-sensing heuristic.

Expected geometric shrinkage is used ONLY to rank optional extra measurements.
No probability value can remove a source, certify a clear, or authorize exit.
"""
from __future__ import annotations
import bisect,math
import q4_baseline as b
from q4_state import Engine
from q4_belief import SceneFactory,PosteriorUnavailable
from radius_direction import marginalize

class RBEngine(Engine):
    def __init__(self,device,config=None,state=None,*,share_threshold=10.,quadrature=12):
        super().__init__(device,config,state)
        self.rb_factory=SceneFactory(draws=96,seed=49511)
        self.share_threshold=share_threshold;self.quadrature=quadrature
        self.rb_share_checks=0;self.rb_share_failures=0

    def posterior_gain(self,c,q):
        tr=self.state.pending[c];cen,rad=b.enclosing_circle(tr.poly)
        pool=self.rb_factory.pool(self.state,c)
        ws=[.5*(o+d) for o,d in zip(pool.omni_weights,pool.dir_weights)]
        total=sum(ws)
        if total<=0:raise PosteriorUnavailable('Empty sharing belief')
        cdf=[];v=0.
        for w in ws:v+=w;cdf.append(v)
        value=0.
        nodes=[(-math.sqrt(3/5)*b.DELTA,5/18),(0.,4/9),(math.sqrt(3/5)*b.DELTA,5/18)]
        # Deterministic equal-mass posterior stratification, not three endpoints.
        for j in range(self.quadrature):
            i=min(len(cdf)-1,bisect.bisect_left(cdf,total*(j+.5)/self.quadrature))
            g=pool.positions[i];old=pool.marginals[i].mixture_evidence
            new=marginalize(g,pool.positives+(q,),pool.negatives,radius_prior=self.rb_factory.radius_prior).mixture_evidence
            prob=min(1.,new/old) if old>0 else 0.
            if prob<=0:continue
            if b.dist(g,q)<=5:value+=prob*rad;continue
            true=math.atan2(g[1]-q[1],g[0]-q[0]);gain=0.
            for error,weight in nodes:
                poly=b.clip_bearing(tr.poly,q,true+error,1500.)
                if poly:gain+=weight*max(0.,rad-b.enclosing_circle(poly)[1])
            value+=prob*gain
        return value/self.quadrature

    def shared(self,exclude=None):
        st=self.state;dev=self.device;cfg=self.config
        from q4_information import expected_radius_gain
        for c in sorted(st.pending,key=lambda x:(x!=dev.channel,x)):
            if c==exclude:continue
            tr=st.pending[c]
            if tr.obs.status=='strong':continue
            if min(b.dist(dev.position,p) for p in tr.measured)<cfg.share_min_distance:continue
            cen,rad=b.enclosing_circle(tr.poly)
            if rad<=b.CLEAR_CERT_RADIUS or b.dist(cen,dev.position)>cfg.channel_potential_radius+rad:continue
            self.rb_share_checks+=1
            try:gain=self.posterior_gain(c,dev.position)
            except PosteriorUnavailable:
                self.rb_share_failures+=1
                gain=expected_radius_gain(tr.poly,tr.positives,tr.negatives,dev.position)
            if gain<self.share_threshold:continue
            obs=dev.detect(c);st.shared_count+=1;tr.add(dev.position,obs)
