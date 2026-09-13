"""Plan extra discovery scans by estimated avoided late-discovery detours.
No source is eliminated on probabilistic evidence. The fixed coverage certificate
and all costed failed scans are kept. This is a heuristic action-value estimate.
"""
import math,random,time
import q4_baseline as b
from q4_state import Engine,Action
from q4_belief import SceneFactory,PosteriorUnavailable

class DiscoveryValue:
    def __init__(self,draws=256,samples=64,cost_ratio=1.5,bonus=0.,max_extra=12,min_sep=200.,seed=188881):
        self.factory=SceneFactory(draws,seed);self.rng=random.Random(seed)
        self.samples=samples;self.cost_ratio=cost_ratio;self.bonus=bonus;self.max_extra=max_extra;self.min_sep=min_sep
        self.extras=0;self.calls=0;self.failures=0;self.last_key=None;self.seconds=0.;self.trace=[]
    def try_scan(self,engine,order):
        s=engine.state;p=engine.device.position
        key=(p,len(s.pending)+len(s.cleared),tuple(sorted(s.unknown)))
        if key==self.last_key:return False
        self.last_key=key
        if self.extras>=self.max_extra or not s.unknown or not s.visited:return False
        if min((b.dist(p,q) for q in s.scanpoints),default=1e9)<self.min_sep:return False
        k=len(s.pending)+len(s.cleared)
        if k==16:return False
        started=time.perf_counter();self.calls+=1
        c=min(s.unknown)
        try:pool=self.factory.pool(s,c)
        except PosteriorUnavailable:self.failures+=1;return False
        expected_n=(max(10,k)+16)/2;expected_unseen=expected_n-k
        if expected_unseen<=0:return False
        pts=[p]+[(s.sites[a.key] if a.kind=='site' else s.pending[a.key].center()) for a in order]
        routesites=[(i+1,s.sites[a.key]) for i,a in enumerate(order) if a.kind=='site']
        po=.5*pool.omni_evidence;pd=.5*pool.dir_evidence;values=[];seen=0
        if po+pd<=0:return False
        for _ in range(self.samples):
            typ=1 if self.rng.random()<po/(po+pd) else 2
            g,r,u=pool.sample(typ,self.rng,self.factory.radius_prior)
            visible=lambda q:b.dist(q,g)<=r and (u is None or b.dot(u,b.sub(q,g))>=0)
            if not visible(p):values.append(0.);continue
            seen+=1
            first=next((i for i,q in routesites if visible(q)),None)
            if first is None:values.append(0.);continue # incomplete numerical support: no optimistic credit
            ins=[b.dist(a,g)+b.dist(g,z)-b.dist(a,z) for a,z in zip(pts,pts[1:])]+[b.dist(pts[-1],g)]
            old=min(ins[first:]) if first<len(ins) else b.dist(pts[-1],g)
            new=min(ins)
            values.append(max(0.,old-new)/5+self.bonus)
        gain=sum(values)/self.samples*expected_unseen
        cost=6*len(s.unknown)-(engine.device.channel in s.unknown)
        self.seconds+=time.perf_counter()-started
        if gain<=cost*self.cost_ratio:return False
        self.extras+=1
        self.trace.append(dict(point=p,estimated_saving=gain,cost=cost,positive_fraction=seen/self.samples,known=k))
        engine.discovery();engine.shared()
        return True

class DiscoveryEngine(Engine):
    def __init__(self,device,cfg,ratio=1.5,bonus=0.,min_sep=200.):
        super().__init__(device,cfg);self.discovery_planner=DiscoveryValue(cost_ratio=ratio,bonus=bonus,min_sep=min_sep)
    def run_base(self,max_actions=None):
        while not self.done():
            order=self.ordered_actions()
            if self.discovery_planner.try_scan(self,order):continue
            self.execute(order[0])
        return dict(**self.report(),extra_scans=self.discovery_planner.extras,discovery_trace=self.discovery_planner.trace)
