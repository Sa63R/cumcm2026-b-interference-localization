"""Intermediate RF stops with nonprobabilistic coverage and progress fallback.
Experimental version. All measurements cost time; the original 21-site proof
and actual success confirmation are retained.
"""
from __future__ import annotations
import math,random,time,dataclasses
import q4_baseline as b
from q4_state import Action
from q4_rb_sharing import RBEngine
from q4_transit import TransitEngine
from q4_v5 import default_config
from q4_belief import SceneFactory,PosteriorUnavailable
from q4_proposals import local_candidates
from q4_information import expected_radius_gain

class TransitRBEngine(RBEngine):
    entry=TransitEngine.entry
    def __init__(self,device,cfg,threshold=100.,maxstops=16,project=False,refine=False,trace=False,scope="all",lock=False):
        super().__init__(device,cfg);self.threshold=threshold;self.maxstops=maxstops
        self.project=project;self.refine=refine;self.transits=0;self.transit_detects=0;self.transit_trace=[];self.trace=trace;self.scope=scope;self.lock=lock;self.last_intercept=None
    def intercept(self,a):
        if self.transits>=self.maxstops or not self.state.pending:return False
        if self.lock and self.last_intercept==(a.kind,a.key):return False
        if self.scope=="sites" and a.kind!="site":return False
        if self.scope=="targets" and a.kind=="site":return False
        start=self.device.position;end=self.entry(a);dist=b.dist(start,end)
        if dist<200:return False
        fs=[.25,.5,.75]
        if self.project:
            v=b.sub(end,start)
            for tr in self.state.pending.values():
                if tr.obs.status=='strong':continue
                f=b.dot(b.sub(tr.center(),start),v)/(dist*dist)
                fs.extend(max(.12,min(.88,f+dd/dist)) for dd in [-150,0,150])
            fs=sorted(set(round(f,4) for f in fs))
        candidates=[]
        for f in fs:
            q=b.add(start,b.mul(f,b.sub(end,start)));chs=[];score=0.
            for c,tr in self.state.pending.items():
                if self.scope=="others" and a.kind=="target" and c==a.key:continue
                if tr.obs.status=='strong':continue
                cen,rad=b.enclosing_circle(tr.poly)
                if rad<35 or b.dist(q,cen)>800+rad or min(b.dist(q,p) for p in tr.measured)<60:continue
                gain=expected_radius_gain(tr.poly,tr.positives,tr.negatives,q)
                if gain>self.threshold:
                    chs.append(c);score+=gain-self.threshold
            if chs:candidates.append((score,q,chs))
        if not candidates:return False
        candidates.sort(reverse=True,key=lambda x:x[0])
        best=candidates[0]
        if self.refine:
            best=None
            for _,q,cs in candidates[:3]:
                total=0.;chs=[]
                for c in cs:
                    try:g=self.posterior_gain(c,q)
                    except PosteriorUnavailable:g=expected_radius_gain(self.state.pending[c].poly,self.state.pending[c].positives,self.state.pending[c].negatives,q)
                    if g>self.threshold:chs.append(c);total+=g-self.threshold
                if chs and (best is None or total>best[0]):best=(total,q,chs)
            if best is None:return False
        _,q,cs=best;self.transits+=1;self.last_intercept=(a.kind,a.key)
        before=self.device.position;self.device.move(q);updates=[]
        for c in sorted(cs,key=lambda c:(c!=self.device.channel,c)):
            if c not in self.state.pending:continue
            tr=self.state.pending[c];oldrad=b.enclosing_circle(tr.poly)[1]
            obs=self.device.detect(c);tr.add(q,obs);self.state.shared_count+=1;self.transit_detects+=1
            updates.append(dict(channel=c,status=obs.status,radius_before=oldrad,radius_after=b.enclosing_circle(tr.poly)[1] if obs.status!='strong' else 5.))
            if obs.status=='strong':
                b.checked_clear(self.device,c,q)
                self.finish(c,dict(certificate='transit_strong',stages=tr.rf_rounds))
        if self.trace:self.transit_trace.append(dict(previous=before,planned_end=end,point=q,updates=updates))
        return True

def solve_transit_v5(device,threshold=100.,maxstops=16,project=False,refine=False,trace=False,scope="all",lock=False):
    cfg=default_config();engine=TransitRBEngine(device,cfg.base,threshold,maxstops,project,refine,trace,scope,lock)
    factory=SceneFactory(cfg.position_draws,cfg.seed);rng=random.Random(cfg.seed)
    calls=accepts=0;elapsed=0.;failures={}
    while not engine.done():
        order=engine.ordered_actions();base=order[0];a=base
        if engine.intercept(base):continue
        if base.kind=='target' and calls<cfg.max_decisions and elapsed<cfg.total_planning_seconds:
            tr=engine.state.pending[base.key]
            if tr.obs.status!='strong' and b.enclosing_circle(tr.poly)[1]>b.CLEAR_CERT_RADIUS:
                t=time.perf_counter();calls+=1
                try:
                    pool=factory.pool(engine.state,base.key)
                    pp=local_candidates(engine,base.key,pool,order,rng,samples=cfg.scenes,top=1)
                    if pp and pp[0][1]>cfg.min_gain:
                        if pp[0][0].kind!='probe' or engine.state.free_probes<cfg.max_free_probes:a=pp[0][0];accepts+=1
                except PosteriorUnavailable as e:failures[str(e)]=failures.get(str(e),0)+1
                elapsed+=time.perf_counter()-t
        engine.execute(a)
        engine.last_intercept=None
    return dict(**engine.report(),planner_calls=calls,planner_accepts=accepts,planner_seconds=elapsed,planner_failures=failures,
                transit_stops=engine.transits,transit_detections=engine.transit_detects,transit_trace=engine.transit_trace)
