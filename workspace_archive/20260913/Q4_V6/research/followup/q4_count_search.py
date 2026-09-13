"""Acquire extra discovery measurements near the known source-count upper bound.
Only positive, distinct-channel observations count. An unresponsive channel is
never removed by a probability or by the count trigger. The original coverage
and verified source-clear requirements are retained.
"""
from __future__ import annotations
import math,random,time
import q4_baseline as b
from q4_v5 import default_config
from q4_state import Engine
from q4_rb_sharing import RBEngine
from q4_belief import SceneFactory,PosteriorUnavailable
from q4_proposals import local_candidates
from q4_transit_critic import Critic,proposals,apply_probe
from q4_route_critic import route_features

class CountSearch:
    def __init__(self,minimum_known=15,max_scans=8,min_separation=150.):
        self.minimum_known=minimum_known;self.max_scans=max_scans;self.min_separation=min_separation
        self.scans=0;self.detections=0;self.discoveries=0;self.triggered_upper=False;self.log=[]
    def try_scan(self,engine):
        s=engine.state;p=engine.device.position;k=len(s.pending)+len(s.cleared)
        if self.scans>=self.max_scans or not s.remaining or not s.unknown or not s.visited:return False
        if not self.minimum_known<=k<16:return False
        if min((b.dist(p,q)for q in s.scanpoints),default=1e9)<self.min_separation:return False
        before=sum(len(x)for x in s.rf.values());unknown=len(s.unknown)
        self.scans+=1;engine.discovery();engine.shared();engine.sync_bound()
        newk=len(s.pending)+len(s.cleared);nd=sum(len(x)for x in s.rf.values())-before
        self.detections+=nd;self.discoveries+=newk-k;self.triggered_upper|=newk==16
        self.log.append(dict(point=p,known_before=k,known_after=newk,unknown_before=unknown,rf_operations_including_sharing=nd))
        return True

def solve_count_search(device,minimum_known=15,max_scans=8,min_separation=150.,v5=True,transit=False,route=False,min_visits=0):
    cfg=default_config();engine=RBEngine(device,cfg.base)if v5 else Engine(device,cfg.base)
    cs=CountSearch(minimum_known,max_scans,min_separation);tc=Critic('transit_critic_big_extra.json')if transit else None;rc=Critic('route_critic_extra.json')if route else None
    factory=SceneFactory(cfg.position_draws,cfg.seed);rng=random.Random(cfg.seed)
    calls=accepts=rchanges=0;elapsed=0.;stops=tests=0;last=None
    while not engine.done():
        cs.try_scan(engine)
        order=engine.ordered_actions()
        active=len(engine.state.visited)>=min_visits
        if rc and active:
            items=route_features(engine,order)
            if items:
                vv=[rc.predict(f)for a,f in items];j=max(range(len(items)),key=lambda j:vv[j])
                if j and vv[j]>vv[0]:
                    a=items[j][0];order=[a]+[a0 for a0 in order if a0!=a];rchanges+=1
        base=order[0];a=base
        if tc and active and stops<16 and last!=(base.kind,base.key):
            ps=proposals(engine,base)
            if ps:
                vv=[tc.predict(p['features'])for p in ps];j=max(range(len(ps)),key=lambda j:vv[j])
                if vv[j]>0:
                    tests+=apply_probe(engine,ps[j]);stops+=1;last=(base.kind,base.key);continue
        if v5 and base.kind=='target' and calls<cfg.max_decisions and elapsed<cfg.total_planning_seconds:
            tr=engine.state.pending[base.key]
            if tr.obs.status!='strong' and b.enclosing_circle(tr.poly)[1]>b.CLEAR_CERT_RADIUS:
                t=time.perf_counter();calls+=1
                try:
                    pool=factory.pool(engine.state,base.key);pp=local_candidates(engine,base.key,pool,order,rng,samples=cfg.scenes,top=1)
                    if pp and pp[0][1]>cfg.min_gain:
                        if pp[0][0].kind!='probe' or engine.state.free_probes<cfg.max_free_probes:a=pp[0][0];accepts+=1
                except PosteriorUnavailable:pass
                elapsed+=time.perf_counter()-t
        engine.execute(a);last=None
    return dict(**engine.report(),planner_calls=calls,planner_accepts=accepts,planner_seconds=elapsed,transit_stops=stops,transit_detections=tests,
                route_accepts=rchanges,count_search_scans=cs.scans,count_search_detections=cs.detections,count_search_discoveries=cs.discoveries,
                count_search_triggered_upper=cs.triggered_upper,count_search_log=cs.log)
