"""Observation-only learned cost-to-go ranking with verified fallback actions.
Offline labels are from full feedback-driven synthetic continuations; online
execution never reads source positions, radii, types, counts, or scenario seeds.
"""
import random,time
import q4_baseline as b
from q4_v5 import default_config
from q4_state import Engine
from q4_rb_sharing import RBEngine
from q4_belief import SceneFactory,PosteriorUnavailable
from q4_proposals import local_candidates
from q4_route_critic import route_features
from q4_transit_critic import Critic,proposals,apply_probe


def solve_learned_route(device, model='route_critic_huber.json', threshold=10.,
                        v5=True, transit=None, transit_threshold=0., maxstops=16):
    cfg=default_config();engine=RBEngine(device,cfg.base) if v5 else Engine(device,cfg.base)
    critic=Critic(model);tc=Critic(transit) if transit else None
    factory=SceneFactory(cfg.position_draws,cfg.seed);rng=random.Random(cfg.seed)
    calls=accepts=route_calls=route_accepts=0;elapsed=0.;stops=tests=0;last=None;chosen=[]
    while not engine.done():
        order=engine.ordered_actions();items=route_features(engine,order)
        if items:
            vals=[critic.predict(f) for a,f in items];route_calls+=1
            j=max(range(len(items)),key=lambda j:vals[j])
            if j!=0 and vals[j]-vals[0]>threshold:
                a=items[j][0];order=[a]+[x for x in order if x!=a];route_accepts+=1
        base=order[0];a=base
        if tc and stops<maxstops and last!=(base.kind,base.key):
            ps=proposals(engine,base)
            if ps:
                vals=[tc.predict(p['features'])for p in ps];j=max(range(len(ps)),key=lambda j:vals[j])
                if vals[j]>transit_threshold:
                    tests+=apply_probe(engine,ps[j]);stops+=1;last=(base.kind,base.key);chosen.append(vals[j]);continue
        if v5 and base.kind=='target' and calls<cfg.max_decisions and elapsed<cfg.total_planning_seconds:
            tr=engine.state.pending[base.key]
            if tr.obs.status!='strong' and b.enclosing_circle(tr.poly)[1]>b.CLEAR_CERT_RADIUS:
                t=time.perf_counter();calls+=1
                try:
                    pool=factory.pool(engine.state,base.key)
                    pp=local_candidates(engine,base.key,pool,order,rng,samples=cfg.scenes,top=1)
                    if pp and pp[0][1]>cfg.min_gain:
                        if pp[0][0].kind!='probe' or engine.state.free_probes<cfg.max_free_probes:a=pp[0][0];accepts+=1
                except PosteriorUnavailable:pass
                elapsed+=time.perf_counter()-t
        engine.execute(a);last=None
    return dict(**engine.report(),transit_stops=stops,transit_detections=tests,critic_values=chosen,
                route_calls=route_calls,route_accepts=route_accepts,
                planner_calls=calls,planner_accepts=accepts,planner_seconds=elapsed)
