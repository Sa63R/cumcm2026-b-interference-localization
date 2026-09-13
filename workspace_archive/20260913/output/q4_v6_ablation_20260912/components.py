"""Ablation-only V6 controller with independent rollout and RB-sharing switches.

The loop follows the frozen q4_guarded_solver. No vendor file is edited. Full,
V5, V4 and exposed-switch variants are checked against their original traces.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
import benchmark as bench
import q4_baseline as b
from q4_state import Engine
from q4_rb_sharing import RBEngine
from q4_v5 import default_config
from q4_belief import SceneFactory, PosteriorUnavailable
from q4_proposals import local_candidates
from q4_route_critic import route_features
from q4_transit_critic import Critic, proposals, apply_probe


@dataclass(frozen=True)
class Components:
    route: bool=True
    transit: str='learned'  # off, learned, geometry
    guard_stations: int=4
    local_rollout: bool=True
    rb_sharing: bool=True


FULL=Components()
VARIANTS={
    'v4':Components(False,'off',0,False,False),
    'v5':Components(False,'off',4,True,True),
    'full':FULL,
    'no_route':replace(FULL,route=False),
    'no_transit':replace(FULL,transit='off'),
    'no_guard':replace(FULL,guard_stations=0),
    'no_rollout':replace(FULL,local_rollout=False),
    'no_rb':replace(FULL,rb_sharing=False),
    'no_rollout_rb':replace(FULL,local_rollout=False,rb_sharing=False),
    'transit_geometry':replace(FULL,transit='geometry'),
    'route_only_unguarded':Components(True,'off',0,True,True),
    'transit_only_unguarded':Components(False,'learned',0,True,True),
}


def solve(device,components=FULL):
    if components.transit not in ('off','learned','geometry'):
        raise ValueError('Invalid transit mode')
    cfg=default_config()
    engine=RBEngine(device,cfg.base) if components.rb_sharing else Engine(device,cfg.base)
    critic=Critic(bench.V6_CONFIG.route_model) if components.route else None
    tc=Critic(bench.V6_CONFIG.transit_model) if components.transit=='learned' else None
    factory=SceneFactory(cfg.position_draws,cfg.seed)
    rng=random.Random(cfg.seed)
    calls=accepts=route_calls=route_accepts=0
    elapsed=0.;stops=tests=0;last=None;chosen=[];guarded_steps=0;activation=None
    failures={};planner_budget_reached=False
    while not engine.done():
        order=engine.ordered_actions()
        active=len(engine.state.visited)>=components.guard_stations
        if not active:guarded_steps+=1
        elif activation is None:
            activation=dict(visited_stations=len(engine.state.visited),known_sources=len(engine.state.pending)+len(engine.state.cleared))
        items=route_features(engine,order) if active and critic else []
        if items:
            vals=[critic.predict(f) for a,f in items];route_calls+=1
            j=max(range(len(items)),key=lambda j:vals[j])
            if j!=0 and vals[j]-vals[0]>bench.V6_CONFIG.route_threshold:
                a=items[j][0];order=[a]+[x for x in order if x!=a];route_accepts+=1
        base=order[0];a=base
        if active and components.transit!='off' and stops<bench.V6_CONFIG.max_transit_stops and last!=(base.kind,base.key):
            ps=proposals(engine,base)
            if ps:
                # Same candidate set and stop budget; geometry replaces the
                # learned choice AND rejection decision, without fitting a threshold.
                vals=([tc.predict(p['features']) for p in ps] if tc else [p['geometry_gain'] for p in ps])
                j=max(range(len(ps)),key=lambda j:vals[j])
                if vals[j]>bench.V6_CONFIG.transit_threshold:
                    tests+=apply_probe(engine,ps[j]);stops+=1;last=(base.kind,base.key);chosen.append(vals[j]);continue
        if components.local_rollout and elapsed>=cfg.total_planning_seconds:
            planner_budget_reached=True
        if components.local_rollout and base.kind=='target' and calls<cfg.max_decisions and elapsed<cfg.total_planning_seconds:
            tr=engine.state.pending[base.key]
            if tr.obs.status!='strong' and b.enclosing_circle(tr.poly)[1]>b.CLEAR_CERT_RADIUS:
                t=time.perf_counter();calls+=1
                try:
                    pool=factory.pool(engine.state,base.key)
                    pp=local_candidates(engine,base.key,pool,order,rng,samples=cfg.scenes,top=1)
                    if pp and pp[0][1]>cfg.min_gain:
                        if pp[0][0].kind!='probe' or engine.state.free_probes<cfg.max_free_probes:a=pp[0][0];accepts+=1
                except PosteriorUnavailable as error:
                    failures[str(error)]=failures.get(str(error),0)+1
                elapsed+=time.perf_counter()-t
        engine.execute(a);last=None
    return dict(**engine.report(),transit_stops=stops,transit_detections=tests,critic_values=chosen,
        route_calls=route_calls,route_accepts=route_accepts,planner_calls=calls,planner_accepts=accepts,
        planner_seconds=elapsed,guarded_steps=guarded_steps,activation=activation,
        minimum_stations=components.guard_stations,components=asdict(components),
        planner_failures=failures,planner_budget_reached=planner_budget_reached,
        rb_share_checks=getattr(engine,'rb_share_checks',0),rb_share_failures=getattr(engine,'rb_share_failures',0))
