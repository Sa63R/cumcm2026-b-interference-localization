"""Hierarchical rollout with an optional global lookahead at scan decisions.

Local source control uses RB-conditioned feedback rollouts plus a route-entry
terminal approximation. Ambiguous station choices can additionally use full
all-channel V4 continuations. The deterministic safety kernel is unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass,field,asdict
import random,time
import q4_baseline as b
from q4_state import Engine
from q4_v4_solver import V4Config
from q4_rollout import RolloutConfig,RolloutPlanner
from q4_belief import PosteriorUnavailable
from q4_proposals import local_candidates

@dataclass
class HybridConfig:
    global_config:RolloutConfig=field(default_factory=lambda:RolloutConfig(base=V4Config(),scenes=12,validation_scenes=6,position_draws=128,max_decisions=10,candidates=6,rich_candidates=True,proposal_scenes=12,min_gain=.3,risk_weight=.01,stderr_weight=.1,decision_seconds=15.))
    local_scenes:int=24
    local_min_gain:float=2.
    local_calls:int=40
    global_at_targets:bool=False


def solve_hybrid(device,config=None):
    cfg=config or HybridConfig();planner=RolloutPlanner(cfg.global_config)
    engine=Engine(device,cfg.global_config.base)
    rng=random.Random(67541);calls=accepts=0;elapsed=0.;localtrace=[];localfails={}
    while not engine.done():
        order=engine.ordered_actions();a=base=order[0]
        if base.kind=='target' and calls<cfg.local_calls:
            tr=engine.state.pending[base.key]
            if tr.obs.status!='strong' and b.enclosing_circle(tr.poly)[1]>b.CLEAR_CERT_RADIUS:
                start=time.perf_counter();calls+=1
                try:
                    pool=planner.factory.pool(engine.state,base.key)
                    pp=local_candidates(engine,base.key,pool,order,rng,samples=cfg.local_scenes,top=1)
                    if pp and pp[0][1]>cfg.local_min_gain:
                        if pp[0][0].kind!='probe' or engine.state.free_probes<cfg.global_config.max_free_probes:
                            a=pp[0][0];accepts+=1
                    localtrace.append(dict(action=engine.state.actions,base=asdict(base),chosen=asdict(a),predicted_gain=pp[0][1] if pp else 0.))
                except PosteriorUnavailable as err:localfails[str(err)]=localfails.get(str(err),0)+1
                elapsed+=time.perf_counter()-start
        if base.kind=='site' or cfg.global_at_targets:
            a=planner.choose(engine,order)
        engine.execute(a)
    return dict(**engine.report(),planner_calls=planner.calls,planner_accepts=planner.accepts,
        planner_seconds=planner.seconds+elapsed,imagined_continuations=planner.rollouts,
        planner_failures=dict(planner.failures),planner_trace=planner.trace,
        local_calls=calls,local_accepts=accepts,local_trace=localtrace,local_failures=localfails,
        configuration=asdict(cfg))
