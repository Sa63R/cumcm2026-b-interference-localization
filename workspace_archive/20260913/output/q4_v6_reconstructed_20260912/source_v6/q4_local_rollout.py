"""Fast hierarchical rollout: global V4 route, posterior-conditioned local control.

The terminal approximation is travel to the next current route task, NOT an
oracle route through hidden target positions. Local imagined trajectories end
only on actual successful optical feedback. Global coverage proof is unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass,field,asdict
import random,time
import q4_baseline as b
from q4_v4_solver import V4Config
from q4_state import Engine
from q4_belief import SceneFactory,PosteriorUnavailable
from q4_proposals import local_candidates

@dataclass
class LocalRolloutConfig:
    base:V4Config=field(default_factory=V4Config)
    scenes:int=32
    position_draws:int=128
    min_gain:float=5.
    max_decisions:int=40
    max_free_probes:int=40
    seed:int=67541
    total_planning_seconds:float=120.
    dynamic_coverage:bool=False
    coverage_updates:int=5
    rb_sharing:bool=False
    share_threshold:float=10.


def solve_local_rollout(device,config=None):
    cfg=config or LocalRolloutConfig()
    if cfg.rb_sharing:
        from q4_rb_sharing import RBEngine
        engine=RBEngine(device,cfg.base,share_threshold=cfg.share_threshold)
    else:engine=Engine(device,cfg.base)
    factory=SceneFactory(cfg.position_draws,cfg.seed);rng=random.Random(cfg.seed)
    calls=accepts=0;elapsed=0.;trace=[];failures={}
    from q4_dynamic_coverage import CoverageExchanger,CoverageConfig
    exchanger=CoverageExchanger(CoverageConfig(max_updates=cfg.coverage_updates))
    last_coverage_key=None
    while not engine.done():
        # Only retry when a source was discovered or cleared; bounded planning work.
        key=(len(engine.state.pending)+len(engine.state.cleared),len(engine.state.cleared))
        if cfg.dynamic_coverage and key!=last_coverage_key and engine.state.visited:
            exchanger.attempt(engine);last_coverage_key=key
        order=engine.ordered_actions();base=order[0];a=base
        if base.kind=='target' and calls<cfg.max_decisions and elapsed<cfg.total_planning_seconds:
            tr=engine.state.pending[base.key]
            if tr.obs.status!='strong' and b.enclosing_circle(tr.poly)[1]>b.CLEAR_CERT_RADIUS:
                start=time.perf_counter();calls+=1
                try:
                    pool=factory.pool(engine.state,base.key)
                    pp=local_candidates(engine,base.key,pool,order,rng,samples=cfg.scenes,top=1)
                    if pp and pp[0][1]>cfg.min_gain:
                        if pp[0][0].kind!='probe' or engine.state.free_probes<cfg.max_free_probes:
                            a=pp[0][0];accepts+=1
                    trace.append(dict(action=engine.state.actions,base=asdict(base),chosen=asdict(a),predicted_gain=pp[0][1] if pp else 0.))
                except PosteriorUnavailable as err:
                    failures[str(err)]=failures.get(str(err),0)+1
                elapsed+=time.perf_counter()-start
        engine.execute(a)
    return dict(**engine.report(),planner_calls=calls,planner_accepts=accepts,planner_seconds=elapsed,
                planner_failures=failures,planner_trace=trace,configuration=asdict(cfg),
                coverage_updates=exchanger.updates,coverage_checks=exchanger.checks,coverage_seconds=exchanger.seconds,coverage_trace=exchanger.trace,
                rb_share_checks=getattr(engine,"rb_share_checks",0),rb_share_failures=getattr(engine,"rb_share_failures",0))
