"""Closed-loop, history-conditioned base-policy rollout with geometric fallback.

The candidate's first macro-action is followed by the ACTUAL resumable V4 policy
until a legal completion certificate. A sampled environment never reveals its
hidden sources to that continuation. Scores use mean(C/N), not E[C]/E[N].
All candidates share scene draws and error fields; an independent scene batch
rechecks the winner. No claim of POMCPOW/VOPP reproduction or SOTA is made.
"""
from __future__ import annotations
from dataclasses import dataclass,field,asdict
import math
import random
import statistics
import time
from collections import Counter
import q4_baseline as b
from q4_v4_solver import V4Config
from q4_v4_local import LocalConfig
from q4_state import Action,Engine
from q4_belief import SceneFactory,PosteriorUnavailable

@dataclass
class RolloutConfig:
    base:V4Config=field(default_factory=lambda:V4Config(local=LocalConfig(optical_cover_limit=6)))
    scenes:int=8
    validation_scenes:int=4
    candidates:int=7
    position_draws:int=96
    max_decisions:int=24
    max_free_probes:int=24
    min_gain:float=.7  # seconds per source, estimated paired gain
    risk_weight:float=.04
    stderr_weight:float=.25
    validation_stderr_weight:float=0.
    seed:int=34071
    decision_seconds:float=8.
    total_planning_seconds:float=180.
    shared_candidates:bool=True
    pair_candidates:bool=True
    alternative_routes:bool=True
    validate_winner:bool=True
    every:int=1
    rich_candidates:bool=False
    proposal_scenes:int=10

    def __post_init__(self):
        if self.scenes<2 or self.validation_scenes<2 or self.candidates<1 or self.max_decisions<0:
            raise ValueError('Invalid rollout counts')
        if not 0<=self.max_free_probes<=64:raise ValueError('Invalid free-probe budget')
        if min(self.decision_seconds,self.total_planning_seconds)<=0:raise ValueError('Invalid wall-clock budget')


def projected(p,a,z):
    d=b.sub(z,a);den=b.dot(d,d)
    u=max(0.,min(1.,b.dot(b.sub(p,a),d)/max(den,1e-20)))
    return b.add(a,b.mul(u,d))


def candidates(engine:Engine,ordered,prepared,cfg):
    st=engine.state;p=engine.device.position;pools=prepared[0]
    base=ordered[0];out=[base]
    def put(a):
        if a not in out:out.append(a)
    # Nearest route alternatives allow information-induced reordering.
    if cfg.alternative_routes:
        for a in ordered[1:3]:put(a)
    target=base.key if base.kind=='target' else next((a.key for a in ordered if a.kind=='target'),None)
    if target is not None and cfg.rich_candidates:
        from q4_proposals import local_candidates
        proposals=local_candidates(engine,target,pools[target],ordered,
            random.Random(cfg.seed+st.actions*997),samples=cfg.proposal_scenes,top=2)
        for a,gain in proposals:
            if a.kind!='probe' or st.free_probes<cfg.max_free_probes:put(a)
    if target is not None and not cfg.rich_candidates:
        tr=st.pending[target]
        if tr.obs.status!='strong' and b.enclosing_circle(tr.poly)[1]>b.CLEAR_CERT_RADIUS:
            if cfg.pair_candidates:
                put(Action('pair',target,axial=.60,lateral=.075))
                put(Action('pair',target,axial=.45,lateral=.15))
            if cfg.shared_candidates and st.free_probes<cfg.max_free_probes:
                mean=pools[target].mean()
                upper=max(b.dist(tr.anchor,v) for v in tr.poly)
                e=b.unit(tr.obs.theta);normal=(-e[1],e[0])
                probes=[b.add(mean,b.mul(sign*min(80.,.04*upper),normal)) for sign in [-1,1]]
                q=min(probes,key=lambda x:b.dist(p,x))
                if min(b.dist(q,x) for x in tr.measured)>20.:put(Action('probe',target,q))
    if cfg.shared_candidates and st.free_probes<cfg.max_free_probes and len(st.pending)>=2:
        # A low-detour shared observation along the first remaining station leg.
        from q4_information import expected_radius_gain
        sites=[a for a in ordered if a.kind=='site']
        if sites:
            z=st.sites[sites[0].key];options=[]
            for c,tr in st.pending.items():
                if tr.obs.status=='strong':continue
                g=pools[c].mean();q=projected(g,p,z)
                if b.dist(p,q)<35 or b.dist(z,q)<35:continue
                if min(b.dist(q,x) for x in tr.measured)<25:continue
                score=0.;contributors=0
                for tt in st.pending.values():
                    if tt.obs.status=='strong':continue
                    gain=expected_radius_gain(tt.poly,tt.positives,tt.negatives,q)
                    if gain>15:contributors+=1;score+=gain
                if contributors>=2:options.append((score,c,q))
            if options:
                _,c,q=max(options);put(Action('probe',c,q))
    return out[:cfg.candidates]


def paired_score(deltas,cfg,stderr_weight=None):
    avg=statistics.mean(deltas)
    se=statistics.stdev(deltas)/math.sqrt(len(deltas)) if len(deltas)>1 else 0.
    k=max(1,math.ceil(.2*len(deltas)))
    tail=statistics.mean(sorted([max(0.,d) for d in deltas],reverse=True)[:k])
    w=cfg.stderr_weight if stderr_weight is None else stderr_weight
    return avg+cfg.risk_weight*tail+w*se

class RolloutPlanner:
    def __init__(self,config=None):
        self.config=config or RolloutConfig();c=self.config
        self.factory=SceneFactory(c.position_draws,c.seed)
        self.rng=random.Random(c.seed)
        self.calls=0;self.accepts=0;self.failures=Counter();self.action_counts=Counter()
        self.seconds=0.;self.rollouts=0;self.trace=[]

    def _evaluate(self,engine,action,scene):
        dev=scene.device(engine.device.position,engine.device.channel)
        continuation=Engine(dev,engine.config,engine.state.clone())
        continuation.execute(action)
        continuation.run_base()
        # End only when POLICY has a legal certificate, not when simulator's
        # hidden target counter happens to become zero.
        if not continuation.done():raise RuntimeError('Incomplete policy continuation')
        self.rollouts+=1
        return dev.virtual_seconds/scene.count

    def choose(self,engine,ordered):
        cfg=self.config;a0=ordered[0];st=engine.state
        if (not st.pending or not st.visited or self.calls>=cfg.max_decisions or
            self.seconds>=cfg.total_planning_seconds or st.actions%cfg.every):return a0
        # Don't spend a search budget on a near, already certified clear.
        if a0.kind=='target':
            tr=st.pending[a0.key]
            if tr.obs.status=='strong':return a0
            cen,r=b.enclosing_circle(tr.poly)
            if r<=19.5 and b.dist(cen,engine.device.position)<100:return a0
        begin=time.perf_counter();deadline=begin+min(cfg.decision_seconds,cfg.total_planning_seconds-self.seconds)
        self.calls+=1;chosen=a0;record={'call':self.calls,'global_action':st.actions,'base':asdict(a0)}
        try:
            prepared=self.factory.prepare(st)
            aa=candidates(engine,ordered,prepared,cfg)
            if len(aa)<2:return a0
            scenes=self.factory.sample(st,cfg.scenes,self.rng,prepared)
            scores=[[] for a in aa]
            for w in scenes:
                for i,a in enumerate(aa):
                    if time.perf_counter()>deadline:raise TimeoutError('Decision budget')
                    scores[i].append(self._evaluate(engine,a,w))
            diffs=[[x-z for x,z in zip(v,scores[0])] for v in scores]
            ranks=[0.]+[paired_score(d,cfg) for d in diffs[1:]]
            j=min(range(len(aa)),key=lambda i:ranks[i])
            record.update(candidates=[asdict(a) for a in aa],scores=ranks,estimated_gains=[-statistics.mean(d) for d in diffs])
            if j and ranks[j]<-cfg.min_gain:
                if cfg.validate_winner:
                    ww=self.factory.sample(st,cfg.validation_scenes,self.rng,prepared)
                    vd=[]
                    for w in ww:
                        if time.perf_counter()>deadline:raise TimeoutError('Validation budget')
                        vc=self._evaluate(engine,aa[j],w);vb=self._evaluate(engine,a0,w)
                        vd.append(vc-vb)
                    vscore=paired_score(vd,cfg,cfg.validation_stderr_weight)
                    record['validation_score']=vscore
                    if vscore<-cfg.min_gain:chosen=aa[j]
                else:chosen=aa[j]
        except (PosteriorUnavailable,TimeoutError) as err:
            self.failures[type(err).__name__+': '+str(err)]+=1
            record['fallback_reason']=str(err)
            chosen=a0
        finally:
            elapsed=time.perf_counter()-begin;self.seconds+=elapsed
            record.update(chosen=asdict(chosen),wall_seconds=elapsed)
            self.trace.append(record)
        if chosen!=a0:self.accepts+=1
        self.action_counts[chosen.kind]+=1
        return chosen


def solve_rollout(device,config=None):
    planner=RolloutPlanner(config);engine=Engine(device,planner.config.base)
    while not engine.done():
        ordered=engine.ordered_actions()
        action=planner.choose(engine,ordered)
        engine.execute(action)
    return dict(**engine.report(),planner_calls=planner.calls,planner_accepts=planner.accepts,
        planner_seconds=planner.seconds,imagined_continuations=planner.rollouts,
        planner_failures=dict(planner.failures),selected_action_kinds=dict(planner.action_counts),
        planner_trace=planner.trace,configuration=asdict(planner.config))
