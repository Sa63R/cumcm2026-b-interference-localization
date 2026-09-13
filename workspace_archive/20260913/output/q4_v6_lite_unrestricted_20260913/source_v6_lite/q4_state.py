"""Resumable, feedback-only V4 execution kernel.

Safety does not depend on probabilities: unchanged outer polygons, bounded RF
progress, actual successful clears and the 21-site coverage certificate remain.
Setting no action selector reproduces V4, including action order and tie breaks.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import copy
import math
from typing import Callable
import q4_baseline as b
from q4_v4_solver import V4Config, Belief
from q4_v4_local import localize_step
from q4_v3_solver import validate_layout
from q4_route_cached import multi_route

@dataclass(frozen=True)
class RFRecord:
    point: b.Point
    obs: b.Observation

@dataclass(frozen=True)
class OpticalRecord:
    point: b.Point
    success: bool

@dataclass(frozen=True)
class Action:
    kind: str  # site, target, pair, probe
    key: int
    point: b.Point | None = None
    axial: float | None = None
    lateral: float | None = None

@dataclass
class State:
    sites: list[b.Point]
    remaining: set[int]
    unknown: set[int] = field(default_factory=lambda:set(range(1,21)))
    pending: dict[int,Belief] = field(default_factory=dict)
    cleared: set[int] = field(default_factory=set)
    cleared_tracks: dict[int,Belief] = field(default_factory=dict)
    negatives: dict[int,list[b.Point]] = field(default_factory=lambda:{c:[] for c in range(1,21)})
    rf: dict[int,list[RFRecord]] = field(default_factory=lambda:{c:[] for c in range(1,21)})
    optical: dict[int,list[OpticalRecord]] = field(default_factory=lambda:{c:[] for c in range(1,21)})
    visited: list[int] = field(default_factory=list)
    scanpoints: list[b.Point] = field(default_factory=list)
    localizations: list[dict] = field(default_factory=list)
    shared_count: int = 0
    replans: int = 0
    actions: int = 0
    free_probes: int = 0

    def clone(self) -> 'State':
        def track(t):
            q=copy.copy(t)
            q.poly=t.poly[:];q.measured=t.measured[:]
            q.positives=t.positives[:];q.negatives=t.negatives[:]
            return q
        # Coordinates, Observation, RFRecord and OpticalRecord are immutable.
        # Mutable lists/dicts and track progress are independent across candidates.
        return State(sites=self.sites[:],remaining=self.remaining.copy(),unknown=self.unknown.copy(),
            pending={c:track(t) for c,t in self.pending.items()},cleared=self.cleared.copy(),
            cleared_tracks={c:track(t) for c,t in self.cleared_tracks.items()},
            negatives={c:p[:] for c,p in self.negatives.items()},
            rf={c:r[:] for c,r in self.rf.items()},optical={c:r[:] for c,r in self.optical.items()},
            visited=self.visited[:],scanpoints=self.scanpoints[:],localizations=self.localizations[:],
            shared_count=self.shared_count,replans=self.replans,actions=self.actions,free_probes=self.free_probes)

class Recorder:
    """Expose only the documented device capabilities and record returned data."""
    __slots__=('__device','state')
    def __init__(self,device:b.Device,state:State):
        self.__device=device;self.state=state
    @property
    def position(self):return self.__device.position
    @property
    def channel(self):return self.__device.channel
    def move(self,p):self.__device.move(p)
    def detect(self,c):
        obs=self.__device.detect(c)
        self.state.rf[c].append(RFRecord(self.position,obs))
        return obs
    def clear(self,c):
        ok=self.__device.clear(c)
        self.state.optical[c].append(OpticalRecord(self.position,ok))
        return ok

class Engine:
    def __init__(self,device:b.Device,config:V4Config|None=None,state:State|None=None):
        self.config=config or V4Config()
        if state is None:
            n1,n2,r1,r2=self.config.ring_sites
            sites=[(0.,0.)]+[b.mul(r,b.unit(2*math.pi*k/n)) for n,r in [(n1,r1),(n2,r2)] for k in range(n)]
            validate_layout(tuple(sites))
            state=State(sites,set(range(len(sites))))
        self.state=state;self.device=Recorder(device,state)

    def sync_bound(self):
        s=self.state
        if len(s.cleared)+len(s.pending)==16:
            s.remaining.clear();s.unknown.clear()

    def done(self):
        self.sync_bound()
        return not self.state.remaining and not self.state.pending

    def ordered_actions(self) -> list[Action]:
        self.sync_bound();s=self.state
        if not s.visited:return [Action('site',0)]
        keys=[('site',i) for i in sorted(s.remaining)]+[('target',c) for c in sorted(s.pending)]
        if not keys:return []
        points=[s.sites[k] if kind=='site' else s.pending[k].center() for kind,k in keys]
        route=multi_route(self.device.position,list(range(len(keys))),points,self.config.route_starts)
        return [Action(*keys[j]) for j in route]

    def discovery(self):
        s=self.state;dev=self.device;cfg=self.config
        for c in sorted(s.unknown,key=lambda x:(x!=dev.channel,x)):
            if len(s.cleared)+len(s.pending)==16:break
            obs=dev.detect(c)
            if obs.status=='none':
                s.negatives[c].append(dev.position);continue
            s.unknown.remove(c)
            poly=b.initial_polygon(dev.position,obs.theta,1500.) if obs.status=='bearing' else []
            tr=Belief(channel=c,anchor=dev.position,obs=obs,poly=poly,
                      measured=[dev.position],positives=[dev.position],negatives=s.negatives[c][:],
                      negative_enabled=cfg.directional_negatives,direction_sectors=cfg.direction_sectors)
            tr.tighten();s.pending[c]=tr
        s.scanpoints.append(dev.position)

    def shared(self,exclude=None):
        from q4_information import expected_radius_gain
        s=self.state;dev=self.device;cfg=self.config
        for c in sorted(s.pending,key=lambda x:(x!=dev.channel,x)):
            if c==exclude:continue
            tr=s.pending[c]
            if tr.obs.status=='strong':continue
            if min(b.dist(dev.position,p) for p in tr.measured)<cfg.share_min_distance:continue
            center,radius=b.enclosing_circle(tr.poly)
            if radius<=b.CLEAR_CERT_RADIUS:continue
            if b.dist(center,dev.position)>cfg.channel_potential_radius+radius:continue
            if cfg.share_min_gain>0 and expected_radius_gain(tr.poly,tr.positives,tr.negatives,dev.position)<cfg.share_min_gain:continue
            obs=dev.detect(c);s.shared_count+=1;tr.add(dev.position,obs)

    def finish(self,c,stats):
        s=self.state
        # Fail closed. Completion is possible only if a clear(True) was recorded.
        if not s.optical[c] or not s.optical[c][-1].success:
            raise RuntimeError('Target completion without actual successful clear')
        s.localizations.append(dict(channel=c,**stats))
        s.cleared.add(c);s.cleared_tracks[c]=s.pending.pop(c)
        if len(s.cleared)<16 and self.config.shared_at_clears:self.shared()

    def execute(self,a:Action):
        s=self.state;dev=self.device;cfg=self.config;s.actions+=1
        if s.actions>len(s.sites)+16*14+65:
            raise RuntimeError('Finite-action safeguard exceeded')
        if a.kind=='site':
            if a.key not in s.remaining:raise ValueError('Already visited site')
            dev.move(s.sites[a.key]);s.remaining.remove(a.key);s.visited.append(a.key)
            self.discovery()
            if cfg.shared_at_stations:self.shared()
        elif a.kind in ('target','pair'):
            tr=s.pending[a.key]
            local=cfg.local if a.kind=='target' else replace(cfg.local,axial=a.axial,lateral=a.lateral)
            stats=localize_step(dev,tr,local,on_probe=(lambda:self.shared(exclude=a.key)) if cfg.shared_at_probes else None)
            if stats.get('complete') is False:s.replans+=1
            else:self.finish(a.key,stats)
        elif a.kind=='probe':
            if s.free_probes>=64:raise ValueError('Free-probe budget exhausted')
            if a.point is None or a.key not in s.pending:raise ValueError('Invalid free probe')
            s.free_probes+=1
            dev.move(a.point);obs=dev.detect(a.key);tr=s.pending[a.key];tr.add(dev.position,obs)
            if obs.status=='strong':
                b.checked_clear(dev,a.key,dev.position)
                self.finish(a.key,dict(certificate='free_probe_strong',stages=tr.rf_rounds,pair_failures=tr.pair_failures))
            elif cfg.shared_at_probes:self.shared(exclude=a.key)
        else:raise ValueError(f'Unknown action kind: {a.kind}')

    def run_base(self,max_actions:int|None=None):
        done=0
        while not self.done():
            self.execute(self.ordered_actions()[0]);done+=1
            if max_actions is not None and done>=max_actions:break
        return self.report()

    def report(self):
        s=self.state
        return dict(stop_certificate='source_upper_bound' if len(s.cleared)==16 else ('coverage_complete' if self.done() else 'in_progress'),
            visited_stations=s.visited,localizations=s.localizations,shared_detections=s.shared_count,
            scanpoints=len(s.scanpoints),replans_after_pairs=s.replans,global_actions=s.actions,
            free_probes=s.free_probes)

def solve_resumable(device,config=None):return Engine(device,config).run_base()
