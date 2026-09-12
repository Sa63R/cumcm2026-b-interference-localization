"""Resumable V4 and experimental policies; original archive remains unchanged."""
from __future__ import annotations
import copy
import math
import time
from dataclasses import dataclass, field
import common
import q4_baseline as b
from q4_v4_solver import Belief, V4Config
from q4_v3_solver import validate_layout
from q4_v4_local import localize_step
from q4_route_cached import multi_route
from q4_information import expected_radius_gain, heuristic_visibility
from radius_direction import marginalize, visibility_probability


def analytic_gain(poly, positives, negatives, p):
    """Same three positions and radius score as V4; replace visibility only."""
    cen, rad = b.enclosing_circle(poly)
    if rad <= b.CLEAR_CERT_RADIUS: return 0.
    if min((b.dist(s, p) for s in positives + negatives), default=10000) < 20: return 0.
    far = max(poly, key=lambda v: b.dist(v, cen))
    other = max(poly, key=lambda v: b.dist(v, far))
    samples = [cen, b.add(b.mul(.3, cen), b.mul(.7, far)), b.add(b.mul(.3, cen), b.mul(.7, other))]
    gain = 0.
    for g in samples:
        try:
            prob = visibility_probability(g, positives, negatives, p)
        except ValueError:
            # A quadrature point with zero mass is not a safety claim. Preserve
            # baseline ranking if this point is numerically impossible.
            prob = heuristic_visibility(g, positives, negatives, p)
        if prob <= 0: continue
        if b.dist(p, g) <= 5:
            gain += prob * rad
            continue
        theta = math.atan2(g[1]-p[1], g[0]-p[0])
        new = b.clip_bearing(poly, p, theta, 1500.)
        if new:
            gain += prob * max(0., rad-b.enclosing_circle(new)[1])
    return gain / len(samples)


class HistoryDevice:
    """Observation-only adapter; solver never receives the true environment."""
    def __init__(self, device, history):
        self.__device = device
        self.history = history
    @property
    def position(self): return self.__device.position
    @property
    def channel(self): return self.__device.channel
    def move(self, p): self.__device.move(p)
    def detect(self, c):
        obs = self.__device.detect(c)
        self.history.setdefault(c, []).append(('detect', self.position, obs.status, obs.theta))
        return obs
    def clear(self, c):
        ok = self.__device.clear(c)
        self.history.setdefault(c, []).append(('clear', self.position, ok, None))
        return ok


@dataclass
class State:
    config: V4Config = field(default_factory=V4Config)
    analytic: bool = False
    sites: list = field(default_factory=list)
    remaining: set = field(default_factory=set)
    unknown: set = field(default_factory=lambda: set(range(1,21)))
    pending: dict = field(default_factory=dict)
    cleared: set = field(default_factory=set)
    archived: dict = field(default_factory=dict)
    negatives: dict = field(default_factory=lambda: {c: [] for c in range(1,21)})
    history: dict = field(default_factory=dict)
    visited: list = field(default_factory=list)
    scanpoints: list = field(default_factory=list)
    localizations: list = field(default_factory=list)
    shared_count: int = 0
    replans: int = 0
    actions: int = 0
    extra_actions: int = 0
    shifted_sites: int = 0

    def __post_init__(self):
        if not self.sites:
            n1,n2,r1,r2 = self.config.ring_sites
            self.sites = [(0.,0.)] + [b.mul(r,b.unit(2*math.pi*k/n)) for n,r in [(n1,r1),(n2,r2)] for k in range(n)]
            validate_layout(tuple(self.sites))
            self.remaining = set(range(len(self.sites)))

    def prepare(self):
        if len(self.cleared)+len(self.pending) == 16:
            self.remaining.clear()
            self.unknown.clear()
        return bool(self.remaining or self.pending)

    def ordered_actions(self, device):
        if not self.visited: return [('site',0)]
        keys = [('site',i) for i in sorted(self.remaining)] + [('target',c) for c in sorted(self.pending)]
        points = [self.sites[k] if typ=='site' else self.pending[k].center() for typ,k in keys]
        order = multi_route(device.position, list(range(len(keys))), points, self.config.route_starts)
        return [keys[i] for i in order]

    def discovery(self, device):
        for c in sorted(self.unknown, key=lambda x:(x!=device.channel,x)):
            if len(self.cleared)+len(self.pending)==16: break
            obs = device.detect(c)
            if obs.status=='none':
                self.negatives[c].append(device.position)
                continue
            self.unknown.remove(c)
            poly = b.initial_polygon(device.position,obs.theta,1500.) if obs.status=='bearing' else []
            tr = Belief(channel=c,anchor=device.position,obs=obs,poly=poly,
                        measured=[device.position],positives=[device.position],negatives=self.negatives[c][:],
                        negative_enabled=self.config.directional_negatives,direction_sectors=self.config.direction_sectors)
            tr.tighten()
            self.pending[c] = tr
        self.scanpoints.append(device.position)

    def shared(self, device, exclude=None):
        gain_fn = analytic_gain if self.analytic else expected_radius_gain
        for c in sorted(self.pending, key=lambda x:(x!=device.channel,x)):
            if c==exclude: continue
            tr = self.pending[c]
            if tr.obs.status=='strong': continue
            if min(b.dist(device.position,p) for p in tr.measured)<self.config.share_min_distance: continue
            center,radius = b.enclosing_circle(tr.poly)
            if radius<=b.CLEAR_CERT_RADIUS: continue
            if b.dist(center,device.position)>self.config.channel_potential_radius+radius: continue
            if self.config.share_min_gain>0 and gain_fn(tr.poly,tr.positives,tr.negatives,device.position)<self.config.share_min_gain: continue
            obs=device.detect(c)
            self.shared_count+=1
            tr.add(device.position,obs)

    def execute(self, raw_device, action):
        device = HistoryDevice(raw_device, self.history)
        self.actions += 1
        if self.actions > len(self.sites)+16*14+1+self.extra_actions:
            raise RuntimeError('Progress invariant violated')
        typ,k,*rest = action
        if typ=='shared':
            self.extra_actions += 1
            device.move(k)
            self.shared(device)
            return
        if typ=='shift':
            self.sites[k] = rest[0]
            self.shifted_sites += 1
            typ='site'
        if typ=='site':
            device.move(self.sites[k])
            self.remaining.remove(k)
            self.visited.append(k)
            self.discovery(device)
            if self.config.shared_at_stations: self.shared(device)
        else:
            tr = self.pending[k]
            stats = localize_step(device,tr,self.config.local,
                    on_probe=(lambda:self.shared(device,exclude=k)) if self.config.shared_at_probes else None)
            if stats.get('complete') is False:
                self.replans+=1
                return
            self.localizations.append(dict(channel=k,**stats))
            self.cleared.add(k)
            self.archived[k]=tr
            del self.pending[k]
            if len(self.cleared)==16: return
            if self.config.shared_at_clears: self.shared(device)

    def report(self):
        return dict(stop_certificate='source_upper_bound' if len(self.cleared)==16 else 'coverage_complete',
                    visited_stations=self.visited, localizations=self.localizations,
                    shared_detections=self.shared_count,scanpoints=len(self.scanpoints),
                    replans_after_pairs=self.replans,global_actions=self.actions,
                    extra_actions=self.extra_actions,shifted_sites=self.shifted_sites,
                    actual_scanpoints=self.scanpoints,unknown_channels=sorted(self.unknown))


def finish(state, device, first=None, deadline=math.inf):
    if first is not None: state.execute(device, first)
    while state.prepare():
        if time.perf_counter() >= deadline: raise TimeoutError('rollout budget')
        state.execute(device,state.ordered_actions(device)[0])
    return state.report()


def solve_resumable(device, analytic=False, planner=None):
    state=State(analytic=analytic)
    while state.prepare():
        actions=state.ordered_actions(device)
        selected=actions[0] if planner is None else planner.choose(state,device,actions)
        state.execute(device,selected)
    report=state.report()
    if planner is not None: report['planning']=planner.stats
    return report
