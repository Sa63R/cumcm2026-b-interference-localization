#!/usr/bin/env python3
"""Q3 v3: bounded, cost-aware sensing and faster online route planning.

Default configuration was frozen after development seeds 0--79, BEFORE the
fresh validation set. All results are from our SELF-BUILT simulator, not the
competition simulator. The official HTTP adapter still requires Annexes 1/2.

Angles are radians, coordinates metres, and virtual action times seconds.
"""
from __future__ import annotations
import math
import numpy as np
import q3_base as b
import q3_optimized as o
from probe_score import rank_probes
from route_search import search_route


class PublicBackend:
    """Expose only public state and allowed operations to the policy."""
    def __init__(self, backend: b.Backend):
        self.__backend = backend
    @property
    def pos(self) -> np.ndarray:
        return self.__backend.pos
    @property
    def channel(self) -> int:
        return self.__backend.channel
    def move(self, p: np.ndarray) -> None:
        self.__backend.move(p)
    def detect(self, c: int) -> b.Observation:
        return self.__backend.detect(c)
    def clear(self, c: int) -> bool:
        return self.__backend.clear(c)


class FastAgent(o.OptimizedAgent):
    """Geometry retains v2's safety certificates; all new scores are heuristics.

    At most four non-centre probes per source, followed by centre contraction.
    No speculative optical attempts, repeated-location noise averaging, moving
    detections, or access to hidden source locations/receiver ranges is used.
    """
    def __init__(self, env: b.Backend, initial_channels: int = 0,
                 expanded_probes: bool = True, route_trials: int = 40,
                 through_clear: bool = True, range_prior: bool = False):
        if not 0 <= initial_channels <= 20:
            raise ValueError('initial_channels must be 0..20')
        if route_trials < 0:
            raise ValueError('route_trials must be nonnegative')
        super().__init__(env, **o.CONFIGS['combined'])
        self.initial_channels = initial_channels
        self.expanded_probes = expanded_probes
        self.route_trials = route_trials
        self.through_clear = through_clear
        self.range_prior = range_prior
        self.startup_done = False
        self.route_successor: np.ndarray | None = None
        # Essential: deleting an origin scan is valid only when future stops
        # alone cover the centre as well as the boundary of the domain.
        if initial_channels < 20 and not o.certified_cover(self.cover):
            raise ValueError('Future scan positions alone do not cover the domain')

    def scan_stop(self, cover_id: int | None = None) -> None:
        if not self.startup_done:
            self.startup_done = True
            if self.initial_channels < 20:
                for c in range(1, self.initial_channels + 1):
                    if c in self.unseen:
                        self.observe(c)
                # A partial scan MUST NOT enter the common full-scan ledger.
                return
        super().scan_stop(cover_id)

    def order(self, start: np.ndarray, points: list[np.ndarray]) -> list[int]:
        if self.route_trials == 0 or len(points) <= 1:
            return super().order(start, points)
        return [int(x) for x in search_route(np.array([start] + list(points)),
                                             self.route_trials)]

    def select_probe(self, c: int) -> tuple[np.ndarray, bool]:
        if not self.expanded_probes:
            return super().select_probe(c)
        t, p = self.tracks[c], self.env.pos
        if self.local_steps[c] >= 4 or t.radius < 22:
            return t.center.copy(), True
        delta = t.center - p
        dist = b.norm(delta)
        if dist < 1e-7:
            return t.center.copy(), True
        u = delta / dist
        v = np.array([-u[1], u[0]])
        candidates = [t.center.copy()]
        for longitudinal in (-.7, -.35, 0., .35, .7):
            for lateral in (-.4, -.15, 0., .15, .4):
                if longitudinal == 0. and lateral == 0.:
                    continue
                candidates.append(t.center + t.radius *
                                  (longitudinal * u + lateral * v))
        candidates.append(p.copy())
        candidates = [q for q in candidates
                      if np.max(np.linalg.norm(t.poly - q, axis=1)) < 999.9
                      and b.coord_key(q) not in t.measured_at]
        if not candidates:
            return t.center.copy(), True
        candidates_array = np.array(candidates)
        scores = rank_probes(
            t.poly, p, candidates_array, 2, .5, 1,
            np.array(self.positive_positions[c]).reshape(-1, 2),
            np.array(self.negative_positions[c]).reshape(-1, 2),
            self.range_prior)
        q = candidates_array[int(np.argmin(scores))]
        return q.copy(), b.norm(q - t.center) < 1e-6

    def go_clear(self, c: int) -> None:
        if not self.through_clear or self.route_successor is None:
            return super().go_clear(c)
        t, p, nxt = self.tracks[c], self.env.pos.copy(), self.route_successor
        if t.radius > b.CLEAR_GUARD:
            raise RuntimeError('No certified clear point available')
        qa = o.nearest_clear_point(p, t.poly, t.center)
        qb = o.nearest_clear_point(nxt, t.poly, t.center)
        def cost(q: np.ndarray) -> float:
            return b.norm(q - p) + b.norm(q - nxt)
        best, best_cost = qa, cost(qa)
        # Both endpoints are feasible; the disk intersection is convex.
        for lam in np.linspace(0, 1, 21):
            q = (1 - lam) * qa + lam * qb
            score = cost(q)
            if score < best_cost:
                best, best_cost = q, score
        # When the future route segment intersects the entire safe region,
        # clearing needs no detour relative to that segment.
        d = nxt - p
        dd = float(d @ d)
        lo, hi = 0., 1.
        if dd > 1e-12:
            for vertex in t.poly:
                w = p - vertex
                bb = 2 * float(d @ w)
                cc = float(w @ w) - (20 - 1e-5)**2
                discriminant = bb * bb - 4 * dd * cc
                if discriminant < 0:
                    lo = 2.
                    break
                root = math.sqrt(discriminant)
                lo = max(lo, (-bb - root) / (2 * dd))
                hi = min(hi, (-bb + root) / (2 * dd))
                if lo > hi:
                    break
            if lo <= hi:
                q = p + d * ((lo + hi) / 2)
                if cost(q) < best_cost:
                    best = q
        if np.linalg.norm(t.poly - best, axis=1).max() > 20 - 1e-7:
            raise RuntimeError('Clear point failed the full-vertex check')
        self.env.move(best)
        self.do_clear(c)

    def run(self) -> None:
        self.scan_stop()
        while self.unseen or self.tracks:
            self.steps += 1
            if self.steps > 600:
                raise RuntimeError('Finite-progress guard exceeded')
            self.clear_at_current_position()
            if not self.unseen and not self.tracks:
                break
            covers = self.remaining_cover if self.unseen else []
            if self.relocate and covers:
                self.relocate_cover_points()
            covers = self.remaining_cover if self.unseen else []
            nodes = [('cover', k, self.cover[k]) for k in covers]
            nodes += [('target', c, t.center) for c, t in sorted(self.tracks.items())]
            if not nodes:
                raise RuntimeError('Unresolved channels without a next task')
            order = self.order(self.env.pos, [n[2] for n in nodes])
            kind, ident, q = nodes[order[0]]
            self.route_successor = nodes[order[1]][2].copy() if len(order) > 1 else None
            if kind == 'cover':
                self.env.move(q)
                self.scan_stop(ident)
            else:
                t = self.tracks[ident]
                if t.radius <= b.CLEAR_GUARD:
                    self.go_clear(ident)
                    self.opportunistic_scan()
                else:
                    q, center = self.select_probe(ident)
                    self.env.move(q)
                    self.local_steps[ident] += 1
                    if not center:
                        self.lateral_steps += 1
                    self.observe(ident, center_step=center)
                    if ident in self.tracks and self.tracks[ident].radius <= b.CLEAR_GUARD:
                        self.go_clear(ident)
                    self.opportunistic_scan(exclude=ident)
        if self.unseen or self.tracks or len(self.cleared | self.absent) != 20:
            raise RuntimeError('Incomplete channel certificate')


CONFIGS = {
    'v3': {},
    'v3_origin20': {'initial_channels': 20},
    'v3_no_probe': {'expanded_probes': False},
    'v3_no_route': {'route_trials': 0},
    'v3_no_through': {'through_clear': False},
    # Optional simulator-specific prior, NOT enabled in the default policy.
    'v3_prior': {'range_prior': True},
}


def make_agent(env: b.Backend, mode: str = 'v3') -> b.Agent:
    backend = PublicBackend(env)
    if mode == 'v1':
        return b.Agent(backend)
    if mode == 'v2':
        return o.OptimizedAgent(backend, **o.CONFIGS['combined'])
    if mode not in CONFIGS:
        raise ValueError(f'Unknown mode {mode!r}; choose v1, v2, or {list(CONFIGS)}')
    return FastAgent(backend, **CONFIGS[mode])


def warmup() -> None:
    """Compile/load optional numerical kernels without touching any simulator."""
    polygon = np.array([[0., 0.], [200., -4.], [200., 4.]])
    rank_probes(polygon, np.zeros(2), np.array([[100., 0.], [100., 20.]]),
                2, .5, 1, np.empty((0, 2)), np.empty((0, 2)), False)
    search_route(np.array([[0., 0.], [100., 0.], [100., 100.], [0., 100.]]), 40)
