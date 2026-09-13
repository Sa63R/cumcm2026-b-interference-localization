"""Bounded experimental schedulers implementing the two proposed Q3 directions.

Only observations enter safety state. Sampled worlds and future scans are
planning hypotheses, never clearance or absence certificates.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from probe_score import njit  # Same optional JIT fallback as the downloaded v3.
import q3_base as b
import q3_optimized as o
import q3_v3 as v
from probe_score import clip_small, approx_center_radius, sample_area
from route_search import search_route


@njit(cache=True)
def posterior(poly, q, theta, half_angle):
    nx, ny = math.sin(theta-half_angle), -math.cos(theta-half_angle)
    pp = clip_small(poly, nx, ny, nx*q[0]+ny*q[1])
    nx, ny = -math.sin(theta+half_angle), math.cos(theta+half_angle)
    return clip_small(pp, nx, ny, nx*q[0]+ny*q[1])


@njit(cache=True)
def initial_region(q, theta, half_angle):
    u = np.array([math.cos(theta), math.sin(theta)])
    side = np.array([-u[1], u[0]]) * 1500*math.tan(half_angle)
    poly = np.empty((3, 2))
    poly[0], poly[1], poly[2] = q, q+1500*u-side, q+1500*u+side
    for k in range(64):
        angle = 2*math.pi*k/64
        poly = clip_small(poly, math.cos(angle), math.sin(angle), 1800.)
    return poly


@njit(cache=True)
def greedy_tail(start, points):
    """Cheap open route surrogate, always visiting every supplied point."""
    n = len(points)
    done = np.zeros(n, dtype=np.bool_)
    p = start.copy()
    total = 0.
    for _ in range(n):
        best, chosen = 1e30, -1
        for j in range(n):
            if not done[j]:
                d = math.sqrt(((p-points[j])**2).sum())
                if d < best:
                    best, chosen = d, j
        done[chosen] = True
        total += best
        p = points[chosen]
    return total


@dataclass
class Action:
    kind: str
    ident: int
    q: np.ndarray
    scan: bool = False
    channels: tuple = ()


class PlanningAgent(v.FastAgent):
    def __init__(self, env, scenario=False, future_cover=False, scenarios=16, candidates=10):
        super().__init__(env)
        self.scenario = scenario
        self.future_cover = future_cover
        self.scenarios = scenarios
        self.candidate_cap = candidates
        # Independent of case seed and hidden simulator state.
        self.rng = np.random.default_rng(274916)
        self.forecast_scans = set()
        self.information_stops = 0
        self.scenario_decisions = 0
        self.plan_changes = 0

    def task_nodes(self):
        covers = list(self.remaining_cover) if self.unseen else []
        targets = [('target', c, t.center.copy()) for c, t in sorted(self.tracks.items())]
        self.forecast_scans = set()
        baseline = [('cover', k, self.cover[k]) for k in covers] + targets
        if not self.future_cover or not covers or not targets:
            return baseline

        def score(scans):
            nodes = [('cover', ident, q) for kind, ident, q in scans if kind == 'cover'] + targets
            points = [n[2] for n in nodes]
            order = self.order(self.env.pos, points)
            path = [self.env.pos] + [points[j] for j in order]
            travel = sum(b.norm(x-y) for x, y in zip(path, path[1:])) / 5
            # Upper estimate of scan cost; discoveries can reduce later scans.
            return travel + len(scans)*len(self.unseen)*6, nodes

        scans0 = [('cover', k, self.cover[k]) for k in covers]
        best_cost, best_nodes = score(scans0)
        best_scans = scans0
        candidate = scans0 + targets
        # Greedy deletion is recertified after every deletion. These are planned
        # points only; the actual remaining_cover and full_scans stay intact.
        for entry in scans0 + list(reversed(targets)):
            trial = [x for x in candidate if (x[0], x[1]) != (entry[0], entry[1])]
            if o.certified_cover(self.full_scans + [x[2] for x in trial]):
                candidate = trial
        cost, nodes = score(candidate)
        if cost < best_cost - 1:
            best_nodes, best_scans = nodes, candidate
            self.plan_changes += 1
        self.forecast_scans = {ident for kind, ident, q in best_scans if kind == 'target'}
        return best_nodes

    def base_action(self, node):
        kind, ident, q = node
        if kind == 'cover':
            return Action('cover', ident, q.copy(), True)
        t = self.tracks[ident]
        if t.radius <= b.CLEAR_GUARD:
            q = o.nearest_clear_point(self.env.pos, t.poly, t.center)
            return Action('clear', ident, q, ident in self.forecast_scans)
        q, _ = self.select_probe(ident)
        return Action('probe', ident, q, ident in self.forecast_scans)

    def channels_for(self, action):
        candidates = []
        for c, t in self.tracks.items():
            if action.kind == 'clear' and c == action.ident:
                continue
            if b.coord_key(action.q) in t.measured_at:
                continue
            if c == action.ident and action.kind == 'probe':
                candidates.append((1e20, c)); continue
            if t.radius <= b.CLEAR_GUARD or self.local_steps[c] >= 4:
                continue
            delta = t.center-action.q
            if b.norm(delta)-t.radius > 1500:
                continue
            pp = posterior(t.poly, action.q, math.atan2(delta[1], delta[0]), b.A)
            _, radius = approx_center_radius(pp)
            gain = t.radius-radius
            if radius <= b.CLEAR_GUARD or gain > max(30., .2*t.radius):
                candidates.append((gain, c))
        return tuple(c for _, c in sorted(candidates, reverse=True)[:4])

    def sample_worlds(self):
        """Approximate posterior: polygon quadrature + fixed-radius likelihood.

        Unknown-channel count uses the 10..16 prior and negative-scan survival;
        no simulator seed, true count, locations, or radii are accessible here.
        """
        worlds = [dict() for _ in range(self.scenarios)]
        for c, t in self.tracks.items():
            points, weights = sample_area(t.poly)
            lower = np.full(len(points), 1000.)
            upper = np.full(len(points), 1500.)
            for p in self.positive_positions[c]:
                lower = np.maximum(lower, np.linalg.norm(points-p, axis=1))
            for p in self.negative_positions[c]:
                upper = np.minimum(upper, np.linalg.norm(points-p, axis=1))
            weights = weights*np.maximum(0., upper-lower)*(np.linalg.norm(points, axis=1) <= 1800.00001)
            if weights.sum() <= 1e-12:
                return None  # Finite quadrature failed; use the certified base action.
            weights /= weights.sum()
            indices = self.rng.choice(len(points), self.scenarios, p=weights)
            for w, j in zip(worlds, indices):
                w[c] = (points[j], self.rng.uniform(lower[j], upper[j]), self.rng.uniform(-1, 1)*math.pi/180)
        found = len(self.cleared) + len(self.tracks)
        if not self.unseen:
            return worlds
        # All still-unknown channels share every completed full scan. A finite
        # sample approximates this region for planning, never for absence.
        angle = self.rng.uniform(0, 2*math.pi, 512)
        radius = 1800*np.sqrt(self.rng.random(512))
        points = radius[:, None]*np.column_stack([np.cos(angle), np.sin(angle)])
        ranges = self.rng.uniform(1000, 1500, 512)
        live = np.ones(512, dtype=bool)
        for p in self.full_scans:
            live &= np.linalg.norm(points-p, axis=1) > ranges
        support = np.flatnonzero(live)
        if not len(support):
            return None
        survival = len(support)/512
        totals = list(range(max(10, found), 17))
        unseen = sorted(self.unseen)
        weights = np.array([math.comb(len(unseen), n-found)/math.comb(20, n)*survival**(n-found) for n in totals])
        weights /= weights.sum()
        for w, n in zip(worlds, self.rng.choice(totals, self.scenarios, p=weights)):
            channels = self.rng.choice(unseen, int(n)-found, replace=False)
            for c in channels:
                j = int(self.rng.choice(support))
                w[int(c)] = (points[j], ranges[j], self.rng.uniform(-1, 1)*math.pi/180)
        return worlds

    def evaluate_action(self, action, worlds, nodes):
        channels = set(action.channels)
        if action.scan:
            channels |= self.unseen
        current_channel = self.env.channel
        ordered = sorted(channels, key=lambda c: (c != current_channel, c))
        immediate = b.norm(action.q-self.env.pos)/5
        for c in ordered:
            immediate += 5 + (c != current_channel)
            current_channel = c
        if action.kind == 'clear':
            immediate += 5
        covers = [n[2] for n in nodes if n[0] == 'cover' and not (action.kind == 'cover' and n[1] == action.ident)]
        future = [t.center for c, t in self.tracks.items() if c in self.forecast_scans and c != action.ident]
        scan_points = covers + future
        # Actual full-scan evidence may eliminate more fallback search points.
        if action.scan:
            keep = self.prune_for(self.full_scans+[action.q], self.remaining_cover)
            if not self.future_cover:
                covers = [self.cover[k] for k in keep]
                scan_points = covers
        scores = []
        for world in worlds:
            tail_points = []
            tail_service = 0.
            remaining_unknown = len(self.unseen)
            cleared_now = int(action.kind == 'clear')
            active_count = 0
            for c in sorted(set(self.tracks) | (set(world) & self.unseen)):
                if action.kind == 'clear' and c == action.ident:
                    continue
                g, reception, error = world[c]
                known = c in self.tracks
                measured = c in channels
                d = b.norm(g-action.q)
                if measured and d <= reception:
                    if not known:
                        remaining_unknown -= 1
                    if d <= 5:
                        tail_service += 5
                        cleared_now += 1
                        continue
                    theta = math.atan2(g[1]-action.q[1], g[0]-action.q[0]) + error
                    theta = round(math.degrees(theta) % 360, 2)*math.pi/180
                    poly = posterior(self.tracks[c].poly, action.q, theta, b.A) if known else initial_region(action.q, theta, b.A)
                    center, uncertainty = approx_center_radius(poly)
                elif known:
                    center, uncertainty = self.tracks[c].center, self.tracks[c].radius
                else:
                    # Approximate extra service after a future discovery. The
                    # sampled source remains undiscovered in the policy state.
                    stops = scan_points if scan_points else [action.q]
                    dist = min(b.norm(g-p) for p in stops)
                    tail_service += dist/5 + 6*math.log2(750/19.5) + 5
                    continue
                active_count += 1
                tail_points.append(center)
                residual = max(0., math.log2(max(uncertainty, 19.5)/19.5))
                tail_service += 5 + 6*residual
                if uncertainty > b.CLEAR_GUARD:
                    tail_service += b.norm(g-center)/5 + .25*uncertainty/5
            discovered = len(self.cleared) + cleared_now + active_count
            if discovered >= 16:
                remaining_unknown = 0
            if remaining_unknown:
                tail_points.extend(covers)
                tail_service += 6*remaining_unknown*len(scan_points)
            route = greedy_tail(action.q, np.asarray(tail_points).reshape(-1, 2))/5
            total_sources = len(self.cleared) + len(world)
            scores.append((immediate+tail_service+route)/max(1, total_sources))
        return float(np.mean(scores))

    def choose_action(self, nodes, order):
        base = self.base_action(nodes[order[0]])
        if not self.scenario or self.information_stops >= 8 or not self.tracks:
            return base
        worlds = self.sample_worlds()
        if worlds is None:
            return base
        candidates = []
        def add(a):
            if len(candidates) >= self.candidate_cap:
                return
            if a.kind == 'probe' and self.local_steps[a.ident] >= 4:
                return
            a.channels = self.channels_for(a)
            if a.kind == 'probe' and a.ident not in a.channels:
                return
            key = (a.kind, a.ident, b.coord_key(a.q), a.scan, a.channels)
            if not any((x.kind, x.ident, b.coord_key(x.q), x.scan, x.channels) == key for x in candidates):
                candidates.append(a)
        # Retain the original first action even after a target's heuristic cap.
        base.channels = self.channels_for(base)
        candidates.append(base)
        for j in order[:5]:
            add(self.base_action(nodes[j]))
        for j in order[:3]:
            kind, c, q = nodes[j]
            if kind == 'target' and self.tracks[c].radius > b.CLEAR_GUARD:
                add(Action('probe', c, q.copy(), c in self.forecast_scans))
        active = [c for c in self.tracks if self.tracks[c].radius > b.CLEAR_GUARD and self.local_steps[c] < 4]
        active.sort(key=lambda c: b.norm(self.tracks[c].center-self.env.pos))
        if len(active) >= 2:
            c, d = active[:2]
            q = .5*(self.tracks[c].center+self.tracks[d].center)
            if np.linalg.norm(self.tracks[c].poly-q, axis=1).max() < 999.9:
                add(Action('probe', c, q))
        # Also compare the baseline stop with only the primary channel. This
        # directly tests whether paying for the other channels is worthwhile.
        if base.kind == 'probe' and len(base.channels) > 1 and len(candidates) < self.candidate_cap:
            candidates.append(Action(base.kind, base.ident, base.q, base.scan, (base.ident,)))
        scores = [self.evaluate_action(a, worlds, nodes) for a in candidates]
        chosen = candidates[int(np.argmin(scores))]
        self.scenario_decisions += 1
        if chosen is not base:
            self.information_stops += 1
        return chosen

    def run(self):
        self.scan_stop()
        while self.unseen or self.tracks:
            self.steps += 1
            if self.steps > 600:
                raise RuntimeError('Finite progress guard')
            self.clear_at_current_position()
            if not self.unseen and not self.tracks:
                break
            if self.relocate and self.unseen:
                self.relocate_cover_points()
            nodes = self.task_nodes()
            if not nodes:
                raise RuntimeError('Missing fallback tasks')
            order = self.order(self.env.pos, [n[2] for n in nodes])
            action = self.choose_action(nodes, order)
            self.route_successor = nodes[order[1]][2].copy() if len(order) > 1 else None
            if action.kind == 'clear':
                if not self.scenario:
                    self.go_clear(action.ident)
                else:
                    self.env.move(action.q)
                    self.do_clear(action.ident)
            elif action.kind == 'cover':
                self.env.move(action.q)
                self.scan_stop(action.ident)
            else:
                self.env.move(action.q)
                c = action.ident
                center = b.norm(action.q-self.tracks[c].center) < 1e-6
                self.local_steps[c] += 1
                self.observe(c, center_step=center)
            # These measurements were evaluated at action.q and are executed
            # before moving to clear a newly localized source.
            for c in sorted(action.channels, key=lambda c: (c != self.env.channel, c)):
                if c != action.ident and c in self.tracks and b.coord_key(self.env.pos) not in self.tracks[c].measured_at:
                    self.local_steps[c] += 1
                    self.observe(c)
            if action.scan and action.kind != 'cover' and self.unseen:
                self.extra_full_scans += 1
                self.scan_stop()
            if action.kind == 'probe' and action.ident in self.tracks and self.tracks[action.ident].radius <= b.CLEAR_GUARD:
                self.go_clear(action.ident)
            if action.kind != 'cover':
                self.opportunistic_scan(exclude=action.ident if action.kind == 'probe' else None)
        if self.unseen or self.tracks or len(self.cleared | self.absent) != 20:
            raise RuntimeError('Incomplete terminal certificate')


def warmup():
    q = np.zeros(2)
    p = initial_region(q, .1, 1.005*math.pi/180)
    posterior(p, np.array([100., 20.]), .1, 1.005*math.pi/180)
    greedy_tail(q, np.array([[1., 2.], [3., 4.]]))
