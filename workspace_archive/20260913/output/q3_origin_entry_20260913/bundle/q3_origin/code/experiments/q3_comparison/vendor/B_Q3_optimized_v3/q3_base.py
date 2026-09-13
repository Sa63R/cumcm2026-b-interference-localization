#!/usr/bin/env python3
"""Problem B, Q3: covering search + set-membership localization + online routing.

Run: python q3_solver.py --cases 100 --out results
Requires: Python >=3.10, numpy. No official simulator protocol is assumed.
`Backend` is the only interface seen by the agent. `ToySimulator` is a separately
labelled, self-built test environment, NOT the competition simulator.
Distances: metres; angles: radians internally; times: seconds.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Protocol
import numpy as np

A = math.pi / 180.0
EPS = 1e-8
CLEAR_GUARD = 19.5
CHANNELS = set(range(1, 21))
COVER = [999.0 * np.array([math.cos(2*math.pi*k/7),
                           math.sin(2*math.pi*k/7)]) for k in range(7)]


def norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def clip(poly: np.ndarray, normal: np.ndarray, bound: float) -> np.ndarray:
    """Convex polygon intersected with n.x <= b; outward numerical slack."""
    if len(poly) == 0:
        return poly
    bound += EPS
    out = []
    x0 = poly[-1]
    f0 = float(normal @ x0 - bound)
    for x1 in poly:
        f1 = float(normal @ x1 - bound)
        if (f0 <= 0) != (f1 <= 0):
            out.append(x0 + (x1 - x0) * (f0 / (f0 - f1)))
        if f1 <= 0:
            out.append(x1.copy())
        x0, f0 = x1, f1
    if not out:
        return np.empty((0, 2))
    ans = [out[0]]
    for x in out[1:]:
        if norm(x - ans[-1]) > 1e-7:
            ans.append(x)
    if len(ans) > 1 and norm(ans[0] - ans[-1]) < 1e-7:
        ans.pop()
    return np.array(ans)


def wedge(poly: np.ndarray, p: np.ndarray, theta: float) -> np.ndarray:
    """Measured bearing +/- 1 degree; two half planes, correct across 0/360."""
    lo, hi = theta - A, theta + A
    n1 = np.array([math.sin(lo), -math.cos(lo)])
    n2 = np.array([-math.sin(hi), math.cos(hi)])
    return clip(clip(poly, n1, float(n1 @ p)), n2, float(n2 @ p))


def first_polygon(p: np.ndarray, theta: float) -> np.ndarray:
    """A conservative triangle: 0 <= projection along bearing <= 1500.
    It contains the exact radius-1500 sector; no inward polygon approximation.
    Intersect with an OUTER 64-gon containing the deployment disk.
    """
    u = np.array([math.cos(theta), math.sin(theta)])
    v = np.array([-u[1], u[0]])
    poly = np.array([p, p + 1500*u - 1500*math.tan(A)*v,
                       p + 1500*u + 1500*math.tan(A)*v])
    for k in range(64):
        ang = 2*math.pi*k/64
        n = np.array([math.cos(ang), math.sin(ang)])
        poly = clip(poly, n, 1800.0)
    return poly


def mec(poly: np.ndarray) -> tuple[np.ndarray, float]:
    """Minimum enclosing circle of a small convex polygon.
    Enumerate one-, two-, and three-vertex supporting circles.  Every returned
    radius is rechecked against ALL vertices and rounded outward. For these
    narrow bearing polygons the vertex count is small; no random solver needed.
    """
    n = len(poly)
    if n == 0:
        raise RuntimeError('Empty feasible set: verify protocol, units and error bound.')
    if n == 1:
        return poly[0].copy(), 1e-7
    best_c = np.mean(poly, axis=0)
    best_r2 = float(np.max(np.sum((poly - best_c)**2, axis=1)))

    def consider(c: np.ndarray, support_r2: float) -> None:
        nonlocal best_c, best_r2
        if support_r2 > best_r2 + 1e-7:
            return
        r2 = float(np.max(np.sum((poly-c)**2, axis=1)))
        if r2 <= support_r2 + 1e-5 and r2 < best_r2:
            best_c, best_r2 = c.copy(), r2

    for i, j in itertools.combinations(range(n), 2):
        c = (poly[i] + poly[j]) / 2
        consider(c, float(np.sum((poly[i]-c)**2)))
    for i, j, k in itertools.combinations(range(n), 3):
        p, q, r = poly[i], poly[j], poly[k]
        u, v = q-p, r-p
        det = float(u[0]*v[1] - u[1]*v[0])
        if abs(det) <= 1e-12 * max(1.0, norm(u)*norm(v)):
            continue
        u2, v2 = float(u@u), float(v@v)
        off = np.array([u2*v[1]-v2*u[1], u[0]*v2-v[0]*u2])/(2*det)
        consider(p+off, float(off@off))
    # Recalculation protects the containment certificate from roundoff.
    radius = float(np.max(np.linalg.norm(poly-best_c, axis=1))) + 1e-7
    return best_c, radius


@dataclass
class Observation:
    kind: str                      # 'none', 'bearing', 'strong'
    bearing: float | None = None  # radians, only for 'bearing'


class Backend(Protocol):
    """Implement this adapter after reading the official annexes 1 and 2.
    Positions/channel changes must reflect the official responses. The example
    interface charges a radio retune before a channel-targeted clear as well.
    """
    pos: np.ndarray
    channel: int
    def move(self, target: np.ndarray) -> None: ...
    def detect(self, channel: int) -> Observation: ...
    def clear(self, channel: int) -> bool: ...


@dataclass
class Track:
    poly: np.ndarray
    center: np.ndarray = field(init=False)
    radius: float = field(init=False)
    positives: int = 1
    center_updates: int = 0
    measured_at: set[tuple[float, float]] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.center, self.radius = mec(self.poly)

    def update(self, p: np.ndarray, theta: float) -> None:
        new_poly = wedge(self.poly, p, theta)
        if not len(new_poly):
            raise RuntimeError('Inconsistent bearings; do not silently reset the target.')
        self.poly = new_poly
        self.center, self.radius = mec(self.poly)
        self.positives += 1


def coord_key(p: np.ndarray) -> tuple[float, float]:
    return tuple(float(round(x, 6)) for x in p)


def route_order(start: np.ndarray, points: list[np.ndarray]) -> list[int]:
    """Nearest-neighbour seed + open-path 2-opt; no return to the origin."""
    if not points:
        return []
    coords = np.array([start] + points)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    remaining, route, last = set(range(1, len(points)+1)), [], 0
    while remaining:
        nxt = min(remaining, key=lambda j: (d[last, j], j))
        route.append(nxt)
        remaining.remove(nxt)
        last = nxt
    for _ in range(30):
        best_delta, pair = -1e-7, None
        for i in range(len(route)-1):
            prev = 0 if i == 0 else route[i-1]
            for j in range(i+1, len(route)):
                a, b = route[i], route[j]
                delta = d[prev, b]-d[prev, a]
                if j+1 < len(route):
                    nxt = route[j+1]
                    delta += d[a, nxt]-d[b, nxt]
                if delta < best_delta:
                    best_delta, pair = delta, (i, j)
        if pair is None:
            break
        i, j = pair
        route[i:j+1] = reversed(route[i:j+1])
    return [j-1 for j in route]


class Agent:
    def __init__(self, env: Backend, mode: str = 'joint', scan_origin: bool = True):
        if mode not in {'joint', 'phased'}:
            raise ValueError('mode must be joint or phased')
        self.env, self.mode = env, mode
        self.scan_origin = scan_origin
        self.unseen = set(CHANNELS)
        self.absent: set[int] = set()
        self.cleared: set[int] = set()
        self.tracks: dict[int, Track] = {}
        self.remaining_cover = list(range(7))
        self.negative_cover: dict[int, set[int]] = {c: set() for c in CHANNELS}
        self.maximum_center_updates = 0
        self.steps = 0

    def count_certificate(self) -> None:
        if len(self.cleared) + len(self.tracks) == 16:
            self.absent |= self.unseen
            self.unseen.clear()

    def do_clear(self, c: int) -> None:
        if not self.env.clear(c):
            raise RuntimeError('Certified optical clear failed; stop and inspect the adapter.')
        if c in self.tracks:
            self.maximum_center_updates = max(self.maximum_center_updates,
                                               self.tracks[c].center_updates)
        self.tracks.pop(c, None)
        self.unseen.discard(c)
        self.cleared.add(c)
        self.count_certificate()

    def observe(self, c: int, cover_id: int | None = None,
                center_step: bool = False) -> None:
        p = self.env.pos.copy()
        ans = self.env.detect(c)
        if ans.kind == 'none':
            if center_step:
                raise RuntimeError('No signal at certified centre (<1000 m from target).')
            if c in self.unseen and cover_id is not None:
                self.negative_cover[c].add(cover_id)
            return
        if ans.kind == 'strong':
            self.do_clear(c)
            return
        if ans.kind != 'bearing' or ans.bearing is None:
            raise RuntimeError(f'Unexpected observation: {ans!r}')
        if c in self.unseen:
            self.unseen.remove(c)
            self.tracks[c] = Track(first_polygon(p, ans.bearing))
        elif c in self.tracks:
            old_r = self.tracks[c].radius
            self.tracks[c].update(p, ans.bearing)
            if center_step:
                t = self.tracks[c]
                t.center_updates += 1
                if t.radius > old_r/(2*math.cos(A)) + 1e-3:
                    raise RuntimeError('Centre-contraction numerical check failed.')
        else:
            raise RuntimeError('Bearing returned on an absent/cleared channel.')
        self.tracks[c].measured_at.add(coord_key(p))
        self.count_certificate()

    def useful_bearing(self, c: int) -> bool:
        t, p = self.tracks[c], self.env.pos
        if t.radius <= CLEAR_GUARD or coord_key(p) in t.measured_at:
            return False
        if norm(t.center-p)-t.radius > 1500:
            return False
        # A heuristic ONLY for saving queries; containment/proofs do not use it.
        delta = t.center-p
        if norm(delta) < 1e-6:
            return True
        pred = wedge(t.poly, p, math.atan2(delta[1], delta[0]))
        if len(pred) == 0:
            return False
        _, r = mec(pred)
        return r < 0.80*t.radius or r <= CLEAR_GUARD

    def scan_stop(self, cover_id: int | None = None) -> None:
        old_active = set(self.tracks)
        # A batch can start with the already tuned channel, saving one retune.
        order = sorted(self.unseen, key=lambda c: (c != self.env.channel, c))
        for c in order:
            if c in self.unseen:
                self.observe(c, cover_id)
        if self.mode == 'phased':
            active = sorted(old_active)
        else:
            active = sorted(c for c in old_active if c in self.tracks
                            and self.useful_bearing(c))
        active.sort(key=lambda c: (c != self.env.channel, c))
        for c in active:
            if c in self.tracks:
                self.observe(c)
        if cover_id is not None:
            self.remaining_cover.remove(cover_id)
        for c in sorted(self.unseen.copy()):
            if len(self.negative_cover[c]) == 7:
                self.unseen.remove(c)
                self.absent.add(c)
        self.count_certificate()

    def clear_at_current_position(self) -> None:
        for c in sorted(list(self.tracks)):
            if c in self.tracks:
                farthest = float(np.max(np.linalg.norm(self.tracks[c].poly-self.env.pos,
                                                       axis=1)))
                if farthest <= 20-1e-6:
                    self.do_clear(c)

    def go_clear(self, c: int) -> None:
        t = self.tracks[c]
        if t.radius > CLEAR_GUARD:
            raise RuntimeError('Target is not yet certified for a single safe clear point.')
        # Inscribed safe-clear disk B(center, 20-radius). Approach its near edge.
        slack = 20-t.radius-1e-5
        delta = self.env.pos-t.center
        dist = norm(delta)
        q = self.env.pos.copy() if dist <= slack else t.center + delta*(slack/dist)
        self.env.move(q)
        self.do_clear(c)

    def run(self) -> None:
        if self.scan_origin:
            self.scan_stop()
        while self.unseen or self.tracks:
            self.steps += 1
            if self.steps > 500:
                raise RuntimeError('Progress bound exceeded.')
            self.clear_at_current_position()
            if not self.unseen and not self.tracks:
                break
            covers = self.remaining_cover if self.unseen else []
            if self.mode == 'phased' and covers:
                k = covers[0]
                self.env.move(COVER[k])
                self.scan_stop(k)
                continue
            nodes: list[tuple[str, int, np.ndarray]] = [('cover', k, COVER[k]) for k in covers]
            nodes += [('target', c, t.center) for c, t in sorted(self.tracks.items())]
            if not nodes:
                raise RuntimeError('Unresolved channels but no remaining coverage point.')
            idx = route_order(self.env.pos, [n[2] for n in nodes])[0]
            kind, ident, q = nodes[idx]
            if kind == 'cover':
                self.env.move(q)
                self.scan_stop(ident)
            else:
                t = self.tracks[ident]
                if t.radius <= CLEAR_GUARD:
                    self.go_clear(ident)
                else:
                    self.env.move(t.center)
                    self.observe(ident, center_step=True)
                    if ident in self.tracks and self.tracks[ident].radius <= CLEAR_GUARD:
                        self.go_clear(ident)
        if self.unseen or self.tracks or len(self.cleared | self.absent) != 20:
            raise RuntimeError('Termination has no complete channel certificate.')


@dataclass
class Source:
    channel: int
    xy: np.ndarray
    radius: float


class ToySimulator:
    """Self-built test only. Hidden states are never read by Agent.
    Spatially fixed errors: same position and channel -> same bounded error.
    Optical clear targets one channel. Each failed optical attempt costs 3 s;
    successful attempts cost 3+2 s. The solver must never need a failed attempt.
    """
    def __init__(self, sources: list[Source], seed: int, noise: str = 'hash',
                 record: bool = False):
        self._sources = {s.channel: s for s in sources}
        self._live = set(self._sources)
        self.seed, self.noise, self.record = seed, noise, record
        self.pos = np.array([0.0, 0.0])
        self.channel = 1
        self.distance = 0.0
        self.detects = self.switches = self.optical = self.clears = self.failed = 0
        self.log: list[dict] = []

    @property
    def virtual_time(self) -> float:
        return self.distance/5 + self.detects*5 + self.switches + self.optical*3 + self.clears*2

    def _log(self, action: str, **data) -> None:
        if self.record:
            self.log.append(dict(action=action, xy=self.pos.tolist(),
                                 virtual_seconds=self.virtual_time, **data))

    def _tune(self, c: int) -> None:
        if c not in CHANNELS:
            raise ValueError('Invalid channel')
        if c != self.channel:
            self.switches += 1
            self.channel = c
            self._log('switch', channel=c)

    def move(self, target: np.ndarray) -> None:
        target = np.array(target, dtype=float)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError('Invalid movement coordinates')
        self.distance += norm(target-self.pos)
        self.pos = target.copy()
        self._log('move')

    def _error(self, c: int) -> float:
        if self.noise == 'plus':
            return A
        if self.noise == 'minus':
            return -A
        if self.noise == 'alternating':
            return A if (c % 2) else -A
        if self.noise == 'smooth':
            x, y = self.pos
            return A*(0.6*math.sin(x/180+y/250+c+self.seed)
                      + 0.4*math.cos(x/95-y/130+2*c-self.seed))
        if self.noise != 'hash':
            raise ValueError('Unknown noise mode')
        key = f'{self.seed}:{c}:{self.pos[0]:.6f}:{self.pos[1]:.6f}'.encode()
        integer = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big')
        return A*(2*integer/(2**64-1)-1)

    def detect(self, channel: int) -> Observation:
        self._tune(channel)
        self.detects += 1
        if channel not in self._live:
            ans = Observation('none')
        else:
            s = self._sources[channel]
            delta = s.xy-self.pos
            d = norm(delta)
            if d > s.radius:
                ans = Observation('none')
            elif d <= 5:
                ans = Observation('strong')
            else:
                ans = Observation('bearing', (math.atan2(delta[1], delta[0])
                                             + self._error(channel)) % (2*math.pi))
        self._log('detect', channel=channel, result=ans.kind, bearing_radians=ans.bearing)
        return ans

    def clear(self, channel: int) -> bool:
        self._tune(channel)
        self.optical += 1
        ok = channel in self._live and norm(self._sources[channel].xy-self.pos) <= 20
        if ok:
            self._live.remove(channel)
            self.clears += 1
        else:
            self.failed += 1
        self._log('clear', channel=channel, success=ok)
        return ok


def make_case(seed: int, stress: bool = False) -> list[Source]:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 17))
    channels = rng.choice(np.arange(1, 21), n, replace=False)
    angles = rng.uniform(0, 2*math.pi, n)
    if stress:
        radius = np.full(n, 1800.0)
        ranges = np.full(n, 1000.0)
        # Include angular gaps between adjacent coverage points.
        for k in range(min(7, n)):
            angles[k] = (2*k+1)*math.pi/7
    else:
        radius = 1800*np.sqrt(rng.random(n))  # uniform in area
        ranges = rng.uniform(1000, 1500, n)
    return [Source(int(c), np.array([r*math.cos(t), r*math.sin(t)]), float(d))
            for c, r, t, d in zip(channels, radius, angles, ranges)]


def evaluate(sources: list[Source], seed: int, mode: str, noise: str = 'hash',
             record: bool = False, scan_origin: bool = True) -> tuple[dict, ToySimulator]:
    env = ToySimulator(sources, seed, noise, record)
    agent = Agent(env, mode, scan_origin)
    t0 = perf_counter()
    agent.run()
    runtime = perf_counter()-t0
    row = dict(seed=seed, mode=mode, noise=noise, true_sources=len(sources),
               cleared=env.clears, clearance_ratio=env.clears/len(sources),
               virtual_seconds=env.virtual_time, average_seconds=env.virtual_time/env.clears,
               movement_metres=env.distance, detects=env.detects, switches=env.switches,
               optical_attempts=env.optical, failed_clears=env.failed,
               local_runtime_seconds=runtime,
               max_center_updates=agent.maximum_center_updates,
               coverage_points_visited=7-len(agent.remaining_cover))
    if env._live:
        raise AssertionError(f'Uncleared sources in self-test: {env._live}')
    return row, env


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=100)
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--stress-cases', type=int, default=10)
    args = parser.parse_args()
    if args.cases < 1 or args.stress_cases < 0:
        parser.error('cases must be positive; stress-cases nonnegative')
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(args.cases):
        sources = make_case(seed)
        for mode in ['phased', 'joint']:
            row, env = evaluate(sources, seed, mode, record=(seed < 3 and mode == 'joint'))
            rows.append(row)
            if seed < 3 and mode == 'joint':
                name = f'SELF_BUILT_seed_{seed}_NOT_OFFICIAL.json'
                (args.out/name).write_text(json.dumps(dict(metadata=row, actions=env.log),
                                                       ensure_ascii=False, indent=2), encoding='utf-8')
        if (seed+1) % 10 == 0:
            print(f'Completed paired cases: {seed+1}', flush=True)
    save_csv(args.out/'random_cases.csv', rows)
    stress = []
    for noise in ['hash', 'smooth', 'plus', 'minus', 'alternating']:
        for seed in range(args.stress_cases):
            row, _ = evaluate(make_case(10000+seed, stress=True), 10000+seed, 'joint', noise)
            row['case_type'] = 'boundary_1000m_range'
            stress.append(row)
    # Strong-signal and co-located-source edge case, including the origin.
    sources = [Source(c, np.array([0.0, 0.0]) if c <= 5 else np.array([999.0, 0.0]),
                      1000.0) for c in range(1, 11)]
    row, _ = evaluate(sources, 20000, 'joint', 'plus')
    row['case_type'] = 'strong_signal_and_colocation'
    stress.append(row)
    save_csv(args.out/'stress_cases.csv', stress)
    summary = {'warning': 'SELF-BUILT SIMULATION ONLY. Not official competition test results.'}
    for mode in ['phased', 'joint']:
        subset = [r for r in rows if r['mode'] == mode]
        summary[mode] = dict(cases=len(subset), all_cleared=all(r['clearance_ratio']==1 for r in subset),
                            mean_virtual_seconds=float(np.mean([r['virtual_seconds'] for r in subset])),
                            mean_case_average_seconds=float(np.mean([r['average_seconds'] for r in subset])),
                            pooled_seconds_per_source=sum(r['virtual_seconds'] for r in subset)/sum(r['cleared'] for r in subset),
                            mean_movement_metres=float(np.mean([r['movement_metres'] for r in subset])),
                            mean_detects=float(np.mean([r['detects'] for r in subset])),
                            mean_local_runtime_seconds=float(np.mean([r['local_runtime_seconds'] for r in subset])),
                            max_local_runtime_seconds=max(r['local_runtime_seconds'] for r in subset))
    summary['stress'] = dict(cases=len(stress), all_cleared=all(r['clearance_ratio']==1 for r in stress),
                             failed_clears=sum(r['failed_clears'] for r in stress))
    (args.out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
