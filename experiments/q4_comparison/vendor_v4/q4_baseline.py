#!/usr/bin/env python3
"""Q4: certified coverage and paired-probe localization.

Pure Python 3.10+; no third-party dependencies.
This is an ORIGINAL LOCAL SIMULATOR, not the competition simulator.
Implement the Device Protocol to connect the solver to an official interface.
No guessed HTTP endpoint or official response schema is used here.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

Point = tuple[float, float]
DELTA = math.pi / 180.0
COS_D = math.cos(DELTA)
TAN_D = math.tan(DELTA)
KAPPA = 0.568  # strictly above the geometric bound 0.5669025561
CLEAR_CERT_RADIUS = 19.5  # numerical safety margin below the physical 20 m
EPS = 1e-8


def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def mul(s: float, a: Point) -> Point:
    return (s * a[0], s * a[1])


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def norm(a: Point) -> float:
    return math.hypot(*a)


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(a: float) -> Point:
    return (math.cos(a), math.sin(a))


def clip_halfplane(poly: list[Point], n: Point, b: float) -> list[Point]:
    """Keep n dot x <= b. Expand by EPS to avoid excluding boundary truth."""
    if not poly:
        return []
    b += EPS
    out: list[Point] = []
    a = poly[-1]
    fa = dot(n, a) - b
    for z in poly:
        fz = dot(n, z) - b
        ia, iz = fa <= 0.0, fz <= 0.0
        if ia != iz:
            w = fa / (fa - fz)
            out.append(add(a, mul(w, sub(z, a))))
        if iz:
            out.append(z)
        a, fa = z, fz
    # Consecutive duplicate vertices are not needed by the circle routine.
    cleaned: list[Point] = []
    for p in out:
        if not cleaned or dist(p, cleaned[-1]) > 1e-8:
            cleaned.append(p)
    if len(cleaned) > 1 and dist(cleaned[0], cleaned[-1]) < 1e-8:
        cleaned.pop()
    return cleaned


def clip_bearing(poly: list[Point], s: Point, theta: float,
                 upper: float) -> list[Point]:
    """Outer triangle: 0 <= axial <= upper, |lateral| <= axial*tan(delta).

    A target inside the radius-upper bearing sector is inside this triangle.
    The triangle is a conservative OUTER approximation, never inscribed.
    """
    e = unit(theta)
    v = (-e[1], e[0])
    for n, offset in ((sub(v, mul(TAN_D, e)), 0.0),
                      (sub(mul(-1.0, v), mul(TAN_D, e)), 0.0),
                      (e, upper)):
        poly = clip_halfplane(poly, n, dot(n, s) + offset)
    return poly


def triangle_from_bearing(s: Point, theta: float, upper: float) -> list[Point]:
    e = unit(theta)
    v = (-e[1], e[0])
    front = add(s, mul(upper, e))
    return [s, add(front, mul(upper * TAN_D, v)),
            add(front, mul(-upper * TAN_D, v))]


def initial_polygon(s: Point, theta: float, upper: float) -> list[Point]:
    poly = triangle_from_bearing(s, theta, upper)
    # Circumscribed 64-gon around the target disk (never an inscribed polygon).
    for k in range(64):
        n = unit(2.0 * math.pi * k / 64)
        poly = clip_halfplane(poly, n, 1800.0)
    if not poly:
        raise RuntimeError('Inconsistent initial observation / numerical failure')
    return poly


def circumcircle(a: Point, b: Point, c: Point) -> tuple[Point, float] | None:
    u, v = sub(b, a), sub(c, a)
    det = 2.0 * (u[0] * v[1] - u[1] * v[0])
    if abs(det) < 1e-10:
        return None
    uu, vv = dot(u, u), dot(v, v)
    center = add(a, ((uu * v[1] - vv * u[1]) / det,
                     (u[0] * vv - v[0] * uu) / det))
    return center, dist(center, a)


def enclosing_circle(poly: list[Point]) -> tuple[Point, float]:
    """Enumerate support sets of <=3 vertices; exact geometric candidates.

    These polygons are normally very small. A final radius recomputation is
    used for certification even when floating-point comparisons tie.
    """
    if not poly:
        raise RuntimeError('Empty feasible polygon; cannot certify a clear')
    best_center = poly[0]
    best_radius = max(dist(best_center, p) for p in poly)

    def consider(c: Point, r: float) -> None:
        nonlocal best_center, best_radius
        if r > best_radius + EPS:
            return
        actual_r = max(dist(c, p) for p in poly)
        if actual_r <= r + 1e-6 and actual_r < best_radius:
            best_center, best_radius = c, actual_r

    for a, b in itertools.combinations(poly, 2):
        c = mul(0.5, add(a, b))
        consider(c, dist(c, a))
    for a, b, c in itertools.combinations(poly, 3):
        circle = circumcircle(a, b, c)
        if circle is not None:
            consider(*circle)
    return best_center, best_radius + 1e-7


def coverage_sites() -> tuple[list[Point], list[tuple[int, int, int]]]:
    """25 sites, 36 triangles, max edge <1000 m; hull contains B(0,1800)."""
    h, outer_mid = 980.0, 1840.0
    sites: list[Point] = [(0.0, 0.0)]
    # A inner ring; B old side midpoints; C outer corners; D expanded midpoints.
    for radius, offset in ((h, 0.0), (math.sqrt(3) * h, math.pi / 6),
                           (2 * h, 0.0), (outer_mid, math.pi / 6)):
        sites.extend(mul(radius, unit(offset + k * math.pi / 3)) for k in range(6))
    tris: list[tuple[int, int, int]] = []
    for k in range(6):
        j = (k + 1) % 6
        a, an, b, c, cn, d = 1+k, 1+j, 7+k, 13+k, 13+j, 19+k
        tris.extend([(0, a, an), (a, c, b), (a, b, an),
                     (an, b, cn), (c, b, d), (b, cn, d)])
    return sites, tris


def coverage_certificate() -> dict:
    sites, tris = coverage_sites()
    max_edge = max(dist(sites[a], sites[b]) for t in tris
                   for a, b in itertools.combinations(t, 2))
    a, b, angle = 1960.0, 1840.0, math.pi / 6
    hull_edge = math.sqrt(a*a + b*b - 2*a*b*math.cos(angle))
    inradius = a*b*math.sin(angle)/hull_edge
    assert max_edge < 1000.0 and inradius > 1800.0
    return {'sites': sites, 'triangles': tris, 'num_sites': len(sites),
            'num_triangles': len(tris), 'max_triangle_edge_m': max_edge,
            'hull_inradius_m': inradius, 'max_station_radius_m': 1960.0,
            'local_contraction_constant': KAPPA,
            'tight_geometric_constant': math.sqrt(21/16 - COS_D
                                                   + math.sin(DELTA)/2)}


@dataclass(frozen=True)
class Observation:
    status: Literal['none', 'bearing', 'strong']
    theta: float | None = None  # radians, counterclockwise from east


class Device(Protocol):
    """Abstract semantics only. The official HTTP protocol was NOT supplied.

    An official adapter must parse the actual documented responses, track
    position/current channel, and do required channel switches in detect().
    clear() means the 3 s optical localization and, on success, 2 s laser step.
    """
    position: Point
    channel: int
    def move(self, p: Point) -> None: ...
    def detect(self, channel: int) -> Observation: ...
    def clear(self, channel: int) -> bool: ...


@dataclass
class Target:
    channel: int
    position: Point
    radius: float
    direction: Point | None
    cleared: bool = False


class LocalSimulator:
    """Synthetic environment; its private target data is NOT given to solve()."""
    def __init__(self, targets: list[Target], seed: int, error_mode: str = 'hash',
                 trace: bool = False):
        self._targets = {t.channel: t for t in targets}
        self._seed = seed
        self._error_mode = error_mode
        self.position = (0.0, 0.0)
        self.channel = 1
        self.distance_m = 0.0
        self.detect_count = self.switch_count = 0
        self.optical_count = self.clear_count = self.failed_clear_count = 0
        self.virtual_seconds = 0.0
        self._trace_enabled = trace
        self.trace: list[dict] = []

    def _record(self, action: str, **extra) -> None:
        if self._trace_enabled:
            self.trace.append({'action': action, 'position': self.position,
                               'channel': self.channel,
                               'virtual_seconds': self.virtual_seconds, **extra})

    def move(self, p: Point) -> None:
        if not all(math.isfinite(x) and abs(x) <= 2000000 for x in p):
            raise ValueError('Invalid move coordinates')
        length = dist(self.position, p)
        self.distance_m += length
        self.virtual_seconds += length / 5.0
        self.position = p
        self._record('move')

    def _error(self, c: int) -> float:
        if self._error_mode == 'plus':
            return DELTA
        if self._error_mode == 'minus':
            return -DELTA
        if self._error_mode != 'hash':
            raise ValueError('Unknown error mode')
        # Same channel and location always yield the SAME error.
        key = f'{self._seed}|{c}|{self.position[0]:.9f}|{self.position[1]:.9f}'.encode()
        x = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big')
        return DELTA * (2 * (x / ((1 << 64) - 1)) - 1)

    def detect(self, channel: int) -> Observation:
        if not 1 <= channel <= 20:
            raise ValueError('Channel out of range')
        if self.channel != channel:
            self.channel = channel
            self.switch_count += 1
            self.virtual_seconds += 1.0
        self.detect_count += 1
        self.virtual_seconds += 5.0
        t = self._targets.get(channel)
        obs = Observation('none')
        if t is not None and not t.cleared:
            d = dist(self.position, t.position)
            illuminated = (t.direction is None
                           or dot(t.direction, sub(self.position, t.position)) >= -1e-9)
            if d <= t.radius + 1e-9 and illuminated:
                if d <= 5.0 + 1e-9:
                    obs = Observation('strong')
                else:
                    direction = sub(t.position, self.position)
                    theta = (math.atan2(direction[1], direction[0])
                             + self._error(channel)) % (2*math.pi)
                    obs = Observation('bearing', theta)
        self._record('detect', status=obs.status, theta=obs.theta)
        return obs

    def clear(self, channel: int) -> bool:
        # Optical/laser action does not require changing the RF receiver channel.
        self.optical_count += 1
        self.virtual_seconds += 3.0
        t = self._targets.get(channel)
        ok = (t is not None and not t.cleared
              and dist(self.position, t.position) <= 20.0 + 1e-8)
        if ok:
            t.cleared = True
            self.clear_count += 1
            self.virtual_seconds += 2.0
        else:
            self.failed_clear_count += 1
        self._record('optical_clear', target_channel=channel, success=ok)
        return ok

    def summary(self) -> dict:
        total = len(self._targets)
        return {'targets': total,
                'directional_targets': sum(t.direction is not None for t in self._targets.values()),
                'cleared': self.clear_count,
                'cleared_fraction': self.clear_count / total,
                'virtual_seconds': self.virtual_seconds,
                'seconds_per_cleared': self.virtual_seconds / self.clear_count if self.clear_count else None,
                'distance_m': self.distance_m,
                'detections': self.detect_count,
                'switches': self.switch_count,
                'optical_attempts': self.optical_count,
                'failed_clears': self.failed_clear_count}


@dataclass
class Track:
    channel: int
    anchor: Point
    observation: Observation


def checked_clear(device: Device, channel: int, point: Point) -> None:
    device.move(point)
    if not device.clear(channel):
        # Never silently claim success or discard a source on an unexpected failure.
        raise RuntimeError('Certified optical clear failed: inspect adapter or bounded-error assumptions')


def localize_and_clear(device: Device, track: Track, use_intersections: bool = True) -> dict:
    c, s, obs = track.channel, track.anchor, track.observation
    if obs.status == 'strong':
        checked_clear(device, c, s)
        return {'stages': 0, 'pair_failures': 0, 'certificate': 'strong'}
    if obs.status != 'bearing' or obs.theta is None:
        raise ValueError('Localization requires a positive initial observation')
    theta, upper = obs.theta, 1500.0
    poly = initial_polygon(s, theta, upper) if use_intersections else []
    pair_failures = 0
    for stage in range(9):
        # The sector alone has a certified enclosing circle.
        sector_radius = upper / (2 * COS_D)
        sector_center = add(s, mul(sector_radius, unit(theta)))
        if sector_radius <= CLEAR_CERT_RADIUS:
            checked_clear(device, c, sector_center)
            return {'stages': stage, 'pair_failures': pair_failures, 'certificate': 'sector'}
        if use_intersections:
            center, radius = enclosing_circle(poly)
            if radius <= CLEAR_CERT_RADIUS:
                checked_clear(device, c, center)
                return {'stages': stage, 'pair_failures': pair_failures, 'certificate': 'intersection'}
            upper = min(upper, max(dist(s, p) for p in poly) + 1e-7)
        e = unit(theta)
        v = (-e[1], e[0])
        t, b = upper/2, upper/4
        midpoint = add(s, mul(t, e))
        probes = [add(midpoint, mul(b, v)), add(midpoint, mul(-b, v))]
        probes.sort(key=lambda q: dist(device.position, q))
        success = False
        for q in probes:
            device.move(q)
            new = device.detect(c)
            if new.status == 'strong':
                checked_clear(device, c, q)
                return {'stages': stage+1, 'pair_failures': pair_failures, 'certificate': 'strong'}
            if new.status == 'bearing':
                if new.theta is None:
                    raise RuntimeError('Malformed bearing observation')
                upper *= KAPPA
                s, theta = q, new.theta
                if use_intersections:
                    poly = clip_bearing(poly, s, theta, upper)
                    if not poly:
                        raise RuntimeError('Inconsistent bearing intersection')
                success = True
                break
        if not success:
            pair_failures += 1
            # BOTH negative probes certify axial target coordinate < old upper/2.
            if use_intersections:
                poly = clip_halfplane(poly, e, dot(e, s) + t)
                if not poly:
                    raise RuntimeError('Inconsistent paired-negative inference')
            upper = t / COS_D + 1e-7
    raise RuntimeError('Contraction safeguard violated; do not continue indefinitely')


def route_order(position: Point, remaining: list[int], sites: list[Point]) -> list[int]:
    """Nearest-neighbor open route plus two passes of open-path 2-opt.

    Heuristic, not a claim of globally optimal routing.
    """
    todo = set(remaining)
    order: list[int] = []
    here = position
    while todo:
        k = min(todo, key=lambda j: (dist(here, sites[j]), j))
        order.append(k)
        todo.remove(k)
        here = sites[k]
    n = len(order)
    for _ in range(2):
        changed = False
        for i in range(n-1):
            before = position if i == 0 else sites[order[i-1]]
            first = sites[order[i]]
            for j in range(i+1, n):
                last = sites[order[j]]
                old, new = dist(before, first), dist(before, last)
                if j+1 < n:
                    after = sites[order[j+1]]
                    old += dist(last, after)
                    new += dist(first, after)
                if new < old - 1e-7:
                    order[i:j+1] = reversed(order[i:j+1])
                    first = sites[order[i]]
                    changed = True
        if not changed:
            break
    return order


def solve(device: Device, use_intersections: bool = True) -> dict:
    """Online solver sees only Device feedback, not source count/location/type.

    Every unknown channel is scanned at every visited coverage station. Sources
    detected at that station are then cleared. A channel is removed from the
    unknown set only on a positive observation, never on a single negative.
    """
    sites, _ = coverage_sites()
    unknown = set(range(1, 21))
    remaining = list(range(25))
    scanned: dict[int, set[int]] = {c: set() for c in range(1, 21)}
    cleared: set[int] = set()
    localization_stats: list[dict] = []
    visited: list[int] = []
    while remaining:
        index = 0 if not visited else route_order(device.position, remaining, sites)[0]
        station = sites[index]
        device.move(station)
        channels = sorted(unknown, key=lambda c: (c != device.channel, c))
        tracks: list[Track] = []
        for c in channels:
            obs = device.detect(c)
            scanned[c].add(index)
            if obs.status != 'none':
                unknown.remove(c)
                tracks.append(Track(c, station, obs))
        # Strong observations first; otherwise nearest current estimated midpoint.
        while tracks:
            def key(track: Track) -> float:
                if track.observation.status == 'strong':
                    return -1e10 + dist(device.position, track.anchor)
                assert track.observation.theta is not None
                guess = add(track.anchor, mul(750.0, unit(track.observation.theta)))
                return dist(device.position, guess)
            track = min(tracks, key=key)
            tracks.remove(track)
            stats = localize_and_clear(device, track, use_intersections)
            localization_stats.append({'channel': track.channel, **stats})
            cleared.add(track.channel)
        remaining.remove(index)
        visited.append(index)
        if len(cleared) == 16:
            return {'stop_certificate': 'known_upper_bound_16', 'visited_stations': visited,
                    'localizations': localization_stats}
    assert all(len(scanned[c]) == 25 for c in unknown)
    return {'stop_certificate': '25_station_coverage_for_all_unknown_channels',
            'visited_stations': visited, 'localizations': localization_stats}


def make_case(seed: int, fraction: float, placement: str = 'uniform') -> list[Target]:
    rng = random.Random(seed)
    n = rng.randint(10, 16)
    channels = rng.sample(range(1, 21), n)
    nd = min(n-1, max(1, int(fraction*n + 0.5)))
    directional_indices = set(rng.sample(range(n), nd))
    targets: list[Target] = []
    for i, c in enumerate(channels):
        angle = rng.random() * 2 * math.pi
        if placement == 'uniform':
            radial = 1800.0 * math.sqrt(rng.random())
        elif placement == 'outward_boundary':
            radial = 1800.0 - rng.random()*1e-4
        else:
            raise ValueError('Unknown placement')
        p = mul(radial, unit(angle))
        radius = 1000.0 if placement == 'outward_boundary' else rng.uniform(1000, 1500)
        direction = None
        if i in directional_indices:
            direction = unit(angle if placement == 'outward_boundary'
                             else rng.random()*2*math.pi)
        targets.append(Target(c, p, radius, direction))
    return targets


def summarize_runs(runs: list[dict]) -> dict:
    times = [r['seconds_per_cleared'] for r in runs]
    return {'cases': len(runs),
            'sources_total': sum(r['targets'] for r in runs),
            'cleared_total': sum(r['cleared'] for r in runs),
            'full_clear_cases': sum(r['cleared'] == r['targets'] for r in runs),
            'mean_seconds_per_cleared_casewise': statistics.mean(times),
            'median_seconds_per_cleared': statistics.median(times),
            'max_seconds_per_cleared': max(times),
            'mean_virtual_seconds': statistics.mean(r['virtual_seconds'] for r in runs),
            'mean_distance_m': statistics.mean(r['distance_m'] for r in runs),
            'mean_detections': statistics.mean(r['detections'] for r in runs),
            'mean_switches': statistics.mean(r['switches'] for r in runs),
            'mean_local_wall_seconds': statistics.mean(r['local_wall_seconds'] for r in runs),
            'mean_visited_stations': statistics.mean(r['visited_station_count'] for r in runs),
            'max_localization_stages': max(r['max_localization_stages'] for r in runs),
            'failed_clears_total': sum(r['failed_clears'] for r in runs)}


def run_experiments(count: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    settings = [('mixed_25pct', 0.25, 'uniform', 'hash'),
                ('mixed_50pct', 0.50, 'uniform', 'hash'),
                ('mixed_75pct', 0.75, 'uniform', 'hash'),
                ('outward_boundary', 0.75, 'outward_boundary', 'hash'),
                ('constant_plus1deg', 0.50, 'uniform', 'plus'),
                ('constant_minus1deg', 0.50, 'uniform', 'minus')]
    all_runs: list[dict] = []
    aggregates: dict[str, dict] = {}
    for gi, (name, fraction, placement, error_mode) in enumerate(settings):
        modes = [('full', True), ('sector_only', False)] if name == 'mixed_50pct' else [('full', True)]
        for mode, intersections in modes:
            group: list[dict] = []
            for i in range(count):
                seed = 20260910 + gi*100000 + i
                local_trace = name == 'mixed_50pct' and mode == 'full' and i < 3
                sim = LocalSimulator(make_case(seed, fraction, placement), seed,
                                     error_mode, trace=local_trace)
                tic = time.perf_counter()
                report = solve(sim, use_intersections=intersections)
                wall = time.perf_counter() - tic
                row = {'group': name, 'mode': mode, 'seed': seed, **sim.summary(),
                       'local_wall_seconds': wall,
                       'visited_station_count': len(report['visited_stations']),
                       'max_localization_stages': max(x['stages'] for x in report['localizations']),
                       'paired_negative_events': sum(x['pair_failures'] for x in report['localizations']),
                       'stop_certificate': report['stop_certificate']}
                if row['cleared'] != row['targets']:
                    raise AssertionError(f'Not fully cleared: {row}')
                if sim.virtual_seconds >= 100*3600:
                    raise AssertionError('Synthetic case exceeded virtual 100 h limit')
                assert abs(sim.virtual_seconds - (sim.distance_m/5 + 5*sim.detect_count
                                                   + sim.switch_count + 3*sim.optical_count
                                                   + 2*sim.clear_count)) < 1e-6
                group.append(row)
                all_runs.append(row)
                if local_trace:
                    (out / f'LOCAL_ONLY_seed_{seed}.json').write_text(json.dumps(
                        {'warning': 'SELF-BUILT LOCAL SIMULATION, NOT AN OFFICIAL TEST LOG',
                         'result': row, 'solver_report': report, 'actions': sim.trace},
                        ensure_ascii=False, indent=2), encoding='utf-8')
            aggregates[f'{name}:{mode}'] = summarize_runs(group)
            print(name, mode, json.dumps(aggregates[f'{name}:{mode}'], ensure_ascii=False), flush=True)
    result = {'warning': 'Synthetic local verification only. NOT official practice/formal test results.',
              'assumptions': {'count': 'uniform integer 10..16',
                 'channels': 'distinct uniform sample without replacement from 1..20',
                 'uniform_placement': 'uniform in area of radius-1800 disk',
                 'reception_radius': 'uniform 1000..1500, or exactly 1000 in boundary stress group',
                 'direction_count': 'round(requested_fraction*N), clipped to 1..N-1',
                 'direction_angle': 'uniform 0..2pi, or radial outward in boundary stress group',
                 'hash_error': 'location/channel/seed-deterministic uniform-like error in [-1,1] degrees',
                 'plus_minus_error': 'constant +1 or -1 degrees at every position',
                 'timing': 'movement/5 + 5*detections + switches + 3*optical + 2*successful_clears',
                 'wall_time': 'local Python process, no HTTP/network or competition simulator overhead'},
              'coverage_certificate': coverage_certificate(), 'aggregates': aggregates, 'runs': all_runs}
    (out / 'local_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'coverage_sites.json').write_text(json.dumps(coverage_certificate(), indent=2), encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=100, help='cases per setting; total 7*cases incl. ablation')
    parser.add_argument('--out', type=Path, default=Path('q4_local_results'))
    args = parser.parse_args()
    if args.cases < 1:
        parser.error('--cases must be positive')
    run_experiments(args.cases, args.out)


if __name__ == '__main__':
    main()
