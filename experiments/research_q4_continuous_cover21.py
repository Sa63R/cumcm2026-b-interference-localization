"""Bounded public-geometry constraint generation; no scenario/client imports."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEST = ROOT / 'research/q4_continuous_cover21'
sys.path.insert(0, str(ROOT / 'src'))
from planning.q4_directional_cover import certify_directional_cover, verify_directional_cover_certificate

INPUT = WORKSPACE / 'q4-round2/research/q4_round2/R30_ASYMMETRIC_COUNT_DIAGNOSTIC.json'
INPUT_SHA = '6ee1259d4748b2a0b50c0d4b1225ed60455fb537c2e14e8f346966b35c985c6c'
STATION_SHA = 'fe142ffa26a0496ccb80b19fbf8ad75dda5b1e7eb6d104080c8aeb1de6f2c726'
SOURCE_SHA = 'e863b3f0fb83e1deba8d1a24c4b929b8fe1caca9927f8b99e514b1deea6da5c1'
BUDGET = dict(arena_radius=1800., reception_radius=1000., max_depth=24,
              max_cells=800000, range_margin_m=1e-5, orientation_margin_m=1e-7)
DEADLINE_UNIX = 1789178377.0  # 2026-09-12 01:59:37 UTC; checked by CLI.


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def write_json(path, value):
    with path.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def identities():
    source = ROOT / 'src/planning/q4_directional_cover.py'
    if sha(INPUT.read_bytes()) != INPUT_SHA or sha(source.read_bytes()) != SOURCE_SHA:
        raise ValueError('Frozen input/source mismatch')
    old = json.loads(INPUT.read_bytes())['results'][5]
    points = old['ordered_stations']
    if sha(json.dumps(sorted(points), separators=(',', ':'), allow_nan=False).encode()) != STATION_SHA:
        raise ValueError('Original station identity mismatch')
    return points, old, dict(input_sha256=INPUT_SHA, station_sha256=STATION_SHA,
        source_sha256=SOURCE_SHA, script_sha256=sha(Path(__file__).read_bytes()),
        protocol_sha256=sha((DEST / 'PROTOCOL.md').read_bytes()))


def independent_counter(points, witness):
    if witness is None:
        return None
    x, n = witness['source'], witness['normal']
    if math.hypot(*x) > 1800. or abs(math.hypot(*n)-1.) > 1e-10:
        raise ValueError('Invalid witness location/normal')
    rows = []
    for p in points:
        distance = math.dist(x, p)
        projection = sum(n[k]*(p[k]-x[k]) for k in (0, 1))
        if distance <= 1000.+1e-5 and projection >= -1e-5:
            raise ValueError('Witness is not strict for every station')
        rows.append(dict(position=p, distance_m=distance, projection_m=projection,
                         exclusion='range' if distance > 1000.+1e-5 else 'behind'))
    return dict(passed=True, source=x, normal=n, stations=rows)


def certify(points, name, context, identity):
    if time.time() >= DEADLINE_UNIX:
        raise TimeoutError('Fixed research deadline; no new candidate started')
    proof = certify_directional_cover(points, **BUDGET)
    check = verify_directional_cover_certificate(points, proof) if proof['passed'] else None
    counter = independent_counter(points, proof['counterexample'])
    raw = encoded(proof)
    packed = gzip.compress(raw, mtime=0)
    proof_dir = DEST / 'proofs'
    proof_dir.mkdir(exist_ok=True)
    filename = name+'.json.gz'
    with (proof_dir / filename).open('xb') as f:
        f.write(packed)
    row = dict(name=name, points=points, context=context, identity=identity,
        certificate_summary={k: v for k, v in proof.items() if k not in ('stations', 'leaves')},
        proof_file='proofs/'+filename, proof_raw_sha256=sha(raw), proof_gzip_sha256=sha(packed),
        independent_leaf_verification=check, independent_counterexample=counter,
        route_length_m=math.fsum(math.dist(a, b) for a, b in zip(points, points[1:])))
    write_json(DEST / (name+'.json'), row)
    print(json.dumps(dict(name=name, status=proof['status'], reason=proof['reason'],
        visited_cells=proof['visited_cells'], runtime_s=proof['runtime_s'],
        working_margin=context.get('working_margin_m'))), flush=True)
    return row, proof


def hull(points):
    pts = sorted(set(tuple(p) for p in points))
    if len(pts) < 2:
        return pts
    def cross(a, b, p):
        return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
    lower, upper = [], []
    for sequence, out in ((pts, lower), (list(reversed(pts)), upper)):
        for p in sequence:
            while len(out) >= 2 and cross(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
    return lower[:-1]+upper[:-1]


def support_normals(points, x):
    """Public local hull facets; these are proposal constraints, never proof."""
    local = [p for p in points if math.dist(p, x) <= 1000.+1e-5]
    h = hull(local)
    normals = []
    for a, b in zip(h, h[1:]+h[:1]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        size = math.hypot(dx, dy)
        if size:
            normals.append((dy/size, -dx/size))
    if len(h) <= 2:
        for p in h:
            dx, dy = x[0]-p[0], x[1]-p[1]
            size = math.hypot(dx, dy)
            if size:
                normals.append((dx/size, dy/size))
    if not normals:
        normals = [(1., 0.), (-1., 0.), (0., 1.), (0., -1.)]
    return normals


class WorkingSet:
    def __init__(self):
        self.rows = []
        self.keys = set()
        self.positions = {}

    def add(self, x, n, origin, evidence_kind):
        x, n = list(map(float, x)), list(map(float, n))
        norm = math.hypot(*n)
        if not norm or not all(map(math.isfinite, x+n)):
            raise ValueError('Nonfinite/zero proposal constraint')
        n = [v/norm for v in n]
        key = tuple(round(v, 10) for v in x+n)
        if key not in self.keys:
            self.keys.add(key)
            self.rows.append(dict(source=x, normal=n, origin=origin, evidence_kind=evidence_kind))
        self.positions[tuple(x)] = origin

    def absorb(self, proof, points, origin):
        witness = proof['counterexample']
        if witness:
            self.add(witness['source'], witness['normal'], origin, 'verified_strict_counterexample')
        cell = proof['unresolved_cell']
        if cell:
            a, b, c, d = cell['box']
            for x in ((a, c), (b, c), (b, d), (a, d), ((a+b)/2, (c+d)/2)):
                if math.hypot(*x) <= 1800.:
                    for n in support_normals(points, x):
                        self.add(x, n, origin, 'unresolved_proposal_only')
        self.refresh_normals(points)
        if not self.rows:
            raise ValueError('No explicit public constraint; stop')

    def refresh_normals(self, points):
        for x, origin in list(self.positions.items()):
            for n in support_normals(points, x):
                self.add(x, n, origin, 'updated_support_direction_proposal_only')

    def compile(self):
        return np.array([r['source'] for r in self.rows]), np.array([r['normal'] for r in self.rows])


def joint_margin(points, compiled):
    positions, normals = compiled
    v = np.asarray(points)[None, :, :] - positions[:, None, :]
    radial = 1000. - np.sqrt(np.sum(v*v, axis=2))
    projections = np.sum(v*normals[:, None, :], axis=2)
    return float(np.min(np.max(np.minimum(radial, projections), axis=1)))


def build_points(z, bounds, route_ids, local_ids):
    values = bounds[:, 0] + np.asarray(z)*(bounds[:, 1]-bounds[:, 0])
    ri, ro, phase = values[:3]
    pts = [[0., 0.]]
    for number, radius, offset in ((7, ri, 0.), (13, ro, math.radians(phase))):
        for i in range(number):
            a = offset+2*math.pi*i/number
            pts.append([float(radius*math.cos(a)), float(radius*math.sin(a))])
    for j, index in enumerate(local_ids):
        pts[index][0] += float(values[3+2*j])
        pts[index][1] += float(values[4+2*j])
    return [pts[i] for i in route_ids], values.tolist()


def nelder_mead(function, start, step, max_evaluations):
    """Bounded deterministic simplex; every evaluation counted, no certificate."""
    def bounded(p):
        # Reflect at faces: clipping can collapse every vertex onto one face
        # and silently lock a coordinate even for a simple convex quadratic.
        q = np.mod(p, 2.)
        return np.where(q <= 1., q, 2.-q)
    start = np.clip(np.asarray(start, dtype=float), 0., 1.)
    n = len(start)
    simplex = [start.copy()]
    for i in range(n):
        p = start.copy()
        p[i] = p[i]+step if p[i]+step <= 1. else p[i]-step
        simplex.append(np.clip(p, 0., 1.))
    calls = 0
    def evaluate(p):
        nonlocal calls
        if calls >= max_evaluations or time.time() >= DEADLINE_UNIX:
            raise StopIteration
        calls += 1
        return function(p)
    scored = []
    for p in simplex:
        scored.append((evaluate(p), p))
    status = 'evaluation_budget'
    try:
        while calls < max_evaluations:
            scored.sort(key=lambda item: item[0])
            best, worst = scored[0][1], scored[-1][1]
            if (max(float(np.max(np.abs(p-best))) for _, p in scored) < 1e-7
                    and scored[-1][0]-scored[0][0] < 1e-9):
                status = 'simplex_tolerance'
                break
            centroid = np.mean([p for _, p in scored[:-1]], axis=0)
            reflect = bounded(centroid+(centroid-worst))
            fr = evaluate(reflect)
            if fr < scored[0][0]:
                expand = bounded(centroid+2.*(reflect-centroid))
                fe = evaluate(expand)
                scored[-1] = (fe, expand) if fe < fr else (fr, reflect)
            elif fr < scored[-2][0]:
                scored[-1] = (fr, reflect)
            else:
                outside = fr < scored[-1][0]
                contract = bounded(centroid+.5*((reflect if outside else worst)-centroid))
                fc = evaluate(contract)
                if fc < (fr if outside else scored[-1][0]):
                    scored[-1] = (fc, contract)
                else:
                    scored = [scored[0]]+[(evaluate(p), p) for p in
                        [best+.5*(p-best) for _, p in scored[1:]]]
    except StopIteration:
        status = 'deadline' if time.time() >= DEADLINE_UNIX else 'evaluation_budget'
    scored.sort(key=lambda item: item[0])
    return scored[0][1], dict(evaluations=calls, status=status, objective=float(scored[0][0]))


def initial():
    points, old, identity = identities()
    write_json(DEST / 'input.json', dict(points=points, old_layout=old, identity=identity,
                                      budget=BUDGET, deadline_unix=DEADLINE_UNIX))
    certify(points, 'initial', dict(kind='exact_original_coordinates_once'), identity)


def optimize():
    original, old, identity = identities()
    initial_row = json.loads((DEST / 'initial.json').read_bytes())
    proof = json.loads(gzip.decompress((DEST / initial_row['proof_file']).read_bytes()))
    if proof['passed']:
        print('Original fully certified; protocol stops before optimization.')
        return
    if list(DEST.glob('candidate-*.json')):
        raise ValueError('Existing candidate records: no implicit rerun/resume')
    work = WorkingSet()
    work.absorb(proof, original, 'initial')
    bounds3 = np.array([[970., 1020.], [old['outer_radius_m']-4.5, old['outer_radius_m']+15.],
                        [old['outer_phase_deg']-3., old['outer_phase_deg']+3.]])
    base_values = np.array([999., old['outer_radius_m'], old['outer_phase_deg']])
    previous = original
    rows = []
    stop = 'twelve_certificate_candidates_exhausted'
    for iteration in range(1, 13):
        if time.time() >= DEADLINE_UNIX:
            stop = 'fixed_wall_deadline'
            break
        local_ids = []
        if iteration > 6:
            last = proof['counterexample']['source'] if proof['counterexample'] else [
                (proof['unresolved_cell']['box'][0]+proof['unresolved_cell']['box'][1])/2,
                (proof['unresolved_cell']['box'][2]+proof['unresolved_cell']['box'][3])/2]
            native = {native_id: previous[j] for j, native_id in enumerate(old['native_station_route_ids'])}
            local_ids = sorted(range(1, 21), key=lambda i: (math.dist(native[i], last), i))[:2]
        bounds = np.vstack([bounds3, np.tile([-10., 10.], (2*len(local_ids), 1))])
        start_values = np.concatenate([base_values, np.zeros(2*len(local_ids))])
        best_z = (start_values-bounds[:, 0])/(bounds[:, 1]-bounds[:, 0])
        runs = []
        objective_calls = 0
        began = time.perf_counter()
        for step in (.06, .02, .006):
            compiled = work.compile()
            def objective(z):
                counts[0] += 1
                return -joint_margin(build_points(z, bounds, old['native_station_route_ids'], local_ids)[0], compiled)
            counts = [0]
            seed_value = objective(best_z)
            proposal, result = nelder_mead(objective, best_z, step, 2600)
            if objective(proposal) < seed_value-1e-10:
                best_z = proposal
            runs.append(result)
            objective_calls += counts[0]
            pts, _ = build_points(best_z, bounds, old['native_station_route_ids'], local_ids)
            work.refresh_normals(pts)
            if time.time() >= DEADLINE_UNIX:
                break
        points, values = build_points(best_z, bounds, old['native_station_route_ids'], local_ids)
        margin = joint_margin(points, work.compile())
        objective_calls += 1
        assert objective_calls <= 8000
        name = f'candidate-{iteration:02d}'
        work_path = DEST / (name+'-working-set.json')
        write_json(work_path, dict(constraints=work.rows, positions=len(work.positions)))
        context = dict(iteration=iteration, kind='continuous_counterexample_constraint_generation',
            bounds=bounds.tolist(), parameters=values, local_native_indices=local_ids,
            working_set_file=work_path.name, working_set_sha256=sha(work_path.read_bytes()),
            working_margin_m=margin, optimization_runs=runs, optimizer_wall_s=time.perf_counter()-began,
            objective_calls=objective_calls,
            note='Finite support constraints are proposal evidence, never full coverage')
        if time.time() >= DEADLINE_UNIX:
            write_json(DEST / 'unsubmitted-proposal.json', dict(points=points, context=context))
            stop = 'fixed_wall_deadline_before_certificate'
            break
        row, proof = certify(points, name, context, identity)
        rows.append(row)
        if proof['passed']:
            stop = 'first_full_certificate_verified'
            break
        previous = points
        base_values = np.array(values[:3])
        work.absorb(proof, points, name)
    write_json(DEST / 'optimization-summary.json', dict(stop_reason=stop, candidates=len(rows),
        statuses=[r['certificate_summary']['status'] for r in rows], identity=identity,
        total_working_constraints=len(work.rows), completed_at_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('initial', 'optimize'))
    args = parser.parse_args()
    # Fail early on a typo in the registered human-readable deadline.
    assert time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(DEADLINE_UNIX)) == '2026-09-12 01:59:37'
    (initial if args.stage == 'initial' else optimize)()
