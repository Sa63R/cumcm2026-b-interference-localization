"""Offline physical/observation/bound audit after independent identity audit.

No policy, simulator or scenario generator is imported. Explicit archive lists
only; held-out archives require the later final identity audit and opt-in flag.
"""
from collections import defaultdict
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from localization.omni import OmniCandidateRegion
from planning.disk_cover import disk_cover_radius
from research.theory_gap_v1.audit_gaps import helpers, PINNED_HELPERS, sha

VERSION = 'q3-offline-physical-observation-audit-v1'
MARGIN_M = 1e-5


def polygon_distance(vertices, q):
    """Distance to a closed convex CCW polygon, segment or point."""
    if not vertices:
        return math.inf
    if len(vertices) >= 3 and all((b[0]-a[0])*(q[1]-a[1])-(b[1]-a[1])*(q[0]-a[0]) >= -1e-8
                               for a, b in zip(vertices, vertices[1:]+vertices[:1])):
        return 0.
    answer = math.inf
    for a, b in zip(vertices, vertices[1:]+vertices[:1]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        size = dx*dx+dy*dy
        t = max(0., min(1., ((q[0]-a[0])*dx+(q[1]-a[1])*dy)/size)) if size else 0.
        answer = min(answer, math.dist(q, (a[0]+t*dx, a[1]+t*dy)))
    return answer


def mixed_disk_cover(disks, arena=1800.):
    """Finite power-diagram witness test for a union of unequal closed disks.

    Each radius is first SHRUNK by MARGIN_M. The maximum over the arena of
    min_i (distance(x,p_i)^2-r_i^2) is attained at a clipped power-cell vertex
    or an antipode on a boundary arc. Pair radical axes replace bisectors.
    This is floating geometry with an explicit margin, not interval arithmetic.
    """
    disks = sorted(set(tuple(d) for d in disks))
    if not disks:
        return dict(certified=False, method='no_exclusion_disks', worst_power=None)
    if all(r == 1000. for _, _, r in disks):
        radius = disk_cover_radius([(x, y) for x, y, _ in disks], arena)
        return dict(certified=radius <= 1000.-MARGIN_M, method='equal_radius_voronoi',
                    covering_radius_m=radius, margin_m=1000.-radius)
    disks = [(x, y, r-MARGIN_M) for x, y, r in disks]
    candidates = [(0., 0.), (arena, 0.), (-arena, 0.), (0., arena), (0., -arena)]
    for i, (ax, ay, ar) in enumerate(disks):
        norm = math.hypot(ax, ay)
        if norm > 1e-12:
            candidates.append((-arena*ax/norm, -arena*ay/norm))
        for j in range(i+1, len(disks)):
            bx, by, br = disks[j]
            ux, uy = bx-ax, by-ay
            norm2 = ux*ux+uy*uy
            if norm2 <= 1e-20:
                continue
            rhs = (ux*(ax+bx)+uy*(ay+by)+ar*ar-br*br)/2
            fx, fy = rhs*ux/norm2, rhs*uy/norm2
            rest = arena*arena-fx*fx-fy*fy
            if rest >= -1e-7:
                t = math.sqrt(max(0., rest)/norm2)
                candidates += [(fx-uy*t, fy+ux*t), (fx+uy*t, fy-ux*t)]
            for cx, cy, cr in disks[j+1:]:
                vx, vy = cx-ax, cy-ay
                det = ux*vy-uy*vx
                if det == 0.:
                    continue
                first = (norm2+ar*ar-br*br)/2
                second = (vx*vx+vy*vy+ar*ar-cr*cr)/2
                x, y = ax+(first*vy-uy*second)/det, ay+(ux*second-first*vx)/det
                if x*x+y*y <= arena*arena+1e-6:
                    candidates.append((x, y))
    worst = max(min((x-a)**2+(y-b)**2-r*r for a, b, r in disks) for x, y in candidates)
    return dict(certified=worst <= -1e-5, method='mixed_radius_power_diagram',
                worst_power=worst, radius_shrink_m=MARGIN_M)


def observation_audit(record, cover_cache=None):
    """Prefix-only proof replay: no ground_truth access, no synthetic measures."""
    cover_cache = {} if cover_cache is None else cover_cache
    summary = record.get('summary') or {}
    reported = summary.get('action_history', [])
    actual = [a for a in record['history'] if a['action'] in ('/measure', '/clear')]
    if len(reported) != len(actual):
        raise ValueError('Physical versus strategy action-history length mismatch')
    regions, near = {}, defaultdict(list)
    negatives, failed = defaultdict(list), defaultdict(list)
    detected, cleared, clear_checks = set(), set(), []
    inferences = defaultdict(list)
    for event in summary.get('strategy_parameters', {}).get('inferred_no_signal_constraints', []):
        count = event['after_actual_action_count']
        if type(count) is not int or not 0 <= count <= len(actual):
            raise ValueError('Inference references a nonexistent action prefix')
        inferences[count].append(event)
    inference_count = 0

    def apply_inferences(count):
        nonlocal inference_count
        for event in inferences.pop(count, []):
            channel = event['channel']
            region = regions.get(channel)
            if (event.get('physical_measurement') is not False or channel not in detected-cleared
                    or not region or not region.vertices):
                raise ValueError('Unjustified inferred no-signal event')
            distance = polygon_distance(region.vertices, tuple(event['position']))
            if distance <= 1500.+MARGIN_M:
                raise ValueError('Inferred silence lacks a >1500m observed-region certificate')
            region.observe_no_signal(event['position'])
            inference_count += 1
            # No physical/absence coverage ledger entry is added here.

    apply_inferences(0)
    for ordinal, (action, report) in enumerate(zip(actual, reported), start=1):
        channel, response = action['channel'], action['response']
        kind, q = action['action'][1:], (action['position']['x'], action['position']['y'])
        outcome = response['measure_result' if kind == 'measure' else 'clear_result']
        if (report['action'] != kind or report['channel'] != channel
                or tuple(report['position']) != q or report['result'] != outcome
                or abs(report['virtual_time_s']-response['virtual_time_s']) > 2e-6):
            raise ValueError('Physical versus strategy action-history content mismatch')
        region = regions.setdefault(channel, OmniCandidateRegion())
        if kind == 'measure' and channel not in cleared:
            if outcome == 'direction':
                if report['bearing_deg'] != response['svd_deg']:
                    raise ValueError('Strategy bearing differs from physical response')
                detected.add(channel)
                region.observe(q, response['svd_deg'])
            elif outcome == 'near':
                detected.add(channel)
                near[channel].append(q)
            else:
                negatives[channel].append(q)
                region.observe_no_signal(q)
            if channel in detected and not region.vertices:
                raise ValueError('Observed positive source has an empty conservative region')
        elif kind == 'clear':
            radius_bound = max((math.dist(q, p) for p in region.vertices), default=math.inf)
            near_bound = min((math.dist(q, p)+5. for p in near[channel]), default=math.inf)
            bound = min(radius_bound, near_bound)
            certified = channel not in cleared and bound <= 20.-MARGIN_M
            phase = report.get('phase', '')
            claimed = 'certified_clear' in phase or 'near_clear' in phase
            if claimed and not certified:
                raise ValueError(f'Claimed clear certificate cannot be replayed: channel {channel}, bound {bound}')
            clear_checks.append(dict(action_index=action['index'], channel=channel, outcome=outcome,
                                     phase=phase, prefix_certified=certified, radius_upper_m=bound))
            if outcome == 'success':
                cleared.add(channel)
            else:
                failed[channel].append(q)
        apply_inferences(ordinal)
    assert not inferences
    # A successful clear itself establishes removal, even if an earlier optical
    # attempt was not guaranteed. Pre-clear proof and terminal proof are distinct.
    cap = len(cleared) == 16
    absence = []
    for channel in sorted(set(range(1, 21))-cleared):
        disks = tuple(sorted(set((x, y, 1000.) for x, y in negatives[channel]) |
                             set((x, y, 20.) for x, y in failed[channel])))
        if cap:
            certificate = dict(certified=True, method='16_actual_successful_clears')
        else:
            if disks not in cover_cache:
                cover_cache[disks] = mixed_disk_cover(disks)
            certificate = cover_cache[disks]
        absence.append(dict(channel=channel, real_negative_measurements=len(negatives[channel]),
                            failed_clear_disks=len(failed[channel]), **certificate))
    terminal = not (detected-cleared) and all(x['certified'] for x in absence)
    if summary.get('completion_certified_under_model') and not terminal:
        raise ValueError('Claimed terminal certificate lacks actual per-channel absence proof')
    if summary.get('completion_reason') == 'source_count_upper_bound_reached' and not cap:
        raise ValueError('Source-count stop requires 16 clears, not 16 detections')
    return dict(physical_actions=len(actual), detected=len(detected), cleared=len(cleared),
                inferred_silence_verified=inference_count, inferred_coverage_credits=0,
                source_cap_actual_clears=cap, terminal_certified=terminal,
                clear_attempts=clear_checks, absence=absence,
                successful_clear_without_prefix_certificate=sum(
                    c['outcome'] == 'success' and not c['prefix_certified'] for c in clear_checks))


def audit_paths(items, *, theory_dir, cache_path, cited_six_disk=False, seed_cache=None):
    """items: explicit dictionaries {label,path,sha256}; caller grants scope."""
    if not items:
        raise ValueError('No archives supplied; an empty batch cannot pass')
    for module_name, relative in (('geometry', 'geometry/__init__.py'),
                                  ('localization', 'localization/__init__.py'),
                                  ('localization.omni', 'localization/omni.py'),
                                  ('planning.disk_cover', 'planning/disk_cover.py')):
        if Path(sys.modules[module_name].__file__).resolve() != (ROOT/'src'/relative).resolve():
            raise ValueError('Geometry imported from another checkout; use a fresh process in q3-geometric')
    started = time.perf_counter()
    base, cited = helpers(Path(theory_dir))
    cache_path = Path(cache_path)
    cache = base.load_cache(cache_path)
    if seed_cache:
        for key, value in base.load_cache(Path(seed_cache)).items():
            if key in cache and cache[key] != value:
                raise ValueError('Conflicting cached geometry bounds')
            cache[key] = value
    hits = misses = 0
    rows, seen, cover_cache = [], set(), {}
    for item in items:
        path = Path(item['path'])
        row = dict(label=item['label'], input_file=str(path))
        try:
            row['input_sha256'] = sha(path)
            if row['input_sha256'] != item['sha256']:
                raise ValueError('Archive changed since independent identity audit')
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                record = json.load(stream)
            raw = record['row']
            identity = (item['label'], raw['case_sha256'])
            if identity in seen:
                raise ValueError('Duplicate label/case archive')
            seen.add(identity)
            row.update(seed=raw['seed'], case_id=raw['case_id'], case_sha256=raw['case_sha256'],
                       successful=raw['successful'], virtual_time_s=raw['virtual_time_s'],
                       penalized_time_s=raw['penalized_time_s'])
            if not math.isclose(raw['penalized_time_s'], raw['virtual_time_s'] if raw['successful'] else 360000., abs_tol=1e-6):
                raise ValueError('Failure penalty must be retained as 360000s')
            sources, physical_count = base.audit_record(record)
            observations = observation_audit(record, cover_cache)
            row['observations'] = observations
            if raw['successful'] and not observations['terminal_certified']:
                raise ValueError('Successful row lacks an independently replayed terminal certificate')
            geometry = [[c, sources[c]['x'], sources[c]['y']] for c in sorted(sources)]
            key = base.digest(dict(version=base.VERSION, geometry=geometry))
            if key in cache:
                physical = cache[key]
                hits += 1
            else:
                physical = base.physical_bounds(sources.values())
                cache[key] = physical
                misses += 1
                base.atomic_json(cache_path, dict(version=base.VERSION, entries=cache, entries_sha256=base.digest(cache)))
            bound = base.improved_bound(physical['source_route_lower_m'], sources)
            rounding = .5e-6*physical_count+1e-6
            lower = max(0., bound['lower_bound_continuous_s']-rounding)
            eligible = raw['successful'] and observations['terminal_certified']
            row.update(geometry_cache_key=key, physical_bounds=physical, original_bound=bound,
                movement_rounding_allowance_s=rounding, original_lower_s=lower,
                eligible_for_full_clear_ratio=eligible,
                time_over_original_lower=raw['virtual_time_s']/lower if eligible else None)
            if eligible and raw['virtual_time_s']+1e-8 < lower:
                raise ValueError('Successful certified trajectory below original conditional bound')
            if cited_six_disk:
                enhanced = cited.enhance_row(dict({**physical, **bound}, source_total=len(sources),
                    movement_rounding_allowance_s=rounding, conditional_machine_lower_s=lower,
                    successful=raw['successful'], eligible_for_full_clear_ratio=eligible,
                    virtual_time_s=raw['virtual_time_s']))['cited_six_disk']
                row['cited_six_disk'] = enhanced
                if eligible and raw['virtual_time_s']+1e-8 < enhanced['conditional_machine_lower_s']:
                    raise ValueError('Successful certified trajectory below cited conditional bound')
            row['audit_passed'] = True
        except (ValueError, AssertionError, KeyError, TypeError, OSError) as error:
            row.update(audit_passed=False, error=f'{type(error).__name__}: {error}')
        rows.append(row)
    base.atomic_json(cache_path, dict(version=base.VERSION, entries=cache, entries_sha256=base.digest(cache)))
    return dict(version=VERSION, simulator_requests_sent=False, records=len(rows),
        audit_passed=all(r['audit_passed'] for r in rows), failed_audits=sum(not r['audit_passed'] for r in rows),
        cache_hits=hits, cache_misses=misses, unique_cover_configurations=len(cover_cache), wall_s=time.perf_counter()-started,
        helper_sha256=PINNED_HELPERS, cited_six_disk_enabled=cited_six_disk,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__), ROOT/'src/geometry/__init__.py',
            ROOT/'src/localization/__init__.py', ROOT/'src/localization/omni.py', ROOT/'src/planning/disk_cover.py')},
        bound_scope='Conditional all-legal-scene certification bound, not an achieved optimum; source-edge DP lower-bounds continuous disk visitation. N=16 excludes empty certification. Ideal expected information bound is not added.',
        certificate_scope='Observed-prefix sufficient proofs with conservative polygons and floating-point margins; unproved optical attempts remain distinct from physical failures.', rows=rows)


def load_inputs(path, *, allow_heldout=False):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if isinstance(data, dict) and data.get('kind') == 'final_identity_archive_audit':
        if not allow_heldout:
            raise ValueError('Final archives require explicit --allow-heldout after identity audit')
        payload = {k:v for k, v in data.items() if k != 'payload_sha256'}
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        if hashlib.sha256(encoded.encode()).hexdigest() != data['payload_sha256']:
            raise ValueError('Final identity audit payload hash mismatch')
        if not data.get('identity_and_archive_checks_passed'):
            raise ValueError('Independent final identity audit did not pass')
        return [dict(label=f'{split}/{label}', path=str(Path(item['directory'])/name), sha256=checksum)
                for split, part in data['partitions'].items() for label, item in part['evaluations'].items()
                for name, checksum in item['evidence']['archives_sha256'].items()]
    if allow_heldout:
        raise ValueError('Held-out input must come from independent final identity audit')
    if not isinstance(data, list) or not data:
        raise ValueError('Development input requires a nonempty explicit archive list')
    for item in data:
        match = re.fullmatch(r'case-(\d+)\.json\.gz', Path(item['path']).name)
        if not match or not (6000 <= int(match[1]) <= 6047 or 113001 <= int(match[1]) <= 113112):
            raise ValueError('Development payload is outside opened validation / training stress scope')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--seed-cache', type=Path)
    parser.add_argument('--theory-dir', type=Path, default=ROOT.parent/'q3-state-search/research/theory_v1')
    parser.add_argument('--cited-six-disk', action='store_true')
    parser.add_argument('--allow-heldout', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    items = load_inputs(args.inputs, allow_heldout=args.allow_heldout)
    result = audit_paths(items, theory_dir=args.theory_dir, cache_path=args.cache,
                         cited_six_disk=args.cited_six_disk, seed_cache=args.seed_cache)
    result.update(input_manifest_sha256=sha(args.inputs), allow_heldout=args.allow_heldout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('records', 'audit_passed', 'failed_audits', 'cache_hits', 'cache_misses', 'wall_s')}))
    raise SystemExit(0 if result['audit_passed'] else 1)


if __name__ == '__main__':
    main()
