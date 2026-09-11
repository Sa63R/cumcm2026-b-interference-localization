"""Fixed 204 public coverage geometries; no strategies or scenario feedback."""
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

from R19_D6_COVER_DIAGNOSTIC import (
    ROOT, concentric_stations, certify_directional_cover,
    strict_counterexample, verify_directional_cover_certificate,
)

PHASES = (0., 5., 10., 15.)
RADII = tuple(range(800, 1201, 25))
EXCESSES = (.5, 5., 15.)
BASE_OUTER = 1800./math.cos(math.pi/12)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def adjudicate(spec, depth):
    stations = concentric_stations(spec['inner_radius_m'], spec['outer_radius_m'],
        inner_count=6, outer_count=12, inner_phase_deg=spec['inner_phase_deg'],
        outer_phase_deg=0.)
    certificate = certify_directional_cover(stations, max_depth=depth,
                                           max_cells=200000, include_leaves=True)
    result = dict(spec=spec, certificate=certificate,
                  full_certificate_sha256=fingerprint(certificate))
    if certificate['passed']:
        checked = verify_directional_cover_certificate(stations, certificate)
        assert checked['passed'] is True
        result['independent_full_leaf_verification'] = checked
        print(json.dumps(dict(event='FULL_PASSED', candidate=spec,
            max_depth=depth, verifier=checked)), flush=True)
    elif certificate['status'] == 'counterexample':
        result['independent_counterexample'] = strict_counterexample(stations, certificate)
    elif certificate['status'] != 'inconclusive':
        raise AssertionError('Unexpected certifier outcome')
    # Partial leaves prove neither full coverage nor its failure. Retain their
    # count/hash and exact unresolved box, but avoid duplicating unused traces.
    if not certificate['passed']:
        result['partial_leaf_count'] = len(certificate['leaves'])
        result['partial_leaves_sha256'] = fingerprint(certificate['leaves'])
        result['certificate'] = {k: v for k, v in certificate.items() if k != 'leaves'}
    return result


def main():
    output = Path(__file__).with_name('R20_D6_GRID_results.json')
    if output.exists():
        raise ValueError('Do not overwrite previous geometry evidence')
    specs = [dict(candidate_id=i, inner_phase_deg=phase, outer_phase_deg=0.,
        inner_radius_m=float(radius), outer_excess_m=excess,
        outer_radius_m=BASE_OUTER+excess, inner_count=6, outer_count=12, origin=True)
        for i, (phase, radius, excess) in enumerate(
            (p, r, e) for p in PHASES for r in RADII for e in EXCESSES)]
    assert len(specs) == 204 and len({fingerprint(s) for s in specs}) == 204
    began = time.perf_counter()
    timestamp = datetime.now().astimezone().isoformat()
    results = []
    for spec in specs:
        results.append(adjudicate(spec, 16))
        if len(results) % 12 == 0:
            print(json.dumps(dict(completed=len(results), counts=dict(Counter(
                r['certificate']['status'] for r in results)),
                elapsed_s=time.perf_counter()-began)), flush=True)
    deeper = []
    unresolved = [r['spec'] for r in results if r['certificate']['status'] == 'inconclusive'][:3]
    for spec in unresolved:
        deeper.append(adjudicate(spec, 22))
    final = {r['spec']['candidate_id']:r['certificate']['status'] for r in results}
    final.update({r['spec']['candidate_id']:r['certificate']['status'] for r in deeper})
    result = dict(
        checked_local=timestamp, completed_local=datetime.now().astimezone().isoformat(),
        public_geometry_only=True, policy_runs=0, scenario_seeds=[],
        source_sha256=hashlib.sha256((ROOT/'src/planning/q4_directional_cover.py').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        witness_checker_sha256=hashlib.sha256(Path(__file__).with_name('R19_D6_COVER_DIAGNOSTIC.py').read_bytes()).hexdigest(),
        plan=dict(inner_phases_deg=PHASES, inner_radii_m=RADII, outer_excesses_m=EXCESSES,
            order='phase ascending, then radius ascending, then excess ascending',
            initial_max_depth=16, max_cells=200000, followup='Only first three inconclusive IDs; depth22, unchanged geometry'),
        planned=specs, planned_sha256=fingerprint(specs), results=results,
        depth22_followups=deeper, initial_counts=dict(Counter(r['certificate']['status'] for r in results)),
        final_counts=dict(Counter(final.values())), final_status_by_id=final,
        elapsed_s=time.perf_counter()-began,
        actual_T_s=None, actual_T_per_source_s=None, actual_T_over_LB=None,
        scope='Full proof requires all leaves and independent replay. Unresolved is never passage or a failure proof. Finite grid says nothing about untested geometries or policy time.')
    with output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k:result[k] for k in ('initial_counts', 'final_counts', 'elapsed_s')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
