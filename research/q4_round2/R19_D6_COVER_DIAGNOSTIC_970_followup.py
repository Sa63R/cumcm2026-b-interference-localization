"""One deeper adjudication of the unchanged 970 m public proposal."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

from R19_D6_COVER_DIAGNOSTIC import (
    OUTER, ROOT, concentric_stations, certify_directional_cover,
    strict_counterexample, verify_directional_cover_certificate,
)


def main():
    output = Path(__file__).with_suffix('.json')
    if output.exists():
        raise ValueError('Keep previous follow-up evidence unchanged')
    began = time.perf_counter()
    original = Path(__file__).with_name('R19_D6_COVER_DIAGNOSTIC.json')
    stations = concentric_stations(970., OUTER, inner_count=6, outer_count=12,
                                  inner_phase_deg=0., outer_phase_deg=0.)
    certificate = certify_directional_cover(
        stations, max_depth=22, max_cells=200000, include_leaves=True)
    result = dict(
        checked_local=datetime.now().astimezone().isoformat(),
        public_geometry_only=True, policy_runs=0, scenario_seeds=[],
        original_evidence_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((ROOT/'src/planning/q4_directional_cover.py').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inner_radius_m=970., outer_radius_m=OUTER, inner_count=6, outer_count=12,
        inner_phase_deg=0., outer_phase_deg=0., certificate=certificate,
        public_analytic_witness=strict_counterexample(stations, {'counterexample': {
            'source': [-1464., -1047.],
            'normal': [math.sqrt(.5), math.sqrt(.5)],
        }}),
        actual_T_s=None, actual_T_per_source_s=None, actual_T_over_LB=None,
    )
    if certificate['passed']:
        result['independent_verify'] = verify_directional_cover_certificate(stations, certificate)
        raise AssertionError('Whole-cover claim conflicts with independently verified witness')
    if certificate['status'] == 'counterexample':
        result['independent_counterexample'] = strict_counterexample(stations, certificate)
    result['final_status'] = 'counterexample'
    result['scope'] = ('The deeper checker status is preserved independently. The final rejection '
                       'follows from a strict, all-19-station analytic receiver counterexample, '
                       'not from treating an inconclusive leaf as failure.')
    result['elapsed_s'] = time.perf_counter() - began
    with output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('final_status', 'elapsed_s')}))
    print(json.dumps({k: certificate.get(k) for k in ('status', 'reason', 'visited_cells', 'unresolved_cell')}))
    print(json.dumps({k: result['public_analytic_witness'][k] for k in (
        'source', 'orientation_deg', 'arena_distance_m', 'in_range_stations',
        'max_in_range_projection_m')}))


if __name__ == '__main__':
    main()
