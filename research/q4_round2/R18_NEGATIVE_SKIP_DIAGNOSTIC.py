"""Bounded old-prefix diagnostic; no policy runs, truth decode, or new cases."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

CORE = Path(__file__).resolve().parents[2]
SOURCE = CORE.parent / 'q4-r12-joint-continuation'
sys.path[:0] = [str(SOURCE/'src'), str(SOURCE)]
from localization import CandidateRegion
from planning.joint_visibility_region import joint_visibility_outer
from strategies.q4_range_pruning import region_distance_lower
from experiments.diagnose_q4_joint_visibility import observation_record
from experiments.audit_q4_joint_continuation import wire_prefix
from experiments.audit_q4_joint_visibility import (
    require, convex_polygon, exact_intersection_points, distance_squared,
    box_corners, orientation_outer, intersection, TAU)

STAGES = {'development': list(range(621001, 621025)),
          'development-stress': list(range(621031, 621045))}
LABEL = 'compact_joint_continuation'
SPEC = {'entrypoint': 'strategies.q4_joint_continuation:run_q4_joint_continuation',
        'kwargs': {'config': 'after_active_miss_optical', 'max_expansions': 200}}
MAX_CANDIDATES = 100


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_sha(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def audit_empty_partition(e):
    """Independently prove empty cells, not the producer's fallback status.

    Exact closed polygon/box intersection plus independently constructed angle
    union for every constraint. Like R9, float trigonometry has outward guards;
    this is not a formal interval-arithmetic certificate.
    """
    original = e['canonical_vertices']; positive = e['positive_positions']; negative = e['negative_positions']
    polygon = convex_polygon(original)
    require(positive and negative and e['grid_size'] == 16 and len(e['cells']) == 256,
            'Missing complete partition/evidence')
    scale = max(1., *(abs(x) for p in original+positive+negative for x in p))
    guard = 1e-7+128*math.ulp(scale)
    require(e['distance_guard_m'] == guard, 'Wrong outward distance guard')
    bbox = (math.nextafter(min(p[0] for p in original)-1e-4, -math.inf),
            math.nextafter(min(p[1] for p in original)-1e-4, -math.inf),
            math.nextafter(max(p[0] for p in original)+1e-4, math.inf),
            math.nextafter(max(p[1] for p in original)+1e-4, math.inf))
    xs = [bbox[0]+(bbox[2]-bbox[0])*i/16 for i in range(17)]
    ys = [bbox[1]+(bbox[3]-bbox[1])*i/16 for i in range(17)]
    xs[0], xs[-1], ys[0], ys[-1] = bbox[0], bbox[2], bbox[1], bbox[3]
    require(e['x_edges'] == xs and e['y_edges'] == ys and tuple(e['bbox']) == bbox,
            'Incomplete/altered partition')
    require(all(a < b for edges in (xs, ys) for a,b in zip(edges, edges[1:])), 'Collapsed grid')
    count = 0
    for j, cell in enumerate(e['cells']):
        ix, iy = j%16, j//16
        box = (xs[ix], ys[iy], xs[ix+1], ys[iy+1])
        require((cell['id'], cell['ix'], cell['iy']) == (j, ix, iy) and tuple(cell['bbox']) == box,
                'Missing/reordered/altered cell')
        clipped = exact_intersection_points(polygon, box)
        if not clipped:
            require(cell['reason'] == 'outside_canonical_region_exact' and not cell['removed'],
                    'False outside-cell record')
            continue
        minimums = [math.hypot(max(box[0]-p[0], 0., p[0]-box[2]),
                               max(box[1]-p[1], 0., p[1]-box[3])) for p in positive]
        lower = max(1000., *minimums)-guard
        uppers = [max(math.dist(n,q) for q in box_corners(box))+guard for n in negative]
        require(F(lower)**2 <= max(F(1000)**2, *(distance_squared(p,box)[0] for p in positive)),
                'Nonconservative common radius lower')
        require(all(F(u)**2 >= distance_squared(n,box)[1] for u,n in zip(uppers,negative)),
                'Nonconservative negative distance upper')
        forced = [i for i,u in enumerate(uppers) if u < lower-1e-4]
        require(forced, 'A nonempty cell still permits an omni source')
        allowed = ((0., TAU),)
        for p in positive:
            allowed = intersection(allowed, orientation_outer(p,box,True,guard)[0])
        for i in forced:
            allowed = intersection(allowed, orientation_outer(negative[i],box,False,guard)[0])
        require(not allowed, 'A closed shared orientation remains in a nonempty cell')
        require(cell['removed'] is True and cell['reason'] == 'forced_directional_orientation_empty',
                'Producer did not complete the same whole-cell contradiction')
        count += 1
    require(count > 0, 'Vacuous empty input is not a reception contradiction')
    return dict(passed=True, cells=256, nonempty_cells=count,
                coverage='All exact C intersections exhausted; no cell-centre sampling')


def candidates(record, seed, stage, path):
    require(record['spec'] == SPEC, 'Unexpected old record specification')
    h, before = wire_prefix(record)
    require(record['summary']['completion_certified_under_model'], 'Need completed old prefix record')
    regions, positives, negatives = {}, defaultdict(list), defaultdict(list)
    known, cleared, near = set(), set(), set()
    output, counts = [], Counter()
    for n,a in enumerate(h):
        c, p = a['channel'], tuple(a['position'])
        region = regions.get(c)
        if (a['action'] == 'measure' and a['phase'] == 'coverage' and c in known-cleared-near
                and region and region.vertices and region.enclosing_disk().radius > 19.9):
            counts['known_unready_actual_coverage'] += 1
            lower = region_distance_lower(p,region.vertices)
            if lower > 1500.+1e-5:
                raise ValueError('Existing range predicate should have removed this actual measurement')
            if lower <= 5.+1e-5:
                counts['near_region_not_tested'] += 1
            elif not negatives[c]:
                counts['no_previous_negative_not_tested'] += 1
            else:
                output.append(dict(seed=seed,stage=stage,record_path=str(path),
                    after_actual_action_count=n,channel=c,point=list(p),
                    canonical_vertices=[list(v) for v in region.vertices],
                    positive_positions=[list(v) for v in positives[c]],
                    negative_positions=[list(v) for v in negatives[c]],
                    distance_to_C_lower_m=lower,observed_radius_m=region.enclosing_disk().radius,
                    current_feedback=a['result'],
                    immediate_detection_and_switch_s=5.+int(before[n][1] != c),
                    positive_count=len(positives[c]),negative_count=len(negatives[c])))
        if a['action'] == 'clear':
            if a['result'] == 'success':
                cleared.add(c)
        elif c not in cleared:
            if a['result'] == 'direction':
                regions.setdefault(c,CandidateRegion()).observe(p,a['bearing_deg'])
                positives[c].append(p); known.add(c)
            elif a['result'] == 'near':
                positives[c].append(p); known.add(c); near.add(c)
            else:
                negatives[c].append(p)
    counts['eligible_after_cheap_screens'] = len(output)
    return output, dict(counts)


def self_test():
    v = ((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.))
    # One actual front observation and two actual back observations. The
    # hypothetical receiver in the opposite direction contradicts every cell.
    _, proof = joint_visibility_outer(v, [(100.,0.),(-100.,0.)], [(-50.,100.),(-50.,-100.)])
    audit_empty_partition(proof)
    original = proof['cells'][-1]['removed']; proof['cells'][-1]['removed'] = False
    try:
        audit_empty_partition(proof)
    except ValueError:
        pass
    else:
        raise AssertionError('Tampered unproved cell accepted')
    proof['cells'][-1]['removed'] = original
    _, omni = joint_visibility_outer(v, [(100.,0.),(-100.,0.)], [(2000.,0.)])
    try:
        audit_empty_partition(omni)
    except ValueError:
        pass
    else:
        raise AssertionError('An omni-compatible region was removed')
    return dict(constructed_empty_partition=True, altered_cell_rejected=True, omni_branch_retained=True)


def main(output):
    require(not output.exists(), 'Preserve previous diagnostic')
    test = self_test()
    began=time.perf_counter()
    per_case, metadata, inventory = [], [], {}
    for stage,seeds in STAGES.items():
        directory=SOURCE/'results/q4_joint_continuation'/stage
        manifest=json.loads((directory/'manifest.json').read_bytes())
        require(manifest['seeds']==seeds and manifest['specs'][LABEL]==SPEC, 'Unexpected opened development set')
        require(json.loads((directory/'independent_audit.json').read_bytes())['all_passed'] is True,
                'Original physical audit must pass')
        for relative in ('src/geometry/__init__.py','src/localization/__init__.py'):
            require(manifest['source_sha256'][relative]==sha(SOURCE/relative), 'Geometry source changed')
        inventory[stage]={name:sha(directory/name) for name in ('manifest.json','freeze.json','source.zip','independent_audit.json')}
        for seed in seeds:
            path=directory/'records'/f'{LABEL}-{seed}.json.gz'
            rows,counts=candidates(observation_record(path),seed,stage,path)
            per_case.append(rows)
            metadata.append(dict(seed=seed,stage=stage,record_sha256=sha(path),**counts))
    # Fixed round-robin: (within-case ascending prefix ordinal, seed, prefix,
    # channel), capped before observing any hypothetical-geometry result.
    planned=sorted((i,row['seed'],row['after_actual_action_count'],row['channel'],row)
                   for rows in per_case for i,row in enumerate(rows))[:MAX_CANDIDATES]
    events=[]
    for ordinal,seed,prefix,channel,candidate in planned:
        event=dict(candidate,within_case_eligible_ordinal=ordinal)
        start=time.perf_counter()
        # Appended point is an explicit hypothesis, never an actual observation.
        _, proof=joint_visibility_outer(candidate['canonical_vertices'],
            candidate['positive_positions']+[candidate['point']], candidate['negative_positions'])
        event.update(geometry_status=proof['status'],fallback_reason=proof['fallback_reason'],
                     geometric_runtime_s=time.perf_counter()-start,
                     deleted_cells=proof.get('deleted_intersecting_cells',0),
                     retained_cells=proof.get('retained_intersecting_cells'),certified_silent=False)
        if (len(proof['cells'])==256 and proof.get('retained_intersecting_cells')==0
                and proof.get('deleted_intersecting_cells',0)>0):
            start=time.perf_counter()
            event['independent_empty_partition']=audit_empty_partition(proof)
            event['audit_runtime_s']=time.perf_counter()-start
            event['certified_silent']=True
            require(event['current_feedback']=='no_signal','Contradiction with actual feedback; reject diagnostic')
        events.append(event)
    grouped={}
    for stage,seeds in STAGES.items():
        subset=[e for e in events if e['stage']==stage]
        hits=[e for e in subset if e['certified_silent']]
        grouped[stage]=dict(cases=len(seeds),tested=len(subset),cases_with_test=len({e['seed'] for e in subset}),
            certified_silent=len(hits),cases_with_hit=len({e['seed'] for e in hits}),
            direct_detection_only_s=5*len(hits),direct_detection_plus_current_switch_s=sum(e['immediate_detection_and_switch_s'] for e in hits),
            optimistic_local_fee_ceiling_s_per_original_case=6*len(hits)/len(seeds),
            interpretation='Only tested-prefix local fee ceiling; not extrapolated trigger rate or a whole-run saving guarantee')
    result=dict(checked_local=datetime.now().astimezone().isoformat(),stage='old_development_prefix_only',
        method='Hypothetical positive receptor plus true P/N; independent exhaustive cell contradiction',
        actual_feedback_used_only_after_certificate=True,policy_executed=False,truth_decoded=False,
        candidate_limit=MAX_CANDIDATES,selection_order='(within-case eligible ordinal, seed, actual prefix, channel)',
        public_radius_bounds_m=[1000,1500],radius_upper_used_in_new_contradiction=False,
        near_guard_m=5.00001,self_test=test,source_sha256={str(p.relative_to(SOURCE)):sha(p) for p in
            [SOURCE/'src/planning/joint_visibility_region.py',SOURCE/'experiments/audit_q4_joint_visibility.py',
             SOURCE/'experiments/diagnose_q4_joint_visibility.py',SOURCE/'experiments/audit_q4_joint_continuation.py',
             SOURCE/'src/strategies/q4_range_pruning.py']},script_sha256=sha(__file__),
        inventory=inventory,old_records=metadata,groups=grouped,events=events,
        geometry_runtime_s=sum(e['geometric_runtime_s'] for e in events),total_runtime_s=time.perf_counter()-began)
    with output.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('checked_local','groups','geometry_runtime_s','total_runtime_s')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_suffix('.json'))
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    print(json.dumps(self_test())) if args.self_test else main(args.output.resolve())
