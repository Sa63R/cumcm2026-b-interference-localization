"""Nine preregistered public ring layouts; no scenario or observation input.

Fixed product (970,990,999)x(1900,1950,1999), one process, original R12
station-index route retained. Old certificate budgets are not increased after
seeing a failure: max_depth16/max_cells200000. Every outcome is retained.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

WORKSPACE=Path(__file__).resolve().parents[3]
R12=WORKSPACE/'q4-r12-joint-continuation'
sys.path.insert(0,str(R12/'src'))
from planning.q4_directional_cover import (certified_cover_points, concentric_stations,
    certify_directional_cover, verify_directional_cover_certificate)

INNERS=(970.,990.,999.)
OUTERS=(1900.,1950.,1999.)
BUDGET=dict(max_depth=16,max_cells=200000,range_margin_m=1e-5,orientation_margin_m=1e-7)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def radial_boundary(outer):
    a,R,step=1800.,1000.,2*math.pi/14
    heading=math.acos(a/outer)
    range_angle=math.acos(max(-1.,min(1.,(outer*outer+a*a-R*R)/(2*outer*a))))
    alpha=min(heading,range_angle)
    ratio=2*alpha/step
    # Closed arc endpoints have zero angular measure; minimum over rotations
    # is floor(length/spacing), including an integer-ratio interval.
    minimum=math.floor(ratio)
    fraction=min(1.,max(0.,ratio-1.))
    return dict(outer_radius_m=outer,heading_halfwidth_deg=math.degrees(heading),
        range_halfwidth_deg=math.degrees(range_angle),receiving_halfwidth_deg=math.degrees(alpha),
        grid_step_deg=math.degrees(step),arc_length_over_step=ratio,
        minimum_receiving_outer_stations=minimum,
        angular_fraction_with_at_least_two=fraction,
        angular_measure_at_least_two_deg=360.*fraction,
        min_station_source_distance_m=outer-a,
        two_minimum_heading_threshold_m=a/math.cos(step),
        heading_margin_at_one_step_m=outer*math.cos(step)-a,
        range_distance_at_one_step_m=math.sqrt(outer*outer+a*a-2*outer*a*math.cos(step)))


def verify_counterexample(points,witness):
    """Directly check every station, not the generator's local hull witness."""
    if witness is None:return None
    x=tuple(witness['source']);n=tuple(witness['normal'])
    assert math.hypot(*x)<=1800. and abs(math.hypot(*n)-1.)<1e-8
    near=[];outside=[]
    for p in points:
        d=math.hypot(p.x-x[0],p.y-x[1]);dot=n[0]*(p.x-x[0])+n[1]*(p.y-x[1])
        if d<=1000.+1e-5:
            assert dot < -1e-5
            near.append(dot)
        else:outside.append(d-1000.)
    return dict(passed=True,source_radius_m=math.hypot(*x),
        stations_within_enlarged_radius=len(near),max_local_projection_m=max(near) if near else None,
        min_excluded_range_margin_m=min(outside) if outside else None)


def analytic_midspoke_counterexample(inner,outer):
    low=inner*math.cos(math.pi/7);high=outer-1000.
    if high<=low:return None
    radial=(low+high)/2;angle=math.pi/7
    x=(radial*math.cos(angle),radial*math.sin(angle))
    return dict(source=list(x),normal=[math.cos(angle),math.sin(angle)],
        source_radius_m=radial,radial_gap_interval_m=[low,high],
        positive_halfplane_max_inner_projection_m=low-radial,
        outer_min_distance_excess_m=outer-radial-1000.,
        explanation='All inner stations and origin lie strictly behind radial emission; every outer station is beyond 1000m')


def run(destination):
    proof_dir=destination.with_name(destination.stem+'-proofs')
    if destination.exists() or proof_dir.exists():raise ValueError('Preserve existing R28 outputs')
    proof_dir.mkdir()
    original,old_certificate=certified_cover_points('compact_22')
    old_outer=1800./math.cos(math.pi/14)+5.
    native=concentric_stations(970.,old_outer,inner_count=7,outer_count=14)
    native_ids={p:i for i,p in enumerate(native)}
    order=[native_ids[p] for p in original]
    assert sorted(order)==list(range(22)) and order[0]==0
    rows=[]
    for inner in INNERS:
        for outer in OUTERS:
            native=concentric_stations(inner,outer,inner_count=7,outer_count=14)
            route=tuple(native[i] for i in order)
            certificate=certify_directional_cover(route,**BUDGET)
            verified=(verify_directional_cover_certificate(route,certificate) if certificate['passed'] else None)
            counter=verify_counterexample(route,certificate['counterexample'])
            analytic=analytic_midspoke_counterexample(inner,outer)
            if analytic is not None:verify_counterexample(route,analytic)
            raw=encoded(certificate)
            proof_path=proof_dir/f'inner{int(inner)}-outer{int(outer)}.json.gz'
            compressed=gzip.compress(raw,mtime=0);proof_path.write_bytes(compressed)
            length=math.fsum(a.distance_to(b) for a,b in zip(route,route[1:]))
            rows.append(dict(inner_radius_m=inner,outer_radius_m=outer,
                stations_in_original_index_order=[[p.x,p.y] for p in route],
                route_length_m=length,pure_movement_s=length/5.,
                extra_pure_movement_s=(length-old_certificate['route_length_m'])/5.,
                certificate={k:v for k,v in certificate.items() if k not in ('leaves','stations','runtime_s')},
                generation_runtime_s=certificate['runtime_s'],independent_leaf_verification=verified,
                independent_counterexample_check=counter,analytic_midspoke_counterexample=analytic,
                proof_file=proof_path.relative_to(destination.parent).as_posix(),
                proof_gzip_sha256=sha(compressed),proof_uncompressed_sha256=sha(raw),
                radial_boundary=radial_boundary(outer)))
            print(json.dumps({'inner':inner,'outer':outer,'status':certificate['status'],
                'cells':certificate['visited_cells'],'route_m':length,
                'boundary_min_count':rows[-1]['radial_boundary']['minimum_receiving_outer_stations']},ensure_ascii=False),flush=True)
    result=dict(scope='Pure public 22-station geometry; no scene, source truth, observation, policy execution or fitted parameter',
        protocol=dict(inner_radii_m=INNERS,outer_radii_m=OUTERS,inner_count=7,outer_count=14,
            phases_deg=[0.,0.],arena_m=1800.,reception_m=1000.,certificate_budget=BUDGET,
            process_count=1,route='Exact original R12 native-station index permutation; no candidate NN/2-opt'),
        original=dict(native_station_permutation=order,route_length_m=old_certificate['route_length_m'],
            station_sha256=old_certificate['station_sha256'],radial_boundary=radial_boundary(old_outer)),
        script_sha256=sha(Path(__file__).read_bytes()),source_sha256={str(p.relative_to(WORKSPACE)):sha(p.read_bytes())
            for p in (R12/'src/planning/q4_directional_cover.py',R12/'src/planning/coverage.py',R12/'src/simulator_client/state.py')},
        candidates=rows,summary=dict(attempted=len(rows),
            certified=sum(r['certificate']['passed'] for r in rows),
            counterexamples=sum(r['certificate']['status']=='counterexample' for r in rows),
            inconclusive=sum(r['certificate']['status']=='inconclusive' for r in rows)),
        limitations=['Full coverage proof is a real convexity argument with floating safety margins, not interval arithmetic',
            'Boundary formula assumes precisely radial outward emission at radius1800 and R1000; not arbitrary orientation double coverage',
            'Angular fractions are geometric arc measures, not official-source probabilities',
            'More positive receiving stations is not a proof of better error geometry, guaranteed clearing or task-time improvement',
            'Failed or budget-inconclusive layouts cannot be used as certified complete discovery routes'])
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result['summary']))


if __name__=='__main__':run(Path(__file__).with_suffix('.json'))
