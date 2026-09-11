"""Six fixed public layouts from GEOMETRY_PROTOCOL.md; no scene/client input."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from planning.q4_directional_cover import (concentric_stations, certify_directional_cover,
                                         verify_directional_cover_certificate)

BUDGET=dict(max_depth=16,max_cells=200000,range_margin_m=1e-5,orientation_margin_m=1e-7)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def boundary(k,rho):
    step=math.pi/k
    heading=math.acos(1800./rho)
    range_angle=math.acos((rho*rho+1800.**2-1000.**2)/(2*rho*1800.))
    alpha=min(heading,range_angle)
    ratio=2*alpha/step
    return dict(angular_step_deg=math.degrees(step),halfwidth_deg=math.degrees(alpha),
                heading_halfwidth_deg=math.degrees(heading),range_halfwidth_deg=math.degrees(range_angle),
                receiving_arc_over_step=ratio,minimum_receiving_outer_stations=math.floor(ratio),
                angle_fraction_at_least_two=min(1.,max(0.,ratio-1.)),
                heading_two_station_threshold_m=1800./math.cos(step),
                range_at_one_step_m=math.sqrt(rho*rho+1800.**2-2*rho*1800.*math.cos(step)),
                min_source_station_distance_m=rho-1800.)


def ordered_layout(k,rho):
    native=concentric_stations(999.,rho,inner_count=k,outer_count=2*k)
    ids=[0]+list(range(1,k+1))+[k+1+((2*k-2-j)%(2*k)) for j in range(2*k)]
    assert sorted(ids)==list(range(1+3*k))
    route=tuple(native[i] for i in ids)
    length=math.fsum(a.distance_to(b) for a,b in zip(route,route[1:]))
    closed_form=rho+2*(k-1)*999.*math.sin(math.pi/k)+2*(2*k-1)*rho*math.sin(math.pi/(2*k))
    assert abs(length-closed_form)<1e-7
    return ids,route,length,closed_form


def check_witness(points,witness):
    if witness is None:return None
    x=witness['source'];n=witness['normal']
    assert math.hypot(*x)<=1800. and abs(math.hypot(*n)-1.)<1e-8
    near=[];far=[]
    for p in points:
        distance=math.hypot(p.x-x[0],p.y-x[1])
        projection=n[0]*(p.x-x[0])+n[1]*(p.y-x[1])
        if distance<=1000.+1e-5:
            assert projection < -1e-5
            near.append(projection)
        else:far.append(distance-1000.)
    return dict(passed=True,source_radius_m=math.hypot(*x),local_stations=len(near),
                max_local_projection_m=max(near) if near else None,
                min_unreachable_margin_m=min(far) if far else None)


def run():
    destination=HERE/'geometry_results.json';proofs=HERE/'geometry-proofs'
    if destination.exists() or proofs.exists():raise ValueError('Preserve existing six-layout outputs')
    protocol=HERE/'GEOMETRY_PROTOCOL.md'
    assert protocol.is_file()
    protocol_sha=sha(protocol.read_bytes())
    proofs.mkdir()
    results=[]
    for k in (8,9,10):
        for rho in (1900.,1920.):
            ids,points,length,formula=ordered_layout(k,rho)
            certificate=certify_directional_cover(points,**BUDGET)
            verification=verify_directional_cover_certificate(points,certificate) if certificate['passed'] else None
            witness=check_witness(points,certificate['counterexample'])
            raw=encoded(certificate);compressed=gzip.compress(raw,mtime=0)
            name=f'inner{k}-outer{2*k}-r999-{int(rho)}.json.gz'
            (proofs/name).write_bytes(compressed)
            result=dict(k=k,station_count=len(points),inner_radius_m=999.,outer_radius_m=rho,
                native_station_route_ids=ids,ordered_stations=[[p.x,p.y] for p in points],
                route_length_m=length,route_closed_form_m=formula,pure_movement_s=length/5.,
                boundary_radial_outward=boundary(k,rho),
                certificate={key:value for key,value in certificate.items() if key not in {'leaves','stations','runtime_s'}},
                independent_leaf_verification=verification,independent_witness_check=witness,
                certificate_runtime_s=certificate['runtime_s'],proof_file='geometry-proofs/'+name,
                proof_gzip_sha256=sha(compressed),proof_raw_sha256=sha(raw),
                radial_midspoke_necessary_outer_limit_m=1000.+999.*math.cos(math.pi/k))
            results.append(result)
            print(json.dumps({'k':k,'rho':rho,'status':certificate['status'],'cells':certificate['visited_cells'],
                              'route_m':length,'boundary':result['boundary_radial_outward']},ensure_ascii=False),flush=True)
    assert sha(protocol.read_bytes())==protocol_sha
    source_names=['src/planning/q4_directional_cover.py','src/simulator_client/state.py']
    output=dict(scope='Six fixed public layouts; no source scene, observed history, policy execution or parameter fitting',
        protocol_sha256=protocol_sha,script_sha256=sha(Path(__file__).read_bytes()),
        source_sha256={p:sha((ROOT/p).read_bytes()) for p in source_names},
        certificate_budget=BUDGET,results=results,
        summary=dict(attempted=6,passed=sum(r['certificate']['passed'] for r in results),
                     counterexamples=sum(r['certificate']['status']=='counterexample' for r in results),
                     inconclusive=sum(r['certificate']['status']=='inconclusive' for r in results)),
        scope_limits=['Route is fixed two-ring order, not asserted shortest',
            'Complete coverage is one receiver for every source and every orientation; not two receivers for every source',
            'Boundary two-receiver fractions are exact angular measures under precisely radial outward orientation',
            'No task-time, expected-cost, clearing, or bearing-error performance claim follows'])
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(output['summary']))


if __name__=='__main__':run()
