"""Eighteen fixed asymmetric-count public layouts; no case/history input."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

WORKSPACE=Path(__file__).resolve().parents[3]
R12=WORKSPACE/'q4-r12-joint-continuation'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(R12/'src'))
from planning.q4_directional_cover import (concentric_stations,certify_directional_cover,
                                         verify_directional_cover_certificate)
BUDGET=dict(arena_radius=1800.,reception_radius=1000.,max_depth=16,max_cells=200000,
            range_margin_m=1e-5,orientation_margin_m=1e-7)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def layout(inner_count,outer_count,inner_radius,half_phase):
    outer_radius=1800./math.cos(math.pi/outer_count)+5.
    phase=180./outer_count if half_phase else 0.
    native=concentric_stations(inner_radius,outer_radius,inner_count=inner_count,
                              outer_count=outer_count,inner_phase_deg=0.,outer_phase_deg=phase)
    last=2*math.pi*(inner_count-1)/inner_count
    angles=[math.radians(phase)+2*math.pi*j/outer_count for j in range(outer_count)]
    distances=[abs(math.remainder(angle-last,2*math.pi)) for angle in angles]
    first=min(range(outer_count),key=lambda j:(distances[j],j))
    ids=[0]+list(range(1,inner_count+1))+[inner_count+1+(first-j)%outer_count for j in range(outer_count)]
    assert sorted(ids)==list(range(1+inner_count+outer_count))
    points=tuple(native[i] for i in ids)
    length=math.fsum(a.distance_to(b) for a,b in zip(points,points[1:]))
    # Formula separates origin, inner chords, the chosen cross-ring edge, and
    # outer chords. It checks the reported index route, not an optimal route.
    formula=(inner_radius+2*(inner_count-1)*inner_radius*math.sin(math.pi/inner_count)
             +native[inner_count].distance_to(native[inner_count+1+first])
             +2*(outer_count-1)*outer_radius*math.sin(math.pi/outer_count))
    assert abs(length-formula)<1e-7
    return points,dict(inner_count=inner_count,outer_count=outer_count,station_count=len(points),
        inner_radius_m=inner_radius,outer_radius_m=outer_radius,inner_phase_deg=0.,outer_phase_deg=phase,
        outer_half_step_phase=bool(half_phase),first_outer_native_index=first,
        first_outer_angle_difference_deg=math.degrees(distances[first]),native_station_route_ids=ids,
        ordered_stations=[[p.x,p.y] for p in points],route_length_m=length,pure_movement_s=length/5.)


def check_counterexample(points,witness):
    if witness is None:return None
    x=witness['source'];n=witness['normal']
    assert math.hypot(*x)<=1800. and abs(math.hypot(*n)-1.)<1e-8
    local=[];far=[]
    for p in points:
        distance=math.hypot(p.x-x[0],p.y-x[1]);dot=n[0]*(p.x-x[0])+n[1]*(p.y-x[1])
        if distance<=1000.+1e-5:
            assert dot < -1e-5
            local.append(dot)
        else:far.append(distance-1000.)
    return dict(passed=True,source_radius_m=math.hypot(*x),receivable_range_station_count=len(local),
                max_local_projection_m=max(local) if local else None,
                min_unreachable_distance_margin_m=min(far) if far else None)


def run():
    dest=HERE/'R30_ASYMMETRIC_COUNT_DIAGNOSTIC.json'
    proofdir=HERE/'R30_ASYMMETRIC_COUNT_PROOFS'
    if dest.exists() or proofdir.exists():raise ValueError('Preserve existing R30 outputs')
    protocol=HERE/'R30_ASYMMETRIC_COUNT_PROTOCOL.md';protocol_sha=sha(protocol.read_bytes())
    proofdir.mkdir();rows=[]
    for ni,no in ((7,13),(6,14),(6,15)):
        for ri in (970.,990.,999.):
            for phase_half in (False,True):
                points,row=layout(ni,no,ri,phase_half)
                proof=certify_directional_cover(points,**BUDGET)
                verify=verify_directional_cover_certificate(points,proof) if proof['passed'] else None
                counter=check_counterexample(points,proof['counterexample'])
                raw=encoded(proof);compressed=gzip.compress(raw,mtime=0)
                name=f'i{ni}-o{no}-r{int(ri)}-half{int(phase_half)}.json.gz'
                (proofdir/name).write_bytes(compressed)
                row.update(certificate={k:v for k,v in proof.items() if k not in {'leaves','stations','runtime_s'}},
                    independent_leaf_verification=verify,independent_counterexample_check=counter,
                    certificate_runtime_s=proof['runtime_s'],proof_file=proofdir.name+'/'+name,
                    proof_gzip_sha256=sha(compressed),proof_raw_sha256=sha(raw))
                rows.append(row)
                print(json.dumps(dict(ni=ni,no=no,ri=ri,half=int(phase_half),status=proof['status'],
                                      cells=proof['visited_cells'],route_m=row['route_length_m'])),flush=True)
    assert len(rows)==18 and sha(protocol.read_bytes())==protocol_sha
    output=dict(scope='18 new asymmetric-count public layouts; no scenario, observation, truth or policy execution',
        protocol_sha256=protocol_sha,script_sha256=sha(Path(__file__).read_bytes()),certificate_budget=BUDGET,
        source_sha256={str(p.relative_to(WORKSPACE)):sha(p.read_bytes()) for p in (
            R12/'src/planning/q4_directional_cover.py',R12/'src/simulator_client/state.py')},
        results=rows,summary=dict(attempted=18,passed=sum(r['certificate']['passed'] for r in rows),
            counterexamples=sum(r['certificate']['status']=='counterexample' for r in rows),
            inconclusive=sum(r['certificate']['status']=='inconclusive' for r in rows)),
        limits=['Failure/unknown here covers only these fixed 18 layouts, not every asymmetric ring placement',
                'Continuous coverage proof uses conservative floats, not interval arithmetic',
                'Pure-route cost is not a source-service or complete-task performance result'])
    with dest.open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(output['summary']))


if __name__=='__main__':run()
