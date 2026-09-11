"""Three fixed public 19-station proposals, never a policy or scenario run."""
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from planning.q4_directional_cover import (concentric_stations, certify_directional_cover,
                                          verify_directional_cover_certificate)
from planning.coverage import nearest_order, improve_open_route

RADII=(970.,990.,1000.)
OUTER=1800./math.cos(math.pi/12)+5.
OLD_COMMIT='f0b2c6a7'
OLD_PATH='research/q4_cover_geometry/results-v1/results.json'


def strict_counterexample(stations, certificate):
    w=certificate['counterexample']; s=tuple(w['source']); n=tuple(w['normal'])
    if not math.hypot(*s)<=1800.-1e-5 or abs(math.hypot(*n)-1.)>1e-12:
        raise ValueError('Witness outside arena or non-unit normal')
    angle=math.degrees(math.atan2(n[1],n[0]))%360.
    co,si=math.cos(math.radians(angle)),math.sin(math.radians(angle))
    rows=[]
    for p in stations:
        dx,dy=p.x-s[0],p.y-s[1]
        distance=math.hypot(dx,dy); projection=co*dx+si*dy
        within_enlarged_radius=distance<=1000.+1e-5
        # Recompute from the angle actually accepted by the engine model, not
        # the certifier's stored maximum or original normal components.
        if within_enlarged_radius and not projection < -1e-12*max(1.,distance)-1e-7:
            raise ValueError('No strict all-station receiver counterexample')
        rows.append(dict(station=[p.x,p.y],distance_m=distance,projection_m=projection,
                         within_enlarged_radius=within_enlarged_radius))
    return dict(passed=True,source=list(s),orientation_deg=angle,reception_radius_m=1000.,
        arena_distance_m=math.hypot(*s),all_stations=rows,
        in_range_stations=sum(r['within_enlarged_radius'] for r in rows),
        max_in_range_projection_m=max((r['projection_m'] for r in rows if r['within_enlarged_radius']),default=None))


def main():
    output=Path(__file__).with_suffix('.json')
    if output.exists(): raise ValueError('Preserve previous geometry diagnostic')
    old_bytes=subprocess.check_output(['git','show',f'{OLD_COMMIT}:{OLD_PATH}'],cwd=ROOT)
    old=json.loads(old_bytes)
    previous=old['records']
    duplicate=[r for r in previous if r['spec'].get('ni')==6 and r['spec'].get('no')==12]
    if duplicate: raise ValueError('D6 family already recorded; inspect old evidence before repeating')
    began=time.perf_counter();results=[]
    for inner in RADII:
        if time.perf_counter()-began>300: raise TimeoutError('Public geometry budget reached before next candidate')
        stations=concentric_stations(inner,OUTER,inner_count=6,outer_count=12,
                                      inner_phase_deg=0.,outer_phase_deg=0.)
        certificate=certify_directional_cover(stations,max_depth=16,max_cells=200000,include_leaves=True)
        entry=dict(inner_radius_m=inner,outer_radius_m=OUTER,inner_count=6,outer_count=12,
            origin=True,inner_phase_deg=0.,outer_phase_deg=0.,certificate=certificate,
            actual_T_s=None,actual_T_per_source_s=None,actual_T_over_LB=None)
        if certificate['passed']:
            entry['independent_verify']=verify_directional_cover_certificate(stations,certificate)
            route=improve_open_route(nearest_order(stations))
            previous_point=(0.,0.);length=0.
            for p in route:
                length+=math.dist(previous_point,(p.x,p.y));previous_point=(p.x,p.y)
            entry.update(route=[[p.x,p.y] for p in route],open_route_upper_m=length,
                         pure_full_route_movement_s=length/5.)
        elif certificate['status']=='counterexample':
            entry['independent_counterexample']=strict_counterexample(stations,certificate)
        results.append(entry)
        print(json.dumps(dict(inner_radius_m=inner,status=certificate['status'],
            visited_cells=certificate['visited_cells'],runtime_s=certificate['runtime_s'],
            witness=entry.get('independent_counterexample',{}).get('source')),ensure_ascii=False),flush=True)
    result=dict(checked_local=datetime.now().astimezone().isoformat(),
        public_geometry_only=True,policy_runs=0,scenario_seeds=[],old_commit=OLD_COMMIT,
        old_public_records_path=OLD_PATH,old_public_records_sha256=hashlib.sha256(old_bytes).hexdigest(),
        old_records=len(previous),old_family_counts=dict(Counter(str((r['spec'].get('ni'),r['spec'].get('no'))) for r in previous)),
        previous_D6_family_records=len(duplicate),fixed_candidate_radii=list(RADII),
        source_sha256=hashlib.sha256((ROOT/'src/planning/q4_directional_cover.py').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=results,
        elapsed_s=time.perf_counter()-began,
        scope='Certify every closed leaf or exhibit a strict radius/direction counterexample; inconclusive is not failure proof or passage. No T/N performance claim.')
    with output.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
