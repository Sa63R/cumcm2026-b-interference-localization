"""Analytic closure of the ten unchanged, previously undecided public geometries."""
from pathlib import Path
import hashlib
import json
import math


def main():
    source=Path(__file__).with_name('R20_D6_GRID_results.json')
    original=json.loads(source.read_bytes());records=[]
    for item in original['results']:
        spec=item['spec'];identity=str(spec['candidate_id'])
        if original['final_status_by_id'][identity]!='inconclusive':continue
        points=item['certificate']['stations']
        assert len(points)==19 and sum(p==[0.,0.] for p in points)==1
        # The source is ten metres east of the origin and emits eastwards.
        # The reverse triangle inequality puts every nonzero station beyond
        # 1000 m. The sole nearby station (origin) is behind the source.
        assert spec['inner_radius_m']>=1050.
        checked=[]
        for p in points:
            radial=math.hypot(*p);distance=math.dist(p,(10.,0.));projection=p[0]-10.
            blocked=distance>1000.+1e-5 or projection < -1e-5
            assert blocked
            if radial:assert radial>=1050.-1e-7 and distance>=1040.-1e-7
            checked.append(dict(station=p,radial_distance_m=radial,distance_to_source_m=distance,
                                emission_projection_m=projection,strictly_blocked=blocked))
        records.append(dict(candidate_id=spec['candidate_id'],spec=spec,source=[10.,0.],
            radius_m=1000.,direction_deg=0.,in_arena=True,stations_checked=checked,
            analytic_nonzero_station_distance_lower_bound_m=spec['inner_radius_m']-10.,
            origin_projection_m=-10.,strict_counterexample=True))
    assert len(records)==10
    out=Path(__file__).with_suffix('.json')
    with out.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(dict(original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            public_geometry_only=True,new_parameters_tested=0,policy_runs=0,
            original_depth_budget_unchanged=True,additional_analytic_witnesses=records,
            combined_counts=dict(strict_counterexample=204,inconclusive=0,full_coverage=0),
            actual_T_s=None,actual_T_per_source_s=None,actual_T_over_LB=None,
            limitation='Only the already fixed 204 candidates; no claim about all 19-station layouts.'),f,indent=2,allow_nan=False)
        f.write('\n')
    print('All ten remaining fixed candidates have the same strict central witness; total 204 counterexamples.')

if __name__=='__main__':main()
