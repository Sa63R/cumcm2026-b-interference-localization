"""Finite OLD-development belief-support diagnostic; never runs a policy.

Only summary.action_history is used from the already-opened 621 development set.
Continuous volumes use ideal half-plane interiors and the stated uniform model;
they are numerical diagnostics, not certified region pruning or calibrated truth.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[3]
R4 = WORKSPACE/'q4-r4-observation'
R12 = WORKSPACE/'q4-r12-joint-continuation'
sys.path.insert(0,str(R4/'src'))
from localization import CandidateRegion
from planning.q4_observation_tree import _kind, _spatial_nodes, make_belief, unique_prefix


def arc(angle):
    left = (angle-math.pi/2) % math.tau
    right = left+math.pi
    return [(left,right)] if right <= math.tau else [(left,math.tau),(0.,right-math.tau)]


def intersection(a,b):
    return [(max(l,x),min(r,y)) for l,r in a for x,y in b if max(l,x)<min(r,y)]


def volume(p,prefix,error):
    positive,negative = [],[]
    for obs in prefix:
        q=obs['position']; d=math.dist(p,q)
        if obs['result']=='no_signal': negative.append((q,d)); continue
        if obs['result']=='near' and d>5. or obs['result']=='direction' and d<=5.: return 0.
        if obs['result']=='direction':
            bearing=math.degrees(math.atan2(p[1]-q[1],p[0]-q[0])) % 360.
            if abs((bearing-obs['bearing_deg']+180.)%360.-180.)>error: return 0.
        positive.append((q,d))
    low=max([1000.]+[d for _,d in positive])
    if low>=1500.: return 0.  # Boundary-only support has zero Lebesgue volume.
    omni=max(0.,min([1500.]+[d for _,d in negative])-low)/500.
    allowed=[(0.,math.tau)]
    for q,d in positive:
        if d>0: allowed=intersection(allowed,arc(math.atan2(q[1]-p[1],q[0]-p[0])))
    breaks=sorted({low,1500.}|{d for _,d in negative if low<d<1500.})
    directional=0.
    for a,b in zip(breaks,breaks[1:]):
        feasible=list(allowed)
        for q,d in negative:
            if d<=(a+b)/2:
                if d==0: feasible=[]; break
                feasible=intersection(feasible,arc(math.atan2(q[1]-p[1],q[0]-p[0])+math.pi))
        directional+=(b-a)*sum(y-x for x,y in feasible)/(500.*math.tau)
    return .5*(omni+directional)


def matches(h,prefix,error):
    for obs in prefix:
        kind,bearing=_kind(h,obs['position'])
        expected='bearing' if obs['result']=='direction' else obs['result']
        if kind!=expected: return False
        if kind=='bearing' and abs((bearing-obs['bearing_deg']+180.)%360.-180.)>error: return False
    return True


def run():
    rows,inputs=[],{}
    cases=[('development',s) for s in range(621001,621025)]+[('development-stress',s) for s in range(621031,621045)]
    for stage,seed in cases:
        if len(rows)>=24: break
        path=R12/f'results/q4_joint_continuation/{stage}/records/compact_joint_continuation-{seed}.json.gz'
        raw=path.read_bytes(); inputs[str(path)]=hashlib.sha256(raw).hexdigest()
        history=json.loads(gzip.decompress(raw))['summary']['action_history']
        regions={}; used=0
        for index,action in enumerate(history):
            c=action['channel']
            if action['action']=='measure' and action['phase']=='active_localization' and c in regions and used<8 and len(rows)<24:
                region=regions[c].copy(); prefix=unique_prefix(history[:index],c)
                belief=make_belief(region,prefix); nodes=_spatial_nodes(region)
                strict=[h for h in belief if matches(h,prefix,region.error_deg)]
                hard_positions={h.position for h in strict}
                volumes={p:volume(p,prefix,region.error_deg) for p in nodes}
                positive_positions={p for p,v in volumes.items() if v>1e-14}
                rows.append(dict(seed=seed,prefix=index,channel=c,observations=len(prefix),
                    spatial_nodes=len(nodes),continuous_positive_volume_nodes=len(positive_positions),
                    coarse_hard_compatible_nodes=len(hard_positions),
                    positive_volume_but_no_coarse_atom=len(positive_positions-hard_positions),
                    r4_soft_mass_on_history_mismatch=sum(h.weight for h in belief if not matches(h,prefix,region.error_deg))))
                used+=1
            if action['action']=='measure' and action['result']=='direction':
                regions.setdefault(c,CandidateRegion()).observe(action['position'],action['bearing_deg'])
    values=[r['r4_soft_mass_on_history_mismatch'] for r in rows]
    return dict(scope='Fixed 621001..024 then 621031..044 old development order, first 24 ACTUAL active prefixes total and at most eight per case; no truth access, new feedback or strategy runs',
        assumptions='Uniform R on [1000,1500], type half/half, uniform directional angle; ideal interior-volume arithmetic with hard bearing support, not official posterior',
        input_sha256=inputs,model_sha256=hashlib.sha256((R4/'src/planning/q4_observation_tree.py').read_bytes()).hexdigest(),
        summary=dict(prefixes=len(rows),prefixes_with_coarse_aliasing=sum(r['positive_volume_but_no_coarse_atom']>0 for r in rows),
            missed_spatial_nodes=sum(r['positive_volume_but_no_coarse_atom'] for r in rows),
            mean_soft_incompatible_mass=sum(values)/len(values) if values else None,max_soft_incompatible_mass=max(values) if values else None,
            prefixes_all_continuous_volumes_zero=sum(r['continuous_positive_volume_nodes']==0 for r in rows)),rows=rows)


if __name__=='__main__':
    result=run()
    destination=Path(__file__).with_name('R17_OLD_PREFIX_DIAGNOSTIC.json')
    with destination.open('x',encoding='utf-8') as stream: json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result['summary']))
