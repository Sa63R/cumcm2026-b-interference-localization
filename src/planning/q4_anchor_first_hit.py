"""One-radio expected first optical hit on a branch-common geometric route.

Spatial quadrature and radio parameters are an explicit assumed model. Latent
nodes only determine first-hit feedback along the SAME route in a branch; they
never choose routes. No predicted observation is written to a live controller.
"""
import math

from geometry import bearing_halfplanes, clip_polygon, disk_halfplanes, distance, point
from planning.coverage import clearance_grid, _clip_horizontal
from planning.q4_conditional_belief import build_belief, predict_branches, ModelUnavailable


LIMITS = dict(spatial_nodes=24, prior_omni=.5, max_history=64, work_limit=262144,
    candidates=3, bin_deg=1., error_deg=1.005, max_grid_cells=256,
    anchor_forward_m=100., anchor_lateral_m=100., improvement_tolerance_s=1e-9)


class _Work:
    def __init__(self):
        self.used = 0

    def add(self, amount=1):
        self.used += amount
        if self.used > LIMITS['work_limit']:
            raise ModelUnavailable('scoring_work_budget')


def key(p):
    return tuple(round(v,6) for v in point(p))


def radio_prefix(history, channel):
    result, seen = [], {}
    for action in history:
        if action.get('action') != 'measure' or action.get('channel') != channel:
            continue
        p = point(action['position'])
        item = dict(position=p, result=action['result'], bearing_deg=action.get('bearing_deg'))
        if p in seen:
            if seen[p] != item:
                raise ModelUnavailable('conflicting_fixed_position_feedback')
            continue
        seen[p] = item
        result.append(item)
    return tuple(result)


def anchor_candidates(history, channel, current, baseline, observed):
    current, baseline = point(current), point(baseline)
    anchors = [(distance(current,point(a['position'])),i,a) for i,a in enumerate(history)
               if a.get('action')=='measure' and a.get('channel')==channel and a.get('result')=='direction']
    if not anchors:
        raise ModelUnavailable('no_real_direction_anchor')
    _, i, a = min(anchors, key=lambda item:(item[0],item[1]))
    p = point(a['position']); angle = math.radians(a['bearing_deg'])
    u = (math.cos(angle),math.sin(angle)); v = (-u[1],u[0])
    values = [baseline]
    for sign in (1.,-1.):
        q = (p[0]+100.*u[0]+sign*100.*v[0], p[1]+100.*u[1]+sign*100.*v[1])
        if key(q) not in observed and all(key(q)!=key(old) for old in values):
            values.append(q)
    return tuple(values),dict(action_index=i,position=list(p),bearing_deg=a['bearing_deg'])


def branch_region(region, observer, outcome, work=None):
    updated = region.copy()
    if outcome[0]=='no_signal':
        return updated
    if outcome[0]=='near':
        constraints=disk_halfplanes(observer,5.,32,outer=True)
    elif outcome[0]=='bearing':
        angle=(outcome[1]+.5)*LIMITS['bin_deg']
        constraints=bearing_halfplanes(observer,angle,LIMITS['bin_deg']/2+LIMITS['error_deg'])
        constraints+=disk_halfplanes(observer,1500.,32,outer=True)
    else:
        raise ModelUnavailable('unknown_predicted_outcome')
    vertices=list(updated.vertices)
    for hp in constraints:
        if work: work.add(len(vertices)+1)
        vertices=clip_polygon(vertices,hp)
        if not vertices:raise ModelUnavailable('empty_geometric_branch')
    updated.vertices,updated._circle=vertices,None
    return updated


def _grid_count(vertices, bearing, work):
    angle=math.radians(bearing); c,s=math.cos(angle),math.sin(angle)
    rotated=[(x*c+y*s,-x*s+y*c) for x,y in vertices]
    first=math.floor(min(v[1] for v in rotated)/28.)
    last=math.floor(max(v[1] for v in rotated)/28.)
    count=0
    for row in range(first,last+1):
        work.add(2*len(rotated)+1)
        strip=_clip_horizontal(rotated,row*28.,True)
        strip=_clip_horizontal(strip,(row+1)*28.,False)
        if strip:
            count+=math.floor(max(v[0] for v in strip)/28.)-math.floor(min(v[0] for v in strip)/28.)+1
        if count>LIMITS['max_grid_cells']:
            raise ModelUnavailable('common_optical_grid_budget')
    return count


def expected_first_hit_cost(route, observer, nodes, weights, work=None):
    """Absorb each positive-mass node once; any uncovered positive node rejects."""
    route=tuple(point(q) for q in route); observer=point(observer)
    nodes=tuple(point(p) for p in nodes); weights=tuple(weights)
    if (not route or len(route)>LIMITS['max_grid_cells'] or len(nodes)!=len(weights)
            or not nodes or any(not math.isfinite(w) or w<0 for w in weights)
            or abs(math.fsum(weights)-1.)>1e-8):
        raise ModelUnavailable('invalid_first_hit_inputs')
    alive={i for i,w in enumerate(weights) if w>0.}
    first_hit=[None]*len(nodes)
    total=0.; elapsed=0.; previous=observer
    for step,q in enumerate(route):
        elapsed+=distance(previous,q)/5.+3.
        for i in tuple(sorted(alive)):
            if work:work.add()
            if distance(nodes[i],q)<=20.:
                total+=weights[i]*(elapsed+2.)
                first_hit[i]=step
                alive.remove(i)
        previous=q
        if not alive:break
    if alive:
        raise ModelUnavailable('uncovered_positive_mass')
    if not math.isfinite(total) or total<=0:
        raise ModelUnavailable('invalid_first_hit_cost')
    return total,first_hit


def optical_first_hit(region, observer, first_bearing, nodes, weights, work):
    circle=region.enclosing_disk()
    if circle.radius<=19.9:
        route=(circle.center,)
        kind='certified_center'
    else:
        count=_grid_count(region.vertices,first_bearing,work)
        # The inherited nearest-order construction performs at most count^2
        # distance comparisons; charge this before doing the common ordering.
        work.add(count*count)
        route=tuple((p.x,p.y) for p in clearance_grid(region.vertices,
            bearing_deg=first_bearing,spacing=28.,start=observer))
        if len(route)!=count:raise ModelUnavailable('grid_count_mismatch')
        kind='common_grid'
    value,hits=expected_first_hit_cost(route,observer,nodes,weights,work)
    return value,dict(kind=kind,route=[list(p) for p in route],first_hit_indices=hits)


def rank_anchor_probes(region, prefix, *, candidates, current, first_bearing,
                       channel, current_channel):
    region=region.copy(); current=point(current); candidates=tuple(point(p) for p in candidates)
    if not 2<=len(candidates)<=3 or len({key(p) for p in candidates})!=len(candidates):
        raise ModelUnavailable('need_two_or_three_distinct_candidates')
    work=_Work()
    belief=build_belief(region,prefix,node_count=LIMITS['spatial_nodes'],prior_omni=LIMITS['prior_omni'],
        max_history=LIMITS['max_history'],work_limit=LIMITS['work_limit'])
    work.add(belief.work_used)
    nodes=tuple(n.position for n in belief.nodes)
    values=[]
    for index,action in enumerate(candidates):
        branches=predict_branches(belief,action,bin_deg=LIMITS['bin_deg'],error_deg=LIMITS['error_deg'])
        if not branches or abs(math.fsum(b.mass for b in branches)-1.)>1e-8:
            raise ModelUnavailable('invalid_outcome_mass')
        work.add(max(b.work_used for b in branches))
        immediate=distance(current,action)/5.+5.+int(channel!=current_channel)
        continuation=0.; outcomes=[]
        for branch in branches:
            if not math.isfinite(branch.mass) or branch.mass<=0:
                raise ModelUnavailable('invalid_branch_mass')
            if branch.outcome[0]=='near':
                value,hits=expected_first_hit_cost((action,),action,nodes,branch.spatial_weights,work)
                route=dict(kind='near_clear',route=[list(action)],first_hit_indices=hits)
            else:
                child=branch_region(region,action,branch.outcome,work)
                value,route=optical_first_hit(child,action,first_bearing,nodes,branch.spatial_weights,work)
            continuation+=branch.mass*value
            outcomes.append(dict(outcome=list(branch.outcome),mass=branch.mass,
                spatial_weights=list(branch.spatial_weights),first_hit_cost_s=value,**route))
        value=immediate+continuation
        if not math.isfinite(value) or value<immediate:
            raise ModelUnavailable('invalid_expected_cost')
        values.append(dict(index=index,position=list(action),immediate_s=immediate,
            expected_cost_s=value,branches=outcomes))
    best=min(range(len(values)),key=lambda i:(values[i]['expected_cost_s'],i))
    selected=best if values[best]['expected_cost_s']<values[0]['expected_cost_s']-1e-9 else 0
    return selected,dict(nodes=[list(p) for p in nodes],spatial_weights=list(belief.spatial_weights),
        candidate_scores=values,selected_index=selected,work_used=work.used,
        limits=dict(LIMITS),original_expected_cost_s=values[0]['expected_cost_s'],
        selected_expected_cost_s=values[selected]['expected_cost_s'])
