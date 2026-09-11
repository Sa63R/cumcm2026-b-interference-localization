"""Independent fixed-grid outer-region certificate audit; no planner imports.

Rational edge intersections differ from the producer's polygon clipping.
Angular enclosure uses a box-centre reference, not the producer's largest-gap
construction. Trigonometric outward guards are audited, not interval arithmetic.
"""
from fractions import Fraction as F
import math

TAU = 2.*math.pi


def require(ok, message):
    if not ok:
        raise ValueError(message)


def point(p):
    require(isinstance(p, (tuple, list)) and len(p) == 2 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in p),
        'Invalid finite coordinate')
    return tuple(p)


def rational(p):
    return tuple(F(x) for x in p)


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def convex_polygon(vertices):
    p = tuple(map(rational, vertices))
    area = sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(p,p[1:]+p[:1]))
    require(len(p) >= 3 and area != 0, 'Degenerate nonfallback polygon')
    if area < 0:
        p = p[::-1]
    require(all(cross(a,b,q) >= 0 for a,b in zip(p,p[1:]+p[:1]) for q in p),
            'Output/canonical polygon is not convex and ordered')
    return p


def inside(p, polygon):
    return all(cross(a,b,p) >= 0 for a,b in zip(polygon,polygon[1:]+polygon[:1]))


def box_corners(box):
    a,b,c,d = box
    return ((a,b),(c,b),(c,d),(a,d))


def exact_intersection_points(polygon, box):
    """Convex intersection extreme points from endpoints and edge crossings."""
    a,b,c,d = map(F,box)
    rect = box_corners((a,b,c,d))
    out = {p for p in polygon if a <= p[0] <= c and b <= p[1] <= d}
    out.update(p for p in rect if inside(p,polygon))
    for p,q in zip(polygon,polygon[1:]+polygon[:1]):
        u = (q[0]-p[0],q[1]-p[1])
        for r,s in zip(rect,rect[1:]+rect[:1]):
            v = (s[0]-r[0],s[1]-r[1]);w = (r[0]-p[0],r[1]-p[1])
            det = u[0]*v[1]-u[1]*v[0]
            if not det:
                continue  # Collinear overlap endpoints already included above.
            t = (w[0]*v[1]-w[1]*v[0])/det
            z = (w[0]*u[1]-w[1]*u[0])/det
            if 0 <= t <= 1 and 0 <= z <= 1:
                out.add((p[0]+t*u[0],p[1]+t*u[1]))
    # The endpoint enumeration may include a redundant collinear original
    # vertex; only extreme points require individual outward rounding boxes.
    ordered=sorted(out)
    if len(ordered)<3:
        return set(ordered)
    lower,upper=[],[]
    for sequence,chain in ((ordered,lower),(ordered[::-1],upper)):
        for p in sequence:
            while len(chain)>=2 and cross(chain[-2],chain[-1],p)<=0:
                chain.pop()
            chain.append(p)
    return set(lower[:-1]+upper[:-1])


def normalized(intervals, gap=0.):
    items=[]
    for interval in intervals:
        require(isinstance(interval,(list,tuple)) and len(interval)==2, 'Invalid angular interval')
        a,b=interval
        require(all(isinstance(x,(int,float)) and math.isfinite(x) for x in interval)
                and 0. <= a <= b <= TAU, 'Invalid closed circular interval bounds')
        items.append((a,b))
    if any(a==0. for a,b in items):items.append((TAU,TAU))
    if any(b==TAU for a,b in items):items.append((0.,0.))
    out=[]
    for a,b in sorted(items):
        if out and a <= out[-1][1]+gap:
            out[-1]=(out[-1][0],max(out[-1][1],b))
        else:
            out.append((a,b))
    return tuple(out)


def arc(start,width):
    if width >= TAU-1e-12:
        return ((0.,TAU),)
    a=start % TAU;b=a+width
    return normalized(((a,b),) if b<=TAU else ((a,TAU),(0.,b-TAU)))


def contains_intervals(outer, inner, tolerance=2e-12):
    outer=normalized(outer,tolerance);inner=normalized(inner)
    for a,b in inner:
        if not any(x-tolerance <= a and b <= y+tolerance for x,y in outer):
            return False
    return True


def intersection(left,right):
    intervals=[]
    for a,b in normalized(left):
        for c,d in normalized(right):
            low,high=max(a,c),min(b,d)
            if low<=high+1e-12:
                intervals.append((min(low,high),max(low,high)))
    return normalized(intervals,1e-12)


def distance_squared(p,box):
    x,y=map(F,p);a,b,c,d=map(F,box)
    low=max(a-x,F(0),x-c)**2+max(b-y,F(0),y-d)**2
    high=max((x-u)**2+(y-v)**2 for u,v in box_corners((a,b,c,d)))
    return low,high


def orientation_outer(observer,box,positive,guard):
    """Union of all admissible bearings over B, then add one half-circle."""
    a,b,c,d=box;x,y=observer
    dmin=max(0.,math.hypot(max(a-x,0.,x-c),max(b-y,0.,y-d))-guard)
    if dmin <= 1e-7:
        return ((0.,TAU),),dmin,0.
    cx,cy=(a+c)/2.,(b+d)/2.
    reference=math.atan2(y-cy,x-cx)
    deltas=[math.atan2(math.sin(math.atan2(y-v,x-u)-reference),
                       math.cos(math.atan2(y-v,x-u)-reference)) for u,v in box_corners(box)]
    lower,upper=min(deltas),max(deltas)
    if upper-lower >= math.pi:
        return ((0.,TAU),),dmin,0.
    engine=math.asin(min(1.,1e-12*max(1.,1./dmin))) if positive else 0.
    margin=1e-10+engine
    start=reference+lower+(0. if positive else math.pi)-math.pi/2.-margin
    return arc(start,upper-lower+math.pi+2.*margin),dmin,engine


def close(actual,expected,message):
    require(isinstance(actual,(int,float)) and math.isfinite(actual) and
            abs(actual-expected) <= max(1e-10,16.*math.ulp(float(expected))),message)


def audit_joint_visibility_certificate(evidence):
    e=evidence
    require(e['method']=='q4_joint_visibility_fixed_grid_v1' and e['passed'] is True,
            'Wrong certificate method/status')
    original=tuple(map(point,e['canonical_vertices']));output=tuple(map(point,e['output_vertices']))
    positives=tuple(map(point,e['positive_positions']));negatives=tuple(map(point,e['negative_positions']))
    require(e['status'] in {'fallback','unchanged','outer_refined'},'Unknown outer-region status')
    if e['status']=='fallback':
        require(output==original and e.get('fallback_reason'),'Fallback changed canonical C')
        return dict(passed=True,fallback=True,cells=0,deleted_cells=0,retained_cells=0,
                    boundary='Unchanged original C; partial fallback cell evidence is not a pruning proof')
    require(e['grid_size']==16 and positives,'Invalid grid or missing positive evidence')
    expected_params=dict(bbox_margin_m=1e-4,distance_margin_m=1e-7,forced_margin_m=1e-4,
        near_point_m=1e-7,angle_margin_rad=1e-10,intersection_margin_rad=1e-12,
        max_constraint_work=65536,max_vertices=256)
    require(e['parameters']==expected_params,'Changed geometric guards')
    require(len(original)<=256 and 256*(len(positives)+len(negatives))<=65536,'Unbounded nonfallback computation')
    polygon=convex_polygon(original);outer=convex_polygon(output)
    scale=max(1.,*(abs(x) for p in original+positives+negatives for x in p))
    guard=1e-7+128.*math.ulp(scale)
    require(e['distance_guard_m']==guard,'Incorrect distance guard')
    bbox=(math.nextafter(min(p[0] for p in original)-1e-4,-math.inf),
          math.nextafter(min(p[1] for p in original)-1e-4,-math.inf),
          math.nextafter(max(p[0] for p in original)+1e-4,math.inf),
          math.nextafter(max(p[1] for p in original)+1e-4,math.inf))
    require(tuple(e['bbox'])==bbox,'Wrong complete bounding box')
    xs,ys=e['x_edges'],e['y_edges']
    for edges,lo,hi in ((xs,bbox[0],bbox[2]),(ys,bbox[1],bbox[3])):
        require(len(edges)==17 and edges[0]==lo and edges[-1]==hi and
                all(math.isfinite(x) for x in edges) and all(a<b for a,b in zip(edges,edges[1:])),
                'Grid has missing/nonmonotone edges')
        require(all(edges[i]==lo+(hi-lo)*i/16 for i in range(1,16)), 'Grid is not the complete fixed partition')
    cells=e['cells'];require(len(cells)==256,'Missing or extra grid cells')
    deleted=retained=0
    for index,cell in enumerate(cells):
        ix,iy=index%16,index//16
        box=(xs[ix],ys[iy],xs[ix+1],ys[iy+1])
        require(cell['id']==index and cell['ix']==ix and cell['iy']==iy and tuple(cell['bbox'])==box,
                'Duplicate, reordered or altered cell')
        clipped=exact_intersection_points(polygon,box)
        require(type(cell['removed']) is bool,'Invalid removal flag')
        if not clipped:
            require(not cell['removed'] and cell['reason']=='outside_canonical_region_exact' and
                    not cell['retained_intersection_boxes'] and not cell['constraints'],
                    'False outside-cell declaration')
            continue
        minimums=[math.hypot(max(box[0]-p[0],0.,p[0]-box[2]),
                            max(box[1]-p[1],0.,p[1]-box[3])) for p in positives]
        lower=max(1000.,*minimums)-guard
        uppers=[max(math.dist(p,q) for q in box_corners(box))+guard for p in negatives]
        require(len(cell['positive_min_distances_m'])==len(minimums) and
                len(cell['negative_max_distances_upper_m'])==len(uppers),'Missing distance constraints')
        for a,b in zip(cell['positive_min_distances_m'],minimums):close(a,b,'Wrong positive cell minimum distance')
        for a,b in zip(cell['negative_max_distances_upper_m'],uppers):close(a,b,'Wrong negative cell maximum distance')
        close(cell['R_min_lower_m'],lower,'Wrong common-radius lower bound')
        require(F(cell['R_min_lower_m'])**2 <= max(F(1000)**2,*(distance_squared(p,box)[0] for p in positives)),
                'Radius lower bound is not conservative over entire cell')
        require(all(F(u)**2>=distance_squared(p,box)[1] for u,p in zip(cell['negative_max_distances_upper_m'],negatives)),
                'Negative upper distance does not cover whole cell')
        forced=[j for j,u in enumerate(uppers) if u<lower-1e-4]
        require(cell['forced_negative_indices']==forced,'Incorrect forced-negative membership')
        expected=[('positive',j) for j in range(len(positives))]+[('forced_negative',j) for j in forced]
        if not forced:expected=[]
        require([(c['kind'],c['observation_index']) for c in cell['constraints']]==expected,
                'Missing/extra angular constraints or false omni exclusion')
        allowed=((0.,TAU),)
        for constraint,(kind,j) in zip(cell['constraints'],expected):
            observer=positives[j] if kind=='positive' else negatives[j]
            needed,dmin,engine=orientation_outer(observer,box,kind=='positive',guard)
            declared=normalized(constraint['allowed_intervals'])
            require(contains_intervals(declared,needed),'Angular interval cuts feasible locations/orientations')
            close(constraint['dmin_lower_m'],dmin,'Wrong angular distance lower bound')
            if needed != ((0.,TAU),):
                close(constraint['engine_margin_rad'],engine,'Missing receiver-edge tolerance')
                require(constraint.get('angular_margin_rad')==1e-10,'Missing angular outward margin')
            allowed=intersection(allowed,declared)
        actual=normalized(cell['orientation_intersection'])
        require(contains_intervals(actual,allowed) and contains_intervals(allowed,actual),
                'Logged shared orientation intersection differs')
        remove=bool(forced) and not allowed
        require(cell['removed']==remove,'Cell deleted without whole-cell empty orientation proof')
        if remove:
            require(cell['reason']=='forced_directional_orientation_empty' and not cell['retained_intersection_boxes'],
                    'Invalid deleted-cell evidence')
            deleted+=1
            continue
        retained+=1
        require(cell['reason']==('omni_branch_retained' if not forced else 'orientation_outer_nonempty'),
                'Wrong retained source-type branch')
        boxes=cell['retained_intersection_boxes']
        require(boxes,'Kept cell lost its exact intersection')
        rational_boxes=[]
        for b in boxes:
            require(len(b)==4 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in b) and
                    b[0]<=b[2] and b[1]<=b[3],'Invalid outward rounding box')
            rational_boxes.append(tuple(map(F,b)))
        for p in clipped:
            require(any(a<=p[0]<=c and b<=p[1]<=d for a,b,c,d in rational_boxes),
                    'Retained rounding boxes omit an exact intersection vertex')
            require(inside(p,outer),'Final hull omits a retained exact intersection point')
    require(deleted==e['deleted_intersecting_cells'] and retained==e['retained_intersecting_cells'],
            'Wrong retained/deleted counts')
    require((output==original and deleted==0) if e['status']=='unchanged' else deleted>0,
            'Unchanged/refined status does not match geometry')
    for name,vertices in (('old_disk',original),('new_disk',output)):
        disk=e[name];center=point(disk['center']);radius=disk['radius_m']
        maximum=max(math.dist(center,p) for p in vertices)
        require(math.isfinite(radius) and radius>=maximum-1e-9,'Reported disk does not enclose output vertices')
        require(type(disk['ready_under_19_9_m']) is bool and
                (not disk['ready_under_19_9_m'] or maximum<=19.9+1e-9),'False safe-ready disk claim')
    require(e['became_ready']==(not e['old_disk']['ready_under_19_9_m'] and e['new_disk']['ready_under_19_9_m']),
            'False readiness transition')
    return dict(passed=True,fallback=False,cells=256,deleted_cells=deleted,retained_cells=retained,
                boundary='Whole-cell distance/angle outer constraints and exact retained-point enclosure; trigonometry uses outward float guards, not formal interval arithmetic')
