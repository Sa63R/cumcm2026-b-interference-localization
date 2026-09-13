"""Finite area/radius proxy for *scheduling*, never localization certificates."""

from dataclasses import dataclass
from functools import lru_cache
import math

from simulator_client.state import Position


@dataclass(frozen=True)
class RegionMass:
    points: tuple
    weights: tuple
    status: str
    proposed_nodes: int = 0
    unnormalized_mass: float = 0.

    @property
    def mean(self):
        return Position(sum(w*p.x for w,p in zip(self.weights,self.points)),
                        sum(w*p.y for w,p in zip(self.weights,self.points)))

    @property
    def variance_m2(self):
        center=self.mean
        return sum(w*p.distance_to(center)**2 for w,p in zip(self.weights,self.points))


def point_mass(position,status="known_endpoint"):
    return RegionMass((Position.coerce(position),),(1.,),status)


def radical_inverse_two(index):
    result,factor=0.,.5
    while index:
        result+=(index&1)*factor
        index>>=1
        factor*=.5
    return result


def area_nodes(vertices,limit=12):
    """Equal-area stratification in a convex polygon's triangle fan.

    A deterministic low-discrepancy second coordinate spreads each stratum
    within its triangle. This is quadrature, not exact continuous integration.
    """
    if len(vertices)<3:
        return ()
    anchor=vertices[0]
    triangles=[]
    for a,b in zip(vertices[1:-1],vertices[2:]):
        area=abs((a[0]-anchor[0])*(b[1]-anchor[1])-(a[1]-anchor[1])*(b[0]-anchor[0]))/2
        if area>1e-12:
            triangles.append((a,b,area))
    total=sum(area for _,_,area in triangles)
    if total<=1e-12:
        return ()
    points=[]
    for index in range(limit):
        target=(index+.5)*total/limit
        previous=0.
        for a,b,area in triangles:
            if target<=previous+area:
                root=math.sqrt(max(0.,min(1.,(target-previous)/area)))
                second=radical_inverse_two(index+1)
                points.append(Position((1-root)*anchor[0]+root*(1-second)*a[0]+root*second*b[0],
                                       (1-root)*anchor[1]+root*(1-second)*a[1]+root*second*b[1]))
                break
            previous+=area
    return tuple(points)


def radius_interval_width(point,positives,negatives):
    """All readings constrain a single per-source R, not independent draws."""
    positive_distances=[point.distance_to(Position(*p)) for p in positives]
    if point.distance_to(Position(0.,0.))>1800 or any(d<=5 for d in positive_distances):
        return 0.  # Actual direction observations exclude the near response.
    lower=max([1000.]+positive_distances)
    upper=min([1500.]+[point.distance_to(Position(*p)) for p in negatives])
    return max(0.,upper-lower)


@lru_cache(maxsize=2048)
def _region_mass(vertices,positives,negatives,representative):
    nodes=area_nodes(vertices)
    weights=[radius_interval_width(p,positives,negatives)/500 for p in nodes]
    total=sum(weights)
    if total<=1e-12:
        return point_mass(representative,"quadrature_empty_point_fallback")
    retained=[(p,w) for p,w in zip(nodes,weights) if w>0]
    return RegionMass(tuple(p for p,_ in retained),tuple(w/total for _,w in retained),
                      "finite_area_radius_proxy",len(nodes),total/len(nodes))


def region_mass(region,representative):
    # The conservative outer polygon already contains all recorded bearing
    # half-plane constraints. Uniform area inside it is an explicit proxy,
    # not the exact rounded-bearing likelihood of the official environment.
    return _region_mass(tuple(region.vertices),
                        tuple(sorted({tuple(o.position) for o in region.observations})),
                        tuple(sorted(set(region.no_signal_positions))),
                        (representative.x,representative.y))


def travel_matrix(masses,start,mode):
    if mode not in ("mean_point","expected_distance"):
        raise ValueError("invalid region travel mode")
    start=Position.coerce(start)
    if mode=="mean_point":
        masses=[point_mass(m.mean) for m in masses]
    initial=[sum(w*start.distance_to(p)/5 for w,p in zip(m.weights,m.points)) for m in masses]
    matrix=[[0.]*len(masses) for _ in masses]
    variances=[]
    for i,a in enumerate(masses):
        for j,b in enumerate(masses[:i]):
            mean=second=0.
            for wa,pa in zip(a.weights,a.points):
                for wb,pb in zip(b.weights,b.points):
                    t=pa.distance_to(pb)/5
                    mean+=wa*wb*t
                    second+=wa*wb*t*t
            matrix[i][j]=matrix[j][i]=mean
            variances.append(max(0.,second-mean*mean))
    return matrix,initial,{"edge_variance_mean_s2":sum(variances)/len(variances) if variances else 0.,
                           "edge_variance_max_s2":max(variances,default=0.)}
