"""Continuous reception-parameter marginalization at finite spatial nodes.

Uniform R/type/orientation and rectangular bounded-bearing likelihood are an
explicit local model, never a calibrated official posterior or clearance proof.
No policy, simulator, clock, scenario or truth inputs are used.
"""
from dataclasses import dataclass
import math

from geometry import point, distance

DEFAULT_WORK_LIMIT = 262144
MIN_ANGLE_WIDTH = 1e-12
MIN_EVIDENCE = 1e-14
MAX_VERTICES = 256
TAU = math.tau


class ModelUnavailable(Exception):
    """The caller must retain its complete geometric/physical fallback."""


@dataclass(frozen=True)
class RadioObservation:
    position: tuple
    result: str
    bearing_deg: float | None


@dataclass(frozen=True)
class RadiusSegment:
    lower: float
    upper: float
    angles: tuple


@dataclass(frozen=True)
class SpatialNode:
    position: tuple
    omni_interval: tuple | None
    directional_segments: tuple
    compatible_volume: float


@dataclass(frozen=True)
class ConditionalBelief:
    nodes: tuple
    spatial_weights: tuple
    prefix: tuple
    prior_omni: float
    error_deg: float
    evidence: float
    work_used: int
    work_limit: int


@dataclass(frozen=True)
class ObservationBranch:
    outcome: tuple
    mass: float
    spatial_weights: tuple
    work_used: int

    @property
    def spatial_ess(self):
        return 1./math.fsum(w*w for w in self.spatial_weights)


class _Work:
    def __init__(self, limit):
        if type(limit) is not int or limit < 1:
            raise ModelUnavailable('invalid_work_limit')
        self.limit, self.used = limit, 0

    def add(self, amount=1):
        self.used += amount
        if self.used > self.limit:
            raise ModelUnavailable('deterministic_work_budget')


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float,int)) or not math.isfinite(value):
        raise ModelUnavailable('nonfinite_or_invalid_'+name)
    return float(value)


def _point(value):
    try:
        if isinstance(value, dict): value = value['x'], value['y']
        return point(value)
    except (ValueError,TypeError,KeyError,OverflowError) as exc:
        raise ModelUnavailable('invalid_position') from exc


def _prefix(items, max_history, work):
    seen, result, channels = {}, [], set()
    for item in items:
        work.add()
        if item.get('action','measure') != 'measure':
            raise ModelUnavailable('radio_prefix_only')
        if 'channel' in item:
            channels.add(item['channel'])
            if len(channels)>1: raise ModelUnavailable('mixed_channel_prefix')
        p = _point(item['position']); kind = item['result']
        if kind not in {'direction','near','no_signal'}:
            raise ModelUnavailable('unknown_observation')
        angle = _finite(item['bearing_deg'],'bearing') % 360. if kind == 'direction' else None
        observation = RadioObservation(p,kind,angle)
        if p in seen:
            if seen[p] != observation:
                raise ModelUnavailable('conflicting_fixed_position_feedback')
            continue
        seen[p] = observation; result.append(observation)
        if len(result)>max_history: raise ModelUnavailable('history_budget')
    return tuple(result)


def _spatial_nodes(region, count, work):
    """R4's same deterministic area rule, with explicit work/finite validation.

Copied from q4-r4-observation/planning/q4_observation_tree.py:_spatial_nodes,
source SHA256 da2325a911b03dbbe80aad26b91f61805d0369aa8be0adaf50a09a6174bccf65.
The vertices-average fan is only quadrature, not an exact spatial posterior.
"""
    vertices = tuple(_point(p) for p in region.vertices)
    if not vertices: raise ModelUnavailable('empty_spatial_region')
    if len(vertices)>MAX_VERTICES: raise ModelUnavailable('vertex_budget')
    work.add(len(vertices))
    center = tuple(sum(v[i] for v in vertices)/len(vertices) for i in (0,1))
    triangles,total = [],0.
    for a,b in zip(vertices,vertices[1:]+vertices[:1]):
        total += abs((a[0]-center[0])*(b[1]-center[1])-(a[1]-center[1])*(b[0]-center[0]))/2.
        triangles.append((total,a,b))
    if not math.isfinite(total): raise ModelUnavailable('nonfinite_area')
    if total<1e-10:
        work.add(len(vertices)**2)
        a,b = max(((a,b) for a in vertices for b in vertices),key=lambda pair:distance(*pair))
        values = [tuple(a[j]+(i+.5)/count*(b[j]-a[j]) for j in (0,1)) for i in range(count)]
    else:
        values = []
        for i in range(count):
            target = (i+.5)/count*total
            work.add(len(triangles))
            _,a,b = next(t for t in triangles if t[0]>=target)
            radial = math.sqrt(((i+1)*.6180339887498949)%1)
            along = ((i+1)*.7548776662466927)%1
            values.append(tuple((1-radial)*center[j]+radial*((1-along)*a[j]+along*b[j]) for j in (0,1)))
    work.add(count*len(vertices))
    nodes = tuple(dict.fromkeys(_point(p) for p in values if region.contains(p)))
    if not nodes: raise ModelUnavailable('spatial_quadrature_exhausted')
    return nodes


def _half_circle(observer, source, positive):
    if observer == source:
        return ((0.,TAU),) if positive else ()
    angle = math.atan2(observer[1]-source[1],observer[0]-source[0])
    left = (angle+(0. if positive else math.pi)-math.pi/2.) % TAU
    # Outward endpoint rounding; this is not certified interval trigonometry.
    lower = max(0.,math.nextafter(left,-math.inf))
    right = math.nextafter(left+math.pi,math.inf)
    return ((lower,right),) if right<=TAU else ((lower,TAU),(0.,min(TAU,right-TAU)))


def _intersect(a,b,work):
    result=[]
    for left,right in a:
        for low,high in b:
            work.add()
            start,end = max(left,low),min(right,high)
            # Numerical slivers/zero-volume boundaries never revive a model.
            # A genuine subthreshold state is unresolved, not safely excluded.
            if end-start>MIN_ANGLE_WIDTH: result.append((start,end))
    return tuple(sorted(result))


def _angular_size(intervals):
    return math.fsum(high-low for low,high in intervals)


def _node(source,prefix,error,prior,work):
    positives,negatives = [],[]
    for obs in prefix:
        work.add()
        d = distance(source,obs.position)
        if not math.isfinite(d): raise ModelUnavailable('nonfinite_distance')
        if obs.result == 'no_signal':
            negatives.append((d,obs.position)); continue
        if (obs.result == 'near' and d>5.) or (obs.result == 'direction' and d<=5.):
            return SpatialNode(source,None,(),0.)
        if obs.result == 'direction':
            angle = math.degrees(math.atan2(source[1]-obs.position[1],source[0]-obs.position[0]))%360.
            if abs((angle-obs.bearing_deg+180.)%360.-180.)>error:
                return SpatialNode(source,None,(),0.)
        positives.append((d,obs.position))
    lower = max([1000.]+[d for d,_ in positives])
    if lower>=1500.: return SpatialNode(source,None,(),0.)
    upper = min([1500.]+[d for d,_ in negatives])
    omni = (lower,upper) if upper>lower and prior>0. else None
    angles = ((0.,TAU),)
    for _,p in positives:
        angles = _intersect(angles,_half_circle(p,source,True),work)
    segments=[]
    if angles and prior<1.:
        pending = sorted(negatives,key=lambda t:t[0])
        boundaries = sorted({lower,1500.}|{d for d,_ in pending if lower<d<1500.})
        cursor=0
        for a,b in zip(boundaries,boundaries[1:]):
            while cursor<len(pending) and pending[cursor][0]<=a:
                angles = _intersect(angles,_half_circle(pending[cursor][1],source,False),work)
                cursor+=1
            work.add()
            if angles: segments.append(RadiusSegment(a,b,angles))
    mass = (prior*(omni[1]-omni[0])/500. if omni else 0.)
    mass += (1.-prior)*math.fsum((s.upper-s.lower)*_angular_size(s.angles) for s in segments)/(500.*TAU)
    return SpatialNode(source,omni,tuple(segments),mass)


def build_belief(region,prefix,*,node_count=24,prior_omni=.5,max_history=64,work_limit=DEFAULT_WORK_LIMIT):
    """Cache continuous R/theta feasible-volume pieces at R4's finite nodes.

The rectangular directional likelihood is constant on its error support; its
per-observation density factors cancel in normalization. This ignores actual
centidegree rounding likelihood and is explicitly not the official posterior.
"""
    if type(node_count) is not int or not 1<=node_count<=256:
        raise ModelUnavailable('invalid_node_count')
    if type(max_history) is not int or not 1<=max_history<=64:
        raise ModelUnavailable('invalid_history_limit')
    prior = _finite(prior_omni,'type_prior')
    error = _finite(region.error_deg,'bearing_error')
    if not 0.<=prior<=1. or not 0.<error<90.: raise ModelUnavailable('invalid_model_parameter')
    work = _Work(work_limit)
    observations = _prefix(prefix,max_history,work)
    nodes = tuple(_node(p,observations,error,prior,work) for p in _spatial_nodes(region,node_count,work))
    total = math.fsum(n.compatible_volume for n in nodes)
    if not math.isfinite(total) or total/len(nodes)<=MIN_EVIDENCE:
        raise ModelUnavailable('conditional_model_exhausted_or_below_precision')
    return ConditionalBelief(nodes,tuple(n.compatible_volume/total for n in nodes),observations,
                             prior,error,total/len(nodes),work.used,work.limit)


def _visible_mass(node,observer,prior,work):
    d = distance(node.position,observer)
    if not math.isfinite(d): raise ModelUnavailable('nonfinite_candidate_distance')
    work.add()
    amount = 0.
    if node.omni_interval:
        a,b = node.omni_interval
        amount = prior*max(0.,b-max(a,d))/500.
    receiving = _half_circle(observer,node.position,True)
    for segment in node.directional_segments:
        work.add()
        span = segment.upper-max(segment.lower,d)
        if span>0.:
            angles = _intersect(segment.angles,receiving,work)
            amount += (1.-prior)*span*_angular_size(angles)/(500.*TAU)
    if not math.isfinite(amount) or amount>node.compatible_volume+max(node.compatible_volume,1e-12)*1e-11:
        raise ModelUnavailable('inconsistent_conditional_mass')
    return min(node.compatible_volume,max(0.,amount)),d


def _bins(bearing,bin_deg,error,work):
    low,high = bearing-error,bearing+error
    count = round(360./bin_deg)
    outcomes={}
    for index in range(math.floor(low/bin_deg),math.floor(high/bin_deg)+1):
        work.add()
        overlap = max(0.,min(high,(index+1)*bin_deg)-max(low,index*bin_deg))
        if overlap>0.:
            key = ('bearing',index%count)
            outcomes[key] = outcomes.get(key,0.)+overlap/(2.*error)
    return outcomes


def predict_branches(belief,observer,*,bin_deg=10.,error_deg=1.005,work_limit=None):
    """Predict common feedback bins; cached hidden-parameter segments are reused.

All spatial_weights align with belief.nodes. They are branch-conditional, not
particle-specific choices of a continuation action. Same exact positions reuse
their observed result with probability one and never draw another noise sample.
"""
    p = _point(observer); width = _finite(bin_deg,'bin_width'); error = _finite(error_deg,'bearing_error')
    if not 0.<width<=360. or abs(360./width-round(360./width))>1e-10 or 360./width>3600:
        raise ModelUnavailable('invalid_bearing_bins')
    if not 0.<error<90. or abs(error-belief.error_deg)>1e-12:
        raise ModelUnavailable('inconsistent_bearing_error_model')
    work = _Work(belief.work_limit if work_limit is None else work_limit)
    for old in belief.prefix:
        work.add()
        if p==old.position:
            outcome = ('bearing',int(old.bearing_deg//width)%round(360./width)) if old.result=='direction' else (old.result,)
            return (ObservationBranch(outcome,1.,belief.spatial_weights,work.used),)
    count = len(belief.nodes)
    groups = {}
    total = math.fsum(node.compatible_volume for node in belief.nodes)
    for i,node in enumerate(belief.nodes):
        work.add()
        if node.compatible_volume<=0.: continue
        positive,d = _visible_mass(node,p,belief.prior_omni,work)
        contributions = {('no_signal',):(node.compatible_volume-positive)/total}
        if positive>0.:
            if d<=5.: contributions[('near',)] = positive/total
            else:
                bearing = math.degrees(math.atan2(node.position[1]-p[1],node.position[0]-p[0]))%360.
                contributions.update({outcome:mass*positive/total for outcome,mass in _bins(bearing,width,error,work).items()})
        for outcome,mass in contributions.items():
            if mass<=0.: continue
            groups.setdefault(outcome,[0.]*count)[i] += mass
    normalizer = math.fsum(math.fsum(values) for values in groups.values())
    if not math.isfinite(normalizer) or abs(normalizer-1.)>1e-10:
        raise ModelUnavailable('prediction_mass_not_conserved')
    branches=[]
    for outcome,values in sorted(groups.items()):
        mass = math.fsum(values)
        if mass>0.: branches.append(ObservationBranch(outcome,mass/normalizer,tuple(v/mass for v in values),work.used))
    return tuple(branches)
