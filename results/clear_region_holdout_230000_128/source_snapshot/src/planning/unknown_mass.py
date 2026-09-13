"""Finite spatial quadrature for unknown-channel evidence, never a certificate.

Research prior: N uniform 10..16, a uniform N-subset of 20 channels, uniform
area source positions, and one source radius uniform [1000,1500]. Every
negative observation restricts that SAME radius. Known positive-channel
likelihood factors cancel when conditioning N on labeled present channels.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

from simulator_client.state import Position


@lru_cache(maxsize=1)
def spatial_nodes():
    points=[]
    half=math.pi/24
    for band in range(4):
        inner,outer=1800*math.sqrt(band/4),1800*math.sqrt((band+1)/4)
        radial=(2/3)*(outer**3-inner**3)/(outer**2-inner**2)*math.sin(half)/half
        for sector in range(24):
            angle=(sector+.5)*2*math.pi/24
            points.append(Position(radial*math.cos(angle),radial*math.sin(angle)))
    return tuple(points)


def elementary(values):
    coefficients=[1.]+[0.]*len(values)
    for index,value in enumerate(values):
        for degree in range(index+1,0,-1):
            coefficients[degree]+=value*coefficients[degree-1]
    return coefficients


def existence_posterior(evidence, known_count):
    """Exact algebra for this finite-position prior's per-channel evidences."""
    if not 0<=known_count<=16 or len(evidence)!=20-known_count:
        raise ValueError("Known count and unknown channels disagree")
    if any(not math.isfinite(v) or not 0<=v<=1 for v in evidence):
        raise ValueError("Evidence must lie in [0,1]")
    coefficients=elementary(evidence)
    count_weights={n:coefficients[n-known_count]/math.comb(20,n)
                  for n in range(max(10,known_count),17)}
    normalizer=sum(count_weights.values())
    if normalizer<=1e-300:
        return None  # Quadrature exhaustion is NOT real-world absence.
    presence=[]
    for index,value in enumerate(evidence):
        other=elementary(evidence[:index]+evidence[index+1:])
        weight=sum(value*other[n-known_count-1]/math.comb(20,n)
                   for n in range(max(10,known_count+1),17))
        presence.append(weight/normalizer)
    counts={n:weight/normalizer for n,weight in count_weights.items()}
    assert abs(sum(presence)-sum((n-known_count)*p for n,p in counts.items()))<1e-8
    return presence,counts


@lru_cache(maxsize=512)
def negative_radius_caps(positions):
    nodes=spatial_nodes()
    caps=tuple(max(1000.,min([1500.]+[point.distance_to(Position(*p)) for p in positions])) for point in nodes)
    evidence=sum((cap-1000)/500 for cap in caps)/len(nodes)
    return caps,evidence


@dataclass(frozen=True)
class UnknownBelief:
    # For each group, node interval probability is coefficient*(upper-lower).
    groups: tuple
    presence: dict
    count_probabilities: dict
    evidence: dict

    @property
    def expected_unknown(self):
        return sum(self.presence.values())


def build_unknown_belief(history, known_channels):
    known=set(known_channels)
    unknown=sorted(set(range(1,21))-known)
    negatives={channel:set() for channel in unknown}
    for item in history:
        channel=item["channel"]
        if channel in negatives and item["action"]=="measure":
            if item["result"]!="no_signal":
                raise ValueError("A positive observed channel cannot remain unknown")
            negatives[channel].add(tuple(item["position"]))
    caps,evidence={},{}
    for channel in unknown:
        caps[channel],evidence[channel]=negative_radius_caps(tuple(sorted(negatives[channel])))
    posterior=existence_posterior([evidence[c] for c in unknown],len(known))
    if posterior is None:
        return None
    probabilities,counts=posterior
    presence=dict(zip(unknown,probabilities))
    # Channels sharing negative geometry collapse to one additive mass field;
    # correlated source counts need not be treated as independent worlds.
    groups={}
    for channel in unknown:
        if presence[channel]>0 and evidence[channel]>0:
            groups[caps[channel]]=groups.get(caps[channel],0.)+presence[channel]/(len(spatial_nodes())*500*evidence[channel])
    return UnknownBelief(tuple(groups.items()),presence,counts,evidence)


def insertion_recourse(route, belief, *, credit_saved_scans=True):
    """Expected additive insertion correction under the declared finite prior.

    A source may be inserted only AFTER its first receiving cover. Its common
    clearance/localization service constant is omitted because expected N is
    shared across candidate routes. Location-dependent localization work and
    interactions between multiple insertions are approximations, so this is
    neither an admissible lower bound nor an exact full-task value function.
    """
    if not route or not belief.groups:
        return {"correction_s":0.,"accounted_unknown_mass":0.,"missed_unknown_mass":belief.expected_unknown}
    points=[Position.coerce(task.position) for task in route]
    covers=[i for i,task in enumerate(route) if not task.is_source]
    suffix_covers=[sum(not task.is_source for task in route[i+1:]) for i in range(len(route))]
    correction=accounted=0.
    for caps,coefficient in belief.groups:
        for point,initial_cap in zip(spatial_nodes(),caps):
            upper=initial_cap
            if upper<=1000:
                continue
            distances=[point.distance_to(p) for p in points]
            for index in covers:
                lower=max(1000.,distances[index])
                width=max(0.,upper-lower)
                if width:
                    mass=coefficient*width
                    costs=[]
                    for after in range(index,len(route)):
                        increment=distances[after]
                        if after+1<len(route):
                            increment+=distances[after+1]-points[after].distance_to(points[after+1])
                        costs.append(increment/5-(6*suffix_covers[after] if credit_saved_scans else 0))
                    correction+=mass*min(costs)
                    accounted+=mass
                upper=min(upper,distances[index])
                if upper<=1000:
                    break
    return {"correction_s":correction,"accounted_unknown_mass":accounted,
            "missed_unknown_mass":max(0.,belief.expected_unknown-accounted)}
