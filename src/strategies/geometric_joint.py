"""Q3 selective cross-channel observations at already visited locations.

Scores are deterministic geometric quadrature heuristics, not expected-cost
guarantees. Only actual observations update the real feasible polygons. Actual
clearance and full discovery retain EfficientSearch's conservative certificates.
"""
from dataclasses import asdict, dataclass, field
import math

from simulator_client.state import Position

from .efficient import EfficientSearch
from .search import SearchResult


@dataclass(frozen=True)
class JointObservationConfig:
    after_clear: bool = True
    active_points: bool = True
    minimum_net_gain_s: float = 5.0
    max_shared_per_stop: int = 4

    @classmethod
    def parse(cls, values):
        if values is not None and not isinstance(values, dict):
            raise ValueError('joint_config must be a dictionary')
        try:
            result = cls(**(values or {}))
        except TypeError as error:
            raise ValueError(f'Invalid joint_config: {error}') from error
        if type(result.after_clear) is not bool or type(result.active_points) is not bool:
            raise ValueError('Sharing switches must be boolean')
        if (type(result.max_shared_per_stop) is not int
                or not 0 <= result.max_shared_per_stop <= 20):
            raise ValueError('max_shared_per_stop must be an integer in 0..20')
        if (isinstance(result.minimum_net_gain_s, bool)
                or not isinstance(result.minimum_net_gain_s, (int, float))
                or not math.isfinite(result.minimum_net_gain_s)
                or not 0 <= result.minimum_net_gain_s <= 1000):
            raise ValueError('minimum_net_gain_s must be finite in 0..1000')
        return result


@dataclass
class JointObservationResult(SearchResult):
    joint_observations: dict = field(default_factory=dict)


def polygon_quadrature(vertices, limit=9):
    """Equal-mass deterministic selection from triangular area quadrature.

    This defines a planning proxy on the observed polygon, not the actual
    posterior of the hidden source/radius. Degenerate segments use midpoints.
    """
    if not vertices:
        return ()
    if len(vertices) < 3:
        a, b = vertices[0], vertices[-1]
        return tuple((a[0]+(b[0]-a[0])*(i+.5)/limit,
                      a[1]+(b[1]-a[1])*(i+.5)/limit) for i in range(limit))
    anchor = vertices[0]
    samples = []
    for a, b in zip(vertices[1:-1], vertices[2:]):
        twice_area = abs((a[0]-anchor[0])*(b[1]-anchor[1])
                         -(a[1]-anchor[1])*(b[0]-anchor[0]))
        if twice_area <= 1e-12:
            continue
        for weights in ((4, 1, 1), (1, 4, 1), (1, 1, 4)):
            point = tuple((weights[0]*anchor[j]+weights[1]*a[j]+weights[2]*b[j])/6
                          for j in range(2))
            samples.append((point, twice_area/3))
    if not samples:
        return polygon_quadrature((vertices[0], vertices[-1]), limit)
    total = sum(weight for _, weight in samples)
    result, index, accumulated = [], 0, samples[0][1]
    for i in range(limit):
        mass = total*(i+.5)/limit
        while index+1 < len(samples) and accumulated < mass:
            index += 1
            accumulated += samples[index][1]
        result.append(samples[index][0])
    return tuple(result)


def shared_observation_value(region, position):
    """Estimate travel avoided by updating an uncertain destination early.

    For old centre c and hypothetical updated centre c', the nonnegative
    triangle excess |q-c|+|c-c'|-|q-c'| compares two explicit local routes.
    Averaging it over quadrature points/errors is ONLY a ranking heuristic.
    Future coverage/ordering can change the benefit; no global bound is claimed.
    """
    if not region.vertices or not region.observations:
        return None
    q = Position.coerce(position)
    circle = region.enclosing_disk()
    if circle.radius <= 19.9:
        return None
    # A maximum of a convex distance over the outer polygon is attained at a
    # vertex. This check is a sufficient guaranteed-reception certificate.
    if max(math.hypot(v[0]-q.x, v[1]-q.y) for v in region.vertices) > 1000-1e-6:
        return None
    center = Position.coerce(circle.center)
    hypotheses = polygon_quadrature(region.vertices)
    travel, radius, mass = 0., 0., 0.
    for source in hypotheses:
        distance = math.hypot(source[0]-q.x, source[1]-q.y)
        for error, weight in ((-region.error_deg, .25), (0., .5), (region.error_deg, .25)):
            if distance <= 5:
                updated, new_radius = q, 5.
            else:
                bearing = (math.degrees(math.atan2(source[1]-q.y, source[0]-q.x))+error) % 360
                hypothetical = region.copy().observe(q, bearing)
                if not hypothetical.vertices:
                    continue
                new_circle = hypothetical.enclosing_disk()
                updated, new_radius = Position.coerce(new_circle.center), new_circle.radius
            excess = max(0., q.distance_to(center)+center.distance_to(updated)-q.distance_to(updated))
            travel += weight*excess
            radius += weight*new_radius
            mass += weight
    if mass <= 0:
        return None
    return dict(travel_saving_proxy_m=travel/mass, predicted_radius_proxy_m=radius/mass,
                original_radius_m=circle.radius, hypotheses=len(hypotheses),
                guaranteed_reception=True, proxy_kind='polygon_quadrature_local_triangle_excess')


class GeometricJointSearch(EfficientSearch):
    def __init__(self, client, max_actions, max_active_probes, config):
        self.joint_config = JointObservationConfig.parse(config)
        super().__init__(client, max_actions, max_active_probes, None)
        self.variant = 'geometric_joint'
        self.report = JointObservationResult(**asdict(self.report))
        self.report.variant = self.variant
        self.report.strategy_parameters['joint_observation_config'] = asdict(self.joint_config)
        self.report.joint_observations = dict(stops_considered=0, candidates_scored=0,
            shared_measurements=0, actual_sharing_time_s=0., by_trigger={}, decisions=[])
        self._sharing = False

    def _perform(self, action, position, channel, phase):
        response = super()._perform(action, position, channel, phase)
        if not self._sharing and self.joint_config.max_shared_per_stop:
            trigger = ('after_clear' if action == 'clear' and response['clear_result'] == 'success'
                       and self.joint_config.after_clear else
                       'active_point' if action == 'measure' and phase == 'active_localization'
                       and self.joint_config.active_points else None)
            if trigger:
                self._share_at_current_point(channel, trigger)
        return response

    def _share_at_current_point(self, primary_channel, trigger):
        q = self.client.state.position
        key = (round(q.x, 6), round(q.y, 6))
        stats = self.report.joint_observations
        stats['stops_considered'] += 1
        values = {}
        for channel in sorted(self.detected-self.cleared-{primary_channel}):
            if channel in self.near_points or key in self.observed_positions.get(channel, set()):
                continue
            value = shared_observation_value(self.regions[channel], q)
            if value is not None:
                stats['candidates_scored'] += 1
                values[channel] = value
        # Reserve a conservative switch back even if the next action is a clear
        # (which actually need not switch). Only actual simulator cost is charged.
        resume_channel = primary_channel if trigger == 'active_point' else self.client.state.current_channel
        self._sharing = True
        try:
            for _ in range(self.joint_config.max_shared_per_stop):
                if not values:
                    break
                current_channel = self.client.state.current_channel
                def score(channel):
                    overhead = 5+int(channel != current_channel)+int(channel != resume_channel)
                    return values[channel]['travel_saving_proxy_m']/5-overhead
                channel = max(values, key=lambda c: (score(c), -c))
                net = score(channel)
                if net < self.joint_config.minimum_net_gain_s:
                    break
                value = values.pop(channel)
                before = self.client.state.virtual_time_s
                response = self._perform('measure', q, channel, 'geometric_shared_observation')
                charged = self.client.state.virtual_time_s-before
                stats['shared_measurements'] += 1
                stats['actual_sharing_time_s'] += charged
                stats['by_trigger'][trigger] = stats['by_trigger'].get(trigger, 0)+1
                actual_region = self.regions[channel]
                stats['decisions'].append(dict(trigger=trigger, channel=channel,
                    primary_channel=primary_channel, position=[q.x, q.y],
                    estimated_net_gain_s=net, actual_measure_cost_s=charged,
                    response=response['measure_result'],
                    actual_radius_m=actual_region.enclosing_disk().radius if actual_region.vertices else None,
                    **value))
        finally:
            self._sharing = False


def run_joint_search(client, *, problem=3, max_actions=20000, max_active_probes=6, joint_config=None):
    if problem not in (3, 'q3'):
        raise ValueError('Geometric joint observation is Q3-only')
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError('max_actions must be an integer >=2')
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError('max_active_probes must be an integer in 0..30')
    return GeometricJointSearch(client, max_actions, max_active_probes, joint_config).run()
