"""Q3 joint-observation ablation changing only certified-clear waypoints."""
from dataclasses import asdict, dataclass, field

from planning import improve_open_route, nearest_order
from planning.disk_waypoint import two_leg_disk_waypoint
from simulator_client.state import Position

from .geometric_joint import GeometricJointSearch, JointObservationResult
from .search import _Search


@dataclass
class RouteClearResult(JointObservationResult):
    route_clear_decisions: list = field(default_factory=list)


class GeometricRouteClearSearch(GeometricJointSearch):
    def __init__(self, client, max_actions, max_active_probes, joint_config, enabled=True):
        if type(enabled) is not bool:
            raise ValueError('route_clear must be boolean')
        super().__init__(client, max_actions, max_active_probes, joint_config)
        self.route_clear = enabled
        self.variant = 'geometric_route_clear'
        self.report = RouteClearResult(**asdict(self.report))
        self.report.variant = self.variant
        self.report.strategy_parameters['route_clear'] = enabled
        self._remaining_cover = ()

    def _next_task(self, remaining):
        self._remaining_cover = tuple(remaining)
        return super()._next_task(remaining)

    def _successor(self, channel, legacy_point):
        # A sixteenth successful clear immediately terminates the base policy.
        if len(self.cleared) == 15:
            return None
        tasks = [('cover', None, point) for point in self._remaining_cover]
        tasks += [('source', other, target)
                  for other in sorted(self.detected-self.cleared-self.blocked-{channel})
                  if (target := self._target(other)) is not None]
        if not tasks:
            return None
        route = improve_open_route(nearest_order([t[2] for t in tasks], start=legacy_point),
                                   start=legacy_point)
        return next(t for t in tasks if t[2] == route[0])

    def _clear(self, position, channel, phase):
        if not self.route_clear or phase != 'certified_clear':
            return super()._clear(position, channel, phase)
        circle = self.regions[channel].enclosing_disk()
        center, current = Position.coerce(circle.center), self.client.state.position
        safe_radius = max(0., 19.9-circle.radius)
        distance = center.distance_to(current)
        fraction = min(1., safe_radius/distance) if distance else 0.
        legacy = Position(center.x+fraction*(current.x-center.x),
                          center.y+fraction*(current.y-center.y))
        successor = self._successor(channel, legacy)
        if successor is None or safe_radius <= 0:
            return super()._clear(position, channel, phase)
        kind, next_channel, q = successor
        choice = two_leg_disk_waypoint((current.x, current.y), (q.x, q.y), circle.center, safe_radius)
        chosen = Position.coerce(choice['position'])
        # Preserve the original 19.9 m certificate. A numerical residual is
        # used only to reject an inaccurate optimizer, never to justify safety.
        use = (choice['gap_m'] <= 1e-5
               and chosen.distance_to(center) <= safe_radius+1e-10
               and choice['local_length_m'] <= choice['legacy_local_length_m']+1e-9)
        if not use:
            chosen = legacy
        self.report.route_clear_decisions.append(dict(channel=channel,
            start=[current.x, current.y], center=list(circle.center),
            region_radius_m=circle.radius, successor_kind=kind, successor_channel=next_channel,
            successor=[q.x, q.y], accepted=use, actual_position=[chosen.x, chosen.y], **choice))
        # Bypass only EfficientSearch's current-point projection. _Search._clear
        # still dispatches self._perform, retaining joint sharing and all rules.
        return _Search._clear(self, chosen, channel, phase)


def run_route_clear_search(client, *, problem=3, max_actions=20000, max_active_probes=6,
                           joint_config=None, route_clear=True):
    if problem not in (3, 'q3'):
        raise ValueError('Route-aware certified clearing is Q3-only')
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError('max_actions must be an integer >=2')
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError('max_active_probes must be an integer in 0..30')
    return GeometricRouteClearSearch(client, max_actions, max_active_probes, joint_config, route_clear).run()
