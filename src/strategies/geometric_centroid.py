"""Q3 single-variable ablation: safe area-centroid first probe only.

Scheduling representatives stay at the original MEC centres. This is not a
global Euclidean projection or a claim that reduced source error saves travel.
"""
from dataclasses import asdict, dataclass, field
import math

from simulator_client.state import Position

from .efficient import EfficientSearch
from .search import SearchResult


def safe_centroid_point(region):
    """Clamp the MEC-to-area-centroid segment inside the reception-safe set."""
    vertices = region.vertices
    if not vertices:
        return None
    center = region.enclosing_disk().center
    radius = 1000.-1e-6
    def safe(q):
        return max(math.dist(q, v) for v in vertices) <= radius
    if not safe(center):
        return None
    anchor = vertices[0]
    triangles = []
    for a, b in zip(vertices[1:-1], vertices[2:]):
        area2 = abs((a[0]-anchor[0])*(b[1]-anchor[1])-(a[1]-anchor[1])*(b[0]-anchor[0]))
        if area2 > 0:
            triangles.append((area2, tuple((anchor[j]+a[j]+b[j])/3 for j in (0, 1))))
    total = math.fsum(w for w, _ in triangles)
    centroid = (tuple(math.fsum(w*p[j] for w, p in triangles)/total for j in (0, 1))
                if total > 1e-12 else center)
    fraction = 1.
    if not safe(centroid):
        low, high = 0., 1.
        # Intersection of convex disks with this segment is an interval
        # containing t=0, so bisection keeps a directly checked feasible point.
        for _ in range(48):
            t = (low+high)/2
            q = tuple(center[j]+t*(centroid[j]-center[j]) for j in (0, 1))
            if safe(q):
                low = t
            else:
                high = t
        fraction = low
    selected = tuple(center[j]+fraction*(centroid[j]-center[j]) for j in (0, 1))
    if not safe(selected):
        selected, fraction = center, 0.
    return dict(position=selected, raw_centroid=centroid, mec_center=center,
                segment_fraction=fraction, maximum_vertex_distance_m=max(math.dist(selected, v) for v in vertices))


@dataclass
class CentroidProbeResult(SearchResult):
    centroid_first_probes: list = field(default_factory=list)


class GeometricCentroidSearch(EfficientSearch):
    def __init__(self, client, max_actions, max_active_probes):
        super().__init__(client, max_actions, max_active_probes, None)
        self.variant = 'geometric_centroid_first'
        self.report = CentroidProbeResult(**asdict(self.report))
        self.report.variant = self.variant
        self.report.strategy_parameters['first_probe'] = 'safe_MEC_to_area_centroid_segment'
        self.report.strategy_parameters['scheduling_representative'] = 'unchanged_MEC'
        self._first_probe_channels = set()

    def _next_probe(self, channel, index):
        if index == 0 and channel not in self._first_probe_channels:
            self._first_probe_channels.add(channel)
            choice = safe_centroid_point(self.regions[channel])
            if choice is not None:
                q = Position.coerce(choice['position'])
                key = (round(q.x, 6), round(q.y, 6))
                if key not in self.observed_positions.get(channel, set()):
                    self.report.centroid_first_probes.append(dict(channel=channel, **choice))
                    return q
        return super()._next_probe(channel, index)


def run_centroid_search(client, *, problem=3, max_actions=20000, max_active_probes=6):
    if problem not in (3, 'q3'):
        raise ValueError('Safe-centroid first probes are Q3-only')
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError('max_actions must be an integer >=2')
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError('max_active_probes must be an integer in 0..30')
    return GeometricCentroidSearch(client, max_actions, max_active_probes).run()
