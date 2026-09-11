"""Q3 from-scratch experiment: certified V0 and mixed-route V1.

Only the public client state and returned observations enter this policy.
Existing strategy choices, fitted parameters and evaluation labels are unused.
Geometry/client primitives are shared with the repository.
"""

from dataclasses import dataclass, field
from functools import lru_cache
import math

from geometry import (HalfPlane, clip_polygon, disk_halfplanes, disk_polygon,
                      distance, minimum_enclosing_circle, point)
from localization import CandidateRegion


COVER = ((0.0, 0.0),) + tuple(
    (1200 * math.cos(k * math.pi / 3), 1200 * math.sin(k * math.pi / 3))
    for k in range(6))
MARGIN = 1e-5


@lru_cache(maxsize=16384)
def covered(negatives, anchor=None):
    """Prove disk/polygon coverage using clipped nearest-site Voronoi cells.

    The arena and optional anchor disk are OUTER polygons. The maximum of
    distance to a cell's nearest site is attained at a vertex. A false return
    means only 'unproved'; a true return covers the entire continuous domain.
    No sampled grid can certify absence. Each caller supplies one channel's
    negative measurements, never another channel's history.
    """
    if not negatives:
        return False
    domain = disk_polygon((0, 0), 1800, 64, outer=True)
    if anchor is not None:
        for hp in disk_halfplanes(anchor, 1000, 64, outer=True):
            domain = clip_polygon(domain, hp)
    for i, a in enumerate(negatives):
        cell = domain
        for j, b in enumerate(negatives):
            if i == j or distance(a, b) < 1e-9:
                continue
            # |x-a| <= |x-b|, expressed without subtracting large squares.
            dx, dy = b[0] - a[0], b[1] - a[1]
            hp = HalfPlane(dx, dy, dx * (a[0] + b[0]) / 2 + dy * (a[1] + b[1]) / 2)
            cell = clip_polygon(cell, hp)
            if not cell:
                break
        if any(distance(a, v) > 1000 - MARGIN for v in cell):
            return False
    return True


def nearest_safe_clear(vertices, robot, radius=20 - MARGIN):
    """Exact candidate enumeration for projection on intersection of disks.

    A nearest feasible point is the robot, a radial projection on one circle,
    or an intersection of two active circle boundaries. Recheck all vertices;
    never substitute a diameter test for an enclosing-circle certificate.
    """
    vertices = tuple(vertices)
    if not vertices:
        raise ValueError("Empty source region")
    circle = minimum_enclosing_circle(vertices)
    if circle.radius > radius:
        return None
    candidates = [robot, circle.center]
    for v in vertices:
        d = distance(v, robot)
        if d > radius:
            candidates.append((v[0] + radius * (robot[0] - v[0]) / d,
                               v[1] + radius * (robot[1] - v[1]) / d))
    for i, a in enumerate(vertices):
        for b in vertices[i + 1:]:
            d = distance(a, b)
            if not 1e-10 < d <= 2 * radius:
                continue
            h = math.sqrt(max(0.0, radius * radius - d * d / 4))
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            for sign in (-1, 1):
                candidates.append((mid[0] + sign * h * (b[1] - a[1]) / d,
                                   mid[1] - sign * h * (b[0] - a[0]) / d))
    feasible = [q for q in candidates if all(distance(q, v) <= radius + 1e-8 for v in vertices)]
    return min(feasible, key=lambda q: distance(q, robot)) if feasible else circle.center


def open_route(items, start):
    """Nearest-neighbour construction and open-ended 2-opt, no return leg."""
    remaining, route, p = list(items), [], start
    while remaining:
        index = min(range(len(remaining)), key=lambda i: (distance(p, remaining[i][2]), i))
        item = remaining.pop(index)
        route.append(item)
        p = item[2]
    for _ in range(30):
        improved = False
        for i in range(len(route) - 1):
            before = start if i == 0 else route[i - 1][2]
            for j in range(i + 1, len(route)):
                old = distance(before, route[i][2])
                new = distance(before, route[j][2])
                if j + 1 < len(route):
                    old += distance(route[j][2], route[j + 1][2])
                    new += distance(route[i][2], route[j + 1][2])
                if new < old - 1e-6:
                    route[i:j + 1] = reversed(route[i:j + 1])
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return route


@dataclass
class Channel:
    status: str = "unknown"
    region: CandidateRegion = field(default_factory=CandidateRegion)
    negatives: list = field(default_factory=list)
    measurements: set = field(default_factory=set)
    local_probes: int = 0


class FreshQ3:
    def __init__(self, client, version="v0", *, dynamic=True, nearest=True):
        if version not in {"v0", "v1"}:
            raise ValueError("version must be v0 or v1")
        self.client, self.version = client, version
        self.dynamic, self.nearest = dynamic, nearest
        self.channels = {c: Channel() for c in range(1, 21)}
        self.history = []
        self.route_decisions = []
        self.visited = set()
        self.last_clear_time = 0.0
        self.certified = False

    @property
    def position(self):
        return point(self.client.state.position)

    def unknown(self):
        # Discovery of 16 exclusive channels uses the public count upper bound.
        if sum(c.status in {"detected", "cleared"} for c in self.channels.values()) == 16:
            for c in self.channels.values():
                if c.status == "unknown":
                    c.status = "absent"
        return [i for i, c in self.channels.items() if c.status == "unknown"]

    def active(self):
        return [i for i, c in self.channels.items() if c.status == "detected"]

    def _record(self, action, p, channel, response, phase):
        result = response["measure_result" if action == "measure" else "clear_result"]
        row = dict(action=action, position=p, channel=channel, result=result,
                   virtual_time_s=self.client.state.virtual_time_s, phase=phase)
        if result == "direction":
            row["bearing_deg"] = response["svd_deg"]
        self.history.append(row)
        return result

    def measure(self, p, channel, phase):
        c, p = self.channels[channel], point(p)
        if p in c.measurements:
            return  # The identical observation conveys no new information.
        response = self.client.measure(p, channel)
        result = self._record("measure", p, channel, response, phase)
        c.measurements.add(p)
        if result == "no_signal":
            c.negatives.append(p)
            if c.status == "unknown" and (
                all(q in c.negatives for q in COVER) or
                (self.dynamic and self.version != "v0" and covered(tuple(c.negatives)))
            ):
                c.status = "absent"
        else:
            c.status = "detected"
            if result == "near":
                for hp in disk_halfplanes(p, 5, 64, outer=True):
                    c.region.vertices = clip_polygon(c.region.vertices, hp)
                c.region._circle = None
            else:
                c.region.observe(p, response["svd_deg"])
            if not c.region.vertices:
                raise RuntimeError(f"Inconsistent observations on channel {channel}")

    def clear(self, p, channel, phase):
        response = self.client.clear(p, channel)
        result = self._record("clear", point(p), channel, response, phase)
        if result != "success":
            raise RuntimeError("A certified clear failed: do not report completion")
        self.channels[channel].status = "cleared"
        self.last_clear_time = self.client.state.virtual_time_s

    def scan(self, p, phase, *, opportunistic=False):
        p = point(p)
        todo = list(self.unknown())
        if opportunistic:
            # Scheduling threshold only; it never constitutes an absence proof.
            todo = [i for i in todo if not self.channels[i].negatives or
                    min(distance(p, n) for n in self.channels[i].negatives) >= 350]
        for i in todo:
            if self.channels[i].status == "unknown":
                self.measure(p, i, phase)
            self.unknown()
        # Shared observation locations still pay separately for every channel.
        for i in self.active():
            c = self.channels[i]
            if p in c.measurements:
                continue
            circle = c.region.enclosing_disk()
            if circle.radius <= 20 - MARGIN:
                continue
            if c.region.observations and min(distance(p, o.position) for o in c.region.observations) < 25:
                continue
            if distance(p, circle.center) <= 1500 + circle.radius:
                self.measure(p, i, phase)

    def locate_clear(self, channel):
        c = self.channels[channel]
        for _ in range(18):
            circle = c.region.enclosing_disk()
            if circle.radius <= 20 - MARGIN:
                q = (nearest_safe_clear(c.region.vertices, self.position)
                     if self.version == "v1" and self.nearest else circle.center)
                self.clear(q, channel, "clear")
                if self.version == "v1":
                    self.scan(self.position, "opportunistic", opportunistic=True)
                return
            q = circle.center
            # After two centre probes use the explicit halving construction.
            # An enclosing bound uses every outer vertex; the contraction is
            # inserted as an additional conservative disk after moving.
            contraction = None
            if c.local_probes >= 2 or q in c.measurements:
                obs = c.region.observations[-1]
                radius = min(1500.0, max(distance(obs.position, v) for v in c.region.vertices))
                step = radius / (2 * math.cos(math.radians(c.region.error_deg)))
                theta = math.radians(obs.bearing_deg)
                q = (obs.position[0] + step * math.cos(theta),
                     obs.position[1] + step * math.sin(theta))
                contraction = step + MARGIN
                if contraction <= 20 - MARGIN:
                    self.clear(q, channel, "halving_clear")
                    if self.version == "v1":
                        self.scan(self.position, "opportunistic", opportunistic=True)
                    return
            self.measure(q, channel, "localize")
            c.local_probes += 1
            if contraction is not None:
                for hp in disk_halfplanes(q, contraction, 64, outer=True):
                    c.region.vertices = clip_polygon(c.region.vertices, hp)
                c.region._circle = None
                if not c.region.vertices:
                    raise RuntimeError("Inconsistent halving bound")
            if self.version == "v1":
                self.scan(self.position, "opportunistic", opportunistic=True)
        raise RuntimeError("Localizer exceeded its safety budget")

    def needed_anchors(self):
        unknown = self.unknown()
        if not unknown:
            return []
        answer = []
        for k, p in enumerate(COVER):
            if k in self.visited:
                continue
            needed = False
            for i in unknown:
                negatives = tuple(self.channels[i].negatives)
                if p in negatives:
                    continue
                if not self.dynamic or not covered(negatives, p):
                    needed = True
                    break
            if needed:
                answer.append(("cover", k, p))
        return answer

    def run(self):
        if self.client.state.session == "new":
            self.client.enter()
        if self.version == "v0":
            for k, p in enumerate(COVER):
                if not self.unknown():
                    break
                self.scan(p, "cover")
                self.visited.add(k)
            while self.active():
                i = min(self.active(), key=lambda i: distance(
                    self.position, self.channels[i].region.enclosing_disk().center))
                self.locate_clear(i)
        else:
            self.scan(COVER[0], "cover")
            self.visited.add(0)
            for _ in range(150):
                anchors = self.needed_anchors()
                items = anchors + [("source", i, self.channels[i].region.enclosing_disk().center)
                                   for i in self.active()]
                if not items:
                    break
                route = open_route(items, self.position)
                kind, index, p = route[0]
                self.route_decisions.append(dict(position=self.position, route=route))
                if kind == "cover":
                    self.scan(p, "cover")
                    self.visited.add(index)
                else:
                    self.locate_clear(index)
            else:
                raise RuntimeError("Mixed route exceeded its safety budget")
        self.certified = not self.unknown() and not self.active()
        if not self.certified:
            raise RuntimeError("Missing discovery/clearance certificate")
        self.client.exit()
        return dict(version=self.version, completion_certified=True,
                    channels={i: c.status for i, c in self.channels.items()},
                    action_history=self.history, route_decisions=self.route_decisions,
                    visited_cover_indices=sorted(self.visited),
                    confirmation_tail_s=self.client.state.virtual_time_s - self.last_clear_time)


def run_fresh(client, version="v0", **kwargs):
    return FreshQ3(client, version, **kwargs).run()
