"""Finite action abstraction of Q3, driven by a genuinely trainable policy.

The controller uses only legal history and conservative geometry. Its history
compression is not claimed to be a sufficient Bayesian belief state. Discovery
uses certified cover-scan macro actions; source actions are individual active
measurements or certified removals. The neural policy controls their interleaving.
"""

from dataclasses import asdict, dataclass, field
import math
import hashlib
import json
import time

from simulator_client.state import Position
from strategies.efficient import EfficientSearch
from strategies.search import SearchResult, _StopSearch


ALGORITHM_VERSIONS = {"v1": "q3-candidate-ppo-v1", "v2": "q3-candidate-ppo-v2",
                      "v3": "q3-joint-scan-ppo-v3"}
ALGORITHM_VERSION = ALGORITHM_VERSIONS["v2"]
FEATURE_DIMS = {"v1": 24, "v2": 44, "v3": 60}
FEATURE_DIM = FEATURE_DIMS["v2"]
CONTEXT_DIM = 12
GEOMETRY_FEATURE_NAMES = (
    "major_span", "minor_span", "elongation", "axis_cos2", "axis_sin2",
    "radial_span", "transverse_span", "max_vertex_distance", "reception_margin",
    "guaranteed_reception", "center_distance", "angular_strip_ratio",
    "posterior_trace_ratio_proxy", "crossing_angle_sin", "inside_outer_polygon",
    "guaranteed_near", "offset_major", "offset_minor", "box_fill_ratio",
    "posterior_area_ratio_proxy")
SCAN_FEATURE_NAMES = ("single_channel_scan", "channel_id", "channel_detected",
    "point_pending_fraction", "channel_pending_fraction", "current_point_pending_fraction",
    "same_position", "scan_focus_match", "channel_negative_cover_fraction",
    "pending_cover_0", "pending_cover_1", "pending_cover_2", "pending_cover_3",
    "pending_cover_4", "pending_cover_5", "pending_cover_6")


def feature_schema(version):
    """Semantic identity, separate from source-code provenance."""
    if version not in FEATURE_DIMS:
        raise ValueError("unsupported feature version")
    description = dict(version=version, algorithm=ALGORITHM_VERSIONS[version],
                       feature_dim=FEATURE_DIMS[version], context_dim=CONTEXT_DIM,
                       base_features=("v3-single-measure-cover-cost-and-seven-point-normalizer" if version == "v3"
                                      else "3315abf-v1-prefix-unchanged"),
                       appended_features=(GEOMETRY_FEATURE_NAMES + SCAN_FEATURE_NAMES if version == "v3"
                                          else GEOMETRY_FEATURE_NAMES if version == "v2" else ()),
                       cover_geometry=("selected_channel_geometry" if version == "v3"
                                       else "mean_over_detected_uncleared_sources"),
                       action_semantics=("v3-single-channel-cover-no-fixed-origin-scan" if version == "v3"
                                         else "3315abf-candidate-actions-unchanged"))
    description["sha256"] = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()
    return description


def polygon_shape(vertices):
    """Cheap support geometry. Vertex scatter selects axes, not a posterior law."""
    count = len(vertices)
    cx = sum(p[0] for p in vertices) / count
    cy = sum(p[1] for p in vertices) / count
    xx = sum((x - cx) ** 2 for x, y in vertices) / count
    yy = sum((y - cy) ** 2 for x, y in vertices) / count
    xy = sum((x - cx) * (y - cy) for x, y in vertices) / count
    angle = 0.5 * math.atan2(2 * xy, xx - yy)
    ux, uy = math.cos(angle), math.sin(angle)
    along = [(x - cx) * ux + (y - cy) * uy for x, y in vertices]
    across = [-(x - cx) * uy + (y - cy) * ux for x, y in vertices]
    return dict(center=(cx, cy), axis=(ux, uy), xx=xx, xy=xy, yy=yy,
                major=max(along) - min(along), minor=max(across) - min(across))


def geometry_features(region, position, shape, first_bearing):
    """Twenty legal geometry features; only maximum-distance tests certify.

    Shrinkage is a rank-one linear Gaussian *proxy* around the vertex centroid.
    Vertex scatter is not a posterior covariance. Center-coincident probes get
    no invented zero-noise information gain: use unchanged covariance (ratio 1).
    """
    x, y = position.x, position.y
    cx, cy = shape["center"]
    ux, uy = shape["axis"]
    major, minor = shape["major"], shape["minor"]
    dx, dy = cx - x, cy - y
    distance = math.hypot(dx, dy)
    ex, ey = ((dx / distance, dy / distance) if distance > 1e-8 else (ux, uy))
    nx, ny = -ey, ex
    radial = [(vx - cx) * ex + (vy - cy) * ey for vx, vy in region.vertices]
    transverse = [(vx - cx) * nx + (vy - cy) * ny for vx, vy in region.vertices]
    radial_span, transverse_span = max(radial) - min(radial), max(transverse) - min(transverse)
    maximum = max(math.hypot(vx - x, vy - y) for vx, vy in region.vertices)
    strip = 2 * distance * math.tan(math.radians(region.error_deg))
    strip_ratio = min(1.0, strip / max(transverse_span, 1e-9)) if distance > 1e-8 else 1.0
    xx, xy, yy = shape["xx"], shape["xy"], shape["yy"]
    variance = max(0.0, nx * nx * xx + 2 * nx * ny * xy + ny * ny * yy)
    noise = (distance * math.tan(math.radians(region.error_deg))) ** 2 / 3
    denominator = variance + noise
    trace_ratio = area_ratio = 1.0
    if distance > 1e-8 and denominator > 1e-12 and xx + yy > 1e-12:
        snx, sny = xx * nx + xy * ny, xy * nx + yy * ny
        trace_ratio = max(0.0, min(1.0, 1 - (snx * snx + sny * sny) / denominator / (xx + yy)))
        area_ratio = math.sqrt(max(0.0, min(1.0, noise / denominator)))
    first = math.radians(first_bearing)
    crossing = abs(ex * math.sin(first) - ey * math.cos(first)) if distance > 1e-8 else 0.0
    area = shape["area"] if "area" in shape else region.area
    return [major / 3600, minor / 3600, (major - minor) / max(major + minor, 1e-9),
            ux * ux - uy * uy, 2 * ux * uy, radial_span / 3600,
            transverse_span / 3600, maximum / 3600,
            max(-4.0, (1000 - maximum) / 1000), float(maximum <= 1000),
            distance / 3600, strip_ratio, trace_ratio, crossing,
            float(region.contains((x, y))), float(maximum <= 5),
            (dx * ux + dy * uy) / 1800, (-dx * uy + dy * ux) / 1800,
            min(1.0, area / max(major * minor, 1e-9)), area_ratio]


@dataclass(frozen=True)
class Candidate:
    kind: str
    channel: int | None
    point: Position
    option: int = 0
    radius: float = 0.0


@dataclass
class RLResult(SearchResult):
    learning: dict = field(default_factory=dict)


class DeepRLSearch(EfficientSearch):
    def __init__(self, client, policy, *, max_actions=20000,
                 max_decisions=256, max_active_probes=6, recorder=None,
                 action_deadline_epoch=None, feature_version="v2"):
        if feature_version == "v3" and type(self) is DeepRLSearch:
            raise ValueError("v3 requires JointScanRLSearch; use the public run_rl_search factory")
        if not callable(policy):
            raise ValueError("policy must be callable")
        for name, value, lower in (("max_actions", max_actions, 2),
                                   ("max_decisions", max_decisions, 0),
                                   ("max_active_probes", max_active_probes, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        super().__init__(client, max_actions, max_active_probes, None)
        self.variant = "deep_rl"
        self.feature_version = feature_version
        self.schema = feature_schema(feature_version)
        self.report = RLResult(**asdict(self.report))
        self.report.variant = self.variant
        self.report.learning = dict(
            algorithm=ALGORITHM_VERSIONS[feature_version], decisions=0, candidate_count_sum=0,
            action_counts={}, fallback_counts={}, fallback_actions=0,
            fallback_virtual_time_s=0.0, inference_wall_time_s=0.0,
            initial_scan_virtual_time_s=0.0, feature_dim=FEATURE_DIMS[feature_version],
            feature_schema=self.schema, feature_wall_time_s=0.0,
            context_dim=CONTEXT_DIM, max_decisions=max_decisions,
            learned_scope=["cover_point", "source_channel", "probe_point", "clear_order"],
            fixed_scope=["initial_origin_scan", "channel_order_inside_cover_scan",
                         "conservative_clear_geometry", "bounded_optical_fallback"],
        )
        self.policy = policy
        self.recorder = recorder
        self.max_decisions = max_decisions
        self.probe_counts = {}
        self.focus = None
        self.action_deadline_epoch = action_deadline_epoch
        self._shape_cache = {}
        self._area_cache = {}
        self._geometry_cache = {}

    def _region_area(self, channel):
        region = self.regions[channel]
        cached = self._area_cache.get(channel)
        if cached is None or cached[0] is not region.vertices:
            cached = (region.vertices, region.area)
            self._area_cache[channel] = cached
        return cached[1]

    def _geometric_features(self, channel, point):
        region = self.regions.get(channel)
        if region is None or not region.observations or not region.vertices:
            return [0.0] * len(GEOMETRY_FEATURE_NAMES)
        cached = self._shape_cache.get(channel)
        if cached is None or cached[0] is not region.vertices:
            cached = (region.vertices, polygon_shape(region.vertices))
            cached[1]["area"] = self._region_area(channel)
            self._shape_cache[channel] = cached
        bearing = self.first_bearings[channel]
        features = self._geometry_cache.get(channel)
        if (features is None or features[0] is not region.vertices
                or features[1] != bearing or features[2] != region.error_deg):
            features = (region.vertices, bearing, region.error_deg, {})
            self._geometry_cache[channel] = features
        by_point = features[3]
        if point not in by_point:
            by_point[point] = tuple(geometry_features(region, point, cached[1], bearing))
        return by_point[point]

    def _check_budget(self, action, position, channel):
        if self.action_deadline_epoch is not None and time.time() >= self.action_deadline_epoch:
            raise _StopSearch("training_deadline")
        return super()._check_budget(action, position, channel)

    def _source_candidates(self, channel):
        if channel in self.near_points:
            return [Candidate("clear", channel, self.near_points[channel])]
        region = self.regions.get(channel)
        if region is None or not region.vertices:
            return []
        disk = region.enclosing_disk()
        center = Position.coerce(disk.center)
        if disk.radius <= 19.9:
            # Exactly the safe edge-clear disk already used by the baseline.
            current = self.client.state.position
            distance = current.distance_to(center)
            scale = min(1.0, max(0.0, 19.9 - disk.radius) / distance) if distance else 0
            point = Position(center.x + scale * (current.x - center.x),
                             center.y + scale * (current.y - center.y))
            return [Candidate("clear", channel, point, radius=disk.radius)]
        if self.probe_counts.get(channel, 0) >= self.max_active_probes:
            return [Candidate("fallback", channel, center, radius=disk.radius)]
        angle = math.radians(self.first_bearings[channel])
        radius = min(180.0, max(25.0, disk.radius * 0.5))
        current = self.client.state.position
        proposals = [center,
                     Position(center.x - radius * math.sin(angle), center.y + radius * math.cos(angle)),
                     Position(center.x + radius * math.sin(angle), center.y - radius * math.cos(angle)),
                     Position((center.x + current.x) / 2, (center.y + current.y) / 2),
                     current]
        # Canonical teacher probe stays available even after previous centers.
        baseline = super()._next_probe(channel, self.probe_counts.get(channel, 0))
        if baseline is not None:
            proposals.insert(0, baseline)
        seen = set(self.observed_positions.get(channel, set()))
        candidates = []
        for option, point in enumerate(proposals):
            key = (round(point.x, 6), round(point.y, 6))
            if key not in seen:
                seen.add(key)
                candidates.append(Candidate("probe", channel, point, option, disk.radius))
        return candidates or [Candidate("fallback", channel, center, radius=disk.radius)]

    def _candidates(self, remaining):
        candidates = [Candidate("cover", None, point) for point in remaining]
        for channel in sorted(self.detected - self.cleared - self.blocked):
            candidates.extend(self._source_candidates(channel))
        return candidates

    def _teacher(self, candidates, remaining):
        # Committing to the current source reproduces efficient's resolve macro
        # at primitive resolution. This label is NOT fed into the actor input.
        if self.focus in self.detected - self.cleared - self.blocked:
            indices = [i for i, c in enumerate(candidates) if c.channel == self.focus]
            if indices:
                return indices[0]
        task = super()._next_task(remaining)
        if task is not None:
            kind, channel, point = task
            for index, candidate in enumerate(candidates):
                if ((kind == "cover" and candidate.kind == "cover" and candidate.point == point)
                        or (kind == "source" and candidate.channel == channel)):
                    return index
        return 0

    def _features(self, candidates, remaining):
        current = self.client.state.position
        unresolved = self.detected - self.cleared
        targets = [(c, self._target(c)) for c in sorted(unresolved)]
        targets = [(c, p) for c, p in targets if p is not None]
        areas = {c: self._region_area(c) for c in {a.channel for a in candidates} if c in self.regions}
        point_data = {}
        context = [current.x / 1800, current.y / 1800,
                   self.client.state.current_channel / 20,
                   len(self.detected) / 20, len(self.cleared) / 16,
                   len(unresolved) / 16, len(remaining) / 6,
                   self.report.coverage_points_visited / 7,
                   min(self.client.state.virtual_time_s / 10000, 10),
                   self.report.learning["decisions"] / max(1, self.max_decisions),
                   len(self.blocked) / 16, float(self.focus is not None)]
        features = []
        for candidate in candidates:
            channel, point = candidate.channel, candidate.point
            region = self.regions.get(channel)
            if point not in point_data:
                target_distances = [(c, point.distance_to(p)) for c, p in targets]
                cover_distances = [point.distance_to(p) for p in remaining if p != point]
                nearest_cover = min((point.distance_to(p) for p in remaining), default=0)
                point_data[point] = (current.distance_to(point), target_distances,
                                     cover_distances, nearest_cover,
                                     sum(d < 400 for _, d in target_distances))
            distance, target_distances, cover_distances, nearest_cover, density = point_data[point]
            nearest = min([d for c, d in target_distances if c != channel] + cover_distances, default=0.0)
            bearing = math.radians(self.first_bearings.get(channel, 0))
            row = [float(candidate.kind == kind) for kind in ("cover", "probe", "clear", "fallback")]
            row += [point.x / 1800, point.y / 1800,
                    (point.x - current.x) / 3600, (point.y - current.y) / 3600,
                    distance / 3600, candidate.radius / 1800,
                    math.log1p(areas.get(channel, 0)) / 18,
                    len(region.observations) / 10 if region else 0,
                    len(region.no_signal_positions) / 10 if region else 0,
                    self.probe_counts.get(channel, 0) / self.max_active_probes,
                    float(channel == self.client.state.current_channel),
                    float(channel is not None and channel == self.focus),
                    math.sin(bearing), math.cos(bearing), candidate.option / 6,
                    nearest / 3600,
                    density / 16, nearest_cover / 3600,
                    float(channel in self.near_points),
                    (distance / 5 + (6 * (20 - len(self.cleared)) if candidate.kind == "cover" else 6)) / 1000]
            if self.feature_version in {"v2", "v3"}:
                if candidate.kind == "cover" and self.feature_version == "v2":
                    geometry = [self._geometric_features(c, point) for c, _ in targets]
                    row += ([sum(column) / len(geometry) for column in zip(*geometry)]
                            if geometry else [0.0] * len(GEOMETRY_FEATURE_NAMES))
                else:
                    row += self._geometric_features(channel, point)
            assert len(row) == (24 if self.feature_version == "v1" else 44)
            features.append(row)
        return features, context

    def _fallback(self, channel, reason):
        metrics = self.report.learning
        metrics["fallback_counts"][reason] = metrics["fallback_counts"].get(reason, 0) + 1
        before_actions, before_time = self.actions, self.client.state.virtual_time_s
        try:
            return super()._resolve(channel)
        finally:
            metrics["fallback_actions"] += self.actions - before_actions
            metrics["fallback_virtual_time_s"] += self.client.state.virtual_time_s - before_time

    def _execute_candidate(self, candidate, remaining):
        channel = candidate.channel
        if candidate.kind == "cover":
            self._scan(candidate.point)
            remaining.remove(candidate.point)
            self.focus = None
        elif candidate.kind == "clear":
            phase = "near_clear" if channel in self.near_points else "rl_certified_clear"
            if not self._clear(candidate.point, channel, phase):
                self.blocked.add(channel)
            self.focus = None
        elif candidate.kind == "probe":
            self._perform("measure", candidate.point, channel, "rl_active_localization")
            self.probe_counts[channel] = self.probe_counts.get(channel, 0) + 1
            self.focus = channel
        else:
            if not self._fallback(channel, "source_probe_limit"):
                self.blocked.add(channel)
            self.focus = None

    def _finish_with_baseline(self, remaining):
        metrics = self.report.learning
        metrics["fallback_counts"]["decision_limit"] = 1
        before_actions, before_time = self.actions, self.client.state.virtual_time_s
        try:
            task = super()._next_task(remaining)
            while task is not None:
                kind, channel, point = task
                if kind == "cover":
                    self._scan(point)
                    remaining.remove(point)
                elif not super()._resolve(channel):
                    self.blocked.add(channel)
                if not remaining:
                    self.report.coverage_complete = True
                task = super()._next_task(remaining)
        finally:
            metrics["fallback_actions"] += self.actions - before_actions
            metrics["fallback_virtual_time_s"] += self.client.state.virtual_time_s - before_time

    def _execute_plan(self):
        self._scan(self.points[0])
        self.report.learning["initial_scan_virtual_time_s"] = self.client.state.virtual_time_s
        remaining = list(self.points[1:])
        while True:
            if not remaining:
                self.report.coverage_complete = True
            candidates = self._candidates(remaining)
            if not candidates:
                return
            if self.report.learning["decisions"] >= self.max_decisions:
                self._finish_with_baseline(remaining)
                return
            feature_started = time.perf_counter()
            features, context = self._features(candidates, remaining)
            self.report.learning["feature_wall_time_s"] += time.perf_counter() - feature_started
            teacher = self._teacher(candidates, remaining)
            started = time.perf_counter()
            selection = self.policy(features, context, teacher)
            self.report.learning["inference_wall_time_s"] += time.perf_counter() - started
            index = selection[0] if isinstance(selection, tuple) else selection
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(candidates):
                raise ValueError("policy selected an invalid candidate")
            candidate = candidates[index]
            metrics = self.report.learning
            metrics["decisions"] += 1
            metrics["candidate_count_sum"] += len(candidates)
            metrics["action_counts"][candidate.kind] = metrics["action_counts"].get(candidate.kind, 0) + 1
            before = self.client.state.virtual_time_s
            try:
                self._execute_candidate(candidate, remaining)
            finally:
                # Include the last action even when the sixteen-source stopping
                # certificate raises _StopSearch during its successful clear.
                if self.recorder is not None:
                    self.recorder(features, context, index, teacher, selection,
                                  self.client.state.virtual_time_s - before)
