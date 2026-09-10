"""Finite action abstraction of Q3, driven by a genuinely trainable policy.

The controller uses only legal history and conservative geometry. Its history
compression is not claimed to be a sufficient Bayesian belief state. Discovery
uses certified cover-scan macro actions; source actions are individual active
measurements or certified removals. The neural policy controls their interleaving.
"""

from dataclasses import asdict, dataclass, field
import math
import time

from simulator_client.state import Position
from strategies.efficient import EfficientSearch
from strategies.search import SearchResult, _StopSearch


ALGORITHM_VERSION = "q3-candidate-ppo-v1"
FEATURE_DIM = 24
CONTEXT_DIM = 12


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
                 action_deadline_epoch=None):
        if not callable(policy):
            raise ValueError("policy must be callable")
        for name, value, lower in (("max_actions", max_actions, 2),
                                   ("max_decisions", max_decisions, 0),
                                   ("max_active_probes", max_active_probes, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        super().__init__(client, max_actions, max_active_probes, None)
        self.variant = "deep_rl"
        self.report = RLResult(**asdict(self.report))
        self.report.variant = self.variant
        self.report.learning = dict(
            algorithm=ALGORITHM_VERSION, decisions=0, candidate_count_sum=0,
            action_counts={}, fallback_counts={}, fallback_actions=0,
            fallback_virtual_time_s=0.0, inference_wall_time_s=0.0,
            initial_scan_virtual_time_s=0.0, feature_dim=FEATURE_DIM,
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
            others = [p for c, p in targets if c != channel] + [p for p in remaining if p != point]
            nearest = min((point.distance_to(p) for p in others), default=0.0)
            distance = current.distance_to(point)
            bearing = math.radians(self.first_bearings.get(channel, 0))
            row = [float(candidate.kind == kind) for kind in ("cover", "probe", "clear", "fallback")]
            row += [point.x / 1800, point.y / 1800,
                    (point.x - current.x) / 3600, (point.y - current.y) / 3600,
                    distance / 3600, candidate.radius / 1800,
                    math.log1p(region.area if region else 0) / 18,
                    len(region.observations) / 10 if region else 0,
                    len(region.no_signal_positions) / 10 if region else 0,
                    self.probe_counts.get(channel, 0) / self.max_active_probes,
                    float(channel == self.client.state.current_channel),
                    float(channel is not None and channel == self.focus),
                    math.sin(bearing), math.cos(bearing), candidate.option / 6,
                    nearest / 3600,
                    sum(point.distance_to(p) < 400 for _, p in targets) / 16,
                    min((point.distance_to(p) for p in remaining), default=0) / 3600,
                    float(channel in self.near_points),
                    (distance / 5 + (6 * (20 - len(self.cleared)) if candidate.kind == "cover" else 6)) / 1000]
            assert len(row) == FEATURE_DIM
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
            features, context = self._features(candidates, remaining)
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
