"""Q3 v3 prototype: learned joint coverage-point/channel/clear decisions.

No scan macro, including the initial origin scan, is forced. Every (point,
channel) pair remains a legal candidate until actually measured or its unique
source cleared. Completion relies on this ledger and the original cover proof,
not on neural confidence. v1/v2 controllers and their action semantics stay intact.
"""

import time

from simulator_client.state import Position
from strategies.efficient import EfficientSearch
from .controller import Candidate, DeepRLSearch, FEATURE_DIMS, SCAN_FEATURE_NAMES


class JointScanRLSearch(DeepRLSearch):
    def __init__(self, client, policy, *, feature_version="v3", **kwargs):
        if feature_version != "v3":
            raise ValueError("joint scan controller requires v3 semantics")
        super().__init__(client, policy, feature_version=feature_version, **kwargs)
        self.scan_ledger = {point: set() for point in self.points}
        self.negative_scan_ledger = {point: set() for point in self.points}
        self._pending_cache = {point: set(range(1, 21)) - self.cleared for point in self.points}
        self.scan_focus = None
        self._candidate_cache = {}
        self.report.learning.update(
            learned_scope=["initial_point", "cover_point", "unknown_channel",
                           "scan_interruption", "source_channel", "probe_point", "clear_order"],
            fixed_scope=["seven_point_cover_candidates", "conservative_clear_geometry",
                         "bounded_optical_fallback"],
            scan_measurements=0, scan_site_changes=0, interrupted_scans=0,
            coverage_ledger={}, coverage_pairs_remaining=140)

    def _pending(self, point):
        return self._pending_cache[point]

    def _refresh_coverage(self):
        live = set(range(1, 21)) - self.cleared
        self._pending_cache = {point: live - scanned for point, scanned in self.scan_ledger.items()}
        self.report.coverage_points_visited = sum(not self._pending(p) for p in self.points)
        self.report.coverage_complete = self.report.coverage_points_visited == len(self.points)
        self.report.learning["coverage_pairs_remaining"] = sum(len(self._pending(p)) for p in self.points)
        self.report.learning["coverage_ledger"] = {
            f"{p.x:.12g},{p.y:.12g}": sorted(self.scan_ledger[p]) for p in self.points}
        if self.scan_focus is not None and not self._pending(self.scan_focus):
            self.scan_focus = None

    def _perform(self, action, position, channel, phase):
        position = Position.coerce(position)
        before = self.actions
        response = None
        try:
            response = super()._perform(action, position, channel, phase)
            return response
        finally:
            # Actual accepted measurements count even if submitted as a probe
            # at exactly a cover point. Rounded near-matches never count.
            if self.actions > before:
                if action == "measure" and position in self.scan_ledger:
                    self.scan_ledger[position].add(channel)
                    if response is not None and response.get("measure_result") == "no_signal":
                        self.negative_scan_ledger[position].add(channel)
                self._refresh_coverage()

    def _candidates(self, remaining):
        choices = [Candidate("cover", c, p) for p in remaining for c in sorted(self._pending(p))]
        for channel in sorted(self.detected - self.cleared - self.blocked):
            choices.extend(self._source_candidates(channel))
        return choices

    def _source_candidates(self, channel):
        # In a point/channel scan the robot often stays still for many actions.
        # A different channel's new observation cannot change this source's
        # legal candidates. Include every source/state dependency explicitly.
        region = self.regions.get(channel)
        vertices = region.vertices if region is not None else None
        signature = (self.client.state.position, self.near_points.get(channel),
                     self.probe_counts.get(channel, 0), self.max_active_probes,
                     self.first_bearings.get(channel),
                     frozenset(self.observed_positions.get(channel, ())))
        cached = self._candidate_cache.get(channel)
        if cached is None or cached[0] is not vertices or cached[1] != signature:
            cached = (vertices, signature, tuple(super()._source_candidates(channel)))
            self._candidate_cache[channel] = cached
        return cached[2]

    def _teacher_scan(self, candidates, point):
        pending = self._pending(point)
        current = self.client.state.current_channel
        channel = current if current in pending else min(pending)
        return next(i for i, c in enumerate(candidates)
                    if c.kind == "cover" and c.point == point and c.channel == channel)

    def _teacher(self, candidates, remaining):
        if self.focus in self.detected - self.cleared - self.blocked:
            source = [i for i, c in enumerate(candidates) if c.kind != "cover" and c.channel == self.focus]
            if source:
                return source[0]
        if self.scan_focus is not None and self._pending(self.scan_focus):
            return self._teacher_scan(candidates, self.scan_focus)
        if not any(self.scan_ledger.values()):
            return self._teacher_scan(candidates, self.points[0])
        task = EfficientSearch._next_task(self, remaining)
        if task is not None:
            kind, channel, point = task
            if kind == "cover":
                return self._teacher_scan(candidates, point)
            return next(i for i, c in enumerate(candidates) if c.kind != "cover" and c.channel == channel)
        return 0

    def _features(self, candidates, remaining):
        features, context = super()._features(candidates, remaining)
        current = self.client.state.position
        context[6] = len(remaining) / 7
        current_pending = len(self._pending(current)) if current in self.scan_ledger else 0
        channel_bits = {channel: [float(channel in self._pending(p)) for p in self.points]
                        for channel in {c.channel for c in candidates}}
        channel_negatives = {channel: sum(channel in self.negative_scan_ledger[p] for p in self.points) / 7
                             for channel in channel_bits}
        for row, candidate in zip(features, candidates):
            point, channel = candidate.point, candidate.channel
            # v3 changes these v2-prefix semantics deliberately; cross-action
            # weight initialization is prohibited, not silently claimed equal.
            if candidate.kind == "cover":
                row[23] = (current.distance_to(point) / 5 + 5 +
                           int(channel != self.client.state.current_channel)) / 1000
            point_pending = len(self._pending(point)) if point in self.scan_ledger else 0
            bits = channel_bits[channel]
            extras = [float(candidate.kind == "cover"), channel / 20,
                      float(channel in self.detected), point_pending / 20,
                      sum(bits) / 7, current_pending / 20, float(point == current),
                      float(point == self.scan_focus),
                      channel_negatives[channel]] + bits
            assert len(extras) == len(SCAN_FEATURE_NAMES)
            row += extras
            assert len(row) == FEATURE_DIMS["v3"]
        return features, context

    def _execute_candidate(self, candidate, remaining):
        if candidate.kind == "cover":
            current = self.client.state.position
            metrics = self.report.learning
            if candidate.point != current:
                metrics["scan_site_changes"] += 1
            if (self.scan_focus is not None and self.scan_focus != candidate.point
                    and self._pending(self.scan_focus)):
                metrics["interrupted_scans"] += 1
            self.scan_focus = candidate.point
            self._perform("measure", candidate.point, candidate.channel, "coverage")
            self.blocked.clear()
            self.focus = None
            metrics["scan_measurements"] += 1
        else:
            if self.scan_focus is not None and self._pending(self.scan_focus):
                self.report.learning["interrupted_scans"] += 1
            super()._execute_candidate(candidate, remaining)

    def _scan_pending(self, point):
        pending = self._pending(point)
        current = self.client.state.current_channel
        channels = ([current] if current in pending else []) + sorted(pending - {current})
        for channel in channels:
            self._perform("measure", point, channel, "coverage")
            self.report.learning["scan_measurements"] += 1
        self.blocked.clear()

    def _finish_with_baseline(self, remaining):
        metrics = self.report.learning
        metrics["fallback_counts"]["decision_limit"] = 1
        before_actions, before_time = self.actions, self.client.state.virtual_time_s
        try:
            while True:
                remaining = [p for p in self.points if self._pending(p)]
                if len(remaining) > 6:
                    # The inherited exact coverage order intentionally supports
                    # six sites, because v1/v2 forced a completed origin scan.
                    # A partial v3 initial scan may leave all seven unfinished.
                    current = self.client.state.position
                    self._scan_pending(min(remaining, key=current.distance_to))
                    continue
                task = EfficientSearch._next_task(self, remaining)
                if task is None:
                    break
                kind, channel, point = task
                if kind == "cover":
                    self._scan_pending(point)
                elif not EfficientSearch._resolve(self, channel):
                    self.blocked.add(channel)
        finally:
            metrics["fallback_actions"] += self.actions - before_actions
            metrics["fallback_virtual_time_s"] += self.client.state.virtual_time_s - before_time

    def _execute_plan(self):
        # Every initial observation is now selected by the actor and charged.
        while True:
            self._refresh_coverage()
            remaining = [p for p in self.points if self._pending(p)]
            candidates = self._candidates(remaining)
            if not candidates:
                return
            if self.report.learning["decisions"] >= self.max_decisions:
                self._finish_with_baseline(remaining)
                return
            started = time.perf_counter()
            features, context = self._features(candidates, remaining)
            self.report.learning["feature_wall_time_s"] += time.perf_counter() - started
            teacher = self._teacher(candidates, remaining)
            prepare = getattr(self.policy, "prepare_candidates", None)
            if prepare is not None:
                prepare(candidates, self.regions)
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
                if self.recorder is not None:
                    self.recorder(features, context, index, teacher, selection,
                                  self.client.state.virtual_time_s - before)
