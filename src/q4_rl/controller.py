"""Observation-only Q4 joint scan/channel/service policy with certified fallback.

The actor sees only public features. Its finite actions are individual real
measurements and a bounded inherited source resolver. A resolver can contain
real optical misses; these are neither hidden nor changed into successful clears.
The compact directional cover remains a completion certificate, not a forced
scan order. Arbitrary current-position measurements do not earn cover credits.
"""
from dataclasses import dataclass, field
import math
import time

from simulator_client.rules import CHANNELS, MIN_SOURCES, MAX_SOURCES
from simulator_client.state import Position
from strategies.q4_clear_before_probe import Q4ClearBeforeProbe
from strategies.q4_range_pruning import region_distance_lower, RANGE_MARGIN_M
from strategies.search import SearchResult, _StopSearch


FEATURE_SCHEMA_VERSION = "q4-joint-scan-service-v1"
GLOBAL_FEATURE_NAMES = (
    "current_x", "current_y", "elapsed_virtual_fraction", "detected_fraction",
    "cleared_fraction", "unknown_fraction", "ready_fraction",
    "unknown_cover_pairs_fraction", "unfinished_cover_sites_fraction",
    "current_unmeasured_channels_fraction",
)
CANDIDATE_FEATURE_NAMES = (
    "is_measure", "is_service", "relative_x", "relative_y", "distance",
    "minimum_immediate_cost", "same_position", "same_current_channel",
    "known_channel", "ready_channel", "outer_radius", "outer_area",
    "measurement_count", "negative_measurement_count", "site_unknown_fraction",
    "channel_unmeasured_cover_fraction",
)
GLOBAL_DIM = len(GLOBAL_FEATURE_NAMES)
CANDIDATE_DIM = len(CANDIDATE_FEATURE_NAMES)


@dataclass(frozen=True)
class Candidate:
    kind: str
    point: Position
    channel: int
    at_cover: bool = False


@dataclass
class Q4RLResult(SearchResult):
    learning: dict = field(default_factory=dict)


class Q4RLSearch(Q4ClearBeforeProbe):
    """Joint online actor; only the inherited source-service internals are fixed.

    ``policy(global_features, candidate_features) -> index`` receives numeric
    lists, with no source IDs, cases, seeds, truth, or simulator references.
    ``None`` selects the deterministic legal-action heuristic. Budget fallback
    does not create policy decisions, but its whole billed cost is charged to
    the final transition. The raw fallback cost is also reported separately.
    """
    def __init__(self, client, policy=None, *, max_actions=20000,
                 max_active_probes=6, max_decisions=2048, max_expansions=200,
                 allow_current_scan=True, record_transitions=True,
                 action_deadline_epoch=None):
        for name, value, low, high in (
            ("max_actions", max_actions, 2, 1_000_000),
            ("max_active_probes", max_active_probes, 0, 30),
            ("max_decisions", max_decisions, 0, 100_000),
            ("max_expansions", max_expansions, 0, 10000),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid " + name)
        if policy is not None and not callable(policy):
            raise ValueError("policy must be callable or None")
        if type(allow_current_scan) is not bool or type(record_transitions) is not bool:
            raise ValueError("scan and record options must be booleans")
        if action_deadline_epoch is not None and not math.isfinite(action_deadline_epoch):
            raise ValueError("deadline must be finite")
        super().__init__(client, max_actions, max_active_probes,
                         max_expansions=max_expansions, config="center_once")
        self.report = Q4RLResult(**self.report.__dict__)
        self.policy = policy
        self.max_decisions = max_decisions
        self.allow_current_scan = allow_current_scan
        self.record_transitions = record_transitions
        self.action_deadline_epoch = action_deadline_epoch
        self.cover_ledger = {point: set() for point in self.points}
        self.actual_measurements = set()
        self.negative_counts = {c: 0 for c in CHANNELS}
        self.measurement_counts = {c: 0 for c in CHANNELS}
        self._ledger_history_index = 0
        self._run_initial_time = client.state.virtual_time_s
        self.report.learning.update(
            algorithm=FEATURE_SCHEMA_VERSION,
            policy_kind="heuristic" if policy is None else "external_actor",
            feature_schema=dict(version=FEATURE_SCHEMA_VERSION,
                                global_features=list(GLOBAL_FEATURE_NAMES),
                                candidate_features=list(CANDIDATE_FEATURE_NAMES)),
            learned_scope=["initial_position", "scan_position", "scan_channel",
                           "continue_scanning", "source_service_order"],
            fixed_scope=["compact_22_discovery_certificate", "R8_source_resolver",
                         "actual_clear_confirmation", "budget_fallback"],
            decisions=0, action_counts={}, candidate_count_sum=0,
            feature_wall_time_s=0., inference_wall_time_s=0.,
            transitions=[], decision_cost_s=0., fallback_cost_s=0.,
            fallback_actions=0, fallback_reason=None, uncovered_cost_s=0.,
            completion_certificate=None, coverage_ledger={},
        )
        self.report.strategy_parameters.update(
            q4_rl_feature_schema=FEATURE_SCHEMA_VERSION,
            discovery_credits="Only accepted real measurements at exactly certified stations",
            terminal_gate="16 actual clears, or all unknown channels covered and every known source actually cleared",
            service_scope="Inherited R8 finite probe and optical resolver, bounded by global action/time limits",
        )

    def _check_budget(self, action, position, channel):
        if self.action_deadline_epoch is not None and time.time() >= self.action_deadline_epoch:
            raise _StopSearch("training_deadline")
        return super()._check_budget(action, position, channel)

    def _consume_actual_history(self):
        # The R8 interception can perform clear+measure or clear+exception inside
        # one call. Consume the actual action records, not the requested action.
        for item in self.report.action_history[self._ledger_history_index:]:
            if item["action"] != "measure":
                continue
            point = Position.coerce(item["position"])
            channel = item["channel"]
            self.actual_measurements.add((point, channel))
            self.measurement_counts[channel] += 1
            self.negative_counts[channel] += item["result"] == "no_signal"
            if point in self.cover_ledger:
                self.cover_ledger[point].add(channel)
            self.blocked.discard(channel)
        self._ledger_history_index = len(self.report.action_history)

    def _perform(self, action, position, channel, phase):
        try:
            return super()._perform(action, position, channel, phase)
        finally:
            self._consume_actual_history()

    def _unknown(self):
        return set(CHANNELS) - self.detected - self.cleared

    def _pending_unknown(self, point):
        return self._unknown() - self.cover_ledger[point]

    def _refresh_certificate(self):
        unknown = self._unknown()
        absent = {c for c in unknown if all(c in scanned for scanned in self.cover_ledger.values())}
        discovery_done = len(self.detected | self.cleared) == MAX_SOURCES or absent == unknown
        self.report.coverage_complete = absent == unknown
        self.report.coverage_points_visited = sum(not (unknown-scanned)
                                                for scanned in self.cover_ledger.values())
        self.report.learning["coverage_ledger"] = {
            f"{p.x:.17g},{p.y:.17g}": sorted(channels)
            for p, channels in self.cover_ledger.items()}
        self.report.learning["certified_absent_channels"] = sorted(absent)
        self.report.learning["discovery_certified"] = discovery_done
        return discovery_done

    def _terminal_gate(self):
        discovery_done = self._refresh_certificate()
        known = self.detected | self.cleared
        if len(self.cleared) == MAX_SOURCES:
            certificate = "public_16_source_upper_bound_and_16_actual_clears"
        elif (discovery_done and MIN_SOURCES <= len(known) <= MAX_SOURCES
              and self.detected <= self.cleared):
            certificate = "per_unknown_channel_directional_cover_and_all_known_actually_cleared"
        else:
            return False
        self.report.learning["completion_certificate"] = certificate
        self.report.completion_certified_under_model = True
        raise _StopSearch("q4_rl_certified_complete")

    def _can_measure(self, point, channel):
        if channel in self.cleared or self._ready(channel) or (point, channel) in self.actual_measurements:
            return False
        region = self.regions.get(channel)
        if channel in self.detected and region is not None and region.vertices:
            if self._region_distance_lower(point, channel) > 1500. + RANGE_MARGIN_M:
                return False
        return True

    def _region_distance_lower(self, point, channel):
        return region_distance_lower(point, self.regions[channel].vertices)

    def _candidates(self):
        discovery_done = self._refresh_certificate()
        candidates = []
        seen = set()
        for point in self.points:
            for channel in CHANNELS:
                if discovery_done and channel not in self.detected:
                    continue
                if self._can_measure(point, channel):
                    candidates.append(Candidate("measure", point, channel, True))
                    seen.add((point, channel))
        current = self.client.state.position
        if self.allow_current_scan:
            for channel in CHANNELS:
                if discovery_done and channel not in self.detected:
                    continue
                if (current, channel) not in seen and self._can_measure(current, channel):
                    candidates.append(Candidate("measure", current, channel, current in self.cover_ledger))
        for channel in sorted(self.detected - self.cleared - self.blocked):
            target = self._target(channel)
            if target is not None:
                candidates.append(Candidate("service", target, channel))
        return candidates

    def _features(self, candidates):
        current = self.client.state.position
        unknown = self._unknown()
        pending = {p: len(unknown - scanned) for p, scanned in self.cover_ledger.items()}
        station_count = len(self.points)
        # These values depend on a channel or position, not on every candidate
        # pair. Cache within this decision only: a subsequent observation may
        # change either the geometry or the ledger. Feature arithmetic/order
        # remains identical to the original uncached v1 schema.
        channels = {candidate.channel for candidate in candidates}
        ready_by_channel = {c: self._ready(c) for c in channels | (self.detected-self.cleared)}
        channel_geometry = {}
        channel_pending = {}
        for channel in channels:
            region = self.regions.get(channel)
            radius = region.enclosing_disk().radius if region is not None and region.vertices else 1800.
            area = region.area if region is not None and region.vertices else math.pi*1800.**2
            if channel in self.near_points:
                radius, area = 5., math.pi*25.
            channel_geometry[channel] = radius, area
            channel_pending[channel] = sum(channel not in scanned for scanned in self.cover_ledger.values())/station_count
        distances = {point: current.distance_to(point) for point in {candidate.point for candidate in candidates}}
        ready = sum(ready_by_channel[c] for c in self.detected - self.cleared)
        global_features = [current.x/3600., current.y/3600.,
            self.client.state.virtual_time_s/360000., len(self.detected)/20.,
            len(self.cleared)/20., len(unknown)/20., ready/20.,
            sum(pending.values())/(20.*station_count),
            sum(v > 0 for v in pending.values())/station_count,
            sum((current, c) not in self.actual_measurements for c in unknown)/20.]
        rows = []
        for candidate in candidates:
            point, channel = candidate.point, candidate.channel
            radius, area = channel_geometry[channel]
            distance = distances[point]
            cost = distance/5. + 5. + (candidate.kind == "measure" and channel != self.client.state.current_channel)
            rows.append([float(candidate.kind == "measure"), float(candidate.kind == "service"),
                (point.x-current.x)/3600., (point.y-current.y)/3600., distance/3600., cost/1000.,
                float(point == current), float(channel == self.client.state.current_channel),
                float(channel in self.detected), float(ready_by_channel[channel]), radius/1800.,
                area/(math.pi*1800.**2), self.measurement_counts[channel]/50.,
                self.negative_counts[channel]/50., pending.get(point, 0)/20.,
                channel_pending[channel]])
        if not all(math.isfinite(x) for row in [global_features]+rows for x in row):
            raise ValueError("Non-finite public observation features")
        return global_features, rows

    def _heuristic(self, candidates):
        current = self.client.state.position
        fixed_scans = [(i, c) for i, c in enumerate(candidates) if c.kind == "measure" and c.at_cover]
        same_site = [(i, c) for i, c in fixed_scans if c.point == current]
        if same_site:
            return min(same_site, key=lambda pair: (pair[1].channel != self.client.state.current_channel,
                                                   pair[1].channel))[0]
        services = [(i, c) for i, c in enumerate(candidates) if c.kind == "service"]
        ready = [(i, c) for i, c in services if self._ready(c.channel)]
        scans = [(i, c) for i, c in enumerate(candidates) if c.kind == "measure"]
        if ready:
            return min(ready, key=lambda pair: current.distance_to(pair[1].point))[0]
        if self.report.learning["discovery_certified"] and services:
            return min(services, key=lambda pair: current.distance_to(pair[1].point))[0]
        compact = [(i, c) for i, c in services
                   if self.regions.get(c.channel) is not None and self.regions[c.channel].enclosing_disk().radius <= 40.]
        if compact:
            return min(compact, key=lambda pair: current.distance_to(pair[1].point))[0]
        pool = fixed_scans or scans or services
        return min(pool, key=lambda pair: (current.distance_to(pair[1].point),
                                          pair[1].channel != self.client.state.current_channel,
                                          pair[1].channel))[0]

    def _execute_candidate(self, candidate):
        if candidate.kind == "measure":
            self._perform("measure", candidate.point, candidate.channel, "rl_joint_measure")
        elif not self._resolve(candidate.channel):
            self.blocked.add(candidate.channel)

    def _finish_with_baseline(self, reason):
        metrics = self.report.learning
        metrics["fallback_reason"] = reason
        before, before_actions = self.client.state.virtual_time_s, self.actions
        try:
            # Preserve the frozen certified station order, but never invent a
            # measurement for an omitted pair or a cleared unique source.
            for point in self.points:
                if self._refresh_certificate():
                    break
                channels = [c for c in CHANNELS if self._can_measure(point, c)]
                channels.sort(key=lambda c: (c != self.client.state.current_channel, c))
                for channel in channels:
                    self._perform("measure", point, channel, "rl_fallback_coverage")
            self.blocked.clear()
            self._resolve_queue()
            self._terminal_gate()
            self.report.error = "No complete discovery-and-clear certificate after fallback"
            raise _StopSearch("q4_rl_uncertified_or_unresolved")
        finally:
            cost = self.client.state.virtual_time_s-before
            metrics["fallback_cost_s"] += cost
            metrics["fallback_actions"] += self.actions-before_actions
            if metrics["transitions"]:
                metrics["transitions"][-1]["cost_s"] += cost
                metrics["transitions"][-1]["fallback_cost_s"] += cost

    def _execute_plan(self):
        while True:
            self._terminal_gate()
            if self.report.learning["decisions"] >= self.max_decisions:
                self._finish_with_baseline("decision_limit")
            started = time.perf_counter()
            candidates = self._candidates()
            if not candidates:
                self._finish_with_baseline("no_legal_candidates")
            features, rows = self._features(candidates)
            self.report.learning["feature_wall_time_s"] += time.perf_counter()-started
            started = time.perf_counter()
            selected = self._heuristic(candidates) if self.policy is None else self.policy(features, rows)
            self.report.learning["inference_wall_time_s"] += time.perf_counter()-started
            if type(selected) is not int or not 0 <= selected < len(candidates):
                self._finish_with_baseline("invalid_policy_action")
            candidate = candidates[selected]
            metrics = self.report.learning
            metrics["decisions"] += 1
            metrics["candidate_count_sum"] += len(candidates)
            metrics["action_counts"][candidate.kind] = metrics["action_counts"].get(candidate.kind, 0)+1
            before = self.client.state.virtual_time_s
            try:
                self._execute_candidate(candidate)
            finally:
                cost = self.client.state.virtual_time_s-before
                metrics["decision_cost_s"] += cost
                if self.record_transitions:
                    metrics["transitions"].append(dict(global_features=features,
                        candidate_features=rows, action_index=selected, cost_s=cost,
                        fallback_cost_s=0., action_kind=candidate.kind))

    def run(self):
        report = super().run()
        self._refresh_certificate()
        metrics = report.learning
        total = report.virtual_time_s-self._run_initial_time
        accounted = metrics["decision_cost_s"]+metrics["fallback_cost_s"]
        remainder = total-accounted
        # Usually enter/exit are free. Preserve any actual billed terminal tail
        # rather than silently dropping it from the training objective.
        metrics["uncovered_cost_s"] = remainder
        if metrics["transitions"]:
            metrics["transitions"][-1]["cost_s"] += remainder
            metrics["transitions"][-1]["terminal"] = True
        metrics["total_billed_cost_s"] = total
        metrics["training_success"] = bool(report.completion_certified_under_model
                                            and not report.error and not report.exit_error)
        metrics["failed_clear_count"] = sum(
            item.get("result") == "no_target_in_range" for item in report.action_history)
        return report


def run_q4_rl(client, policy=None, *, problem=4, **kwargs):
    if type(problem) is not int or problem != 4:
        raise ValueError("Q4 RL supports problem=4 only")
    return Q4RLSearch(client, policy=policy, **kwargs).run()
