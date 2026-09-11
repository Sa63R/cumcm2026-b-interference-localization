"""Public-observation semi-MDP over fixed-station scan groups and R8 service.

Each group executes real measurements one by one. Planned group members never
earn discovery credit. Grouping deliberately removes within-group policy
interleaving and standalone known-channel/current-position probes; it is an
action representation experiment, not learned per-channel joint control.
"""
from dataclasses import dataclass
import math

from simulator_client.state import Position
from strategies.search import _StopSearch
from . import controller as macro


FEATURE_SCHEMA_VERSION = "q4-scan-bundle-service-g1-v1"
BUNDLE_SEMANTICS_VERSION = "unknown-fixed-station-current-first-v1"
GLOBAL_FEATURE_NAMES = macro.GLOBAL_FEATURE_NAMES
CANDIDATE_FEATURE_NAMES = (
    "is_scan_bundle", "is_service", "relative_x", "relative_y", "distance",
    "scan_group_cost_upper", "same_position", "current_channel_in_group",
    "mean_known_channel", "mean_ready_channel", "mean_outer_radius",
    "mean_outer_area", "mean_measurement_count", "mean_negative_measurement_count",
    "site_unknown_fraction", "mean_channel_unmeasured_cover_fraction",
    "bundle_remaining_fraction", "service_first_request_cost_estimate",
)
GLOBAL_DIM, CANDIDATE_DIM = len(GLOBAL_FEATURE_NAMES), len(CANDIDATE_FEATURE_NAMES)


def feature_schema():
    return dict(version=FEATURE_SCHEMA_VERSION,
        global_features=list(GLOBAL_FEATURE_NAMES),
        candidate_features=list(CANDIDATE_FEATURE_NAMES),
        bundle_semantics_version=BUNDLE_SEMANTICS_VERSION)


@dataclass(frozen=True)
class BundleCandidate:
    kind: str
    point: Position
    channels: tuple[int, ...]

    @property
    def channel(self):
        """Single service channel; representative only for deterministic ties."""
        return self.channels[0]


class Q4BundleSearch(macro.Q4RLSearch):
    def __init__(self, client, policy=None, *, max_actions=20000,
                 max_active_probes=6, max_decisions=128, max_expansions=200,
                 record_transitions=True, action_deadline_epoch=None):
        super().__init__(client, policy=policy, max_actions=max_actions,
            max_active_probes=max_active_probes, max_decisions=max_decisions,
            max_expansions=max_expansions, allow_current_scan=False,
            record_transitions=record_transitions,
            action_deadline_epoch=action_deadline_epoch)
        self.report.learning.update(algorithm=FEATURE_SCHEMA_VERSION,
            feature_schema=feature_schema(), bundle_events=[],
            learned_scope=["initial_station", "next_station", "station_or_source_service",
                           "source_service_order"],
            fixed_scope=["unknown_channel_bundle_current_first_then_ascending",
                         "compact_22_discovery_certificate", "R8_source_resolver",
                         "actual_clear_confirmation", "budget_fallback"])
        self.report.strategy_parameters.update(q4_rl_feature_schema=FEATURE_SCHEMA_VERSION,
            bundle_semantics_version=BUNDLE_SEMANTICS_VERSION,
            bundle_scope="All remaining unknown channels at one certified station; current tuned channel first, then ascending; real replies after each request",
            bundle_stop="Discovery certificate stops remaining scans; actual clears still required for completion",
            bundle_cost="Actual accumulated billed duration; full-group cost feature is an upper bound before possible early stopping",
            bundle_control_restriction="No within-group actor interleaving or standalone known-channel/arbitrary-current-point probes",
            service_cost_feature="First request travel plus five seconds only; not the total resolver cost")

    def _ordered_channels(self, channels):
        current = self.client.state.current_channel
        return tuple(sorted(channels, key=lambda channel: (channel != current, channel)))

    def _candidates(self):
        discovery_done = self._refresh_certificate()
        candidates = []
        if not discovery_done:
            for point in self.points:
                channels = self._ordered_channels(self._pending_unknown(point))
                if channels:
                    candidates.append(BundleCandidate("scan_bundle", point, channels))
        for channel in sorted(self.detected - self.cleared - self.blocked):
            target = self._target(channel)
            if target is not None:
                candidates.append(BundleCandidate("service", target, (channel,)))
        return candidates

    def _features(self, candidates):
        # Reuse the frozen public geometry/count arithmetic. Expansion here is
        # numeric summarization, not a request, observation or coverage credit.
        expanded = []
        spans = []
        for candidate in candidates:
            start = len(expanded)
            expanded.extend(macro.Candidate(
                "measure" if candidate.kind == "scan_bundle" else "service",
                candidate.point, channel, candidate.kind == "scan_bundle")
                for channel in candidate.channels)
            spans.append((start, len(expanded)))
        global_features, individual = super()._features(expanded)
        rows = []
        for candidate, (start, end) in zip(candidates, spans):
            if end == start:
                raise ValueError("Empty bundle candidate")
            group = individual[start:end]
            row = list(group[0])
            is_bundle = candidate.kind == "scan_bundle"
            row[0], row[1] = float(is_bundle), float(not is_bundle)
            row[7] = float(self.client.state.current_channel in candidate.channels)
            for index in range(8, 16):
                row[index] = sum(part[index] for part in group)/len(group)
            distance = self.client.state.position.distance_to(candidate.point)
            movement_upper = math.ceil(distance/5.*1e6)/1e6
            switches = len(group)-int(self.client.state.current_channel in candidate.channels)
            row[5] = (movement_upper+5.*len(group)+switches)/1000. if is_bundle else 0.
            row.extend([len(group)/20. if is_bundle else 0.,
                        0. if is_bundle else group[0][5]])
            rows.append(row)
        if len(global_features) != GLOBAL_DIM or any(len(row) != CANDIDATE_DIM for row in rows):
            raise ValueError("Bundle feature schema mismatch")
        if not all(math.isfinite(value) for row in [global_features]+rows for value in row):
            raise ValueError("Non-finite bundle features")
        return global_features, rows

    def _heuristic(self, candidates):
        current = self.client.state.position
        scans = [(i, c) for i, c in enumerate(candidates) if c.kind == "scan_bundle"]
        same_site = [(i, c) for i, c in scans if c.point == current]
        if same_site:
            return same_site[0][0]
        services = [(i, c) for i, c in enumerate(candidates) if c.kind == "service"]
        ready = [(i, c) for i, c in services if self._ready(c.channel)]
        if ready:
            return min(ready, key=lambda pair: current.distance_to(pair[1].point))[0]
        if self.report.learning["discovery_certified"] and services:
            return min(services, key=lambda pair: current.distance_to(pair[1].point))[0]
        compact = [(i, c) for i, c in services if self.regions.get(c.channel) is not None
                   and self.regions[c.channel].enclosing_disk().radius <= 40.]
        if compact:
            return min(compact, key=lambda pair: current.distance_to(pair[1].point))[0]
        return min(scans or services, key=lambda pair: (
            current.distance_to(pair[1].point), pair[1].channel != self.client.state.current_channel,
            pair[1].channel))[0]

    def _execute_candidate(self, candidate):
        if candidate.kind == "service":
            if len(candidate.channels) != 1 or candidate.channel not in self.detected-self.cleared-self.blocked:
                raise _StopSearch("invalid_bundle_service")
            if not self._resolve(candidate.channel):
                self.blocked.add(candidate.channel)
            return
        if (candidate.kind != "scan_bundle" or candidate.point not in self.cover_ledger
                or not candidate.channels
                or candidate.channels != self._ordered_channels(self._pending_unknown(candidate.point))):
            raise _StopSearch("invalid_scan_bundle")
        event = dict(position=[candidate.point.x, candidate.point.y],
            planned_channels=list(candidate.channels), measured_channels=[],
            start_actual_action_count=len(self.report.action_history),
            end_actual_action_count=len(self.report.action_history), cost_s=0., status="running")
        self.report.learning["bundle_events"].append(event)
        before = self.client.state.virtual_time_s
        try:
            for channel in candidate.channels:
                if self._refresh_certificate():
                    event["status"] = "discovery_certified"
                    break
                # Check every physical request against the current real ledger;
                # never credit the entire planned group in advance.
                if channel not in self._pending_unknown(candidate.point):
                    continue
                self._perform("measure", candidate.point, channel, "rl_scan_bundle_measure")
                event["measured_channels"].append(channel)
                if self._refresh_certificate():
                    event["status"] = "discovery_certified"
                    break
            else:
                event["status"] = "group_finished"
        except BaseException as error:
            event.update(status="interrupted", interruption_type=type(error).__name__,
                         interruption_reason=str(error))
            raise
        finally:
            event["end_actual_action_count"] = len(self.report.action_history)
            event["cost_s"] = self.client.state.virtual_time_s-before
            event["known_source_count_after"] = len(self.detected | self.cleared)
            event["cleared_count_after"] = len(self.cleared)


def run_q4_bundle(client, policy=None, *, problem=4, **kwargs):
    if type(problem) is not int or problem != 4:
        raise ValueError("Q4 bundle controller supports problem=4 only")
    return Q4BundleSearch(client, policy=policy, **kwargs).run()
