"""Experimental adaptive cover actions; frozen v1 controller stays unchanged."""
import math
import time

from simulator_client.state import Position
from .controller import Q4RLSearch, Candidate, GLOBAL_FEATURE_NAMES, CANDIDATE_FEATURE_NAMES
from .adaptive_cover import AdaptiveCoverLedger, position_key, replay_adaptive_artifact


FEATURE_SCHEMA_VERSION = "q4-joint-adaptive-obligations-v2"


class Q4AdaptiveRLSearch(Q4RLSearch):
    def __init__(self, client, policy=None, *, adaptive_cover=True, replacement_actions=True,
                 max_cover_cells=20000, substitute_radius_m=250., cover_episode_wall_s=10.,
                 cover_decision_wall_s=.5, max_cover_searches=64, **kwargs):
        if type(adaptive_cover) is not bool or type(replacement_actions) is not bool:
            raise ValueError("Adaptive cover switches must be booleans")
        if not math.isfinite(substitute_radius_m) or not 0 <= substitute_radius_m <= 1000:
            raise ValueError("Invalid substitution distance budget")
        super().__init__(client, policy=policy, **kwargs)
        self.adaptive_enabled = adaptive_cover
        self.replacement_actions = replacement_actions
        self.substitute_radius_m = substitute_radius_m
        self.adaptive = AdaptiveCoverLedger(self.points, max_cells=max_cover_cells,
            episode_wall_s=cover_episode_wall_s, decision_wall_s=cover_decision_wall_s,
            max_episode_searches=max_cover_searches) if adaptive_cover else None
        self._adaptive_history_index = 0
        self._replacement_map = {}
        self._proposal_cache = {}
        if adaptive_cover:
            self.report.learning["algorithm"] = FEATURE_SCHEMA_VERSION
            self.report.learning["feature_schema"] = dict(version=FEATURE_SCHEMA_VERSION,
                global_features=list(GLOBAL_FEATURE_NAMES), candidate_features=list(CANDIDATE_FEATURE_NAMES),
                changed_semantics="pending fractions count exact-certified residual obligations; candidate sites may be adaptive")
            self.report.learning["fixed_scope"][0] = "exact-certified_residual_cover_with_fixed_fallback"
            self.report.learning["learned_scope"].append("certified_alternative_measurement_point")
            self.report.learning.update(adaptive_same_position_measurements=0,
                adaptive_service_opportunity_measurements=0, adaptive_shifted_measurements=0)
            self.report.strategy_parameters.update(adaptive_cover=True,
                adaptive_feature_schema=FEATURE_SCHEMA_VERSION,
                replacement_actions=replacement_actions,
                exact_arithmetic="rational verifier on submitted IEEE coordinates",
                discovery_credits="Only actual negative measurements; planned points prove residual feasibility only")

    def _consume_actual_history(self):
        super()._consume_actual_history()
        if not getattr(self, "adaptive_enabled", False):
            return
        for index in range(self._adaptive_history_index, len(self.report.action_history)):
            item = self.report.action_history[index]
            c, point = item["channel"], Position.coerce(item["position"])
            self.adaptive.observe(index, item["action"], point, c, item["result"])
            if item["action"] != "measure" or item["result"] != "no_signal" or c not in self.adaptive.unknown():
                continue
            exact = position_key(point)
            proposed = self._replacement_map.get((point, c))
            if proposed is not None:
                self.adaptive.remove(c, proposed)
            # An arbitrary current-position measurement may provide a useful
            # replacement even when it was selected without a prospective proof.
            if exact not in self.adaptive.fixed:
                nearby = sorted(self.adaptive.pending[c], key=lambda p: math.dist(p, exact))[:1]
                for fixed in nearby:
                    if math.dist(fixed, exact) <= self.substitute_radius_m:
                        self.adaptive.remove(c, fixed)
        self._adaptive_history_index = len(self.report.action_history)

    def _refresh_certificate(self):
        if not getattr(self, "adaptive_enabled", False):
            return super()._refresh_certificate()
        absent = self.adaptive.absent()
        unknown = self.adaptive.unknown()
        done = len(self.detected | self.cleared) == 16 or absent == unknown
        self.report.coverage_complete = absent == unknown
        self.report.coverage_points_visited = sum(
            not any(position_key(p) in self.adaptive.pending[c] for c in unknown) for p in self.points)
        self.report.learning.update(certified_absent_channels=sorted(absent), discovery_certified=done,
            adaptive_deleted_pairs=len(self.adaptive.events),
            coverage_ledger={f"{p.x:.17g},{p.y:.17g}": sorted(cs) for p, cs in self.cover_ledger.items()})
        return done

    def _can_measure(self, point, channel):
        if getattr(self, "adaptive_enabled", False) and position_key(point) in self.adaptive.fixed:
            unknown = self.adaptive.unknown()
            key = position_key(point)
            if channel in unknown and key not in self.adaptive.pending[channel]:
                return False
        return super()._can_measure(point, channel)

    def _replacement(self, channel, fixed, current):
        # Cache by exact public state, not by a case or seed. Prospective points
        # do not enter the ledger until their real measurement is accepted.
        key = (channel, fixed, current, frozenset(self.adaptive.negatives[channel]),
               frozenset(self.adaptive.pending[channel]))
        if key in self._proposal_cache:
            return self._proposal_cache[key]
        distance = current.distance_to(fixed)
        result = None
        if distance > 1e-9:
            # A current position that is close can replace a visit entirely;
            # smaller shifts provide a bounded, falsifiable first action family.
            offsets = ([distance] if distance <= self.substitute_radius_m else [])+[20., 5., 1.]
            for offset in dict.fromkeys(min(value, distance) for value in offsets):
                point = Position(fixed.x+(current.x-fixed.x)*offset/distance,
                                 fixed.y+(current.y-fixed.y)*offset/distance)
                if (point, channel) in self.actual_measurements or point == fixed:
                    continue
                if self.adaptive.proposal(channel, fixed, [point]) is not None:
                    result = point
                    break
        self._proposal_cache[key] = result
        return result

    def _candidates(self):
        if self.adaptive_enabled:
            self.adaptive.begin_decision()
        candidates = super()._candidates()
        self._replacement_map = {}
        if not self.adaptive_enabled or not self.replacement_actions:
            return candidates
        current = self.client.state.position
        # One nearest pending station per decision limits geometry overhead;
        # every channel there may choose its certified alternate point.
        unknown = self.adaptive.unknown()
        active = [p for p in self.points if any(position_key(p) in self.adaptive.pending[c] for c in unknown)]
        if not active:
            return candidates
        fixed = min(active, key=current.distance_to)
        seen = {(c.point, c.channel) for c in candidates if c.kind == "measure"}
        for channel in sorted(unknown):
            if position_key(fixed) not in self.adaptive.pending[channel]:
                continue
            point = self._replacement(channel, fixed, current)
            if point is not None:
                self._replacement_map[(point, channel)] = fixed
                if (point, channel) not in seen:
                    candidates.append(Candidate("measure", point, channel, False))
                    seen.add((point, channel))
        return candidates

    def _features(self, candidates):
        global_features, rows = super()._features(candidates)
        if not self.adaptive_enabled:
            return global_features, rows
        unknown = self.adaptive.unknown()
        count = len(self.points)
        pending = {p: sum(p in self.adaptive.pending[c] for c in unknown) for p in self.adaptive.fixed}
        global_features[7] = sum(pending.values())/(20.*count)
        global_features[8] = sum(v > 0 for v in pending.values())/count
        for candidate, row in zip(candidates, rows):
            key = position_key(candidate.point)
            fixed = self._replacement_map.get((candidate.point, candidate.channel))
            row[14] = pending.get(position_key(fixed) if fixed is not None else key, 0)/20.
            row[15] = len(self.adaptive.pending[candidate.channel])/count if candidate.channel in unknown else 0.
        return global_features, rows

    def _heuristic(self, candidates):
        selected = super()._heuristic(candidates)
        if not self.adaptive_enabled or not self.replacement_actions:
            return selected
        old = candidates[selected]
        if old.kind == "measure":
            for index, candidate in enumerate(candidates):
                if (candidate.channel == old.channel and
                        self._replacement_map.get((candidate.point, candidate.channel)) == old.point):
                    return index
        return selected

    def _execute_candidate(self, candidate):
        if self.adaptive_enabled and (candidate.point, candidate.channel) in self._replacement_map:
            name = ("adaptive_same_position_measurements" if candidate.point == self.client.state.position
                    else "adaptive_shifted_measurements")
            self.report.learning[name] += 1
            if candidate.point == self.client.state.position:
                # Find why this position was first reached. Subsequent channels
                # scanned at a deliberately shifted cover site are not counted
                # as naturally occurring source-service opportunities.
                origin = None
                previous = Position(0., 0.)
                for item in self.report.action_history:
                    point = Position.coerce(item["position"])
                    if point != previous:
                        origin = item["phase"]
                    previous = point
                if origin is not None and origin not in {"rl_joint_measure", "rl_fallback_coverage", "coverage"}:
                    self.report.learning["adaptive_service_opportunity_measurements"] += 1
        return super()._execute_candidate(candidate)

    def run(self):
        report = super().run()
        if self.adaptive_enabled:
            artifact = self.adaptive.artifact()
            began = time.perf_counter()
            audit = replay_adaptive_artifact(report.action_history, artifact)
            elapsed = time.perf_counter()-began
            artifact["cost"]["replay_wall_s"] = elapsed
            artifact["cost"]["total_certificate_wall_s"] = (
                artifact["cost"]["initial_certificate_wall_s"]+artifact["cost"]["search_wall_s"]+elapsed)
            report.program_runtime_s += elapsed
            report.learning["adaptive_cover"] = artifact
            report.learning["adaptive_replay_audit"] = audit
            if report.completion_certified_under_model and not audit["complete"]:
                report.completion_certified_under_model = False
                report.learning["training_success"] = False
                report.error = "Independent adaptive completion replay failed"
        return report


def run_q4_adaptive_rl(client, policy=None, *, problem=4, **kwargs):
    if type(problem) is not int or problem != 4:
        raise ValueError("Adaptive Q4 only")
    return Q4AdaptiveRLSearch(client, policy=policy, **kwargs).run()
