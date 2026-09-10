"""Q3 root Monte Carlo macro-task rollout, not a POMCP observation tree.

Only legal observations seed hypothetical worlds. Each candidate shares the
same worlds, executes one macro-task, then the frozen efficient policy finishes
discovery and clearance. Particle probabilities never certify an empty region.
"""

from dataclasses import asdict, dataclass, field
import math
import statistics
import time

from planning.routing import exact_open_route

from .efficient import EfficientSearch
from .search import SearchResult, _StopSearch


@dataclass(frozen=True)
class RolloutConfig:
    particles: int = 2
    candidates: int = 3
    seed: int = 9102026
    min_gain_s: float = 10.0
    uncertainty_penalty: float = 0.5
    max_searches: int = 24
    max_candidate_evaluations: int = 144
    max_planning_s: float = 60.0

    @classmethod
    def parse(cls, values):
        if values is not None and not isinstance(values, dict):
            raise ValueError("rollout_config must be a dictionary")
        try:
            result = cls(**(values or {}))
        except TypeError as error:
            raise ValueError(f"Invalid rollout_config: {error}") from error
        for name, low, high in (("particles", 1, 64), ("candidates", 2, 12),
                                ("seed", 0, 2**63 - 1), ("max_searches", 0, 100),
                                ("max_candidate_evaluations", 0, 10000)):
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}")
        for name, high in (("min_gain_s", 10000), ("uncertainty_penalty", 10),
                           ("max_planning_s", 600)):
            value = getattr(result, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= high):
                raise ValueError(f"{name} must be finite and between 0 and {high}")
        return result


@dataclass
class RolloutResult(SearchResult):
    planning: dict = field(default_factory=dict)


def _task_record(task):
    kind, channel, point = task
    return {"kind": kind, "channel": channel, "position": [point.x, point.y]}


class _Continuation(EfficientSearch):
    """A baseline policy clone; it receives no hidden source geometry."""

    def __init__(self, parent, client, remaining, first_task, deadline):
        super().__init__(client, parent.max_actions, parent.max_active_probes,
                         asdict(parent.config))
        self.regions = {c: region.copy() for c, region in parent.regions.items()}
        self.first_bearings = parent.first_bearings.copy()
        self.near_points = parent.near_points.copy()
        self.observed_positions = {c: points.copy() for c, points in parent.observed_positions.items()}
        self.detected = parent.detected.copy()
        self.cleared = parent.cleared.copy()
        self.blocked = parent.blocked.copy()
        self.actions = parent.actions
        self.report.coverage_points_visited = parent.report.coverage_points_visited
        self.report.coverage_complete = not remaining
        self.remaining = list(remaining)
        self.first_task = first_task
        self.deadline = deadline

    def _check_budget(self, action, position, channel):
        if time.perf_counter() >= self.deadline:
            raise _StopSearch("rollout_compute_budget")
        return super()._check_budget(action, position, channel)

    def _execute_plan(self):
        task = self.first_task
        while task is not None:
            kind, channel, point = task
            if kind == "cover":
                self._scan(point)
                self.remaining.remove(point)
            elif not self._resolve(channel):
                self.blocked.add(channel)
            if not self.remaining:
                self.report.coverage_complete = True
            task = self._next_task(self.remaining)


class RolloutSearch(EfficientSearch):
    def __init__(self, client, max_actions, max_active_probes, config):
        self.rollout_config = RolloutConfig.parse(config)
        # Keep the existing efficient default frozen as both comparator and tail.
        super().__init__(client, max_actions, max_active_probes, None)
        self.variant = "rollout"
        self.report = RolloutResult(**asdict(self.report))
        self.report.variant = "rollout"
        self.report.strategy_parameters = asdict(self.rollout_config)
        self.report.planning = {
            "algorithm": "root_macro_task_monte_carlo_rollout",
            "tail_policy": "efficient_default",
            "belief_model": "approximate_conditional_uniform_q3_prior",
            "searches": 0, "candidate_evaluations": 0,
            "changed_decisions": 0, "planning_wall_time_s": 0.0,
            "fallback_counts": {}, "decisions": [],
        }

    def _candidates(self, remaining, baseline):
        choices = [baseline]
        sources = [("source", c, target)
                   for c in sorted(self.detected - self.cleared - self.blocked)
                   if (target := self._target(c)) is not None]
        sources.sort(key=lambda task: (self.client.state.position.distance_to(task[2]), task[1]))
        covers = ([("cover", None, p) for p in exact_open_route(
            remaining, start=self.client.state.position)] if remaining else [])
        # Always include both useful macro-task families before further options.
        preferred = covers[:1] + sources[:1] + sources[1:] + covers[1:]
        for task in preferred:
            if task not in choices:
                choices.append(task)
            if len(choices) >= self.rollout_config.candidates:
                break
        return choices

    def _fallback(self, reason):
        counts = self.report.planning["fallback_counts"]
        counts[reason] = counts.get(reason, 0) + 1

    def _next_task(self, remaining):
        baseline = super()._next_task(remaining)
        if baseline is None:
            return None
        config, stats = self.rollout_config, self.report.planning
        candidates = self._candidates(remaining, baseline)
        if len(candidates) < 2:
            return baseline
        if stats["searches"] >= config.max_searches:
            self._fallback("search_limit")
            return baseline
        needed = len(candidates) * config.particles
        if stats["candidate_evaluations"] + needed > config.max_candidate_evaluations:
            self._fallback("evaluation_limit")
            return baseline
        time_left = config.max_planning_s - stats["planning_wall_time_s"]
        real_left = getattr(self.client, "remaining_real_time_s", None)
        if real_left is not None:
            time_left = min(time_left, real_left - 5.0)
        if time_left <= 0:
            self._fallback("compute_budget")
            return baseline

        from simulation.q3_branch import make_q3_branch
        from .q3_belief import BeliefSamplingError, BeliefSamplingTimeout, sample_worlds

        started = time.perf_counter()
        deadline = started + time_left
        stats["searches"] += 1
        record = {"search_index": stats["searches"],
                  "after_actions": self.actions,
                  "virtual_time_s": self.client.state.virtual_time_s,
                  "baseline": _task_record(baseline),
                  "selected": _task_record(baseline), "candidates": [],
                  "status": "baseline"}
        stats["decisions"].append(record)
        try:
            try:
                worlds = sample_worlds(self.report.action_history, count=config.particles,
                                       seed=config.seed + 1000003 * stats["searches"],
                                       deadline=deadline)
            except BeliefSamplingTimeout as error:
                self._fallback("compute_budget")
                record.update(status="compute_budget", detail=str(error))
                return baseline
            except BeliefSamplingError as error:
                self._fallback("belief_sampling")
                record.update(status="belief_sampling", detail=str(error))
                return baseline
            costs = []
            for task in candidates:
                values = []
                for world in worlds:
                    if time.perf_counter() >= deadline:
                        self._fallback("compute_budget")
                        record["status"] = "compute_budget"
                        return baseline
                    try:
                        client = make_q3_branch(world, self.client.state, self.report.action_history)
                    except ValueError as error:
                        self._fallback("inconsistent_hypothesis")
                        record.update(status="inconsistent_hypothesis", detail=str(error))
                        return baseline
                    with client:
                        branch = _Continuation(self, client, remaining, task, deadline)
                        result = branch.run()
                    stats["candidate_evaluations"] += 1
                    if (not result.completion_certified_under_model
                            or result.error or result.exit_error):
                        # No short/incomplete trace can win by paying less cost.
                        self._fallback("incomplete_continuation")
                        record.update(status="incomplete_continuation",
                                      detail=result.completion_reason)
                        return baseline
                    values.append(result.virtual_time_s - self.client.state.virtual_time_s)
                costs.append(values)
                record["candidates"].append(dict(_task_record(task),
                    remaining_costs_s=values, mean_remaining_cost_s=statistics.fmean(values)))
            selected, best_margin = 0, 0.0
            for index in range(1, len(candidates)):
                gains = [base - alternative for base, alternative in zip(costs[0], costs[index])]
                mean_gain = statistics.fmean(gains)
                se = statistics.stdev(gains) / math.sqrt(len(gains)) if len(gains) > 1 else 0.0
                margin = mean_gain - config.min_gain_s - config.uncertainty_penalty * se
                record["candidates"][index].update(paired_mean_gain_s=mean_gain,
                    paired_standard_error_s=se, selection_margin_s=margin)
                if margin > best_margin:
                    selected, best_margin = index, margin
            if selected:
                stats["changed_decisions"] += 1
                record.update(status="changed", selected=_task_record(candidates[selected]))
            # The standard-error adjustment is a heuristic, not a confidence
            # guarantee (small particles, approximate model, multiple choices).
            return candidates[selected]
        finally:
            elapsed = time.perf_counter() - started
            stats["planning_wall_time_s"] += elapsed
            record["planning_wall_time_s"] = elapsed
