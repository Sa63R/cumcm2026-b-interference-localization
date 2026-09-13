"""Independent state-search candidate with one-at-a-time future cover relocation."""

from dataclasses import asdict
import time

from planning.coverage_relocation import CoverageOracle, CoverageOracleBudget, feasible_ray_point, project_to_segment
from planning.state_route import RouteTask, solve_state_route
from .inferred_state_search import InferredSilenceSearch


class RelocatingStateSearch(InferredSilenceSearch):
    def __init__(self, client, max_actions, max_active_probes, config, enabled):
        super().__init__(client, max_actions, max_active_probes, config, True)
        if self.state_config.replace_coverage:
            raise ValueError("One-site relocation is isolated from after-clear station deletion")
        self.relocation_enabled = enabled
        self.relocation_log = []
        self.coverage_oracle = CoverageOracle(12000)
        self.report.strategy_parameters.update({
            "future_cover_relocation": enabled, "relocation_log": self.relocation_log,
            "relocation_limits": {"next_route_covers": 2, "nearest_known_targets": 3,
                                  "route_variants": 4, "ray_bisection_steps": 20,
                                  "oracle_calls_per_case": 12000, "minimum_proxy_gain_s": 1.},
            "relocation_scope": "At most one future site changed per real task; complete disk cover re-certified; no planned site is actual discovery evidence",
        })

    def _next_task(self, remaining):
        if not self.relocation_enabled:
            return super()._next_task(remaining)
        began = time.perf_counter()
        self.remaining_covers = remaining
        sources = [("source", c, point) for c in sorted(self.detected-self.cleared-self.blocked)
                   if (point := self._target(c)) is not None]
        covers = [("cover", None, p) for p in remaining]
        if self.state_config.stop_discovery_at_16 and len(self.detected | self.cleared) == 16:
            covers = []
        tasks = covers + sources
        if not tasks:
            return None
        current = self.client.state.position
        background = max(0, 20-len(self.cleared)-len(sources))

        def solve(items):
            finite = [RouteTask(p, kind == "source", 5. if kind == "source" else 6.*background)
                      for kind, _, p in items]
            budget = max(0, min(self.state_config.max_expansions,
                               self.state_config.max_total_expansions-self.total_expansions))
            result = solve_state_route(finite, current, scan_source_s=self.state_config.scan_source_s,
                                       max_expansions=budget)
            self.total_expansions += result.expanded
            return result

        baseline = solve(tasks)
        best_result, best_tasks, best_cost, best_order = baseline, tasks, baseline.cost_s, baseline.order
        selected_relocation = None
        calls_before = self.coverage_oracle.calls
        candidates, budget_exhausted = [], False
        if covers and sources and self.coverage_oracle.calls < self.coverage_oracle.maximum:
            # P consists ONLY of fully executed scans. Each still-unknown
            # channel was physically queried at every such station; inference
            # for already known channels is not part of that certificate.
            indices = [i for i in baseline.order if i < len(covers)][:2]
            seen = set()
            try:
                for index in indices:
                    old = tasks[index][2]
                    source_targets = sorted(sources, key=lambda t: old.distance_to(t[2]))[:3]
                    targets = [current]
                    for _, channel, point in source_targets:
                        targets.extend((point, project_to_segment(old, current, point)))
                    fixed = tuple(self.discovery_stations) + tuple(p for i, p in enumerate(remaining) if i != index)
                    order_index = baseline.order.index(index)
                    previous = current if order_index == 0 else tasks[baseline.order[order_index-1]][2]
                    following = tasks[baseline.order[order_index+1]][2] if order_index+1 < len(tasks) else None
                    for target in targets:
                        choice = feasible_ray_point(fixed, old, target, oracle=self.coverage_oracle)
                        key = (index, round(choice.point.x, 6), round(choice.point.y, 6))
                        if key in seen or old.distance_to(choice.point) < .001:
                            continue
                        seen.add(key)
                        old_length, new_length = previous.distance_to(old), previous.distance_to(choice.point)
                        if following is not None:
                            old_length += old.distance_to(following)
                            new_length += choice.point.distance_to(following)
                        gain = (old_length-new_length)/5
                        if gain > 1.:
                            candidates.append((gain, index, choice))
            except CoverageOracleBudget:
                budget_exhausted = True
        # A bounded short list is selected by improvement of the already
        # feasible frozen order. This screening has no global optimality claim.
        candidates.sort(key=lambda item: (-item[0], item[1], item[2].point.x, item[2].point.y))
        evaluated = []
        for gain, index, choice in candidates[:4]:
            alternate = tasks.copy()
            alternate[index] = ("cover", None, choice.point)
            result = solve(alternate)
            # The baseline order remains a feasible incumbent in the new
            # geometry, even if bounded A*'s own initial heuristic misses it.
            fixed_order_cost = baseline.cost_s-gain
            cost, order = (fixed_order_cost, baseline.order) if fixed_order_cost < result.cost_s else (result.cost_s, result.order)
            evaluated.append({"site_index": index, "position": [choice.point.x, choice.point.y],
                              "fixed_order_gain_s": gain, "proxy_cost_s": cost,
                              "coverage_radius_m": choice.coverage_radius_m,
                              "ray_fraction": choice.fraction, "route_expanded": result.expanded})
            if cost < best_cost-1.:
                best_result, best_tasks, best_cost, best_order = result, alternate, cost, order
                selected_relocation = (index, choice)
        selected = best_tasks[best_order[0]]
        record = {"after_actual_action_count": len(self.report.action_history),
                  "executed_discovery_stations": [[p.x, p.y] for p in self.discovery_stations],
                  "remaining_before": [[p.x, p.y] for p in remaining],
                  "baseline_proxy_s": baseline.cost_s, "selected_proxy_s": best_cost,
                  "candidate_count": len(candidates), "evaluated": evaluated,
                  "oracle_calls": self.coverage_oracle.calls-calls_before,
                  "oracle_budget_exhausted": budget_exhausted, "relocated": selected_relocation is not None}
        if selected_relocation is not None:
            index, choice = selected_relocation
            old = remaining[index]
            # Commit only this one future point; the complete plan was tested
            # with all other executed/future stations unchanged. No scan count
            # or observation evidence is incremented by this planning step.
            remaining[index] = choice.point
            self.report.coverage_points = [[p.x, p.y] for p in self.discovery_stations + remaining]
            record.update(old_position=[old.x, old.y], new_position=[choice.point.x, choice.point.y],
                          certified_cover_radius_m=choice.coverage_radius_m,
                          remaining_after=[[p.x, p.y] for p in remaining])
        self.relocation_log.append(record)
        result_data = {k: v for k, v in asdict(best_result).items() if k != "order"}
        result_data.update(cost_s=best_cost, runtime_s=time.perf_counter()-began)
        self.search_log.append({"task_count": len(tasks), "source_count": len(sources), "cover_count": len(covers),
                               "selected_kind": selected[0], "selected_channel": selected[1],
                               "selected_point": [selected[2].x, selected[2].y], **result_data,
                               "model_gap_s": max(0., best_cost-best_result.lower_bound_s),
                               "bound_scope": "Selected frozen-point variant only; not continuous neighborhoods or Q3"})
        return selected


def run_relocating_state_search(client, *, problem=3, max_actions=10000, max_active_probes=6,
                                config=None, enabled=True):
    if problem != 3:
        raise ValueError("Future cover relocation is Q3 only")
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError("max_actions must be integer >=2")
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError("max_active_probes must be in [0,30]")
    return RelocatingStateSearch(client, max_actions, max_active_probes, config, enabled).run()
