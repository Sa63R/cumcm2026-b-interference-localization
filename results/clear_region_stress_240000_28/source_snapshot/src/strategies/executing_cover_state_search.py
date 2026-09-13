"""Relocate only the next real cover action, retaining the old finite order."""

from dataclasses import asdict
import time

from planning.coverage_relocation import CoverageOracleBudget, feasible_ray_point, project_to_segment
from planning.state_route import RouteTask, solve_state_route
from .inferred_state_search import InferredSilenceSearch
from .relocating_state_search import RelocatingStateSearch


class ExecutingCoverStateSearch(RelocatingStateSearch):
    def __init__(self, client, max_actions, max_active_probes, config, enabled):
        super().__init__(client, max_actions, max_active_probes, config, enabled)
        self.report.strategy_parameters.update(
            relocation_scope="Only baseline next-cover point may change, then it is immediately scanned; the finite task order remains fixed",
            relocation_limits={"next_route_covers": 1, "nearest_known_targets": 3,
                               "route_variants": 0, "ray_bisection_steps": 20,
                               "oracle_calls_per_case": 12000, "minimum_proxy_gain_s": 1.},
            relocation_policy="executing_cover_fixed_order")

    def _next_task(self, remaining):
        if not self.relocation_enabled:
            return InferredSilenceSearch._next_task(self, remaining)
        began = time.perf_counter()
        self.remaining_covers = remaining
        sources = [("source", c, point) for c in sorted(self.detected-self.cleared-self.blocked)
                   if (point := self._target(c)) is not None]
        covers = [("cover", None, p) for p in remaining]
        if self.state_config.stop_discovery_at_16 and len(self.detected | self.cleared) == 16:
            covers = []
        tasks = covers+sources
        if not tasks:
            return None
        current = self.client.state.position
        background = max(0, 20-len(self.cleared)-len(sources))
        finite = [RouteTask(p, kind == "source", 5. if kind == "source" else 6.*background)
                  for kind, _, p in tasks]
        budget = max(0, min(self.state_config.max_expansions,
                           self.state_config.max_total_expansions-self.total_expansions))
        result = solve_state_route(finite, current, scan_source_s=self.state_config.scan_source_s,
                                   max_expansions=budget)
        self.total_expansions += result.expanded
        selected = tasks[result.order[0]]
        baseline_selected = selected
        best_cost, best_choice = result.cost_s, None
        calls_before = self.coverage_oracle.calls
        candidates, budget_exhausted, evaluated = [], False, []
        index = result.order[0]
        if selected[0] == "cover" and sources and self.coverage_oracle.calls < self.coverage_oracle.maximum:
            old = selected[2]
            source_targets = sorted(sources, key=lambda t: old.distance_to(t[2]))[:3]
            targets = [current]
            for _, _, point in source_targets:
                targets.extend((point, project_to_segment(old, current, point)))
            fixed = tuple(self.discovery_stations)+tuple(p for i, p in enumerate(remaining) if i != index)
            following = tasks[result.order[1]][2] if len(tasks) > 1 else None
            seen = set()
            try:
                for target in targets:
                    choice = feasible_ray_point(fixed, old, target, oracle=self.coverage_oracle)
                    key = (round(choice.point.x, 6), round(choice.point.y, 6))
                    if key in seen or old.distance_to(choice.point) < .001:
                        continue
                    seen.add(key)
                    old_length, new_length = current.distance_to(old), current.distance_to(choice.point)
                    if following is not None:
                        old_length += old.distance_to(following)
                        new_length += choice.point.distance_to(following)
                    gain = (old_length-new_length)/5.
                    if gain > 1.:
                        candidates.append((gain, choice))
            except CoverageOracleBudget:
                budget_exhausted = True
            candidates.sort(key=lambda item: (-item[0], item[1].point.x, item[1].point.y))
            for gain, choice in candidates:
                evaluated.append({"site_index": index, "position": [choice.point.x, choice.point.y],
                                  "fixed_order_gain_s": gain, "proxy_cost_s": result.cost_s-gain,
                                  "coverage_radius_m": choice.coverage_radius_m,
                                  "ray_fraction": choice.fraction, "route_expanded": 0})
            if candidates:
                gain, best_choice = candidates[0]
                best_cost = result.cost_s-gain
        record = {"after_actual_action_count": len(self.report.action_history),
                  "executed_discovery_stations": [[p.x, p.y] for p in self.discovery_stations],
                  "remaining_before": [[p.x, p.y] for p in remaining],
                  "baseline_selected_kind": baseline_selected[0],
                  "baseline_selected_channel": baseline_selected[1],
                  "baseline_selected_point": [baseline_selected[2].x, baseline_selected[2].y],
                  "baseline_order": list(result.order), "selected_order": list(result.order),
                  "baseline_proxy_s": result.cost_s, "selected_proxy_s": best_cost,
                  "candidate_count": len(candidates), "evaluated": evaluated,
                  "oracle_calls": self.coverage_oracle.calls-calls_before,
                  "oracle_budget_exhausted": budget_exhausted, "relocated": best_choice is not None}
        result_data = {k: v for k, v in asdict(result).items() if k != "order"}
        if best_choice is not None:
            old = remaining[index]
            remaining[index] = best_choice.point
            selected = ("cover", None, best_choice.point)
            self.report.coverage_points = [[p.x, p.y] for p in self.discovery_stations+remaining]
            record.update(old_position=[old.x, old.y], new_position=[best_choice.point.x, best_choice.point.y],
                          certified_cover_radius_m=best_choice.coverage_radius_m,
                          remaining_after=[[p.x, p.y] for p in remaining])
            # Every open order contains at most two edges incident on this
            # task. Moving it by delta changes any order cost by <=2delta/v.
            # Transfer the OLD finite lower bound, never its exact flag.
            result_data.update(lower_bound_s=max(0., result.lower_bound_s-2*old.distance_to(best_choice.point)/5.),
                               exact=False)
        self.relocation_log.append(record)
        result_data.update(cost_s=best_cost, runtime_s=time.perf_counter()-began)
        self.search_log.append({"task_count": len(tasks), "source_count": len(sources), "cover_count": len(covers),
                               "selected_kind": selected[0], "selected_channel": selected[1],
                               "selected_point": [selected[2].x, selected[2].y], **result_data,
                               "model_gap_s": max(0., best_cost-result_data["lower_bound_s"]),
                               "bound_scope": "Frozen-point task model only; moved-point lower bound transfers by 2delta/5; executed order is held fixed"})
        return selected


def run_executing_cover_state_search(client, *, problem=3, max_actions=10000,
                                     max_active_probes=6, config=None, enabled=True):
    if problem != 3:
        raise ValueError("Executing-cover relocation is Q3 only")
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError("max_actions must be integer >=2")
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError("max_active_probes must be in [0,30]")
    return ExecutingCoverStateSearch(client, max_actions, max_active_probes, config, enabled).run()
