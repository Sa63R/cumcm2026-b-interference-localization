"""Continuous future cover freedom with the original geometric 2-opt route.

The shared oracle certifies a full arena cover, not a good unknown-world route.
There is no compressed-state route solver, new source policy, or learned model.
"""

from collections import Counter
import time

from planning import improve_open_route, nearest_order
from planning.coverage_relocation import (CoverageOracle, CoverageOracleBudget,
                                           feasible_ray_point, project_to_segment)
from .geometric_probe_cost import GeometricProbeCostSearch


def _length(route, current):
    return sum(a.distance_to(b) for a, b in zip((current,)+tuple(route), route))


def _order_for_points(tasks, route):
    """Keep distinct physical tasks even when their representative points tie."""
    pending = list(range(len(tasks)))
    order = []
    for point in route:
        index = next(i for i in pending if tasks[i][2] == point)
        order.append(index)
        pending.remove(index)
    assert not pending
    assert Counter(tasks[i][2] for i in order) == Counter(route)
    return tuple(order)


class GeometricRelocationSearch(GeometricProbeCostSearch):
    def __init__(self, client, max_actions, max_active_probes, joint_config,
                 max_planning_s, enabled):
        super().__init__(client, max_actions, max_active_probes, joint_config,
                         'single', max_planning_s, 'equal')
        self.variant = self.report.variant = 'geometric_probe_relocation'
        self.relocation_enabled = enabled
        self.discovery_stations = []
        self.relocation_log = []
        self.coverage_oracle = CoverageOracle(12000)
        self._last_relocation_action_count = None
        self.report.strategy_parameters.update(
            future_cover_relocation=enabled, relocation_log=self.relocation_log,
            relocation_limits=dict(next_route_covers=2, nearest_known_targets=3,
                route_variants=4, ray_bisection_steps=20, oracle_calls_per_case=12000,
                minimum_proxy_gain_s=1., coverage_margin_m=1e-5),
            relocation_scope='One future site, complete cover re-certified; only completed atomic scans are discovery evidence',
            relocation_route='Original nearest-neighbour plus geometric 2-opt; distance/5 proxy only',
            oracle_source_commit='8aa6206',
            oracle_source_blobs=dict(coverage_relocation='5aeabb27c3faee7d1601b34a69b2f3552737ed30',
                                     disk_cover='524b46065b2570407981c948235e89c384cd65c6'))

    def _scan(self, point):
        # Do not count a future plan or a partially interrupted scan as P.
        # The inherited scan physically measures every uncleared channel.
        super()._scan(point)
        self.discovery_stations.append(point)

    def _next_task(self, remaining):
        if not self.relocation_enabled:
            return super()._next_task(remaining)
        sources = [('source', c, target) for c in sorted(self.detected-self.cleared-self.blocked)
                   if (target := self._target(c)) is not None]
        if not remaining or not sources:
            return super()._next_task(remaining)
        action_count = len(self.report.action_history)
        if action_count == self._last_relocation_action_count:
            return super()._next_task(remaining)
        self._last_relocation_action_count = action_count
        started = time.perf_counter()
        tasks = [('cover', None, p) for p in remaining]+sources
        current = self.client.state.position
        route = improve_open_route(nearest_order([t[2] for t in tasks], start=current), start=current)
        baseline_order = _order_for_points(tasks, route)
        baseline_cost = _length(route, current)/5.
        best_tasks, best_order, best_cost = tasks, baseline_order, baseline_cost
        selected_relocation = None
        candidates, seen = [], set()
        calls_before = self.coverage_oracle.calls
        exhausted = self.coverage_oracle.calls >= self.coverage_oracle.maximum
        if not exhausted:
            try:
                for index in [i for i in baseline_order if i < len(remaining)][:2]:
                    old = tasks[index][2]
                    nearest = sorted(sources, key=lambda t: (old.distance_to(t[2]), t[1]))[:3]
                    targets = [current]
                    for _, _, point in nearest:
                        targets.extend((point, project_to_segment(old, current, point)))
                    fixed = tuple(self.discovery_stations)+tuple(p for i, p in enumerate(remaining) if i != index)
                    route_index = baseline_order.index(index)
                    previous = current if not route_index else tasks[baseline_order[route_index-1]][2]
                    following = tasks[baseline_order[route_index+1]][2] if route_index+1 < len(tasks) else None
                    for target in targets:
                        choice = feasible_ray_point(fixed, old, target, oracle=self.coverage_oracle)
                        key = (index, round(choice.point.x, 6), round(choice.point.y, 6))
                        if key in seen or old.distance_to(choice.point) < .001:
                            continue
                        seen.add(key)
                        old_length = previous.distance_to(old)
                        new_length = previous.distance_to(choice.point)
                        if following is not None:
                            old_length += old.distance_to(following)
                            new_length += choice.point.distance_to(following)
                        gain = (old_length-new_length)/5.
                        if gain > 1.:
                            candidates.append((gain, index, choice))
            except CoverageOracleBudget:
                exhausted = True
        candidates.sort(key=lambda v: (-v[0], v[1], v[2].point.x, v[2].point.y))
        evaluated = []
        for gain, index, choice in candidates[:4]:
            alternate = tasks.copy()
            alternate[index] = ('cover', None, choice.point)
            # Reuse the baseline complete order; a fresh NN construction could
            # make its own unrelated optimization error look like relocation.
            incumbent = tuple(alternate[i][2] for i in baseline_order)
            changed = improve_open_route(incumbent, start=current)
            order, cost = _order_for_points(alternate, changed), _length(changed, current)/5.
            assert cost <= baseline_cost-gain+1e-6
            evaluated.append(dict(site_index=index, position=[choice.point.x, choice.point.y],
                fixed_order_gain_s=gain, proxy_cost_s=cost, coverage_radius_m=choice.coverage_radius_m,
                ray_fraction=choice.fraction))
            if cost < best_cost-1.:
                best_tasks, best_order, best_cost = alternate, order, cost
                selected_relocation = (index, choice)
        selected = best_tasks[best_order[0]]
        record = dict(after_actual_action_count=len(self.report.action_history),
            executed_discovery_stations=[[p.x, p.y] for p in self.discovery_stations],
            remaining_before=[[p.x, p.y] for p in remaining],
            frozen_tasks=[[kind, channel, p.x, p.y] for kind, channel, p in tasks],
            baseline_order=list(baseline_order), selected_order=list(best_order),
            selected_kind=selected[0], selected_channel=selected[1], selected_point=[selected[2].x, selected[2].y],
            baseline_proxy_s=baseline_cost, selected_proxy_s=best_cost,
            candidate_count=len(candidates), evaluated=evaluated,
            oracle_calls=self.coverage_oracle.calls-calls_before, oracle_budget_exhausted=exhausted,
            relocated=selected_relocation is not None)
        if selected_relocation is not None:
            index, choice = selected_relocation
            old = remaining[index]
            remaining[index] = choice.point
            self.report.coverage_points = [[p.x, p.y] for p in self.discovery_stations+remaining]
            record.update(old_position=[old.x, old.y], new_position=[choice.point.x, choice.point.y],
                certified_cover_radius_m=choice.coverage_radius_m,
                remaining_after=[[p.x, p.y] for p in remaining])
        record['planning_s'] = time.perf_counter()-started
        self.relocation_log.append(record)
        return selected


def run_relocation_search(client, *, problem=3, max_actions=20000, max_active_probes=6,
                          joint_config=None, max_planning_s=120., enabled=True):
    if problem not in (3, 'q3'):
        raise ValueError('Geometric cover relocation is Q3-only')
    if type(enabled) is not bool:
        raise ValueError('enabled must be boolean')
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError('max_actions must be an integer >=2')
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError('max_active_probes must be an integer in 0..30')
    return GeometricRelocationSearch(client, max_actions, max_active_probes,
                                     joint_config, max_planning_s, enabled).run()
