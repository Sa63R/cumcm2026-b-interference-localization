"""R12 with exact bounded point-task routing only after truncated A* search."""
from dataclasses import asdict
import time

from planning.chain_route import ChainSource, solve_chain_route
from planning.chain_route_exact import refine_truncated_route
from .q4_joint_continuation import Q4JointContinuation
from .search import _StopSearch


class Q4ExactFrontier(Q4JointContinuation):
    def __init__(self, client, max_actions, max_active_probes, *, max_expansions,
                 config='truncated_dag8'):
        if config != 'truncated_dag8':
            raise ValueError('Unknown exact-frontier config')
        super().__init__(client,max_actions,max_active_probes,max_expansions=max_expansions)
        self.exact_frontier_log=[]
        self.dp_total_states=self.dp_total_arcs=0
        self.report.strategy_parameters.update(exact_frontier_config=config,
            exact_frontier_log=self.exact_frontier_log,
            exact_frontier_limits=dict(max_sources=8,max_covers=22,max_states_per_call=60000,
                                       improvement_tolerance_s=1e-9),
            exact_frontier_scope='Only truncated finite point-task routing; preserve original A* work budget and all R12 physical policies')

    def _refine_route(self, original, covers, sources, channels, background):
        selected, diagnostic=refine_truncated_route(original,covers,sources,self.client.state.position,
            background_scan_s=background,scan_source_s=0.)
        if diagnostic['called']:
            self.dp_total_states+=diagnostic['dp_result']['expanded']
            self.dp_total_arcs+=diagnostic['dp_result']['generated']
        self.exact_frontier_log.append(dict(after_actual_action_count=len(self.report.action_history),
            route_log_index=len(self.route_log),current_position=[self.client.state.position.x,self.client.state.position.y],
            source_channels=list(channels),background_scan_s=background,scan_source_s=0.,
            original_result=asdict(original),selected_result=asdict(selected),
            astar_total_expanded=self.total_expansions,dp_total_states=self.dp_total_states,
            dp_total_arcs=self.dp_total_arcs,**diagnostic))
        return selected

    # Exact copy of frozen Q4R2Scheduling._execute_plan except the marked
    # refinement after its original A* accumulation and before route selection.
    def _execute_plan(self):
        remaining = list(self.points)
        while True:
            if not remaining:
                self.report.coverage_complete = True
            if len(self.cleared) == 16:
                self.report.completion_certified_under_model = True
                raise _StopSearch("source_count_upper_bound_reached")
            known = self.detected | self.cleared
            discovery_done = len(known) == 16
            covers = [] if discovery_done else remaining
            if discovery_done and not self.discovery_stop_log:
                self.discovery_stop_log.append({"after_actual_action_count": len(self.report.action_history),
                    "known_channels": sorted(known), "unresolved_channels": sorted(known-self.cleared),
                    "omitted_cover_stations": len(remaining), "reason": "public_source_count_upper_bound",
                    "discovery_only_not_removal": True})
            if covers:
                candidate = self._early_candidate(covers[0])
                if candidate is not None:
                    self._early_service(candidate)
                    continue
            channels = [c for c in sorted(self.detected-self.cleared-self.blocked)
                        if self._target(c) is not None and (not covers or self._ready(c))]
            if not channels:
                if not covers:
                    return
                self._scan(covers[0])
                remaining.pop(0)
                continue
            began = time.perf_counter()
            sources = [ChainSource(self._target(c), service_s=5.) for c in channels]
            budget = max(0, min(self.max_expansions, 60000-self.total_expansions))
            result = solve_chain_route(covers, sources, self.client.state.position,
                max_expansions=budget, background_scan_s=6.*max(0, 20-len(self.cleared)-len(channels)),
                scan_source_s=0.)
            self.total_expansions += result.expanded
            result = self._refine_route(result, covers, sources, channels,
                6.*max(0, 20-len(self.cleared)-len(channels)))
            kind, index = result.order[0]
            self.route_log.append({"after_actual_action_count": len(self.report.action_history),
                "remaining_covers": [[p.x, p.y] for p in covers], "source_channels": channels,
                "source_positions": [[s.position.x, s.position.y] for s in sources],
                "discovery_count_cap": discovery_done, "selected_kind": kind,
                "selected_channel": channels[index] if kind == "source" else None,
                "result": asdict(result), "runtime_s": time.perf_counter()-began})
            if kind == "cover":
                self._scan(covers[0])
                remaining.pop(0)
            elif not self._resolve(channels[index]):
                self.blocked.add(channels[index])


def run_q4_exact_frontier(client, *, problem=4, config='truncated_dag8', max_actions=20000,
                          max_active_probes=6, max_expansions=200):
    if type(problem) is not int or problem!=4 or config!='truncated_dag8':
        raise ValueError('Q4 only; fixed truncated_dag8 config')
    for value,low,high in ((max_actions,2,1_000_000),(max_active_probes,0,30),(max_expansions,0,10000)):
        if type(value) is not int or not low<=value<=high:
            raise ValueError('Invalid execution budget')
    return Q4ExactFrontier(client,max_actions,max_active_probes,max_expansions=max_expansions,config=config).run()
