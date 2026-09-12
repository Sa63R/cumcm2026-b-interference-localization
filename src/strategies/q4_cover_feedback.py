"""One-cover feedback experiment on the unchanged R12 physical controller.

Only an eligible source/cover incumbent may be vetoed. Its unexecuted route is
retained; a separate ownership event records the actual original cover scan.
Prediction expansions never consume the original 60000-expansion route budget.
"""
from copy import deepcopy
from dataclasses import asdict
import time

from planning.chain_route import ChainSource, solve_chain_route
from planning.q4_cover_feedback import (evaluate_cover_feedback, xy, PREDICTION_LIMIT,
    VETO_LIMIT, EXPANSION_LIMIT, SOLVE_EXPANSIONS, WORLDS)
from .q4_joint_continuation import Q4JointContinuation
from .search import _StopSearch

CONFIG = 'centered_one_cover'


class Q4CoverFeedback(Q4JointContinuation):
    def __init__(self, client, max_actions, max_active_probes, *, max_expansions, config=CONFIG):
        if config != CONFIG:
            raise ValueError('Unknown cover feedback configuration')
        super().__init__(client, max_actions, max_active_probes, max_expansions=max_expansions)
        self.cover_feedback_log = []
        self.prediction_calls = self.prediction_expanded = self.cover_feedback_vetoes = 0
        self.vetoed_channels = set()
        self.report.strategy_parameters.update(cover_feedback_config=config,
            cover_feedback_log=self.cover_feedback_log,
            cover_feedback_limits=dict(predictions=PREDICTION_LIMIT, actual_vetoes=VETO_LIMIT,
                vetoes_per_channel=1, worlds=WORLDS, expansions_per_solve=SOLVE_EXPANSIONS,
                total_prediction_expansions=EXPANSION_LIMIT, solves_per_prediction=11,
                belief_nodes=24, belief_prior_omni=.5, belief_history=64, belief_work=262144),
            cover_feedback_scope='Centered all-known feedback proxy; only real cover or real clear executes; no unknown-source positions or physical scan credit',
            state_scope='Original fixed-chain ready-source incumbent, optionally vetoed by a separately budgeted one-cover feedback proxy')

    def _prediction(self, covers, channels, sources, selected, result):
        known = []
        for channel in sorted(self.detected-self.cleared):
            known.append(dict(channel=channel, region=self.regions.get(channel),
                near=self.near_points.get(channel),
                prefix=[deepcopy(a) for a in self.report.action_history
                        if a['action'] == 'measure' and a['channel'] == channel]))
        return evaluate_cover_feedback(current=self.client.state.position, covers=covers,
            ready_channels=channels, ready_positions=[s.position for s in sources],
            sources=known, cleared_count=len(self.cleared), selected_channel=selected,
            incumbent_cost_s=result.cost_s, expansion_allowance=EXPANSION_LIMIT-self.prediction_expanded)

    def _eligibility(self, covers, result, selected):
        if (len(result.order) < 2 or result.order[0][0] != 'source'
                or tuple(result.order[1]) != ('cover', 0) or not covers):
            return 'not_source_then_next_cover'
        if selected is None or not self._ready(selected):
            return 'selected_source_not_ready'
        if not any(not self._ready(c) for c in self.detected-self.cleared):
            return 'no_known_nonready'
        if selected in self.vetoed_channels:
            return 'channel_veto_limit'
        if self.cover_feedback_vetoes >= VETO_LIMIT:
            return 'session_veto_limit'
        if self.prediction_calls >= PREDICTION_LIMIT:
            return 'prediction_call_limit'
        if self.prediction_expanded + SOLVE_EXPANSIONS > EXPANSION_LIMIT:
            return 'prediction_expansion_limit'
        return None

    def _execute_plan(self):
        # The baseline loop through incumbent construction is copied verbatim in
        # behavior from Q4R2Scheduling. Differences begin at the ownership event.
        remaining = list(self.points)
        while True:
            if not remaining:
                self.report.coverage_complete = True
            if len(self.cleared) == 16:
                self.report.completion_certified_under_model = True
                raise _StopSearch('source_count_upper_bound_reached')
            known = self.detected | self.cleared
            discovery_done = len(known) == 16
            covers = [] if discovery_done else remaining
            if discovery_done and not self.discovery_stop_log:
                self.discovery_stop_log.append({'after_actual_action_count': len(self.report.action_history),
                    'known_channels': sorted(known), 'unresolved_channels': sorted(known-self.cleared),
                    'omitted_cover_stations': len(remaining), 'reason': 'public_source_count_upper_bound',
                    'discovery_only_not_removal': True})
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
                max_expansions=budget, background_scan_s=6.*max(0,20-len(self.cleared)-len(channels)),
                scan_source_s=0.)
            self.total_expansions += result.expanded
            kind, index = result.order[0]
            selected = channels[index] if kind == 'source' else None
            route = {'after_actual_action_count': len(self.report.action_history),
                'remaining_covers': [xy(p) for p in covers], 'source_channels': channels,
                'source_positions': [xy(s.position) for s in sources],
                'discovery_count_cap': discovery_done, 'selected_kind': kind,
                'selected_channel': selected, 'result': asdict(result),
                'runtime_s': time.perf_counter()-began}
            self.route_log.append(route)
            event = dict(id=len(self.cover_feedback_log), route_index=len(self.route_log)-1,
                after_actual_action_count=len(self.report.action_history),
                end_actual_action_count=len(self.report.action_history), channel=selected,
                next_cover=xy(covers[0]) if covers else None, remaining_covers=[xy(p) for p in covers],
                current_position=xy(self.client.state.position), current_channel=self.client.state.current_channel,
                ready_channels=list(channels), known_channels=sorted(known), cleared_channels=sorted(self.cleared),
                blocked_channels=sorted(self.blocked), incumbent=deepcopy(route),
                eligibility_reason=self._eligibility(covers,result,selected), prediction=None,
                prediction_calls_before=self.prediction_calls, prediction_expanded_before=self.prediction_expanded,
                actual_vetoes_before=self.cover_feedback_vetoes,
                vetoed_channels_before=sorted(self.vetoed_channels),
                coverage_visited_before=self.report.coverage_points_visited,
                veto_selected=False, executed_cover=False, executed_kind=kind, status='pending')
            self.cover_feedback_log.append(event)
            route.update(execution_role='original_incumbent', cover_feedback_event_id=event['id'])
            try:
                if event['eligibility_reason'] is None:
                    self.prediction_calls += 1
                    prediction = self._prediction(covers,channels,sources,selected,result)
                    event['prediction'] = prediction
                    self.prediction_expanded += prediction['expanded']
                    event['veto_selected'] = bool(prediction['status'] == 'scored'
                        and prediction['information_changed'] and prediction['recommend_veto'])
                event['decision_wall_s'] = time.perf_counter()-began
                if event['veto_selected']:
                    route['execution_role'] = 'incumbent_not_executed'
                    event['executed_kind'] = 'cover'
                    self._scan(covers[0])
                    # Only a complete normal scan return owns a real veto. An
                    # accepted partial scan remains logged but cannot earn it.
                    remaining.pop(0)
                    event['executed_cover'] = True
                    self.cover_feedback_vetoes += 1
                    self.vetoed_channels.add(selected)
                elif kind == 'cover':
                    self._scan(covers[0])
                    remaining.pop(0)
                    event['executed_cover'] = True
                elif not self._resolve(selected):
                    self.blocked.add(selected)
                event['status'] = 'completed'
            except Exception as error:
                event.update(status='interrupted', interruption_type=type(error).__name__,
                             interruption_reason=str(error))
                raise
            finally:
                event.update(end_actual_action_count=len(self.report.action_history),
                    prediction_calls_after=self.prediction_calls, prediction_expanded_after=self.prediction_expanded,
                    actual_vetoes_after=self.cover_feedback_vetoes, vetoed_channels_after=sorted(self.vetoed_channels),
                    coverage_visited_after=self.report.coverage_points_visited,
                    runtime_s=time.perf_counter()-began)


def run_q4_cover_feedback(client, *, problem=4, config=CONFIG, max_actions=20000,
                          max_active_probes=6, max_expansions=200):
    if type(problem) is not int or problem != 4 or config != CONFIG:
        raise ValueError('Q4 only; use centered_one_cover')
    for value, low, high in ((max_actions,2,1_000_000),(max_active_probes,0,30),(max_expansions,0,10000)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError('Invalid integer execution budget')
    return Q4CoverFeedback(client,max_actions,max_active_probes,
                          max_expansions=max_expansions,config=config).run()
