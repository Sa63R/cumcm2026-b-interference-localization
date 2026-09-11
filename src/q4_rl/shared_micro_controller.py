"""Rule-only service-co-located scans with actual whole-station provenance.

The original micro implementation and its checkpoint schema are unchanged.
Disabling this extension dispatches directly to that original implementation.
"""
import time

from simulator_client.rules import CHANNELS, MIN_SOURCES, MAX_SOURCES
from simulator_client.state import Position
from strategies.search import _StopSearch
from .micro_controller import (Q4MicroSearch, MicroCandidate, run_q4_micro,
    GLOBAL_FEATURE_NAMES as MICRO_GLOBAL_NAMES,
    CANDIDATE_FEATURE_NAMES as MICRO_CANDIDATE_NAMES)
from .shared_cover import SharedCoverLedger, PlannedAction, public_plan_cost, replay_shared_cover, position_key

FEATURE_SCHEMA_VERSION = "q4-shared-micro-g2-rule-v1"
GLOBAL_FEATURE_NAMES = MICRO_GLOBAL_NAMES + (
    "shared_pending_pairs_fraction", "shared_cancelled_station_fraction",
    "shared_packet_remaining_fraction", "shared_certificate_budget_fraction",
)
CANDIDATE_FEATURE_NAMES = MICRO_CANDIDATE_NAMES + (
    "is_shared_service_scan", "is_active_packet_scan",
    "shared_channel_pending_fraction", "shared_site_pending_fraction",
)
GLOBAL_DIM,CANDIDATE_DIM = len(GLOBAL_FEATURE_NAMES),len(CANDIDATE_FEATURE_NAMES)


class Q4SharedMicroSearch(Q4MicroSearch):
    def __init__(self,client,policy=None,*,max_shared_reviews=8,**kwargs):
        if policy is not None:
            raise ValueError("Shared micro G2 is rule-only; old 13/50 checkpoints are incompatible")
        if type(max_shared_reviews) is not int or not 0<=max_shared_reviews<=8:
            raise ValueError("At most eight public schedule reviews")
        super().__init__(client,policy=None,**kwargs)
        self.shared = SharedCoverLedger(self.points)
        self.shared_plan = None
        self.max_shared_reviews = max_shared_reviews
        self.shared_reviews = []
        self.shared_signatures = set()
        self.shared_points = ()
        self.shared_replay_wall_s = 0.
        self.shared_scan_count = 0
        self.shared_added_candidate_scan_count = 0
        self.report.learning.update(algorithm=FEATURE_SCHEMA_VERSION,
            feature_schema=dict(version=FEATURE_SCHEMA_VERSION,
                global_features=list(GLOBAL_FEATURE_NAMES),candidate_features=list(CANDIDATE_FEATURE_NAMES)),
            shared_plan_reviews=self.shared_reviews,
            shared_scope="Rule-only finite service-point scan packets; atomic whole inner-station cancellation")
        self.report.strategy_parameters.update(q4_rl_feature_schema=FEATURE_SCHEMA_VERSION,
            discovery_credits="Actual fixed measurements are separate from exact residual-plan cancellations",
            shared_plan_proxy="Explicit full remaining unknown discovery schedule plus at most four next public source-service actions; future feedback not predicted")

    def _consume_actual_history(self):
        super()._consume_actual_history()
        if not hasattr(self,"shared"):
            return
        for index in range(len(self.shared.history),len(self.report.action_history)):
            item = self.report.action_history[index]
            self.shared.observe(index,item["action"],item["position"],item["channel"],
                item["result"],accepted=True,bearing_deg=item.get("bearing_deg"))
        self._try_commit_packet()

    def _try_commit_packet(self):
        if self.shared_plan is None:
            return
        if self.shared.cancel_station(self.shared_plan):
            self.shared_plan = None
        elif not any(position_key(self.shared_plan.station) in self.shared.pending[c] for c in self._unknown()):
            self.shared_plan = None

    def _pending_unknown(self,point):
        if not hasattr(self,"shared"):
            return super()._pending_unknown(point)
        return {c for c in self._unknown() if position_key(point) in self.shared.pending[c]}

    def _can_measure(self,point,channel):
        if (hasattr(self,"shared") and point in self.cover_ledger and channel in self._unknown()
                and position_key(point) not in self.shared.pending[channel]):
            return False
        return super()._can_measure(point,channel)

    def _refresh_certificate(self):
        if not hasattr(self,"shared"):
            return super()._refresh_certificate()
        super()._refresh_certificate()  # This ledger remains exclusively actual.
        unknown = self._unknown()
        absent = {c for c in unknown if not self.shared.pending[c]}
        discovery = len(self.detected|self.cleared)==MAX_SOURCES or absent==unknown
        self.report.coverage_complete = absent==unknown
        self.report.coverage_points_visited = len({p for p,_ in self.actual_measurements if p in self.cover_ledger})
        self.report.learning.update(certified_absent_channels=sorted(absent),discovery_certified=discovery)
        return discovery

    def _terminal_gate(self):
        discovery = self._refresh_certificate()
        eligible = len(self.cleared)==MAX_SOURCES or (discovery and
            MIN_SOURCES<=len(self.detected|self.cleared)<=MAX_SOURCES and self.detected<=self.cleared)
        if not eligible:
            return False
        started = time.perf_counter()
        verified = replay_shared_cover(self.report.action_history,self.shared.artifact())
        self.shared_replay_wall_s += time.perf_counter()-started
        if not verified["complete"]:
            raise _StopSearch("shared_actual_completion_replay_failed")
        self.report.learning["shared_completion_replay"] = verified
        self.report.learning["completion_certificate"] = (
            "actual_clear_receipts_and_exact_actual_per_channel_shared_cover")
        self.report.completion_certified_under_model = True
        raise _StopSearch("q4_shared_micro_certified_complete")

    def _public_service_candidates(self,base):
        # At most one next service action per source; no invented mandatory
        # sequence of several mutually alternative probes for the same source.
        raw = self.shared.service_points(self.client.state.position)
        used,selected = set(),[]
        for point in raw:
            legal = [c for c in base if c.point==point.position and c.channel==point.source_channel
                     and (c.kind=="certified_clear" or c.role=="localize")]
            if legal and point.source_channel not in used:
                selected.append(point)
                used.add(point.source_channel)
        return tuple(selected)

    def _public_schedule(self,points,base):
        # A deterministic complete discovery schedule with co-located next
        # service requests. This is a public proxy, not a rollout of feedback.
        visits = {}
        for point in self.points:
            owed = self._pending_unknown(point)
            if owed:
                visits[point] = [PlannedAction("measure",point,c,"coverage") for c in sorted(owed)]
        for point in points:
            legal = [c for c in base if c.point==point.position and c.channel==point.source_channel
                     and (c.kind=="certified_clear" or c.role=="localize")]
            if not legal:
                continue
            chosen = min(legal,key=lambda c:(c.kind!="certified_clear",c.role))
            visits.setdefault(point.position,[]).insert(0,PlannedAction(
                "measure" if chosen.kind=="measure" else "clear",chosen.point,chosen.channel,"service"))
        current,tuned = self.client.state.position,self.client.state.current_channel
        actions = []
        while visits:
            point = min(visits,key=lambda p:(current.distance_to(p),p.x,p.y))
            group = visits.pop(point)
            # Clear first; measurements use the current channel when available.
            group.sort(key=lambda a:(a.kind!="clear",a.channel!=tuned,a.channel,a.purpose))
            actions.extend(group)
            for action in group:
                if action.kind=="measure": tuned=action.channel
            current=point
        return tuple(actions)

    def _review_plan(self,base,points):
        if self.shared_plan is not None or len(points)<2 or len(self.shared_reviews)>=self.max_shared_reviews:
            return
        if self.shared.certificate_queries>=self.shared.max_queries-1:
            return
        signature = tuple((p.source_channel,position_key(p.position),p.origin) for p in points)
        if signature in self.shared_signatures:
            return
        self.shared_signatures.add(signature)
        actions = self._public_schedule(points,base)
        before_queries = self.shared.certificate_queries
        started = time.perf_counter()
        plans = self.shared.proposals(points,actions,self.client.state.position,self.client.state.current_channel)
        review = dict(after_action_index=len(self.report.action_history)-1,
            public_service_points=[list(position_key(p.position)) for p in points],
            source_channels=[p.source_channel for p in points],
            full_proxy_cost=public_plan_cost(actions,self.client.state.position,self.client.state.current_channel),
            returned_plans=len(plans),certificate_operations=self.shared.certificate_queries-before_queries,
            review_wall_s=time.perf_counter()-started)
        self.shared_reviews.append(review)
        if plans:
            self.shared_plan = max(plans,key=lambda p:p.estimated_saved_s)
            review.update(selected_station=list(position_key(self.shared_plan.station)),
                          selected_proxy_saving_s=self.shared_plan.estimated_saved_s)

    def _candidates(self):
        base = super()._candidates()
        if self.report.learning["discovery_certified"]:
            return base
        points = self._public_service_candidates(base)
        self._review_plan(base,points)
        active = self.shared_plan.service_points if self.shared_plan is not None else ()
        retained = list(active)
        for point in points:
            if point.position not in {p.position for p in retained} and len(retained)<4:
                retained.append(point)
        self.shared_points = tuple(retained)
        seen = {(c.kind,c.point,c.channel) for c in base}
        result = list(base)
        for point in self.shared_points:
            for channel in sorted(self._unknown()):
                key = "measure",point.position,channel
                if key not in seen and self._can_measure(point.position,channel):
                    result.append(MicroCandidate("measure",point.position,channel,
                        point.position in self.cover_ledger,"shared_service"))
                    seen.add(key)
        self.report.learning["maximum_candidates"] = max(self.report.learning["maximum_candidates"],len(result))
        return result

    def _remaining_packet_pairs(self):
        if self.shared_plan is None:
            return set()
        return {(p.position,c) for p in self.shared_plan.service_points for c in self._unknown()
                if position_key(self.shared_plan.station) in self.shared.pending[c]
                and position_key(p.position) not in self.shared.negatives[c]}

    def _features(self,candidates):
        features,rows = super()._features(candidates)
        packet = self._remaining_packet_pairs()
        unknown = self._unknown()
        features.extend([sum(len(self.shared.pending[c]) for c in unknown)/440.,
            len(self.shared.cancelled_stations)/22.,len(packet)/40.,
            (self.shared.max_queries-self.shared.certificate_queries)/8.])
        for c,row in zip(candidates,rows):
            row.extend([float(c.role=="shared_service"),float((c.point,c.channel) in packet and c.kind=="measure"),
                len(self.shared.pending[c.channel])/22.,
                sum(position_key(c.point) in self.shared.pending[ch] for ch in unknown)/20.])
        assert len(features)==GLOBAL_DIM and all(len(row)==CANDIDATE_DIM for row in rows)
        return features,rows

    def _heuristic(self,candidates):
        packet = self._remaining_packet_pairs()
        if packet:
            current = self.client.state.position
            point = min({p for p,_ in packet},key=lambda p:(current.distance_to(p),p.x,p.y))
            # Real known-source service first, then every unknown channel at
            # the same visit. No candidate ever executes a multi-request macro.
            service = [(i,c) for i,c in enumerate(candidates) if c.point==point
                       and (c.kind=="certified_clear" or c.role=="localize")]
            if service:
                return min(service,key=lambda pair:(pair[1].kind!="certified_clear",pair[1].channel))[0]
            scans = [(i,c) for i,c in enumerate(candidates) if c.kind=="measure" and (c.point,c.channel) in packet and c.point==point]
            if scans:
                return min(scans,key=lambda pair:(pair[1].channel!=self.client.state.current_channel,pair[1].channel))[0]
        # Uncertified extension candidates are available but not selected by
        # this conservative rule. The original heuristic receives its own set.
        original = [(i,c) for i,c in enumerate(candidates) if c.role!="shared_service"]
        return original[super()._heuristic([c for _,c in original])][0]

    def _execute_candidate(self,candidate):
        before = len(self.report.action_history)
        packet_scan = candidate.kind=="measure" and (candidate.point,candidate.channel) in self._remaining_packet_pairs()
        try:
            return super()._execute_candidate(candidate)
        finally:
            if packet_scan or candidate.role=="shared_service":
                self.shared_scan_count += len(self.report.action_history)-before
            if candidate.role=="shared_service":
                self.shared_added_candidate_scan_count += len(self.report.action_history)-before

    def run(self):
        report = super().run()
        report.learning.update(shared_artifact=self.shared.artifact(),
            shared_service_actual_measurements=self.shared_scan_count,
            shared_added_candidate_actual_measurements=self.shared_added_candidate_scan_count,
            shared_completion_replay_wall_s=self.shared_replay_wall_s,
            shared_unfinished_packet_pairs=len(self._remaining_packet_pairs()))
        return report


def run_q4_shared_micro(client,policy=None,*,problem=4,shared_enabled=True,max_shared_reviews=8,**kwargs):
    if type(shared_enabled) is not bool:
        raise ValueError("shared_enabled must be boolean")
    if not shared_enabled:
        return run_q4_micro(client,policy=policy,problem=problem,**kwargs)
    if type(problem) is not int or problem!=4:
        raise ValueError("Shared micro supports Q4 only")
    return Q4SharedMicroSearch(client,policy=policy,max_shared_reviews=max_shared_reviews,**kwargs).run()
