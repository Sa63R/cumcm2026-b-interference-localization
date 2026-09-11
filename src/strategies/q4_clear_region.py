"""Change only certified/near clear positions on the frozen Q4 combo policy."""
import time

from planning.certified_clear_region import choose_certified_clear_point
from simulator_client.state import Position
from .q4_range_scheduling import Q4RangeScheduling


class Q4ClearRegion(Q4RangeScheduling):
    def __init__(self, client, max_actions, max_active_probes, *, config, max_expansions):
        super().__init__(client,max_actions,max_active_probes,config="onroute",max_expansions=max_expansions)
        self.clear_region_config=config
        self.clear_region_log=[]
        self._pending_early_anchor=None
        self._early_anchor=None
        self._resolve_anchor=None
        self.report.strategy_parameters.update(q4_clear_region=config,clear_region_log=self.clear_region_log,
            clear_region_scope="Only actual certified/near clear position; original probes, stations and scheduling",
            clear_region_limits={"uniform_rays":64,"ternary_steps":32,"polygon_search_radius_m":19.99998,
                                 "near_search_radius_m":14.99998,"verification_margin_m":0.00001,
                                 "incoming_method":"boundary_candidates","max_vertices":64})

    def _early_candidate(self,next_cover):
        self._pending_early_anchor=None
        candidate=super()._early_candidate(next_cover)
        if candidate is not None:
            point=Position.coerce(next_cover)
            self._pending_early_anchor={"candidate":candidate,"point":point,
                "source":{"kind":"early_service_next_cover","candidate_prefix":len(self.report.action_history),
                          "channel":candidate[2],"next_cover_position":[point.x,point.y]}}
        return candidate

    def _early_service(self,candidate):
        pending=self._pending_early_anchor
        self._pending_early_anchor=None
        self._early_anchor=(pending if pending is not None and pending["candidate"] == candidate
            and pending["source"]["candidate_prefix"] == len(self.report.action_history) else None)
        try:
            return super()._early_service(candidate)
        finally:
            self._early_anchor=None

    def _route_anchor(self,channel):
        if not self.route_log:
            return None
        index=len(self.route_log)-1
        event=self.route_log[index]
        n=len(self.report.action_history)
        if (event.get("after_actual_action_count") != n or event.get("selected_kind") != "source"
                or event.get("selected_channel") != channel):
            return None
        try:
            order=event["result"]["order"]
            if len(order)<2 or order[0][0] != "source":return None
            channels=event["source_channels"]
            if channels[order[0][1]] != channel:return None
            kind,i=order[1]
            if type(i) is not int or i<0:return None
            if kind == "cover":
                point=Position.coerce(event["remaining_covers"][i])
            elif kind == "source":
                next_channel=channels[i]
                if next_channel not in self.detected-self.cleared or next_channel == channel:return None
                point=Position.coerce(event["source_positions"][i])
                if point != self._target(next_channel):return None
            else:return None
            return {"point":point,"source":{"kind":"route_second_task","route_log_index":index,
                "prefix":n,"channel":channel,"task":[kind,i]}}
        except (KeyError,IndexError,TypeError,ValueError):
            return None

    def _resolve(self,channel):
        # A narrow wrapper preserves the entire parent resolver and its finally
        # semantics; it only binds an observed anchor to this one macro.
        self._resolve_anchor=None
        if self._early_anchor is not None and self._early_anchor["source"]["channel"] == channel:
            self._resolve_anchor=self._early_anchor
        elif self.service_deadline is None:
            self._resolve_anchor=self._route_anchor(channel)
        try:
            return super()._resolve(channel)
        finally:
            self._resolve_anchor=None

    def _clear(self,position,channel,phase):
        if phase not in {"certified_clear","near_clear"}:
            return super()._clear(position,channel,phase)
        began=time.perf_counter()
        original=Position.coerce(position)
        current=self.client.state.position
        n=len(self.report.action_history)
        context=self._resolve_anchor
        anchor=None;source=None;anchor_status="missing"
        if self.clear_region_config == "incoming":
            anchor_status="incoming_configuration"
        elif context is not None:
            source=context["source"]
            if source["kind"] == "route_second_task" and source["prefix"] != n:
                anchor_status="route_prefix_expired";source=None
            else:
                anchor=context["point"];anchor_status="valid"
        kwargs={}
        if channel in self.detected-self.cleared:
            if phase == "near_clear" and channel in self.near_points and original == self.near_points[channel]:
                kwargs={"near_point":self.near_points[channel]}
            elif phase == "certified_clear" and channel in self.regions:
                region=self.regions[channel]
                if region.vertices and original == Position.coerce(region.enclosing_disk().center):
                    kwargs={"vertices":list(region.vertices)}
        selected,geometry=choose_certified_clear_point(original,current,anchor=anchor,
                                                       config=self.clear_region_config,**kwargs)
        event={"after_actual_action_count":n,"channel":channel,"phase":phase,"config":self.clear_region_config,
               "original_position":[original.x,original.y],"current_position":[current.x,current.y],
               "anchor":[anchor.x,anchor.y] if anchor is not None else None,
               "anchor_source":source,"anchor_status":anchor_status,
               "selected_position":[selected.x,selected.y],"geometry":geometry,"executed":False}
        self.clear_region_log.append(event)
        try:
            # Cooperative MRO still reaches Q4R2Scheduling._perform, so its
            # original time-slice gate precedes every physical clear request.
            return super()._clear(selected,channel,phase)
        finally:
            end=len(self.report.action_history)
            event["end_action_count"]=end
            if end == n+1:
                action=self.report.action_history[n]
                if action["action"] == "clear" and action["channel"] == channel and action["position"] == [selected.x,selected.y]:
                    event.update(executed=True,actual_position=action["position"],result=action["result"])
            event["runtime_s"]=time.perf_counter()-began


def run_q4_clear_region(client, *, problem=4, config="incoming", max_actions=20000,
                        max_active_probes=6, max_expansions=200):
    if type(problem) is not int or problem != 4 or config not in {"incoming","anchored"}:
        raise ValueError("Q4 only; unknown clear-region configuration")
    for value,low,high in ((max_actions,2,1000000),(max_active_probes,0,30),(max_expansions,0,10000)):
        if type(value) is not int or not low<=value<=high:raise ValueError("Invalid execution budget")
    return Q4ClearRegion(client,max_actions,max_active_probes,config=config,max_expansions=max_expansions).run()
