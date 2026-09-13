"""Skip physically redundant known-channel reads, record explicit inference."""

from planning.silence_certificate import certify_silence

from .refined_state_search import RefinedStateSearch


class InferredSilenceSearch(RefinedStateSearch):
    def __init__(self,client,max_actions,max_active_probes,config,enabled):
        super().__init__(client,max_actions,max_active_probes,config,"axis_quantile")
        self.inferred_enabled=enabled
        self.inferred=[]
        self.report.strategy_parameters.update({"inferred_silence_enabled":enabled,
            "inferred_no_signal_constraints":self.inferred,
            "inference_scope":"known Q3 channel; d(query,outer_region)>1500+1e-5; geometry-only deduction, not an actual measurement or unknown-channel coverage evidence"})

    def _scan(self,point):
        if not self.inferred_enabled:
            return super()._scan(point)
        channels=[c for c in range(1,21) if c not in self.cleared]
        current=self.client.state.current_channel
        if current in channels:
            channels.remove(current)
            channels.insert(0,current)
        for channel in channels:
            region=self.regions.get(channel)
            certified=(channel in self.near_points or (
                channel in self.detected and region and region.vertices and region.enclosing_disk().radius<=19.9))
            if self.state_config.skip_certified_scans and certified:
                self.skipped_measurements+=1
                continue
            certificate=(certify_silence(region,point) if channel in self.detected and region else None)
            if certificate:
                # No call, no invented response, no client source update and no
                # addition to action_history or actual observed_positions.
                # This is a logical fact derived from existing safe geometry.
                region.observe_no_signal(point)
                self.inferred.append({"inference_kind":"inferred_no_signal",
                    "channel":channel,"position":[point.x,point.y],
                    "after_actual_action_count":len(self.report.action_history),
                    "physical_measurement":False,"virtual_time_s":self.client.state.virtual_time_s,
                    **certificate})
                continue
            self._perform("measure",point,channel,"coverage")
        self.report.coverage_points_visited+=1
        # Every still-unknown channel was physically measured here. A known
        # channel's derived fact is never used as its unknown coverage record.
        self.discovery_stations.append(point)
        self.blocked.clear()
        self.report.strategy_parameters["skipped_certified_measurements"]=self.skipped_measurements


def run_inferred_state_search(client,*,problem=3,max_actions=10000,max_active_probes=6,
                              config=None,enabled=True):
    if problem!=3:
        raise ValueError("inferred silence supports Q3 only")
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if type(max_actions) is not int or max_actions<2:
        raise ValueError("max_actions must be integer >=2")
    if type(max_active_probes) is not int or not 0<=max_active_probes<=30:
        raise ValueError("max_active_probes must be in [0,30]")
    return InferredSilenceSearch(client,max_actions,max_active_probes,config,enabled).run()
