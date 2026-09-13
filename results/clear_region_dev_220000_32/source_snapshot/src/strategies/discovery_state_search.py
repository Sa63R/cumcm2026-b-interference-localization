"""Separate mandatory unknown-channel discovery from optional known readings."""

from simulator_client.state import Position

from .refined_state_search import RefinedStateSearch


class DiscoveryStateSearch(RefinedStateSearch):
    def __init__(self,client,max_actions,max_active_probes,config,unknown_only):
        super().__init__(client,max_actions,max_active_probes,config,"axis_quantile")
        if unknown_only and self.state_config.replace_coverage:
            raise ValueError("discovery masks use the fixed seven-station cover; replacement must be disabled")
        self.unknown_only=unknown_only
        self.station_index={(p.x,p.y):i for i,p in enumerate(self.points)}
        self.station_masks=[0]*len(self.points)
        self.discovery_stats={"scan_calls":0,"known_uncertified_scans_skipped":0,
                              "stations_removed_without_pending":[],"snapshots":[]}
        self.report.strategy_parameters.update({"unknown_only_coverage":unknown_only,
            "station_measured_masks":self.station_masks,"discovery_scan_ledger":self.discovery_stats,
            "discovery_completion_rule":"per-unknown-channel measurements at all seven certified cover stations, or public detected count 16; actual clearances still required",
            "known_information_policy":"original axis localization and after-clear sharing only"})

    def _perform(self,action,position,channel,phase):
        response=super()._perform(action,position,channel,phase)
        if action=="measure" and phase=="coverage":
            p=Position.coerce(position)
            index=self.station_index[(p.x,p.y)]
            self.station_masks[index]|=1<<(channel-1)
        return response

    def _pending_masks(self):
        known=self.detected|self.cleared
        if self.state_config.stop_discovery_at_16 and len(known)==16:
            return [0]*len(self.points)
        unknown=sum(1<<(c-1) for c in range(1,21) if c not in known)
        return [unknown&~mask for mask in self.station_masks]

    def _scan(self,point):
        if not self.unknown_only:
            return super()._scan(point)
        index=self.station_index[(point.x,point.y)]
        stats=self.discovery_stats
        stats["scan_calls"]+=1
        for c in self.detected-self.cleared:
            region=self.regions.get(c)
            if c not in self.near_points and region and region.vertices and region.enclosing_disk().radius>19.9:
                stats["known_uncertified_scans_skipped"]+=1
        pending=self._pending_masks()[index]
        channels=[c for c in range(1,21) if pending&(1<<(c-1))]
        current=self.client.state.current_channel
        if current in channels:
            channels.remove(current)
            channels.insert(0,current)
        for channel in channels:
            # A real positive response elsewhere may remove an obligation.
            # Public count 16 can end discovery even midway through a station.
            if not self._pending_masks()[index]&(1<<(channel-1)):
                continue
            self._perform("measure",point,channel,"coverage")
        self.report.coverage_points_visited+=1
        self.discovery_stations.append(point)
        self.blocked.clear()
        stats["snapshots"].append({"station":index,"measured_mask":self.station_masks[index],
            "pending_masks":self._pending_masks(),"detected_count":len(self.detected|self.cleared),
            "cleared_count":len(self.cleared)})

    def _next_task(self,remaining):
        if self.unknown_only:
            pending=self._pending_masks()
            removed=[p for p in remaining if not pending[self.station_index[(p.x,p.y)]]]
            for p in removed:
                remaining.remove(p)
                self.discovery_stats["stations_removed_without_pending"].append(self.station_index[(p.x,p.y)])
            if not any(pending):
                self.report.coverage_complete=True
        return super()._next_task(remaining)


def run_discovery_state_search(client,*,problem=3,max_actions=10000,max_active_probes=6,
                               config=None,unknown_only=True):
    if problem!=3:
        raise ValueError("discovery-mask strategy supports Q3 only")
    if type(unknown_only) is not bool:
        raise ValueError("unknown_only must be boolean")
    if type(max_actions) is not int or max_actions<2:
        raise ValueError("max_actions must be integer >=2")
    if type(max_active_probes) is not int or not 0<=max_active_probes<=30:
        raise ValueError("max_active_probes must be in [0,30]")
    return DiscoveryStateSearch(client,max_actions,max_active_probes,config,unknown_only).run()
