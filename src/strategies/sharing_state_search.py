"""Independent axis ablation: selective sharing at actual active probe stops."""

from dataclasses import asdict,dataclass
import math
import time

from planning.shared_observation import shared_observation_value

from .refined_state_search import RefinedStateSearch


@dataclass(frozen=True)
class ActiveSharingConfig:
    enabled: bool = True
    minimum_net_gain_s: float = 5.
    max_shared_per_stop: int = 4

    @classmethod
    def parse(cls,values):
        if values is not None and not isinstance(values,dict):
            raise ValueError("sharing_config must be a dictionary")
        try:
            config=cls(**(values or {}))
        except TypeError as error:
            raise ValueError(f"Invalid sharing_config: {error}") from error
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if type(config.max_shared_per_stop) is not int or not 0<=config.max_shared_per_stop<=20:
            raise ValueError("max_shared_per_stop must be an integer in [0,20]")
        if (isinstance(config.minimum_net_gain_s,bool) or not isinstance(config.minimum_net_gain_s,(int,float))
                or not math.isfinite(config.minimum_net_gain_s) or not 0<=config.minimum_net_gain_s<=1000):
            raise ValueError("minimum_net_gain_s must be finite in [0,1000]")
        return config


class SharingStateSearch(RefinedStateSearch):
    def __init__(self,client,max_actions,max_active_probes,config,sharing_config):
        self.sharing_config=ActiveSharingConfig.parse(sharing_config)
        self._active_sharing=False
        super().__init__(client,max_actions,max_active_probes,config,"axis_quantile")
        self.sharing_stats={"stops_considered":0,"candidates_scored":0,"shared_measurements":0,
                            "actual_measure_cost_s":0.,"planning_s":0.,"decisions":[]}
        self.report.strategy_parameters.update({"active_sharing_config":asdict(self.sharing_config),
            "active_sharing":self.sharing_stats,
            "sharing_score_scope":"local triangle-excess quadrature minus conservative switch/measure/return costs; heuristic only",
            "after_clear_sharing":"unchanged original StateSearch rule",
            "task_replanning":"unchanged; atomic current-source resolve, then replan from actual observations"})

    def _perform(self,action,position,channel,phase):
        response=super()._perform(action,position,channel,phase)
        if (self.sharing_config.enabled and self.sharing_config.max_shared_per_stop
                and not self._active_sharing and action=="measure" and phase=="active_localization"):
            self._share_active_point(channel)
        return response

    def _share_active_point(self,primary):
        began=time.perf_counter()
        action_wall=0.
        config=self.sharing_config
        q=self.client.state.position
        key=(round(q.x,6),round(q.y,6))
        stats=self.sharing_stats
        stats["stops_considered"]+=1
        values={}
        for channel in sorted(self.detected-self.cleared-{primary}):
            if channel in self.near_points or key in self.observed_positions.get(channel,set()):
                continue
            value=shared_observation_value(self.regions[channel],q)
            if value is not None:
                values[channel]=value
                stats["candidates_scored"]+=1
        self._active_sharing=True
        try:
            for _ in range(config.max_shared_per_stop):
                if not values:
                    break
                current=self.client.state.current_channel
                def cost(channel):
                    # A switch back is reserved even when the primary is next
                    # cleared without a measurement. With several extra reads
                    # this reserves return more than once, conservatively.
                    return 5+int(channel!=current)+int(channel!=primary)
                def net(channel):
                    return values[channel]["travel_saving_proxy_m"]/5-cost(channel)
                channel=max(values,key=lambda c:(net(c),-c))
                score=net(channel)
                if score<config.minimum_net_gain_s:
                    break
                value=values.pop(channel)
                estimated_cost=cost(channel)
                before=self.client.state.virtual_time_s
                action_began=time.perf_counter()
                try:
                    response=self._perform("measure",q,channel,"axis_active_shared_observation")
                finally:
                    action_wall+=time.perf_counter()-action_began
                charged=self.client.state.virtual_time_s-before
                region=self.regions[channel]
                stats["shared_measurements"]+=1
                stats["actual_measure_cost_s"]+=charged
                stats["decisions"].append({"primary_channel":primary,"channel":channel,
                    "position":[q.x,q.y],"estimated_net_gain_s":score,
                    "conservative_measure_switch_return_s":estimated_cost,
                    "reserved_return_switch_s":int(channel!=primary),
                    "actual_measure_cost_s":charged,"response":response["measure_result"],
                    "actual_radius_m":region.enclosing_disk().radius if region.vertices else None,**value})
        finally:
            self._active_sharing=False
            stats["planning_s"]+=time.perf_counter()-began-action_wall


def run_sharing_state_search(client,*,problem=3,max_actions=10000,max_active_probes=6,
                             config=None,sharing_config=None):
    if problem!=3:
        raise ValueError("axis active sharing supports Q3 only")
    if type(max_actions) is not int or max_actions<2:
        raise ValueError("max_actions must be an integer >=2")
    if type(max_active_probes) is not int or not 0<=max_active_probes<=30:
        raise ValueError("max_active_probes must be an integer in [0,30]")
    return SharingStateSearch(client,max_actions,max_active_probes,config,sharing_config).run()
