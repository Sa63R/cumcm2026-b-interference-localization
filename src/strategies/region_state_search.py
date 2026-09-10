"""Scheduling-only Q3 ablation: mean endpoint versus expected travel matrix."""

from dataclasses import asdict
import time

from planning.region_travel import point_mass, region_mass, travel_matrix
from planning.state_route import RouteTask, solve_state_route

from .refined_state_search import RefinedStateSearch


class RegionStateSearch(RefinedStateSearch):
    def __init__(self,client,max_actions,max_active_probes,config,mode):
        super().__init__(client,max_actions,max_active_probes,config,"axis_quantile")
        self.region_mode=mode
        self.region_log=[]
        self.report.strategy_parameters.update({"region_travel_mode":mode,
            "region_travel_log":self.region_log,
            "model":"frozen_finite_region_travel_proxy",
            "search_bound_scope":"frozen mean/expected travel matrix only; not original Q3",
            "region_prior":"12 stratified area nodes in conservative bearing polygon; one shared R uniform 1000..1500, conditioned on recorded reception signs"})

    def _next_task(self,remaining):
        began=time.perf_counter()
        self.remaining_covers=remaining
        sources=[("source",channel,target) for channel in sorted(self.detected-self.cleared-self.blocked)
                 if (target:=self._target(channel)) is not None]
        covers=[("cover",None,p) for p in remaining]
        if self.state_config.stop_discovery_at_16 and len(self.detected|self.cleared)==16:
            covers=[]
        tasks=covers+sources
        if not tasks:
            return None
        masses=[]
        source_stats=[]
        for kind,channel,point in tasks:
            if kind=="cover" or channel in self.near_points:
                mass=point_mass(point)
            else:
                region=self.regions[channel]
                # Already certified sources terminate at a known safe location;
                # do not deliberately spread that endpoint again.
                mass=(point_mass(point,"certified_endpoint") if region.enclosing_disk().radius<=19.9
                      else region_mass(region,point))
            masses.append(mass)
            if kind=="source":
                source_stats.append({"channel":channel,"status":mass.status,"nodes":len(mass.points),
                    "proposed_nodes":mass.proposed_nodes,"mass":mass.unnormalized_mass,
                    "variance_m2":mass.variance_m2,"mean":[mass.mean.x,mass.mean.y],
                    "mean_shift_from_mec_m":mass.mean.distance_to(point)})
        matrix,initial,variances=travel_matrix(masses,self.client.state.position,self.region_mode)
        background=max(0,20-len(self.cleared)-len(sources))
        finite=[RouteTask(p,kind=="source",5. if kind=="source" else 6.*background) for kind,_,p in tasks]
        budget=max(0,min(self.state_config.max_expansions,self.state_config.max_total_expansions-self.total_expansions))
        result=solve_state_route(finite,self.client.state.position,scan_source_s=self.state_config.scan_source_s,
            max_expansions=budget,travel_times_s=matrix,initial_times_s=initial)
        self.total_expansions+=result.expanded
        chosen=tasks[result.order[0]]
        runtime=time.perf_counter()-began
        self.search_log.append({"task_count":len(tasks),"source_count":len(sources),"cover_count":len(covers),
            "selected_kind":chosen[0],"selected_channel":chosen[1],"selected_point":[chosen[2].x,chosen[2].y],
            **{k:v for k,v in asdict(result).items() if k!="order"},"runtime_s":runtime,
            "matrix_search_runtime_s":result.runtime_s,"model_gap_s":result.cost_s-result.lower_bound_s,
            "bound_scope":"frozen region travel matrix, not original Q3"})
        self.region_log.append({"sources":source_stats,"selected_kind":chosen[0],
                               "selected_channel":chosen[1],"runtime_s":runtime,**variances})
        return chosen


def run_region_state_search(client,*,problem=3,max_actions=10000,max_active_probes=6,config=None,
                            mode="expected_distance"):
    if problem!=3:
        raise ValueError("region travel supports Q3 only")
    if mode not in ("mean_point","expected_distance"):
        raise ValueError("mode must be mean_point or expected_distance")
    if isinstance(max_actions,bool) or not isinstance(max_actions,int) or max_actions<2:
        raise ValueError("max_actions must be integer >=2")
    if isinstance(max_active_probes,bool) or not isinstance(max_active_probes,int) or not 0<=max_active_probes<=30:
        raise ValueError("max_active_probes must be in [0,30]")
    return RegionStateSearch(client,max_actions,max_active_probes,config,mode).run()
