"""Q3 fixed-layout scheduling with conditional unknown-source insertion mass."""

from dataclasses import asdict
import time

from planning.state_route import RouteTask, solve_state_route
from planning.unknown_mass import build_unknown_belief, insertion_recourse

from .refined_state_search import RefinedStateSearch


class RecourseStateSearch(RefinedStateSearch):
    def __init__(self, client, max_actions, max_active_probes, config, *, credit_saved_scans,
                 recourse_candidates, recourse_max_expansions):
        super().__init__(client,max_actions,max_active_probes,config,"axis_quantile")
        self.credit_saved_scans=credit_saved_scans
        self.recourse_candidates=recourse_candidates
        self.recourse_max_expansions=recourse_max_expansions
        self.recourse_log=[]
        self.report.strategy_parameters.update({"recourse_log":self.recourse_log,
            "model":"finite_unknown_mass_and_additive_insertion_surrogate",
            "unknown_prior":"uniform N=10..16; uniform labeled subset; 96 area-centroid nodes; one continuous uniform R per source",
            "recourse_bound_scope":"additive expected insertion surrogate; NOT an original-Q3 bound",
            "credit_saved_scans":credit_saved_scans,"recourse_candidates":recourse_candidates,
            "recourse_max_expansions":recourse_max_expansions})

    def _next_task(self, remaining):
        self.remaining_covers=remaining
        known=self.detected|self.cleared
        if not remaining or len(known)==16:
            return super()._next_task(remaining)
        began=time.perf_counter()
        initial_expanded=self.total_expansions
        belief=build_unknown_belief(self.report.action_history,known)
        if belief is None or belief.expected_unknown<1e-10:
            self.recourse_log.append({"status":"quadrature_empty_or_no_expected_sources",
                "expected_unknown":belief.expected_unknown if belief else None})
            return super()._next_task(remaining)
        sources=[("source",channel,target) for channel in sorted(self.detected-self.cleared-self.blocked)
                 if (target:=self._target(channel)) is not None]
        tasks=[("cover",None,p) for p in remaining]+sources
        background=max(0,20-len(self.cleared)-len(sources))
        finite=[RouteTask(t[2],t[0]=="source",5. if t[0]=="source" else 6.*background) for t in tasks]
        current=self.client.state.position
        budget=max(0,min(self.state_config.max_expansions,self.state_config.max_total_expansions-self.total_expansions))
        baseline=solve_state_route(finite,current,scan_source_s=self.state_config.scan_source_s,max_expansions=budget)
        self.total_expansions+=baseline.expanded
        firsts=[baseline.order[0]]
        covers=sorted(range(len(remaining)),key=lambda i:(current.distance_to(tasks[i][2]),i))
        source_indices=sorted(range(len(remaining),len(tasks)),key=lambda i:(current.distance_to(tasks[i][2]),i))
        for index in covers[:1]+source_indices[:1]+covers[1:]+source_indices[1:]:
            if index not in firsts:
                firsts.append(index)
            if len(firsts)>=self.recourse_candidates:
                break
        alternatives=[]
        for first in firsts:
            if first==baseline.order[0]:
                order,result=baseline.order,baseline
            else:
                indices=[i for i in range(len(tasks)) if i!=first]
                cap=max(0,min(self.recourse_max_expansions,self.state_config.max_total_expansions-self.total_expansions))
                suffix=solve_state_route([finite[i] for i in indices],tasks[first][2],
                    scan_source_s=0.,max_expansions=cap)
                self.total_expansions+=suffix.expanded
                initial=current.distance_to(tasks[first][2])/5+finite[first].service_s
                # scan_source_s is fixed at zero for this research candidate.
                result=type(suffix)((first,)+tuple(indices[i] for i in suffix.order),
                    initial+suffix.cost_s,initial+suffix.lower_bound_s,suffix.expanded,
                    suffix.generated,suffix.dominance_pruned,suffix.bound_pruned,suffix.exact,suffix.runtime_s)
                order=result.order
            correction=insertion_recourse([finite[i] for i in order],belief,credit_saved_scans=self.credit_saved_scans)
            # Missing quadrature coverage may not silently win by omitting work.
            if correction["missed_unknown_mass"]>1e-7:
                self.recourse_log.append({"status":"unaccounted_mass_fallback",**correction})
                return super()._next_task(remaining)
            alternatives.append({"first_index":first,"order":order,"route_result":result,
                "score_s":result.cost_s+correction["correction_s"],**correction})
        chosen=min(alternatives,key=lambda v:(v["score_s"],firsts.index(v["first_index"])))
        task=tasks[chosen["first_index"]]
        result=chosen["route_result"]
        self.search_log.append({"task_count":len(tasks),"source_count":len(sources),
            "cover_count":len(remaining),"selected_kind":task[0],"selected_channel":task[1],
            "selected_point":[task[2].x,task[2].y],
            **{k:v for k,v in asdict(result).items() if k!="order"},
            "frozen_selected_route_runtime_s":result.runtime_s,
            "frozen_selected_route_expanded":result.expanded,
            "runtime_s":time.perf_counter()-began,
            "expanded":self.total_expansions-initial_expanded,
            "model_gap_s":result.cost_s-result.lower_bound_s,
            "bound_scope":"frozen known tasks, selected first fixed; excludes recourse"})
        self.recourse_log.append({"status":"selected","expected_unknown":belief.expected_unknown,
            "known_count":len(known),"channel_presence":belief.presence,
            "count_probabilities":belief.count_probabilities,"compressed_mass_groups":len(belief.groups),
            "selected_kind":task[0],"selected_channel":task[1],
            "changed_first_task":chosen["first_index"]!=baseline.order[0],
            "baseline_first_kind":tasks[baseline.order[0]][0],
            "alternatives":[{k:v for k,v in a.items() if k not in ("route_result",)} for a in alternatives],
            "runtime_s":time.perf_counter()-began})
        return task


def run_recourse_state_search(client, *, problem=3,max_actions=10000,max_active_probes=6,
                              config=None,credit_saved_scans=True,recourse_candidates=4,
                              recourse_max_expansions=100):
    if problem!=3:
        raise ValueError("Unknown-source recourse supports Q3 only")
    if isinstance(max_actions,bool) or not isinstance(max_actions,int) or max_actions<2:
        raise ValueError("max_actions must be integer >=2")
    if isinstance(max_active_probes,bool) or not isinstance(max_active_probes,int) or not 0<=max_active_probes<=30:
        raise ValueError("max_active_probes must be in [0,30]")
    if not isinstance(credit_saved_scans,bool):
        raise ValueError("credit_saved_scans must be boolean")
    if isinstance(recourse_candidates,bool) or not isinstance(recourse_candidates,int) or not 2<=recourse_candidates<=12:
        raise ValueError("recourse_candidates must be in [2,12]")
    if isinstance(recourse_max_expansions,bool) or not isinstance(recourse_max_expansions,int) or not 0<=recourse_max_expansions<=10000:
        raise ValueError("recourse_max_expansions must be in [0,10000]")
    if config is not None and not isinstance(config,dict):
        raise ValueError("config must be a dictionary")
    if config and config.get("scan_source_s",0)!=0:
        raise ValueError("Recourse comparison fixes scan_source_s=0")
    normalized={"scan_source_s":0.,**(config or {})}
    return RecourseStateSearch(client,max_actions,max_active_probes,normalized,
        credit_saved_scans=credit_saved_scans,recourse_candidates=recourse_candidates,
        recourse_max_expansions=recourse_max_expansions).run()
