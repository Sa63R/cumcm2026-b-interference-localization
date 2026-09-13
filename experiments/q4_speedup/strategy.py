import bootstrap

def build_candidate(method):
    if method=='prior_fixed':
        from online import State
        from unrestricted_planner import Planner,PlanningConfig
        s=State(analytic=True)
        s.planner=Planner(90210000,PlanningConfig(coarse_scenarios=8,verify_scenarios=8,max_decisions=8,seconds_per_decision=5.))
        return s
    if method.startswith('sectors'):
        from online import State
        from q4_v4_solver import V4Config
        return State(config=V4Config(direction_sectors=int(method[7:])))
    if method=='domain':
        from domain_candidate import CandidateState
        return CandidateState()
    if method=='nearest_cells':
        from local_candidate import CandidateState
        return CandidateState(local_mode='nearest_cells',local_parameters={'optical_cover_limit':12})
    if method=='flex':
        from coverage_candidate import FlexibleState
        return FlexibleState()
    if method=='tangent':
        from coverage_candidate import TangentialState
        return TangentialState()
    if method in ('cells_flex','cells_domain','combined'):
        from local_candidate import CandidateState as LocalState
        from coverage_candidate import FlexibleState
        from domain_candidate import DomainMixin
        class CellsFlex(FlexibleState,LocalState):pass
        class CellsDomain(DomainMixin,LocalState):pass
        class Combined(DomainMixin,FlexibleState,LocalState):pass
        cls={'cells_flex':CellsFlex,'cells_domain':CellsDomain,'combined':Combined}[method]
        return cls(local_mode='nearest_cells',local_parameters={'optical_cover_limit':12})
    raise ValueError(method)
