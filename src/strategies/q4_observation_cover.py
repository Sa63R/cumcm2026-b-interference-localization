"""R12 feedback policy on two genuinely certified denser observation rings."""
import time

from planning.observation_cover_route import CONFIGS, observation_cover_route
from .q4_joint_continuation import Q4JointContinuation


class Q4ObservationCover(Q4JointContinuation):
    def __init__(self, client, max_actions, max_active_probes, *, max_expansions, config):
        began = time.perf_counter()
        points, certificate, metadata = observation_cover_route(config)
        super().__init__(client, max_actions, max_active_probes, max_expansions=max_expansions)
        if certificate.get("passed") is not True or points[0].x != 0. or points[0].y != 0.:
            raise ValueError("Observation cover needs a completed proof and an origin start")
        if len(points) != certificate["station_count"] or len(set(points)) != len(points):
            raise ValueError("Observation station count differs from its proof")
        self.points = points
        self.report.coverage_points_total = len(points)
        self.report.coverage_points = [[p.x,p.y] for p in points]
        self.report.strategy_parameters.update(
            observation_cover_config=config, observation_cover_route=metadata,
            observation_cover_setup_runtime_s=time.perf_counter()-began,
            q4_compact_profile="observation_"+config,
            directional_cover_certificate=certificate,
            state_scope="Fixed newly certified observation-ring cover chain plus actually observed ready-source subset; R12 feedback/action methods inherited",
            observation_cover_scope="Only construction-time stations, fixed route and matching discovery certificate change; canonical observations, range skips, R8/R12 localization and actual source completion remain inherited",
        )


def run_q4_observation_cover(client, *, problem=4, config="ring_28", max_actions=20000,
                             max_active_probes=6, max_expansions=200):
    if type(problem) is not int or problem != 4:
        raise ValueError("Observation rings are Q4 only")
    if not isinstance(config,str) or config not in CONFIGS:
        raise ValueError("Only ring_28 and ring_31 are permitted")
    for value,low,high in ((max_actions,2,1_000_000),(max_active_probes,0,30),(max_expansions,0,10000)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Invalid integer execution budget")
    return Q4ObservationCover(client,max_actions,max_active_probes,
                               max_expansions=max_expansions,config=config).run()
