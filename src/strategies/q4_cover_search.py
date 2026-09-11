"""Q4 compact certified discovery with joint or deferred source scheduling."""
from simulator_client.state import Position
from .q4_state_search import Q4StateSearch


class Q4CoverSearch(Q4StateSearch):
    def __init__(self, client, max_actions, max_active_probes, *, profile, schedule, max_expansions):
        # Construction validates the geometric cover before any client entry.
        from planning.q4_directional_cover import certified_cover_points
        points, certificate = certified_cover_points(profile)
        if certificate.get("passed") is not True:
            raise ValueError("Q4 discovery needs a completed geometric certificate")
        if schedule not in {"joint", "deferred"}:
            raise ValueError("Unknown compact-cover schedule")
        super().__init__(client, max_actions, max_active_probes,
                         mode="state_pruned", max_expansions=max_expansions)
        self.points = tuple(Position.coerce(p) for p in points)
        self.schedule = schedule
        self.report.coverage_points_total = len(self.points)
        self.report.coverage_points = [[p.x, p.y] for p in self.points]
        self.report.strategy_parameters.update(
            q4_compact_profile=profile, compact_schedule=schedule,
            directional_cover_certificate=certificate,
            state_scope="Fixed certified compact cover chain plus observed source subset",
        )

    def _execute_plan(self):
        if self.schedule == "joint":
            return super()._execute_plan()
        # Ablation: preserve every certified discovery station while withholding
        # clear tasks until the end. Ready channels are still safely skipped.
        for point in self.points:
            self._scan(point)
        self.report.coverage_complete = True
        self.points = ()
        return super()._execute_plan()


def run_q4_cover_search(client, *, problem=4, max_actions=20000, max_active_probes=6,
                        profile="compact", schedule="joint", max_expansions=200):
    if problem != 4:
        raise ValueError("Compact directional cover is Q4 only")
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError("max_actions must be integer >=2")
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError("max_active_probes must be integer in [0,30]")
    if type(max_expansions) is not int or not 0 <= max_expansions <= 10000:
        raise ValueError("max_expansions must be integer in [0,10000]")
    return Q4CoverSearch(client, max_actions, max_active_probes,
                         profile=profile, schedule=schedule, max_expansions=max_expansions).run()
