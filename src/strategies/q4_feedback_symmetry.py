"""Select one equal-length D7 cover order after the real origin full scan."""
import time

from planning.cover_symmetry import CONFIGS, d7_candidates, select_feedback_symmetry
from simulator_client.state import Position
from .q4_joint_continuation import Q4JointContinuation


class Q4FeedbackSymmetry(Q4JointContinuation):
    def __init__(self, client, max_actions, max_active_probes, *, max_expansions,
                 config="bearing_mean"):
        if config not in CONFIGS:
            raise ValueError("Unknown feedback-symmetry configuration")
        super().__init__(client, max_actions, max_active_probes, max_expansions=max_expansions)
        self.symmetry_config = config
        self.symmetry_base_points = tuple(self.points)
        self.symmetry_candidates = d7_candidates(self.symmetry_base_points)
        self.symmetry_log = []
        self.report.strategy_parameters.update(feedback_symmetry_config=config,
            feedback_symmetry_log=self.symmetry_log,
            feedback_symmetry_limits=dict(candidate_count=14, mean_minimum_directions=2,
                mean_minimum_concentration=.5, early_station_count=3,
                early_improvement_tolerance_m=1e-9, route_length_tolerance_m=1e-7),
            feedback_symmetry_scope="One D7 original-coordinate permutation after the paid full origin scan; same station set and pure cover length, no new measurements or inferred coverage")

    def _execute_plan(self):
        # This fresh-session boundary permits exactly the original origin scan.
        # A partly completed scan propagates its parent stop; it never selects
        # an order from an incomplete or outcome-dependent channel prefix.
        if (self.report.action_history or self.symmetry_log or self.detected or self.cleared
                or self.report.coverage_points_visited or self.client.state.position != Position(0., 0.)):
            raise ValueError("Feedback symmetry requires a fresh origin observation prefix")
        actions_before = self.actions
        virtual_before = self.client.state.virtual_time_s
        self._scan(self.symmetry_base_points[0])
        h = self.report.action_history
        if (len(h) != 20 or {a["channel"] for a in h} != set(range(1, 21))
                or any(a["action"] != "measure" or a["phase"] != "coverage"
                       or a["position"] != [0., 0.] for a in h)
                or self.report.coverage_points_visited != 1):
            raise ValueError("Order selection needs the complete real origin scan")
        began = time.perf_counter()
        bearings = [dict(channel=a["channel"], bearing_deg=a["bearing_deg"])
                    for a in sorted(h, key=lambda x: x["channel"]) if a["result"] == "direction"]
        centers = [dict(channel=c, center=list(self.regions[c].enclosing_disk().center))
                   for c in sorted(self.regions)
                   if self.regions[c].observations and self.regions[c].vertices]
        selected, decision = select_feedback_symmetry(self.symmetry_base_points, self.symmetry_candidates,
            config=self.symmetry_config, bearings=[x["bearing_deg"] for x in bearings],
            centers=[x["center"] for x in centers], known_count=len(self.detected | self.cleared))
        ordered = tuple(self.symmetry_base_points[i] for i in decision["selected_permutation"])
        event = dict(config=self.symmetry_config, origin_scan_start=0, origin_scan_end=len(h),
            after_actual_action_count=len(h), end_actual_action_count=len(h),
            current_position=[self.client.state.position.x, self.client.state.position.y],
            current_channel=self.client.state.current_channel,
            known_channels=sorted(self.detected | self.cleared), bearings=bearings, positive_centers=centers,
            original_full_points=[[p.x, p.y] for p in self.symmetry_base_points],
            selected_full_points=[[p.x, p.y] for p in ordered],
            coverage_points_visited=self.report.coverage_points_visited,
            budget=dict(policy_actions_before_scan=actions_before, policy_actions_after_scan=self.actions,
                max_actions=self.max_actions, virtual_before_scan=virtual_before,
                virtual_after_scan=self.client.state.virtual_time_s,
                max_virtual_duration_s=self.client.state.max_virtual_duration_s,
                remaining_real_s=getattr(self.client, "remaining_real_time_s", None)),
            decision_wall_s=time.perf_counter()-began, **decision)
        self.symmetry_log.append(event)
        self.report.coverage_points = [[p.x, p.y] for p in ordered]
        # The parent constructs its local remaining list at entry. Supplying
        # only the unvisited 21 points avoids stale local copies or a free/repeated
        # origin visit, while the report/certificate retain all22 exact points.
        self.points = ordered[1:]
        try:
            return super()._execute_plan()
        finally:
            self.points = ordered


def run_q4_feedback_symmetry(client, *, problem=4, config="bearing_mean", max_actions=20000,
                             max_active_probes=6, max_expansions=200):
    if type(problem) is not int or problem != 4 or config not in CONFIGS:
        raise ValueError("Q4 only; use a fixed feedback-symmetry configuration")
    for value, low, high in ((max_actions, 2, 1_000_000), (max_active_probes, 0, 30),
                             (max_expansions, 0, 10000)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Invalid integer execution budget")
    return Q4FeedbackSymmetry(client, max_actions, max_active_probes,
        max_expansions=max_expansions, config=config).run()
