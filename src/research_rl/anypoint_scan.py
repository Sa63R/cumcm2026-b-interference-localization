"""Bounded unknown-channel measurements away from the seven cover sites.

The actor can discover a new source at its current position or at a small set
of already available source probe/clear destinations. These are real new
physical actions, not a change to the legacy coverage mask. All old actions,
the seven-site completion certificate and baseline fallback stay available.
No hidden source state, source count, or simulated future feedback is read.
"""

from simulator_client.state import Position
from .action_sets import action_schema
from .controller import Candidate
from .joint_scan import JointScanRLSearch


class AnyPointCurrentRLSearch(JointScanRLSearch):
    schema_name = "anypoint_current"

    def __init__(self, client, policy, *, feature_version="v3", **kwargs):
        if feature_version != "v3":
            raise ValueError("anypoint scan requires v3 feature semantics")
        schema = action_schema(self.schema_name)
        if hasattr(policy, "action_schema") and policy.action_schema != schema:
            raise ValueError("anypoint controller requires a matching policy action schema")
        super().__init__(client, policy, feature_version=feature_version, **kwargs)
        self.anypoint_schema = schema
        # This is accepted, actually observed evidence, independent of the
        # inherited rounded source-probe cache and the seven-site cover ledger.
        self.measurement_ledger = {}
        self.anypoint_measurements = 0
        self.report.learning.update(
            action_schema=schema, anypoint_measurements=0,
            anypoint_current_measurements=0, anypoint_target_measurements=0,
            anypoint_candidates_offered_sum=0, anypoint_measurement_limit=32,
            anypoint_virtual_time_s=0.0,
            measurement_ledger=[],
            measurement_count_semantics="scan_measurements=legacy-cover; anypoint=extra-discovery; probes=source-localization")
        self.report.learning["learned_scope"].append("current-position-unknown-channel")
        if schema["max_target_points"]:
            self.report.learning["learned_scope"].append("known-target-position-unknown-channel")
        self.report.learning["fixed_scope"].append("bounded-anypoint-destinations-and-measurements")

    def _perform(self, action, position, channel, phase):
        position = Position.coerce(position)
        before = self.actions
        try:
            return super()._perform(action, position, channel, phase)
        finally:
            # Even an accepted action immediately followed by a model/budget
            # stop must be accounted for. Rejected requests never create proof.
            if action == "measure" and self.actions > before:
                self.measurement_ledger.setdefault(position, set()).add(channel)

    def _unknown_channels(self):
        known = self.detected | self.cleared
        # The public upper bound proves the other channels contain no source.
        # It does not remove ANY old cover action or known-source localization.
        if len(known) >= 16:
            return set()
        return {channel for channel in set(range(1, 21)) - known
                if any(channel not in measured for measured in self.scan_ledger.values())}

    def _candidates(self, remaining):
        original = super()._candidates(remaining)
        if not original or self.anypoint_measurements >= self.anypoint_schema["max_extra_measurements"]:
            return original
        unknown = self._unknown_channels()
        if not unknown:
            return original
        current = self.client.state.position
        # Pair identity is physical (point, channel), never feature rounding or
        # the arbitrary order of candidate rows. Preserve the complete prefix.
        present = {(c.point, c.channel) for c in original if c.kind in {"cover", "probe"}}

        def fresh_channels(point):
            return sorted(channel for channel in unknown - self.measurement_ledger.get(point, set())
                          if (point, channel) not in present)

        sites = [(current, 7)]
        maximum = self.anypoint_schema["max_target_points"]
        if maximum:
            points = {c.point for c in original if c.kind in {"probe", "clear"}
                      and c.point != current and fresh_channels(c.point)}
            sites.extend((point, 8) for point in sorted(
                points, key=lambda p: (current.distance_to(p), p.x, p.y))[:maximum])
        extra = [Candidate("cover", channel, point, option=option)
                 for point, option in sites for channel in fresh_channels(point)]
        self.report.learning["anypoint_candidates_offered_sum"] += len(extra)
        return original + extra

    def _execute_candidate(self, candidate, remaining):
        if candidate.kind != "cover" or candidate.option not in (7, 8):
            return super()._execute_candidate(candidate, remaining)
        if (self.anypoint_measurements >= self.anypoint_schema["max_extra_measurements"]
                or candidate.channel not in self._unknown_channels()
                or candidate.channel in self.measurement_ledger.get(candidate.point, set())):
            raise ValueError("anypoint measurement is stale, repeated, or over budget")
        metrics = self.report.learning
        before_actions, before_time = self.actions, self.client.state.virtual_time_s
        if self.scan_focus is not None and self._pending(self.scan_focus):
            metrics["interrupted_scans"] += 1
        try:
            self._perform("measure", candidate.point, candidate.channel, "rl_anypoint_discovery")
        finally:
            if self.actions > before_actions:
                self.anypoint_measurements += 1
                metrics["anypoint_measurements"] = self.anypoint_measurements
                family = "current" if candidate.option == 7 else "target"
                metrics[f"anypoint_{family}_measurements"] += 1
                metrics["anypoint_virtual_time_s"] += self.client.state.virtual_time_s - before_time
                self.report.action_history[-1]["rl_discovery_family"] = family
        # Do not turn an arbitrary scan position into a fixed coverage focus.
        # New source feedback uses the existing region and localization logic.
        self.blocked.clear()
        self.focus = None

    def run(self):
        try:
            return super().run()
        finally:
            self.report.learning["measurement_ledger"] = [
                {"position": [point.x, point.y], "channels": sorted(channels)}
                for point, channels in sorted(self.measurement_ledger.items(),
                                              key=lambda item: (item[0].x, item[0].y))]


class AnyPointTargetsRLSearch(AnyPointCurrentRLSearch):
    schema_name = "anypoint_targets"
