"""Two isolated, observation-only scan reductions over frozen relocating v1."""

from planning.relative_silence import relative_silence_certificate
from planning.silence_certificate import certify_silence
from simulator_client.state import Position

from .relocating_state_search import RelocatingStateSearch


class DerivedSilenceStateSearch(RelocatingStateSearch):
    def __init__(self, client, max_actions, max_active_probes, config, *,
                 enabled=True, relative_silence=True, stop_scan_at_16=True):
        for name, value in (("enabled", enabled), ("relative_silence", relative_silence),
                            ("stop_scan_at_16", stop_scan_at_16)):
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
        # `enabled` controls only the new mechanisms, never the original v1
        # relocation, inferred silence, measurements or routing configuration.
        super().__init__(client, max_actions, max_active_probes, config, True)
        self.derived_enabled = enabled
        self.relative_silence_enabled = relative_silence
        self.stop_scan_at_16 = stop_scan_at_16
        self.scan_audit = []
        self.derived_stats = {"relative_silence_skips": 0, "count_cap_skips": 0,
                              "completed_geometric_scans": 0, "count_certified_scans": 0,
                              "no_physical_action_scans": 0}
        self.report.strategy_parameters.update({
            "derived_silence_config": {"enabled": enabled, "relative_silence": relative_silence,
                                       "stop_scan_at_16": stop_scan_at_16},
            "derived_silence_stats": self.derived_stats,
            "derived_scan_audit": self.scan_audit,
            "derived_silence_scope": "same known uncleared Q3 channel; prior actual no_signal only; no fabricated client action or unknown-channel coverage",
        })

    def _relative_certificate(self, channel, point, region):
        if channel not in self.detected or channel in self.cleared or region is None:
            return None
        # This report contains only accepted real client actions. In particular,
        # region.no_signal_positions also contains inferred data and is NOT an
        # admissible provenance source for this first, non-chained mechanism.
        history = self.report.action_history
        if any(a["action"] == "clear" and a["channel"] == channel and a["result"] == "success"
               for a in history):
            return None
        if not any(a["action"] == "measure" and a["channel"] == channel
                   and a["result"] in ("direction", "near") for a in history):
            return None
        for index, action in enumerate(history, 1):
            if (action["action"] != "measure" or action["channel"] != channel
                    or action["result"] != "no_signal"):
                continue
            certificate = relative_silence_certificate(region, point, action["position"])
            if certificate is not None:
                return {**certificate, "witness_action_ordinal": index,
                        "witness_virtual_time_s": action["virtual_time_s"],
                        "witness_kind": "prior_actual_measure_no_signal"}
        return None

    def _infer(self, channel, point, region, certificate):
        region.observe_no_signal(point)
        self.inferred.append({"inference_kind": "inferred_no_signal", "channel": channel,
                              "position": [point.x, point.y],
                              "after_actual_action_count": len(self.report.action_history),
                              "physical_measurement": False,
                              "virtual_time_s": self.client.state.virtual_time_s,
                              **certificate})

    def _scan(self, point):
        if not self.derived_enabled or not (self.relative_silence_enabled or self.stop_scan_at_16):
            return super()._scan(point)
        point = Position.coerce(point)
        start = len(self.report.action_history)
        channels = [c for c in range(1, 21) if c not in self.cleared]
        current = self.client.state.current_channel
        if current in channels:
            channels.remove(current)
            channels.insert(0, current)
        log = {"position": [point.x, point.y], "after_actual_action_count_before": start,
               "known_channels_before": sorted(self.detected | self.cleared),
               "actual_measured_channels": [], "count_skipped": [],
               "relative_silence_skipped": [], "old_silence_skipped": [],
               "clear_certified_skipped": [], "completed": False,
               "certified_by_count": False, "credited_geometric_station": False}
        self.scan_audit.append(log)
        try:
            for channel in channels:
                known = self.detected | self.cleared
                if self.stop_scan_at_16 and len(known) == 16 and channel not in known:
                    log["count_skipped"].append({"channel": channel,
                        "after_actual_action_count": len(self.report.action_history),
                        "known_channels": sorted(known)})
                    self.derived_stats["count_cap_skips"] += 1
                    continue
                region = self.regions.get(channel)
                certified = (channel in self.near_points or (
                    channel in self.detected and region and region.vertices
                    and region.enclosing_disk().radius <= 19.9))
                if self.state_config.skip_certified_scans and certified:
                    self.skipped_measurements += 1
                    log["clear_certified_skipped"].append(channel)
                    continue
                certificate = (certify_silence(region, point)
                               if channel in self.detected and region else None)
                if certificate:
                    self._infer(channel, point, region, certificate)
                    log["old_silence_skipped"].append(channel)
                    continue
                certificate = (self._relative_certificate(channel, point, region)
                               if self.relative_silence_enabled else None)
                if certificate:
                    self._infer(channel, point, region, certificate)
                    log["relative_silence_skipped"].append(channel)
                    self.derived_stats["relative_silence_skips"] += 1
                    continue
                self._perform("measure", point, channel, "coverage")
                log["actual_measured_channels"].append(channel)
            log["completed"] = True
            unknown = set(range(1, 21)) - self.detected - self.cleared
            measured = set(log["actual_measured_channels"])
            log["certified_by_count"] = len(self.detected | self.cleared) == 16
            # A geometric station must correspond to actual physical queries on
            # every still-unknown channel, even when the count cap also holds.
            # All-skipped scans never move the dog or create a visited station.
            if measured and unknown <= measured:
                self.report.coverage_points_visited += 1
                self.discovery_stations.append(point)
                log["credited_geometric_station"] = True
                self.derived_stats["completed_geometric_scans"] += 1
            if log["certified_by_count"]:
                self.derived_stats["count_certified_scans"] += 1
            if not measured:
                self.derived_stats["no_physical_action_scans"] += 1
            self.blocked.clear()
        finally:
            log["after_actual_action_count_after"] = len(self.report.action_history)
            log["known_channels_after"] = sorted(self.detected | self.cleared)
            log["physical_action_count"] = len(self.report.action_history) - start
            self.report.strategy_parameters["skipped_certified_measurements"] = self.skipped_measurements


def run_derived_silence_state_search(client, *, problem=3, max_actions=10000,
                                   max_active_probes=6, config=None, enabled=True,
                                   relative_silence=True, stop_scan_at_16=True):
    if problem != 3:
        raise ValueError("Derived silence supports Q3 only")
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError("max_actions must be integer >=2")
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError("max_active_probes must be in [0,30]")
    return DerivedSilenceStateSearch(client, max_actions, max_active_probes, config,
        enabled=enabled, relative_silence=relative_silence,
        stop_scan_at_16=stop_scan_at_16).run()


__all__ = ["DerivedSilenceStateSearch", "run_derived_silence_state_search"]
