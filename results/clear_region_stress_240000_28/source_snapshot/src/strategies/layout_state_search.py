"""Choose a certified discovery layout from legal initial observations only."""

from dataclasses import asdict
import math
import time

from planning import nearest_order
from planning.ring_layout import ring_cover_radius, ring_sites
from planning.state_route import RouteTask, solve_state_route
from simulator_client.state import Position

from .state_search import StateSearch


class LayoutStateSearch(StateSearch):
    def __init__(self, client, max_actions, max_active_probes, config, *,
                 layout_radii, phase_step_deg, reselect_before_outer,
                 layout_max_expansions):
        super().__init__(client, max_actions, max_active_probes, config)
        self.layout_radii = tuple(layout_radii)
        self.phase_step_deg = phase_step_deg
        self.reselect_before_outer = reselect_before_outer
        self.layout_max_expansions = layout_max_expansions
        self.layout_selected = False
        self.layout_log = []
        self.report.strategy_parameters.update({
            "layout_radii": self.layout_radii, "layout_phase_step_deg": phase_step_deg,
            "reselect_before_outer": reselect_before_outer,
            "layout_log": self.layout_log,
            "layout_selection_scope": "frozen observed tasks; future discoveries are not predicted",
        })

    def _choose_layout(self, remaining):
        started = time.perf_counter()
        sources = [(c, target) for c in sorted(self.detected - self.cleared - self.blocked)
                   if (target := self._target(c)) is not None]
        current = self.client.state.position
        background = max(0, 20 - len(self.cleared) - len(sources))
        candidates = [("incumbent", None, None, tuple(remaining))]
        for radius in self.layout_radii:
            for i in range(60 // self.phase_step_deg):
                phase = i * self.phase_step_deg
                points = nearest_order(ring_sites(radius, phase))[1:]
                if tuple(points) != tuple(remaining):
                    candidates.append(("finite_ring", radius, phase, points))
        evaluated, best, chosen = [], math.inf, tuple(remaining)
        for kind, radius, phase, points in candidates:
            tasks = [RouteTask(point, False, 6.0 * background) for point in points]
            tasks.extend(RouteTask(point, True, 5.0) for _, point in sources)
            budget = max(0, min(self.layout_max_expansions,
                               self.state_config.max_total_expansions - self.total_expansions))
            result = solve_state_route(tasks, current,
                scan_source_s=self.state_config.scan_source_s, max_expansions=budget)
            self.total_expansions += result.expanded
            evaluated.append({"kind": kind, "radius_m": radius, "phase_deg": phase,
                              "cover_radius_m": ring_cover_radius(radius) if radius else None,
                              **{k: v for k, v in asdict(result).items() if k != "order"}})
            if result.cost_s < best - 1e-7:
                best, chosen = result.cost_s, tuple(points)
                selected = len(evaluated) - 1
        remaining[:] = chosen
        self.points = (Position(0, 0),) + chosen
        self.report.coverage_points = [[p.x, p.y] for p in self.points]
        self.layout_log.append({
            "position": [current.x, current.y], "observed_source_channels": [c for c, _ in sources],
            "cleared_channels": sorted(self.cleared), "selected": selected,
            "candidates": evaluated, "frozen_union_lower_bound_s": min(v["lower_bound_s"] for v in evaluated),
            "frozen_union_upper_bound_s": best, "runtime_s": time.perf_counter() - started,
            "coverage_certificate": "origin plus six equal-radius equally spaced sites; analytic full disk",
        })
        self.layout_selected = True

    def _next_task(self, remaining):
        # Replacing all six unvisited ring tasks retains an analytic certificate.
        # Once even one outer station was scanned, keep the selected layout;
        # arbitrary mixing of old and rotated subsets would invalidate it.
        can_choose = (len(self.discovery_stations) == 1 and len(remaining) == 6
                      and len(self.detected | self.cleared) < 16)
        if can_choose and (not self.layout_selected or self.reselect_before_outer):
            self._choose_layout(remaining)
        return super()._next_task(remaining)


def run_layout_state_search(client, *, problem=3, max_actions=10000,
                            max_active_probes=6, config=None,
                            layout_radii=(1150.0,), phase_step_deg=10,
                            reselect_before_outer=False, layout_max_expansions=100):
    if problem != 3:
        raise ValueError("layout research supports only Q3")
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 2:
        raise ValueError("max_actions must be an integer >=2")
    if (isinstance(max_active_probes, bool) or not isinstance(max_active_probes, int)
            or not 0 <= max_active_probes <= 30):
        raise ValueError("max_active_probes must be an integer in [0,30]")
    if not isinstance(layout_radii, (tuple, list)) or not 1 <= len(layout_radii) <= 4:
        raise ValueError("layout_radii needs one to four certified radii")
    for radius in layout_radii:
        ring_cover_radius(radius)
    if (isinstance(phase_step_deg, bool) or not isinstance(phase_step_deg, int)
            or phase_step_deg not in (5, 10, 15, 20, 30, 60)):
        raise ValueError("phase step must be one of 5,10,15,20,30,60 degrees")
    if not isinstance(reselect_before_outer, bool):
        raise ValueError("reselect_before_outer must be boolean")
    if (isinstance(layout_max_expansions, bool) or not isinstance(layout_max_expansions, int)
            or not 0 <= layout_max_expansions <= 10000):
        raise ValueError("layout_max_expansions must be an integer in [0,10000]")
    return LayoutStateSearch(client, max_actions, max_active_probes, config,
        layout_radii=layout_radii, phase_step_deg=phase_step_deg,
        reselect_before_outer=reselect_before_outer,
        layout_max_expansions=layout_max_expansions).run()
