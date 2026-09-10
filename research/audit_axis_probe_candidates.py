"""Audit an unselected geometry action extension on training-only teacher paths.

This does not change a controller action or train a network. Proposed points are
never executed. The original teacher trajectory is checked against a separate
unmodified run on each identical training scenario.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from research_rl import run_rl_search
from research_rl.joint_scan import JointScanRLSearch
from simulation import LocalResearchSimulator, random_scenario
from simulator_client.state import Position
from tests.test_strategy import ObservationOnlyClient


def axis_proposals(region):
    """Finite support geometry only, no scoring or hypothetical observations."""
    vertices = region.vertices
    a, b = max(((a, b) for a in vertices for b in vertices),
               key=lambda pair: (pair[0][0]-pair[1][0])**2 + (pair[0][1]-pair[1][1])**2)
    length = math.dist(a, b)
    ux, uy = ((b[0]-a[0])/length, (b[1]-a[1])/length) if length else (1.0, 0.0)
    cx, cy = region.enclosing_disk().center
    coordinates = [(x-cx)*ux + (y-cy)*uy for x,y in vertices]
    low, high = min(coordinates), max(coordinates)
    scale = min(200.0, max(20.0, (high-low)/8))
    offsets = [(low+q*(high-low), across) for q in (.25, .375, .625, .75)
               for across in (-scale, 0.0, scale)] + [(0.0, -scale), (0.0, scale)]
    points = [Position(cx+along*ux-across*uy, cy+along*uy+across*ux) for along,across in offsets]
    # Max norm over vertices bounds every point in the convex outer polygon.
    received = [p for p in points if max(math.hypot(x-p.x,y-p.y) for x,y in vertices) <= 1000.0-1e-7]
    return received, len(points)


def key(point):
    return (round(point.x, 6), round(point.y, 6))


class AuditedTeacher(JointScanRLSearch):
    def __init__(self, client):
        super().__init__(client, lambda f,c,t: t)
        self.axis_cache = {}
        self.signatures = set()
        self.extensions = {}
        self.frames = []
        self.template_calls = self.template_proposals = self.template_received = 0
        self.template_wall_s = self.feature_wall_s = 0.0
        self.old_peak = self.extended_peak = 0

    def _source_candidates(self, channel):
        original = super()._source_candidates(channel)
        if not any(c.kind == "probe" for c in original):
            self.extensions[channel] = []
            return original
        region = self.regions[channel]
        vertices = region.vertices
        cached = self.axis_cache.get(channel)
        if cached is None or cached[0] is not vertices:
            started = time.perf_counter()
            candidates, count = axis_proposals(region)
            self.template_wall_s += time.perf_counter()-started
            self.template_calls += 1
            self.template_proposals += count
            self.template_received += len(candidates)
            cached = (vertices, candidates)
            self.axis_cache[channel] = cached
        old = {key(c.point) for c in original}
        seen = set(self.observed_positions.get(channel, ())) | old
        extra = []
        for point in cached[1]:
            if key(point) not in seen:
                seen.add(key(point))
                extra.append(point)
        self.extensions[channel] = extra
        # A vertex tuple itself is retained in signatures, avoiding Python id reuse.
        signature = (channel, vertices, self.client.state.position,
                     self.probe_counts.get(channel, 0), tuple(sorted(old)))
        if signature not in self.signatures:
            self.signatures.add(signature)
            started = time.perf_counter()
            for point in extra:
                features = self._geometric_features(channel, point)
                assert len(features) == 20 and all(math.isfinite(x) for x in features)
            self.feature_wall_s += time.perf_counter()-started
            self.frames.append(dict(channel=channel, vertices=len(vertices),
                old_probe_candidates=len(original), added_candidates=len(extra),
                received_templates=len(cached[1]),
                old_matching_templates=sum(key(p) in old for p in cached[1])))
        return original  # The proposed extension never changes behavior here.

    def _candidates(self, remaining):
        original = super()._candidates(remaining)
        active = {c.channel for c in original if c.kind == "probe"}
        self.old_peak = max(self.old_peak, len(original))
        self.extended_peak = max(self.extended_peak, len(original)+sum(len(self.extensions[c]) for c in active))
        return original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=110001)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("results/rl/axis_candidate_audit.json"))
    args = parser.parse_args(argv)
    if args.count < 32 or not 100001 <= args.seed_start <= args.seed_start+args.count-1 <= 199999:
        parser.error("Use at least 32 training-only scenarios in 100001..199999")
    rows, frames = [], []
    for seed in range(args.seed_start,args.seed_start+args.count):
        scenario = random_scenario(3, seed)
        simulator = LocalResearchSimulator(scenario)
        controller = AuditedTeacher(ObservationOnlyClient(simulator.client()))
        report = controller.run()
        plain = LocalResearchSimulator(scenario)
        reference = run_rl_search(ObservationOnlyClient(plain.client()),
            policy=lambda f,c,t:t, feature_version="v3")
        assert report.action_history == reference.action_history
        assert report.virtual_time_s == reference.virtual_time_s
        assert simulator.evaluation()["all_cleared"] and simulator.evaluation()["failed_clear_count"] == 0
        assert reference.completion_certified_under_model
        frames.extend(controller.frames)
        rows.append(dict(seed=seed, exact_original_trajectory=True,
            audited_source_states=len(controller.frames), old_candidate_peak=controller.old_peak,
            extended_candidate_peak=controller.extended_peak,
            template_calls=controller.template_calls,
            template_proposals=controller.template_proposals,
            received_templates=controller.template_received,
            template_wall_s=controller.template_wall_s,
            cached_geometry_feature_wall_s=controller.feature_wall_s))
    count = len(frames)
    result = dict(schema=1, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scenario_scope="training teacher trajectories only; no proposed point was executed",
        seed_range=[args.seed_start,args.seed_start+args.count-1], cases=len(rows),
        states=count, states_with_new_points=sum(f["added_candidates"]>0 for f in frames),
        mean_old_probes=statistics.mean(f["old_probe_candidates"] for f in frames),
        mean_added_probes=statistics.mean(f["added_candidates"] for f in frames),
        max_added_probes=max(f["added_candidates"] for f in frames),
        old_matching_template_fraction=sum(f["old_matching_templates"] for f in frames)/sum(f["received_templates"] for f in frames),
        mean_template_generation_ms_per_case=1000*statistics.mean(r["template_wall_s"] for r in rows),
        mean_extra_cached_geometry_ms_per_case=1000*statistics.mean(r["cached_geometry_feature_wall_s"] for r in rows),
        peak_original_candidates=max(r["old_candidate_peak"] for r in rows),
        peak_extended_candidates=max(r["extended_candidate_peak"] for r in rows),
        excludes_cost="network forward, full base features, optimizer padding and grouped distribution kernels",
        limitations="Teacher-state opportunity audit; not improved-policy performance or worst-case CPU proof.",
        cases_detail=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:v for k,v in result.items() if k!="cases_detail"},indent=2))


if __name__ == "__main__":
    main()
