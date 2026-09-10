"""Reproduce a coverage-representation diagnosis from archived public feedback.

Only the explicitly named 6000..6047 development-validation archives are read.
No simulator/policy is run, and no hidden source coordinates are accessed.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from localization.omni import OmniCandidateRegion
from simulator_client.state import Position


def load(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def replay(data):
    # Use the archived public cover configuration exactly: regenerating sin/cos
    # on another OS can change a coordinate by 1e-13 and corrupt an exact ledger.
    sites = tuple(Position(*p) for p in data["summary"]["coverage_points"])
    assert len(sites) == 7
    ledger = {p: set() for p in sites}
    regions, detected, cleared = {}, set(), set()
    current = Position(0, 0)
    counters = Counter()
    source_two = []
    actions = data["summary"]["action_history"]
    for index, item in enumerate(actions):
        pending = {p: set(range(1, 21)) - cleared - ledger[p] for p in sites}
        unknown = set(range(1, 21)) - detected  # Same rule as v4, including 16 discoveries.
        debt = [p for p in sites if pending[p] & unknown]
        remaining = [p for p in sites if pending[p]]
        optional = [p for p in remaining if p not in debt]
        counters["physical_action_states"] += 1
        counters["states_with_only_known_pending_site"] += bool(optional)
        counters["only_known_pending_site_count_sum"] += len(optional)
        for channel in sorted(detected - cleared):
            region = regions[channel]
            if not region.vertices:
                continue
            disk = region.enclosing_disk()
            center = Position(*disk.center)
            old_nearest = min((center.distance_to(p) for p in remaining), default=0.)
            debt_nearest = min((center.distance_to(p) for p in debt), default=0.)
            counters["known_source_states"] += 1
            counters["source_nearest_site_distance_differs"] += abs(debt_nearest-old_nearest) > 1e-7
            counters["source_nearest_distance_difference_over_300m"] += debt_nearest-old_nearest > 300
            if data["row"]["seed"] == 6011 and channel == 2 and index in (84, 131):
                source_two.append(dict(before_action_index=index, next_action=item,
                    current=[current.x,current.y], center=list(disk.center), radius_m=disk.radius,
                    current_to_center_s=current.distance_to(center)/5,
                    old_nearest_pending_site_m=old_nearest, nearest_unknown_check_site_m=debt_nearest,
                    detected_count=len(detected), unresolved=sorted(detected-cleared)))
        point, channel = Position(*item["position"]), item["channel"]
        if item["action"] == "measure":
            if point in ledger:
                ledger[point].add(channel)
            region = regions.setdefault(channel, OmniCandidateRegion())
            if item["result"] == "direction":
                detected.add(channel)
                region.observe(point, item["bearing_deg"])
            elif item["result"] == "near":
                detected.add(channel)
            else:
                region.observe_no_signal(point)
        elif item["result"] == "success":
            cleared.add(channel)
        current = point
    assert len(cleared) == data["row"]["cleared_total"]
    return dict(counters=counters, source_two_example=source_two)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("research/route_debt_evidence/observed_ledger_audit.json"))
    args = parser.parse_args(argv)
    rows, hashes, examples, totals = [], [], [], Counter()
    for seed in range(6000, 6048):
        p, q = (root / f"case-{seed}.json.gz" for root in (args.candidate, args.baseline))
        candidate, baseline = load(p), load(q)
        a, b = candidate["row"], baseline["row"]
        assert a["seed"] == b["seed"] == seed and a["case_sha256"] == b["case_sha256"]
        assert a["successful"] and b["successful"] and a["failed_clear_count"] == b["failed_clear_count"] == 0
        replayed = replay(candidate)
        totals.update(replayed["counters"])
        examples.extend(replayed["source_two_example"])
        rows.append(dict(seed=seed, candidate={key:a[key] for key in
            ("virtual_time_s","movement_s","detection_s","switching_s")},
            baseline={key:b[key] for key in ("virtual_time_s","movement_s","detection_s","switching_s")},
            counts=dict(replayed["counters"])))
        hashes.append(dict(seed=seed, case_sha256=a["case_sha256"],
            candidate_archive_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            baseline_archive_sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
    means = {name:{key:statistics.mean(r[name][key] for r in rows) for key in rows[0][name]}
             for name in ("candidate", "baseline")}
    result = dict(schema=1, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Read-only already exposed 48 validation histories; all scenario truth fields ignored",
        policy="cold-finetune-ppo-002 u384 versus archived rollout",
        unknown_definition="1..20 minus actual detected channels; no additional 16-source pruning",
        coordinate_replay="Exact archived summary.coverage_points; no platform-regenerated float keys",
        limitations=["Physical action states include any fallback actions and are not all actor decisions.",
                    "A pending unknown check is not a proof a future site visit is inevitable; observations can discharge it.",
                    "The 6011 geometric return distance is diagnostic, not a counterfactual achievable cost saving."],
        means=means, counts=dict(totals), source_two_example=examples, inputs=hashes, rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(means=means, counts=dict(totals), example=examples),indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
