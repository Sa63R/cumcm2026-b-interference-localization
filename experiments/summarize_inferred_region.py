"""Paired combination pilot plus independent legal-observation inference audit."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from localization.omni import OmniCandidateRegion
from planning.silence_certificate import certify_silence
from simulator_client.state import Position
from summarize_region_travel import paired_summary, percentile


def audit(report):
    # Only public action records and labelled inferences are read. The full
    # archive's post-run evaluation/truth is deliberately not passed here.
    regions, known, cleared, actual = {}, set(), set(), {}
    inferences = report["strategy_parameters"]["inferred_no_signal_constraints"]
    cursor = physical_measures = 0
    current = inserted_current = 1
    actual_service = inserted_service = 0.
    for index, action in enumerate(report["action_history"]):
        while cursor < len(inferences) and inferences[cursor]["after_actual_action_count"] == index:
            inferred = inferences[cursor]
            channel = inferred["channel"]
            assert channel in known and channel not in cleared
            assert inferred["physical_measurement"] is False
            certificate = certify_silence(regions[channel], inferred["position"])
            assert certificate and certificate["distance_lower_m"] == inferred["distance_lower_m"]
            regions[channel].observe_no_signal(inferred["position"])
            inserted_service += 5 + int(inserted_current != channel)
            inserted_current = channel
            cursor += 1
        channel = action["channel"]
        point = Position(*action["position"])
        if action["action"] == "measure":
            physical_measures += 1
            actual_service += 5 + int(current != channel)
            current = channel
            inserted_service += 5 + int(inserted_current != channel)
            inserted_current = channel
            region = regions.setdefault(channel, OmniCandidateRegion())
            actual.setdefault(channel, {})[tuple(action["position"])] = action["result"]
            if action["result"] == "direction":
                region.observe(point, action["bearing_deg"])
                known.add(channel)
            elif action["result"] == "near":
                known.add(channel)
            elif action["result"] == "no_signal":
                region.observe_no_signal(point)
        elif action["action"] == "clear":
            if action["phase"] == "certified_clear":
                assert regions[channel].vertices
                assert all(point.distance_to(Position(*v)) <= 19.9 + 1e-6 for v in regions[channel].vertices)
            if action["result"] == "success":
                cleared.add(channel)
    assert cursor == len(inferences)
    if len(known) < 16:
        for channel in set(range(1, 21)) - known:
            assert all(actual[channel].get(tuple(p)) == "no_signal" for p in report["coverage_points"])
    assert all(not any(a["phase"] == "coverage" and a["channel"] == item["channel"]
                       and a["position"] == item["position"] for a in report["action_history"]) for item in inferences)
    return {"inferences": len(inferences), "methods": dict(Counter(i["method"] for i in inferences)),
            "all_inferences_recertified_from_legal_history": True,
            "physical_measures": physical_measures,
            "frozen_insertion_service_cost_s": inserted_service - actual_service,
            "frozen_insertion_scope": "Cost of putting labelled silence reads back into this same actual sequence; not an independently executed strategy."}


def summarize(directory):
    rows = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    policies = ("axis_inferred", "mean_point_inferred")
    indexed = {p: {r["seed"]: r for r in rows if r["policy"] == p} for p in policies}
    assert indexed[policies[0]].keys() == indexed[policies[1]].keys()
    overall, cases, methods = {}, [], {p: Counter() for p in policies}
    for policy, subset in indexed.items():
        overall[policy] = {"n": len(subset), "mean_s": statistics.mean(r["time_s"] for r in subset.values()),
            "mean_wall_s": statistics.mean(r["wall_s"] for r in subset.values()),
            "p95_s": percentile([r["time_s"] for r in subset.values()], .95),
            "successes": sum(r["success"] for r in subset.values()),
            "failed_clears": sum(r["failed_clears"] for r in subset.values()),
            "time_breakdown": {k: statistics.mean(r["breakdown"][k] for r in subset.values())
                               for k in next(iter(subset.values()))["breakdown"]}}
    for seed in indexed[policies[0]]:
        reports, audits = {}, {}
        for policy in policies:
            with gzip.open(directory / f"q3-random-{seed}--{policy}.json.gz", "rt", encoding="utf-8") as stream:
                reports[policy] = json.load(stream)["report"]
            audits[policy] = audit(reports[policy])
            methods[policy].update(audits[policy]["methods"])
            assert 5 * audits[policy]["physical_measures"] == indexed[policy][seed]["breakdown"]["detection_s"]
        def task_sequence(report):
            return [(p["selected_kind"], p["selected_channel"], tuple(p["selected_point"]))
                    for p in report["strategy_parameters"]["planning_log"]]
        def clear_order(report):
            return [a["channel"] for a in report["action_history"]
                    if a["action"] == "clear" and a["result"] == "success"]
        cases.append({"seed": seed,
            "saved_s": indexed[policies[0]][seed]["time_s"] - indexed[policies[1]][seed]["time_s"],
            "same_task_sequence": task_sequence(reports[policies[0]]) == task_sequence(reports[policies[1]]),
            "same_clear_order": clear_order(reports[policies[0]]) == clear_order(reports[policies[1]]),
            "audit": audits})
    for policy in policies:
        overall[policy]["inferences"] = sum(c["audit"][policy]["inferences"] for c in cases)
        overall[policy]["inference_methods"] = dict(methods[policy])
        overall[policy]["mean_frozen_insertion_service_cost_s"] = statistics.mean(c["audit"][policy]["frozen_insertion_service_cost_s"] for c in cases)
    summary = {"origin": "synthetic_research_training_only", "platform": "Windows", "overall": overall,
        "pair": paired_summary(indexed[policies[0]], indexed[policies[1]]),
        "same_task_sequence_cases": sum(c["same_task_sequence"] for c in cases),
        "same_clear_order_cases": sum(c["same_clear_order"] for c in cases),
        "all_case_inferences_coverage_and_clearance_replayed": True, "cases": cases}
    (directory / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)
