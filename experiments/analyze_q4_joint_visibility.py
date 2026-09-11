"""Attribute observed paired costs without reading the environment's hidden state."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def actual_costs(record):
    result = defaultdict(lambda: {"time_s": 0., "actions": 0, "silent_measures": 0,
                                   "failed_clears": 0})
    previous = 0.
    for action in record["summary"]["action_history"]:
        item = result[action["phase"]]
        item["time_s"] += action["virtual_time_s"] - previous
        item["actions"] += 1
        item["silent_measures"] += action["action"] == "measure" and action["result"] == "no_signal"
        item["failed_clears"] += action["action"] == "clear" and action["result"] not in ("success", "cleared")
        previous = action["virtual_time_s"]
    return dict(result)


def analyze(directory, candidate, reference):
    rows = json.loads((directory / "summary.json").read_bytes())["rows"]
    groups = {label: {r["seed"]: r for r in rows if r["strategy"] == label}
              for label in (reference, candidate)}
    if not groups[reference] or groups[reference].keys() != groups[candidate].keys():
        raise ValueError("Missing paired rows")
    details, aggregate = [], {label: defaultdict(lambda: defaultdict(float)) for label in groups}
    keys = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")
    for seed in sorted(groups[reference]):
        old, new = (groups[label][seed] for label in (reference, candidate))
        if old["case_sha256"] != new["case_sha256"] or old["common_lower_bound_s"] != new["common_lower_bound_s"]:
            raise ValueError("Different environment or bound")
        costs = {}
        for label in groups:
            record = read(directory / "records" / f"{label}-{seed}.json.gz")
            costs[label] = actual_costs(record)
            for phase, values in costs[label].items():
                for name, value in values.items():
                    aggregate[label][phase][name] += value / len(groups[label])
        details.append(dict(seed=seed, saved_s=old["virtual_time_s"]-new["virtual_time_s"],
            reference_time_s=old["virtual_time_s"], candidate_time_s=new["virtual_time_s"],
            lower_bound_s=old["common_lower_bound_s"],
            reference_ratio=old["time_over_lower_bound"], candidate_ratio=new["time_over_lower_bound"],
            component_savings_s={key: old[key]-new[key] for key in keys}, phase_costs=costs))
    return dict(reference=reference, candidate=candidate, pairs=len(details),
        mean_component_savings_s={key: statistics.mean(d["component_savings_s"][key] for d in details) for key in keys},
        mean_phase_costs=aggregate, regressions=[d for d in details if d["saved_s"] < -1e-6],
        largest_savings=sorted(details, key=lambda d: d["saved_s"], reverse=True)[:5],
        attribution_note="Each elapsed action interval includes travel into that action's phase. These are realized paired accounting differences, not a causal decomposition holding later policy decisions fixed. Hidden source data are not read.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidate", default="compact_joint_probe")
    parser.add_argument("--reference", default="compact_clear_before_probe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.input, args.candidate, args.reference)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("pairs", "mean_component_savings_s")}))
