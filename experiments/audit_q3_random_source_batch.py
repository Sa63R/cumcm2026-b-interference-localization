"""Independently replay billed actions and clear geometry from saved Q3 traces."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def audit(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((folder / "rows.json").read_text(encoding="utf-8"))
    by_seed = {r["seed"]: r for r in rows}
    errors, index = [], []
    total_us, total_sources, cleared_total, actions_total = 0, 0, 0, 0
    if len(by_seed) != len(rows) or set(by_seed) != {c["seed"] for c in manifest["cases"]}:
        errors.append("Missing or duplicate scenario rows")
    if len(list((folder / "cases").glob("case-*.json.gz"))) != len(manifest["cases"]):
        errors.append("Unexpected trace count")
    for item in manifest["cases"]:
        seed = item["seed"]
        path = folder / "cases" / f"case-{seed}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        row, evaluation = record["row"], record["evaluation"]
        case_hash = hashlib.sha256(canonical(evaluation["ground_truth"]).encode()).hexdigest()
        if case_hash != item["case_sha256"] or row != by_seed.get(seed):
            errors.append(f"{seed}: case or row identity mismatch")
        if record["spec"] != manifest["spec"] or not row["successful"]:
            errors.append(f"{seed}: strategy mismatch or unsuccessful outcome")
        if row["source_total"] != item["source_total"]:
            errors.append(f"{seed}: source count mismatch")
        sources = {s["channel"]: s for s in evaluation["ground_truth"]["sources"]}
        cleared, position, channel, clock_us = set(), (0.0, 0.0), 1, 0
        components = dict.fromkeys(("movement_s", "switching_s", "detection_s", "optical_s", "removal_s"), 0)
        for entry in record["history"]:
            action = entry["action"]
            response = entry["response"]
            if action in ("/measure", "/clear"):
                point = (entry["position"]["x"], entry["position"]["y"])
                move = round(math.dist(position, point) / 5 * 1000000)
                components["movement_s"] += move
                clock_us += move
                position = point
                ch = entry["channel"]
                if action == "/measure":
                    switch = 1000000 * (ch != channel)
                    components["switching_s"] += switch
                    components["detection_s"] += 5000000
                    clock_us += switch + 5000000
                    channel = ch
                else:
                    source = sources.get(ch)
                    valid = (source is not None and ch not in cleared and
                             math.dist(point, (source["x"], source["y"])) <= 20.0)
                    reported = response["clear_result"] == "success"
                    if valid != reported:
                        errors.append(f"{seed}: invalid optical feedback at {entry['index']}")
                    components["optical_s"] += 3000000
                    components["removal_s"] += 2000000 * valid
                    clock_us += 3000000 + 2000000 * valid
                    if valid:
                        cleared.add(ch)
            if abs(clock_us / 1000000 - response["virtual_time_s"]) > 1e-6:
                errors.append(f"{seed}: cumulative billing mismatch at {entry['index']}")
        if cleared != set(sources) or not row["completion_certified"] or not row["accepted_exit"]:
            errors.append(f"{seed}: incomplete clearance or exit")
        if abs(clock_us / 1000000 - row["virtual_time_s"]) > 1e-6:
            errors.append(f"{seed}: row total mismatch")
        for key, value in components.items():
            if abs(value / 1000000 - evaluation["time_breakdown_s"][key]) > 1e-6:
                errors.append(f"{seed}: component mismatch: {key}")
        total_us += clock_us
        total_sources += len(sources)
        cleared_total += len(cleared)
        actions_total += len(record["history"])
        index.append({"seed": seed, "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pooled = total_us / 1000000 / total_sources
    if total_sources != summary["total_sources"] or abs(pooled - summary["pooled_time_per_source_s"]) > 1e-9:
        errors.append("Final aggregate mismatch")
    result = {"passed": not errors, "errors": errors, "audited_cases": len(index),
              "audited_actions": actions_total, "total_sources": total_sources,
              "geometrically_verified_clearances": cleared_total,
              "replayed_total_virtual_time_s": total_us / 1000000,
              "replayed_pooled_time_per_source_s": pooled,
              "checks": ["complete fixed manifest", "scenario identity", "strategy identity", "source counts",
                         "every action billing and cumulative time", "time components", "20m successful-clear geometry",
                         "all-source completion", "termination certificate and accepted exit", "aggregate metric"]}
    (folder / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (folder / "trace-index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    audit(parser.parse_args().batch)
