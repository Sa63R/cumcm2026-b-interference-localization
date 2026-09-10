"""Combine frozen three-policy stress results and the later fixed RL append."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics


def summarize(directory):
    original = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    later = json.loads((directory / "rl_append/summary.json").read_text(encoding="utf-8"))
    policies = {**original["policies"], **later["policies"]}
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in
                 (directory / "manifest.json", directory / "rl_append/manifest.json")]
    assert manifests[0]["cases_sha256"] == manifests[1]["cases_sha256"]
    indexed, diagnostics = {}, {}
    for policy in policies:
        folder = directory / policy if policy != "rl_best002_u384" else directory / "rl_append" / policy
        indexed[policy] = {row["case_id"]: row for row in
                          (json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines())}
        planning, fallbacks, reason = [], Counter(), Counter()
        fallback_actions, fallback_virtual_s, fallback_cases = 0, 0., []
        for path in sorted(folder.glob("*.json.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                report = json.load(stream)["report"]
            if report is None:
                reason["missing_report"] += 1
                continue
            reason[report["completion_reason"]] += 1
            if policy == "rollout_frozen":
                data = report["planning"]
                planning.append(data.get("planning_wall_time_s", 0.))
                fallbacks.update(data.get("fallback_counts", {}))
            elif policy == "geo_probe_single":
                data = report["first_probe_planning"]
                planning.append(data["planning_s"])
                fallbacks.update(data["fallbacks"])
            elif policy == "rl_best002_u384":
                data = report["learning"]
                # Store the original mechanism counters for transparent audit;
                # never infer optimality from network confidence or its logits.
                fallbacks.update(data.get("fallback_counts", {}))
                fallback_actions += data.get("fallback_actions", 0)
                fallback_virtual_s += data.get("fallback_virtual_time_s", 0.)
                if data.get("fallback_counts"):
                    fallback_cases.append({"case_id": path.name.removesuffix(".json.gz"),
                                           "reasons": data["fallback_counts"],
                                           "actions": data.get("fallback_actions", 0),
                                           "virtual_time_s": data.get("fallback_virtual_time_s", 0.)})
        diagnostics[policy] = {"completion_reasons": dict(reason), "fallbacks": dict(fallbacks),
                               "max_reported_planning_s": max(planning) if planning else None,
                               "rl_fallback_actions": fallback_actions, "rl_fallback_virtual_s": fallback_virtual_s,
                               "rl_fallback_cases": fallback_cases}
    pairs = {}
    for first, second in (("rollout_frozen", p) for p in policies if p != "rollout_frozen"):
        assert indexed[first].keys() == indexed[second].keys()
        rows = []
        for case_id, old in indexed[first].items():
            new = indexed[second][case_id]
            assert old["case_sha256"] == new["case_sha256"]
            rows.append({"case_id": case_id, "case_sha256": old["case_sha256"], "family": old["family"],
                         "old_s": old["time_s"], "new_s": new["time_s"],
                         "saved_s": old["time_s"]-new["time_s"], "both_successful": old["success"] and new["success"]})
        pairs[second] = {"mean_saved_s": statistics.mean(r["saved_s"] for r in rows),
                          "relative_reduction": sum(r["saved_s"] for r in rows)/sum(r["old_s"] for r in rows),
                          "wins": sum(r["saved_s"] > 1e-6 for r in rows),
                          "losses": sum(r["saved_s"] < -1e-6 for r in rows),
                          "ties": sum(abs(r["saved_s"]) <= 1e-6 for r in rows),
                          "worst_regressions": sorted(rows, key=lambda r: r["saved_s"])[:3], "cases": rows}
    audits = [json.loads(p.read_text(encoding="utf-8")) for p in
              (directory / "history_audit.json", directory / "rl_append/history_audit.json")]
    result = {"scope": "predeclared_training_reliability_not_final_test", "cases_sha256": manifests[0]["cases_sha256"],
              "policies": policies, "paired_vs_rollout": pairs, "diagnostics": diagnostics,
              "history_audit": {key: sum(a[key] for a in audits) for key in
                                ("traces", "passed_traces", "errors", "recertified_inferences")},
              "inference_note": "Families are deliberately chosen stresses, not IID draws from the task law. Descriptive statistics do not estimate a universal expected performance or failure rate."}
    (directory / "all_policies_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"history_audit": result["history_audit"], "diagnostics": diagnostics,
                      "paired_vs_rollout": {p: {k: v for k, v in d.items() if k != "cases"} for p, d in pairs.items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)
