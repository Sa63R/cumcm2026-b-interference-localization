"""Bind implementation checks to exact Q3 candidate bytes before new cases."""
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from round2_runner import ROOT, digest, hashes, read, write


def main():
    repo = ROOT.parent / "q3-r2-observation-tree"
    output = ROOT / "research/round2/diagnoses/OBSERVATION_TREE_MODEL_GATE.json"
    if output.exists():
        raise ValueError("Do not overwrite a model gate")
    current = hashes(repo)
    evidence = {}
    review_path = "research/round2/diagnoses/MODEL_GATE_REVIEW.json"
    review = read(ROOT / review_path)
    if review.get("passed") is not True or review["candidate_source_sha256"] != current:
        raise ValueError("Independent model review did not pass for these source bytes")
    evidence[review_path] = hashlib.sha256((ROOT / review_path).read_bytes()).hexdigest()
    for path, expected in review["evidence_sha256"].items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Reviewed evidence changed: {path}")
        evidence[path] = actual
    smoke_checks = {}
    for name, depth in (("depth2-90s", 2), ("depth1-90s", 1), ("disabled-90s", 0)):
        path = f"results/round2/observation_tree/model_smoke/{name}.json.gz"
        with gzip.open(ROOT / path, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        if (not record["source_unchanged_during_smoke"] or record["source_sha256"] != current
                or record["source_after_sha256"] != current or record["audit"]["errors"]
                or not record["evaluation"]["all_cleared"] or record["evaluation"]["failed_clear_count"]
                or not record["summary"]["completion_certified_under_model"]):
            raise ValueError(f"Invalid implementation smoke: {name}")
        logs = record["summary"]["strategy_parameters"]["observation_tree_log"]
        if depth:
            if len(logs) != 1 or logs[0]["status"] not in ("changed", "evaluated_baseline"):
                raise ValueError(f"Complete root comparison missing: {name}")
            if logs[0]["tail_evaluations"] != 48 or (depth == 2 and not logs[0]["mechanism_active"]):
                raise ValueError(f"Required mechanism calculation missing: {name}")
            if depth == 1 and logs[0]["second_layer_comparisons"]:
                raise ValueError("Root MC unexpectedly used a second layer")
        elif logs:
            raise ValueError("Disabled strategy still planned")
        evidence[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        smoke_checks[name] = {"passed": True, "time_over_physical_lower": record["time_over_physical_lower"]}
    xml_path = "results/round2/observation_tree/model_smoke/implementation-tests.xml"
    suites = ET.parse(ROOT / xml_path).getroot().findall("testsuite")
    total = sum(int(suite.get("tests", 0)) for suite in suites)
    if total != 61 or any(int(suite.get(field, 0)) for suite in suites for field in ("failures", "errors", "skipped")):
        raise ValueError("Required current implementation tests did not all pass")
    evidence[xml_path] = hashlib.sha256((ROOT / xml_path).read_bytes()).hexdigest()
    policies = {}
    for label in ("root_mc", "observation_tree"):
        spec = read(repo / f"experiments/state_search_candidate_{label}_r2.json")
        policies[label] = {"source_sha256": current, "spec_sha256": digest(spec)}
    write(output, {"passed": True, "policies": policies, "evidence_sha256": evidence,
                   "smoke_checks": smoke_checks, "implementation_tests_passed": total,
                   "scope": "Implementation/model feasibility gate only. Old 200114, max_searches=1. No independent performance claim, no exact posterior or POMCPOW equivalence claim."})
    print(json.dumps({"model_gate": str(output), "passed": True, "tests": total}))


if __name__ == "__main__":
    main()
