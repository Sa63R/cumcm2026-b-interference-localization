"""Apply the predeclared fixed-sequence comparison after complete audits."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.run_q4_round2 import hashes, write_json
from experiments.run_study import bootstrap_mean_interval


def comparison(report, candidate, reference):
    base = {r["seed"]: r for r in report["rows"] if r["strategy"] == reference}
    rows = [r for r in report["rows"] if r["strategy"] == candidate]
    if len(rows) != len(base) or {r["seed"] for r in rows} != base.keys():
        raise ValueError("Unpaired decision input")
    savings = [base[r["seed"]]["penalized_time_s"]-r["penalized_time_s"] for r in rows]
    result = {"candidate": candidate, "reference": reference, "pairs": len(rows),
        "all_clear": all(r["successful"] and base[r["seed"]]["successful"] for r in rows),
        "mean_saved_s": statistics.mean(savings),
        "saving_ci95_s": bootstrap_mean_interval(savings, seed=610941, samples=10000),
        "mean_reduction_fraction": 1-report["summaries"][candidate]["mean_time_s"]/report["summaries"][reference]["mean_time_s"],
        "p95_ratio": report["summaries"][candidate]["p95_time_s"]/report["summaries"][reference]["p95_time_s"],
        "wins": sum(s>1e-6 for s in savings), "losses": sum(s < -1e-6 for s in savings),
        "loss_rows": [{"seed": r["seed"], "case_id": r["case_id"], "candidate_time_s": r["virtual_time_s"],
            "reference_time_s": base[r["seed"]]["virtual_time_s"], "lower_bound_s": r["common_lower_bound_s"],
            "candidate_ratio": r["time_over_lower_bound"], "reference_ratio": base[r["seed"]]["time_over_lower_bound"]}
            for r,s in zip(rows,savings) if s < -1e-6]}
    return result


def main():
    selection_path = ROOT/"research/q4_round2/selection.json"
    selection = json.loads(selection_path.read_bytes())
    if hashes() != selection["source_sha256"]:
        raise ValueError("Candidate or auditor changed since independent selection")
    decisions, reports, evidence = {}, {}, {}
    for stage in ("confirmation", "stress"):
        directory = ROOT/"results/q4_round2"/stage
        audit = json.loads((directory/"independent_audit.json").read_bytes())
        manifest = json.loads((directory/"manifest.json").read_bytes())
        if not audit["all_passed"] or manifest["source_sha256"] != selection["source_sha256"] or manifest["specs"] != selection["specs"] or manifest["seeds"] != selection["reserved_seeds"][stage]:
            raise ValueError("Unqualified independent evidence")
        reports[stage] = report = json.loads((directory/"summary.json").read_bytes())
        decisions[stage] = {"primary": comparison(report,"compact_range","compact_baseline"),
                            "secondary": comparison(report,"compact_combo","compact_range"),
                            "combined_vs_original": comparison(report,"compact_combo","compact_baseline")}
        for filename in ("summary.json", "independent_audit.json", "manifest.json", "freeze.json", "source.zip"):
            p = directory/filename
            evidence[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    def gate(key):
        return (decisions["confirmation"][key]["saving_ci95_s"][0] > 0 and
                all(decisions[s][key]["all_clear"] and decisions[s][key]["p95_ratio"] <= 1.05 for s in decisions))
    primary = gate("primary")
    secondary = primary and gate("secondary")
    promoted = "compact_combo" if secondary else "compact_range" if primary else "compact_baseline"
    result = {"primary_passed": primary, "secondary_passed": secondary, "promoted": promoted,
              "decision_rule": selection["testing_order"], "comparisons": decisions,
              "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
              "evidence_sha256": evidence, "spec": selection["specs"][promoted],
              "scope": "Audited local synthetic validation; no official practice performance claim"}
    write_json(ROOT/"research/q4_round2/qualification.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("comparisons","evidence_sha256")}),flush=True)


if __name__ == "__main__":
    main()
