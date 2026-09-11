"""One unchanged generic evaluation, plus a contrast from the same frozen rows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    control = root / "launch/eval-memory-v4"
    plan = json.loads((control / "plan.json").read_text())
    panels = {p["run"]: p for p in plan["panels"]}
    if args.panel not in panels:
        raise ValueError("Undeclared panel")
    panel = panels[args.panel]
    sys.path[:0] = [str(root / "src"), str(root)]
    from scripts.q4_training_pair import require_outer_supervisor
    require_outer_supervisor(root)
    from experiments import q4_rl_evaluate as evaluator
    output = root / "runs" / panel["run"] / "evaluation"
    sys.argv = ["q4_rl_evaluate", "--output", str(output), "--specs", str(control / "specs.json"),
        "--reference", "r9_probe", "--stage", "development", "--start", str(panel["start"]), "--count", str(panel["count"]),
        "--families", *panel["families"], "--source-modes", "mixed", "all_directional", "--workers", "10", "--bootstrap-samples", "5000"]
    code = evaluator.main()
    summary = json.loads((output / "summary.json").read_text())
    evidence = json.loads((output / "evidence.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    expected = panel["scenarios"] * len(plan["arms"])
    if len(summary["rows"]) != expected or len(evidence["record_sha256"]) != expected or len(manifest["requests"]) != panel["scenarios"]:
        raise ValueError("Incomplete fixed scenario/arm matrix")
    # No new episode: all three reference contrasts use the same complete
    # paired rows, source freeze and 5000 seed-cluster bootstrap convention.
    historical = evaluator.report_rows(summary["rows"], reference="r8", samples=5000)
    evaluator.write_json(output / "summary_vs_r8.json", historical)
    prior = evaluator.report_rows(summary["rows"], reference=plan["prior_rl_reference"], samples=5000)
    evaluator.write_json(output / "summary_vs_prior_rl.json", prior)
    evaluator.write_json(output / "panel_complete.json", {"run": panel["run"], "stage": "development",
        "scenarios": panel["scenarios"], "arms": plan["arms"], "complete_rows": expected,
        "evaluator_returncode": code, "reference": "r9_probe", "historical_reference": "r8", "prior_rl_reference": plan["prior_rl_reference"],
        "plan_sha256": hashlib.sha256((control / "plan.json").read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256((output / "summary.json").read_bytes()).hexdigest(),
        "historical_summary_sha256": hashlib.sha256((output / "summary_vs_r8.json").read_bytes()).hexdigest(),
        "prior_summary_sha256": hashlib.sha256((output / "summary_vs_prior_rl.json").read_bytes()).hexdigest(),
        "failed_rows_retained": sum(not row["successful"] for row in summary["rows"]),
        "no_additional_rollouts_for_prior_contrast": True})
    return code

if __name__ == "__main__":
    raise SystemExit(main())
