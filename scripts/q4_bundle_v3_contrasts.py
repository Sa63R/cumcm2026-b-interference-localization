"""Post-audit paired learning-phase contrasts from the fixed completed rows."""
import hashlib
import json
from pathlib import Path
from q4_bundle_v3_review import paired

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (("mlp_v3_bc512", "mlp_v3_ppo512"),
         ("attention_v3_bc512", "attention_v3_ppo512"),
         ("micro_v2_bc512", "micro_v2_ppo512"),
         ("ppo_initialized512", "scst_initialized512"),
         ("mlp_v3_ppo512", "attention_v3_ppo512"))


def main():
    audit_path = ROOT/"results/q4_rl/bundle-v3-independent-review-001.json"
    audit = json.loads(audit_path.read_bytes())
    assert audit["validation"]["physical_wire_replays"] == 6144
    summary_path = ROOT/"results/q4_rl/server-bundle-v3-evaluation-001/summary.json"
    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == audit["summary_sha256"]
    summary = json.loads(summary_path.read_bytes())
    output = ROOT/"results/q4_rl/bundle-v3-learning-contrasts-001.json"
    if output.exists():
        raise ValueError("New contrast output required")
    groups = {}
    for group, predicate in (("all",lambda r:True),("random",lambda r:r["family"]=="random"),
                             ("stress",lambda r:r["family"]!="random")):
        selected = [r for r in summary["rows"] if predicate(r)]
        groups[group] = [paired(selected,source,target) for source,target in PAIRS]
    value = dict(scope="Post-audit descriptive learning-phase contrasts on the same fixed development panel; no new model or scenario evaluation",
        prior_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        summary_sha256=audit["summary_sha256"], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bootstrap=dict(samples=5000,seed=4260911,unit="paired seed cluster"),groups=groups,
        reliability_scope="512/512 is the observed completion count for this fixed panel. Wilson intervals treating runs as independent are descriptive only; 512 correlated transformed scenes are not an independent reliability guarantee. The panel has 32 shared seed clusters.",
        confounding="BC/PPO within architecture tracks the additional training phase. Cross-architecture and SCST/initialized-PPO comparisons also differ in actual training exposure; identical wall allocations do not imply identical data or update counts.")
    output.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    report = ROOT/"research/q4_rl/BUNDLE_V3_EVALUATION_REVIEW.md"
    lines = ["", "## Additional paired learning-phase contrasts", "", value["reliability_scope"], ""]
    for row in groups["all"]:
        lo,hi = row["relative_saving_ci95"]
        lines.append(f"- {row['target']} versus {row['reference']}: mean saving {100*row['relative_saving']:.2f}% (95% paired-cluster CI {100*lo:.2f}% to {100*hi:.2f}%).")
    lines += ["", value["confounding"], "", "Raw failure-mode comparisons (all 512 cases per method):", "",
        "| Method | Mean T s | T/L | Fallback cases | Prefix current scans/case | Prefix localization scans/case | Movement s/case | Detection s/case | Optical s/case |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ("r8","macro_v2_ppo512","micro_v2_ppo512","mlp_v3_ppo512","attention_v3_ppo512","ppo_initialized512","scst_initialized512"):
        raw = audit["raw_failure_modes"]["all"][method]
        roles = raw["mean_policy_prefix_measurement_roles"]
        metrics = audit["groups"]["all"]["arms"][method]
        costs = metrics["components"]
        current = f"{roles.get('current',0.):.2f}" if roles else "n/a"
        localize = f"{roles.get('localize',0.):.2f}" if roles else "n/a"
        lines.append(f"| {method} | {metrics['mean_time_s']:.2f} | {metrics['ratio_of_sums']:.5f} | {sum(raw['fallback_reason_counts'].values())} | {current} | {localize} | {costs['movement_s']:.2f} | {costs['detection_s']:.2f} | {costs['optical_s']:.2f} |")
    lines += ["", "Prefix roles apply to explicit micro actions before fallback; R8/macro service macros have no equivalent role labels, so n/a is not zero. Complete measurement phases, switching/removal costs, fallback reasons and source-count exploratory breakdowns remain in the machine-readable audit.", ""]
    with report.open("a",encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(json.dumps(groups["all"]),flush=True)


if __name__ == "__main__":
    main()
