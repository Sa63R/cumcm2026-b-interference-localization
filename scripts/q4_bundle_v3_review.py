"""Offline audit of all frozen bundle-v3 rows; no policy or simulator execution."""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from experiments.audit_q4_state import wire_audit, digest

FOLDER = ROOT/"results/q4_rl/server-bundle-v3-evaluation-001"
OUTPUT = ROOT/"results/q4_rl/bundle-v3-independent-review-001.json"
REPORT = ROOT/"research/q4_rl/BUNDLE_V3_EVALUATION_REVIEW.md"
SUMMARY_SHA = "2848138b614b571be706f5f3ebe18e28031610886c4c45c0ebce40c59adc2026"
FAMILIES = ("random", "minimum_radius", "boundary_outward", "cluster", "positive_error",
            "negative_error", "alternating_error", "narrow_strip")
ELIGIBLE_KINDS = ("legacy_ppo128", "macro_v2_ppo512", "micro_v2_ppo512", "mlp_v3_ppo512",
                  "attention_v3_ppo512", "ppo_initialized512", "scst_initialized512")
COMPONENTS = ("movement_s", "detection_s", "switching_s", "optical_s", "removal_s")


def sha(blob): return hashlib.sha256(blob).hexdigest()
def read(name): return json.loads((FOLDER/name).read_bytes())
def quantile(values, probability):
    values = sorted(values); at = (len(values)-1)*probability; left = math.floor(at)
    return values[left] + (values[math.ceil(at)]-values[left])*(at-left)


def metrics(rows):
    return dict(runs=len(rows), successful=sum(r["successful"] for r in rows),
        audited=sum(r["audit_passed"] for r in rows), mean_time_s=statistics.mean(r["virtual_time_s"] for r in rows),
        mean_bound_s=statistics.mean(r["common_lower_bound_s"] for r in rows),
        ratio_of_sums=sum(r["penalized_time_s"] for r in rows)/sum(r["common_lower_bound_s"] for r in rows),
        p95_time_s=quantile([r["penalized_time_s"] for r in rows], .95),
        max_time_s=max(r["penalized_time_s"] for r in rows),
        failed_clear_total=sum(r["failed_clear_count"] for r in rows),
        mean_cpu_s=statistics.mean(r["process_cpu_s"] for r in rows),
        mean_wall_s=statistics.mean(r["program_runtime_s"] for r in rows),
        mean_inference_s=statistics.mean(r["inference_wall_s"] for r in rows),
        mean_feature_s=statistics.mean(r["feature_wall_s"] for r in rows),
        mean_audit_s=statistics.mean(r["audit_runtime_s"] for r in rows),
        mean_bound_audit_s=statistics.mean(r["lower_bound_runtime_s"] for r in rows),
        mean_measurements=statistics.mean(r["measurement_count"] for r in rows),
        fallback_runs=sum(r["fallback_virtual_s"] > 1e-6 for r in rows),
        mean_fallback_s=statistics.mean(r["fallback_virtual_s"] for r in rows),
        components={k:statistics.mean(r[k] for r in rows) for k in COMPONENTS})


def paired(rows, baseline, target):
    base = {r["case_id"]:r for r in rows if r["strategy"] == baseline}
    clusters = defaultdict(list)
    for row in rows:
        if row["strategy"] == target:
            clusters[row["seed"]].append((base[row["case_id"]]["penalized_time_s"], row["penalized_time_s"]))
    keys = sorted(clusters)
    assert len({len(values) for values in clusters.values()}) == 1
    # Equal-size clusters allow exact mean/ratio bootstrap from cluster means,
    # avoiding repeated flattening of all 512 rows. No bootstrap sample changes.
    values = {k:(statistics.mean(a for a,b in clusters[k]), statistics.mean(b for a,b in clusters[k])) for k in keys}
    rng = random.Random(4260911); differences = []; ratios = []
    for _ in range(5000):
        selected = rng.choices(keys, k=len(keys))
        a = statistics.mean(values[k][0] for k in selected)
        b = statistics.mean(values[k][1] for k in selected)
        differences.append(a-b); ratios.append(1-b/a)
    a = statistics.mean(values[k][0] for k in keys); b = statistics.mean(values[k][1] for k in keys)
    return dict(reference=baseline, target=target, seed_clusters=len(keys), mean_saved_s=a-b,
        mean_saved_ci95_s=[quantile(differences,.025), quantile(differences,.975)],
        relative_saving=1-b/a, relative_saving_ci95=[quantile(ratios,.025), quantile(ratios,.975)])


def diagnose(record):
    row = record["row"]; report = record["summary"]; learning = report.get("learning", {})
    actions = report["action_history"]
    phases = Counter(action["phase"] for action in actions if action["action"] == "measure")
    result = dict(strategy=row["strategy"], source_total=row["source_total"],
        fallback_reason=learning.get("fallback_reason"), decisions=learning.get("decisions",0),
        all_measurement_phases=dict(phases), prefix_measurement_roles={}, fallback_boundary=None)
    steps = learning.get("micro_steps")
    if steps is None:
        return result
    assert len(steps) == learning["decisions"]
    roles = Counter(); known = set(); cleared = set(); measured = defaultdict(set)
    for step in steps:
        assert step["accepted_requests"] == 1
        assert step["end_actual_action_index"] == step["before_actual_action_index"]+1
        action = actions[step["before_actual_action_index"]]
        assert action["position"] == step["position"] and action["channel"] == step["channel"]
        assert action["action"] == ("measure" if step["kind"] == "measure" else "clear")
        if action["action"] == "measure":
            roles[step["role"]] += 1
            measured[action["channel"]].add(tuple(action["position"]))
            if action["result"] in ("near","direction"):
                known.add(action["channel"])
        elif action["result"] == "success":
            known.add(action["channel"]); cleared.add(action["channel"])
    result["prefix_measurement_roles"] = dict(roles)
    assert math.isclose(sum(step["cost_s"] for step in steps),learning["decision_cost_s"],abs_tol=2e-5)
    for check in learning["certified_clear_checks"]:
        action = actions[check["actual_action_index"]]
        assert action["action"] == "clear" and action["result"] == check["result"] == "success"
        assert check["max_distance_upper_m"] <= check["operational_radius_m"]-check["guard_m"]
    if learning.get("fallback_reason"):
        points = {tuple(point) for point in report["coverage_points"]}
        unknown = set(range(1,21))-known
        pending = sum(len(points-measured[channel]) for channel in unknown)
        discovery = len(known) == 16 or pending == 0
        category = ("not_all_real_sources_found" if len(known) < row["source_total"] else
                    "only_absence_proof_pending" if not discovery else
                    "discovered_but_uncleared" if not known <= cleared else "already_complete")
        result["fallback_boundary"] = dict(category=category, known=len(known), cleared=len(cleared),
            pending_unknown_pairs=pending, discovery_certified_from_prefix=discovery)
    return result


def aggregate_diagnostics(records):
    result = {}
    for mode, predicate in (("all",lambda r:True),("source_total_16",lambda r:r["source_total"]==16),
                            ("source_total_below_16",lambda r:r["source_total"]<16)):
        result[mode] = {}
        for method in sorted({r["strategy"] for r in records}):
            selected = [r for r in records if r["strategy"] == method and predicate(r)]
            phase, roles = Counter(),Counter()
            for row in selected:
                phase.update(row["all_measurement_phases"]); roles.update(row["prefix_measurement_roles"])
            boundaries = [r["fallback_boundary"] for r in selected if r["fallback_boundary"]]
            result[mode][method] = dict(runs=len(selected),
                fallback_reason_counts=dict(Counter(r["fallback_reason"] for r in selected if r["fallback_reason"])),
                mean_decisions=statistics.mean(r["decisions"] for r in selected),
                mean_measurement_phases={k:v/len(selected) for k,v in phase.items()},
                mean_policy_prefix_measurement_roles={k:v/len(selected) for k,v in roles.items()},
                micro_fallback_boundary_counts=dict(Counter(r["category"] for r in boundaries)),
                mean_known_at_micro_fallback=statistics.mean(r["known"] for r in boundaries) if boundaries else None,
                mean_cleared_at_micro_fallback=statistics.mean(r["cleared"] for r in boundaries) if boundaries else None)
    return result


def main():
    began = time.perf_counter()
    if OUTPUT.exists() or REPORT.exists():
        raise ValueError("New independent review outputs required")
    manifest, evidence, readback = read("manifest.json"), read("evidence.json"), read("OBJECT_READBACK.json")
    blob = (FOLDER/"summary.json").read_bytes(); summary = json.loads(blob)
    assert sha(blob) == evidence["summary_sha256"] == SUMMARY_SHA
    assert sha((FOLDER/"manifest.json").read_bytes()) == evidence["manifest_sha256"]
    assert digest(manifest) == read("freeze.json")["manifest_sha256"]
    assert sha((FOLDER/"specs.original.json").read_bytes()) == manifest["raw_specs_sha256"]
    assert read("specs.original.json") == manifest["specs"]
    assert summary["bootstrap"] == dict(samples=5000, seed=4260911, unit="paired seed cluster")
    assert manifest["stage"] == "development" and not manifest["formal_simulator"] and not manifest["practice_simulator"]
    assert manifest["bootstrap_samples"] == 5000 and manifest["numerical_threads_per_worker"] == 1
    assert len(manifest["requests"]) == 512
    assert {(r["seed"], r["family"], r["source_mode"]) for r in manifest["requests"]} == {
        (seed,family,mode) for seed in range(8101000,8101032) for family in FAMILIES for mode in ("mixed", "all_directional")}
    with zipfile.ZipFile(FOLDER/"source.zip") as archive:
        assert archive.testzip() is None and set(archive.namelist()) == set(manifest["source_sha256"])
        for name, expected in manifest["source_sha256"].items():
            assert sha(archive.read(name)) == expected, name
        assert sha(archive.read("research/q4_rl/protocol.json")) == manifest["protocol_sha256"]
    assert sha((ROOT/"experiments/audit_q4_state.py").read_bytes()) == manifest["source_sha256"]["experiments/audit_q4_state.py"]
    for name, expected in manifest["artifact_sha256"].items():
        assert sha((ROOT/"results/q4_rl/bundle-v3-endpoints"/Path(name).name).read_bytes()) == expected, name
    rows = summary["rows"]; methods = sorted(summary["summaries"])
    assert len(rows) == 6144 and len(methods) == 12 and len(manifest["artifact_sha256"]) == 10
    assert {(r["seed"],r["family"],r["source_mode"],r["strategy"]) for r in rows} == {
        (seed,family,mode,method) for seed in range(8101000,8101032) for family in FAMILIES
        for mode in ("mixed","all_directional") for method in methods}
    indexed = {(r["case_id"],r["strategy"]):r for r in rows}
    assert len(indexed) == len(rows)
    identities = {}; checked = set(); checked_bytes = 0; source_counts = defaultdict(set); diagnostics = []
    for obj in readback["objects"]:
        name = obj["name"]; path = FOLDER/name
        assert path.resolve().is_relative_to(FOLDER.resolve())
        raw = path.read_bytes()
        assert len(raw) == obj["bytes"] and sha(raw) == obj["sha256"], name
        checked_bytes += len(raw)
        if not name.startswith("records/"):
            continue
        assert sha(raw) == evidence["record_sha256"][path.name]
        record = json.loads(gzip.decompress(raw)); row = record["row"]
        key = (row["case_id"],row["strategy"])
        assert row == indexed[key] and key not in checked
        assert record["spec"] == manifest["specs"][row["strategy"]]
        assert record["request"] in manifest["requests"]
        assert row["successful"] and row["all_cleared"] and row["audit_passed"] and not row["errors"]
        assert record["audit"]["passed"] and record["audit"]["completion"]["passed"]
        assert record["summary"]["completion_certified_under_model"]
        assert row["penalized_time_s"] == row["virtual_time_s"]
        assert math.isclose(sum(row[k] for k in COMPONENTS), row["virtual_time_s"], abs_tol=1e-6)
        assert record["common_lower_bound"]["common_lower_bound_s"] == row["common_lower_bound_s"]
        assert math.isclose(row["time_over_lower_bound"],row["virtual_time_s"]/row["common_lower_bound_s"],rel_tol=1e-12)
        identity = (row["seed"],row["family"],row["source_mode"],row["stage"],row["case_sha256"],row["common_lower_bound_s"])
        assert identities.setdefault(row["case_id"],identity) == identity
        wire_audit(record)
        learning = record["summary"].get("learning", {})
        if learning:
            assert math.isclose(learning["decision_cost_s"]+learning["fallback_cost_s"]+learning["uncovered_cost_s"],row["virtual_time_s"],abs_tol=2e-5)
        source_counts[row["source_total"]].add(row["seed"])
        diagnostics.append(diagnose(record))
        checked.add(key)
        if len(checked)%1024 == 0:
            print(json.dumps(dict(raw_records_hash_and_physics_verified=len(checked))), flush=True)
    assert len(checked) == len(evidence["record_sha256"]) == 6144
    assert len(readback["objects"]) == readback["files"] and checked_bytes == readback["bytes"]
    scalar_map = {"runs":"runs", "successful":"successful", "mean_time_s":"mean_actual_elapsed_time_s",
        "ratio_of_sums":"ratio_of_sums", "p95_time_s":"p95_penalized_time_s", "failed_clear_total":"failed_clear_total",
        "mean_cpu_s":"mean_process_cpu_s"}
    full = {method:metrics([r for r in rows if r["strategy"] == method]) for method in methods}
    for method, values in full.items():
        for independent, original in scalar_map.items():
            assert math.isclose(values[independent],summary["summaries"][method][original],rel_tol=1e-12,abs_tol=1e-8)
    eligible = sorted((full[label]["mean_time_s"],label) for label in ELIGIBLE_KINDS
                      if full[label]["runs"] == full[label]["successful"] == full[label]["audited"] == 512)
    winner = eligible[0][1] if eligible else None
    groups = {}
    for group, predicate in (("all", lambda r:True), ("random",lambda r:r["family"] == "random"),
                             ("stress",lambda r:r["family"] != "random"),
                             ("mixed",lambda r:r["source_mode"] == "mixed"),
                             ("all_directional",lambda r:r["source_mode"] == "all_directional")):
        selected = [r for r in rows if predicate(r)]
        arm_values = full if group == "all" else {m:metrics([r for r in selected if r["strategy"] == m]) for m in methods}
        comparisons = {ref:{m:paired(selected,ref,m) for m in methods if m != ref} for ref in ("r8","macro_v2_ppo512")}
        groups[group] = dict(arms=arm_values, paired=comparisons)
        if group == "all":
            for method, value in comparisons["r8"].items():
                old = summary["paired"][method]
                assert math.isclose(value["mean_saved_s"],old["mean_saved_s"],abs_tol=1e-7)
                for x,y in zip(value["mean_saved_ci95_s"],old["mean_saving_ci95_s"]): assert abs(x-y) < 1e-7
                for x,y in zip(value["relative_saving_ci95"],old["relative_saving_ci95"]): assert abs(x-y) < 1e-10
    prior_path = ROOT.parent/"q4-rl-micro-actions/results/q4_rl/server-pair-v2-evaluation-001/summary.json"
    prior = json.loads(prior_path.read_bytes())
    old_index = {(r["case_id"],r["strategy"]):r for r in prior["rows"]}
    compared = 0
    for row in rows:
        old_label = {"r8":"r8", "macro_v2_ppo512":"macro_ppo512"}.get(row["strategy"])
        if old_label:
            old = old_index[(row["case_id"],old_label)]
            for field in ("case_sha256", "common_lower_bound_s", "virtual_time_s", "failed_clear_count", "measurement_count", *COMPONENTS):
                assert row[field] == old[field], (row["case_id"],field)
            compared += 1
    review = dict(summary_sha256=SUMMARY_SHA, review_script_sha256=sha(Path(__file__).read_bytes()),
        scope="Completed 512-case DEVELOPMENT panel; offline evidence audit only. No policy reruns or independent-confirmation claim.",
        validation=dict(readback_objects=readback["files"],readback_bytes=checked_bytes,raw_records=6144,
            all_full_clear_and_stored_completion_audits=True,physical_wire_replays=6144,
            expensive_directional_completion_proofs_recomputed=False,
            completion_scope="Stored completion certificates and flags verified for every row; independent physical/feedback/clear/billing replay repeated for every raw record. Expensive geometric completion solver not repeated.",
            frozen_source_files=len(manifest["source_sha256"]),source_zip_crc_and_hashes=True,
            frozen_models=manifest["artifact_sha256"], manifest_specs_freeze_hashes=True,
            panel_pairing_and_summary_scalars=True,full_panel_paired_mean_bootstrap_recomputed=True,
            paired_p95_intervals_scope="Preserved original 5000-cluster bootstrap intervals; raw empirical P95 independently recomputed; P95 bootstrap not redundantly rerun.",
            repeated_r8_macro_v2_rows_match_previous_panel=compared,elapsed_wall_s=time.perf_counter()-began),
        prior_reference_selection=dict(protocol="MEMORY_V4_EVALUATION.md preregistered 2026-09-11 18:10 UTC",
            rule="PPO or SCST, all 512 successful and audited, minimum full-panel mean T, exact tie by label",
            eligible_sorted=[dict(method=label,mean_time_s=value) for value,label in eligible],
            selected=winner, additional_memory_v4_arm_required=winner not in (None,"legacy_ppo128","macro_v2_ppo512"),
            interpretation="Developmental reference selection only; independent superiority not established"),
        groups=groups,original_strata=summary["strata"],raw_failure_modes=aggregate_diagnostics(diagnostics),
        current_qualified_rule_note="R9 current qualification 93922e6f is outside this B panel; R8 is the frozen panel reference, not a claim of the current best rule. R9 is included in subsequent C evaluation.",
        statistical_cautions=["32 seed clusters, 512 correlated transformed scenarios, not 512 independent scenes",
            "Seven stress families and one random family; report ordinary random and stress separately",
            "All methods clear all cases but can incur many failed optical attempts",
            "Full developmental panel was used for method analysis; no promotion without frozen choice and unused confirmation",
            "Different training exposure and warmstarts limit causal attribution across PPO, attention and SCST"])
    OUTPUT.write_text(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    lines = ["# Bundle-v3 completed development evaluation", "",
        "All 6144 runs (12 methods × 512 cases) cleared every source and passed archived audits; all raw physical/feedback/billing histories were independently replayed. The panel contains 32 seeds × eight families × two source modes and is developmental.", "",
        "R9 current qualification 93922e6f is outside this B panel. R8 is the frozen panel reference, not a claim of the current best rule; subsequent C evaluation includes R9.", "",
        "| Method | Clear | Mean T (s) | T/L (ratio of sums) | P95 T (s) | Failed clears | Mean CPU (s) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for method in ["r8","macro_v2_ppo512",*sorted(m for m in methods if m not in ("r8","macro_v2_ppo512"))]:
        value = full[method]
        lines.append(f"| {method} | 512/512 | {value['mean_time_s']:.2f} | {value['ratio_of_sums']:.5f} | {value['p95_time_s']:.2f} | {value['failed_clear_total']} | {value['mean_cpu_s']:.3f} |")
    lines += ["", f"Preregistered eligible prior RL reference: **{winner}**. It is already included in the base memory-v4 specifications, so no additional RL reference arm is required; the separate R9 amendment supplies the thirteenth control. This is reference selection on development data, not a promotion.", ""]
    for method in ("macro_v2_ppo512","attention_v3_ppo512","scst_initialized512","mlp_v3_ppo512","ppo_initialized512"):
        references = ("r8",) if method == "macro_v2_ppo512" else ("r8","macro_v2_ppo512")
        for reference in references:
            value = groups["all"]["paired"][reference][method]; lo,hi = value["relative_saving_ci95"]
            lines.append(f"- {method} versus {reference}: mean saving {100*value['relative_saving']:.2f}% (paired 32-seed-cluster 95% CI {100*lo:.2f}% to {100*hi:.2f}%; negative means slower).")
    lines += ["", "Random and stress strata, full time components, inference/fallback costs and paired contrasts against R8 and the selected RL reference are retained in bundle-v3-independent-review-001.json. Full empirical P95 was independently recomputed; original P95 bootstrap intervals are retained without rerunning the same expensive resampling.", "",
        "All-clear is distinct from zero failed optical clears. Different BC/PPO/SCST training exposure prevents attributing cross-model differences solely to architecture or optimizer. No model is independently qualified by this panel.", "",
        "Object supervisor reports child_returncode=0 but final sync timeout; directory completeness is established by all 6144 evidence hashes and object readback, not by treating sync timeout as success. No model or simulator was rerun during this audit.", ""]
    REPORT.write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(dict(selected_prior_rl=winner,verified_rows=6144,objects=readback["files"],bytes=checked_bytes,
        output=OUTPUT.relative_to(ROOT).as_posix(),elapsed_wall_s=time.perf_counter()-began)),flush=True)


if __name__ == "__main__":
    main()
