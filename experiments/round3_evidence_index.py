"""Read-only index of audited, preregistered round3 development pilots.

No policy/simulator imports, simulations, training, or new lower bounds. Only
fixed controls are pooled; every candidate remains in its own pilot table.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LOCKED = {
    "round3_runner.py": "a4fabe4168675780ed773ae944e028df8d816873124e90dbd640deef7b711494",
    "round2_runner.py": "4119190603ffb4a7d28d7e3bc1e286bdda88c3d6667c59f5659119152d759535",
}
CONTROLS = ("baseline", "original", "rl")
SCOPE = ("Prespecified disjoint 32-case development pilots; fixed controls only are pooled. "
         "Not final acceptance or an official simulator benchmark. No reruns or training. "
         "Candidate comparisons retain their own development batch; no multiplicity-adjusted claim.")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_statistics():
    # These exact modules have no policy/physics imports at module load. Never
    # call their prepare/run/worker/summarize entry points from this script.
    for name, expected in LOCKED.items():
        require(sha(ROOT / "experiments" / name) == expected, f"Changed statistics dependency: {name}")
    spec = importlib.util.spec_from_file_location("_evidence_frozen_statistics", ROOT / "experiments/round3_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        require(expected is None or actual == expected, f"Input SHA mismatch: {path}")
        require(str(path) not in self.hashes or self.hashes[str(path)] == actual, f"Input changed: {path}")
        self.hashes[str(path)] = actual
        return actual

    def verify_unchanged(self):
        for path, expected in self.hashes.items():
            require(sha(path) == expected, f"Input changed during indexing: {path}")


def verify_archives(batch, manifest, runner, inputs):
    require(manifest["runner_sha256"] == LOCKED["round3_runner.py"], "Unexpected runner generation")
    require(manifest["runner_dependencies"] == {"round2_runner.py": LOCKED["round2_runner.py"]}, "Unexpected runner dependency")
    inputs.bind(batch / "runner.py", manifest["runner_sha256"])
    inputs.bind(batch / "round2_runner.py", LOCKED["round2_runner.py"])
    common = None
    for label, policy in manifest["policies"].items():
        for prefix in ("original_spec", "spec"):
            require(runner.digest(policy[prefix]) == policy[prefix + "_sha256"], f"Spec identity mismatch: {label}")
        archive = batch / f"{label}-source.zip"
        inputs.bind(archive, policy["source_archive_sha256"])
        entries = policy["source_sha256"] | policy["evaluation_helper_sha256"]
        with zipfile.ZipFile(archive) as stream:
            expected_names = set(entries) | {"frozen/input_spec.json", "frozen/effective_spec.json"}
            require(set(stream.namelist()) == expected_names and len(stream.namelist()) == len(expected_names), "Source archive contents differ")
            for name, expected in entries.items():
                require(hashlib.sha256(stream.read(name)).hexdigest() == expected, f"Archived source mismatch: {label}/{name}")
            require(json.loads(stream.read("frozen/input_spec.json")) == policy["original_spec"], "Archived input spec mismatch")
            require(json.loads(stream.read("frozen/effective_spec.json")) == policy["spec"], "Archived effective spec mismatch")
        for argument, weight in policy["weights"].items():
            path = (batch / weight["archive_path"]).resolve()
            require(path.is_relative_to(batch) and path.stat().st_size == weight["bytes"], "Invalid archived checkpoint location/size")
            require(Path(policy["spec"]["kwargs"][argument]).resolve() == path, "Checkpoint argument differs from archive")
            inputs.bind(path, weight["sha256"])
        physics = {p: policy["source_sha256"][p] for p in runner.COMMON} | policy["evaluation_helper_sha256"]
        require(common is None or common == physics, "Cross-arm physical/helper identity mismatch")
        common = physics
    return common


def load_batch(batch, runner, inputs):
    batch = batch.resolve()
    for name in ("manifest.json", "summary.json", "audit.json", "protocol.json"):
        inputs.bind(batch / name)
    manifest, summary, audit, protocol = (runner.read(batch / name) for name in
                                         ("manifest.json", "summary.json", "audit.json", "protocol.json"))
    require(manifest["stage"] == summary["stage"] == "pilot", "Only development pilots may be indexed")
    require(summary.get("source_and_weights_verified_after_batch") is True, "Batch lacks completed source verification")
    require(set(CONTROLS) <= set(manifest["policies"]), "Missing fixed control")
    require(runner.digest(protocol) == manifest["protocol_sha256"] and sha(batch / "protocol.json") == manifest["protocol_file_sha256"], "Archived protocol mismatch")
    trial = protocol["created_experiments"][manifest["trial"]]
    require(trial["recipes_sha256"] == runner.digest(manifest["recipes"]) == manifest["recipes_sha256"], "Unregistered recipe set")
    start, stop = trial["pilot_seeds"]
    require(manifest["seeds"] == list(range(start, stop + 1)) and len(manifest["seeds"]) == 32, "Expected complete registered 32-case pilot")
    require(manifest["limits"] == runner.LIMITS and manifest["thread_environment"] == runner.THREAD_ENV, "Evaluation budget/environment differs")
    common = verify_archives(batch, manifest, runner, inputs)
    runner.verify_prior_audit(batch, manifest, summary)
    require(len(audit["batches"]) == 1 and audit["unique_cases"] == 32, "Audit is not this complete pilot")
    require(audit["sqlite_read"] is False and audit["new_simulations"] == 0 and audit["simulator_calls"] == 0, "Unexpected audit data source")
    audit_rows = {(r["strategy"], r["seed"]): r for r in audit["rows"]}
    expected = {(label, seed) for label in manifest["policies"] for seed in manifest["seeds"]}
    require(len(audit_rows) == len(audit["rows"]) and set(audit_rows) == expected, "Duplicate/missing audit episode identity")
    summary_rows = {(r["strategy"], r["seed"]): r for r in summary["rows"]}
    require(len(summary_rows) == len(summary["rows"]) and set(summary_rows) == expected, "Duplicate/missing summary episode identity")
    require(set(audit["bounds_by_seed"]) == {str(s) for s in manifest["seeds"]} == set(summary["case_sha256"]), "Missing/extra case bounds")
    rows, bounds = {label: [] for label in manifest["policies"]}, {}
    for label, policy in manifest["policies"].items():
        for seed in manifest["seeds"]:
            relative = f"records/{label}-{seed}.json.gz"
            path = batch / relative
            inputs.bind(path, summary["records_sha256"][relative])
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                record = json.load(stream)
            runner.verify_record(record, manifest, label, seed)
            row, audited = record["row"], audit_rows[label, seed]
            require(row == summary_rows[label, seed], "Raw and summary rows differ")
            for field in ("strategy", "seed", "case_id", "case_sha256", "source_total", "successful", "failed_clear_count", "virtual_time_s", "program_runtime_s"):
                require(row[field] == audited[field], f"Audited row mismatch: {field}")
            integrity = record["round3_integrity"]
            require(integrity["weights_sha256"] == {k: v["sha256"] for k, v in policy["weights"].items()} and integrity["runner_dependencies"] == manifest["runner_dependencies"] and integrity["thread_environment"] == manifest["thread_environment"], "Record model/runtime integrity mismatch")
            bound = audit["bounds_by_seed"][str(seed)]
            lower = bound["physical_clairvoyant_lower_s"]
            require(math.isfinite(lower) and lower > 0 and lower == audited["physical_lower_s"], "Invalid/inconsistent old lower bound")
            require(row["case_sha256"] == bound["case_sha256"] == summary["case_sha256"][str(seed)], "Paired case identity mismatch")
            for field in ("penalized_time_s", "virtual_time_s", "program_cpu_s", "program_runtime_s"):
                require(math.isfinite(row[field]) and row[field] >= 0, f"Invalid complete cost: {field}")
            bounds[str(seed)] = {"case_sha256": row["case_sha256"], "LB_s": lower}
            rows[label].append(row)
    require(len({b["case_sha256"] for b in bounds.values()}) == len(bounds), "Duplicate case world within pilot")
    return manifest, summary, rows, bounds, common


def metrics(rows, bounds, runner):
    times = [r["penalized_time_s"] for r in rows]
    lower = [bounds[str(r["seed"])]["LB_s"] for r in rows]
    return {"runs": len(rows), "successful": sum(r["successful"] for r in rows),
        "failed_runs": sum(not r["successful"] for r in rows), "failed_clears": sum(r["failed_clear_count"] for r in rows),
        "mean_s": statistics.mean(times), "p95_s": runner.legacy.percentile(times, .95), "max_s": max(times),
        "mean_LB_s": statistics.mean(lower), "mean_case_T_over_LB": statistics.mean(t / b for t, b in zip(times, lower)),
        "sum_T_over_sum_LB": sum(times) / sum(lower), "mean_cpu_s": statistics.mean(r["program_cpu_s"] for r in rows),
        "mean_policy_wall_s": statistics.mean(r["program_runtime_s"] for r in rows),
        "mean_measurements": statistics.mean(r["measurement_count"] for r in rows)}


def render(result):
    lines = ["# Round3 development evidence index", "", result["scope"], "",
        "虚拟时间采用整局失败惩罚口径，失败不剔除。旧 LB 为事后先知物理下界，不能据此宣称逼近线上可达最优。CPU 与 wall 均为每局策略计算耗时；下表单位为秒。", ""]
    def table(title, groups):
        lines.extend([title, "", "|策略|局数/成功/失败清除|均值|P95|max|CPU|wall|均LB|均T/LB|总T/总LB|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for label, m in groups.items():
            lines.append(f"|{label}|{m['runs']}/{m['successful']}/{m['failed_clears']}|{m['mean_s']:.6f}|{m['p95_s']:.6f}|{m['max_s']:.6f}|{m['mean_cpu_s']:.6f}|{m['mean_policy_wall_s']:.6f}|{m['mean_LB_s']:.6f}|{m['mean_case_T_over_LB']:.9f}|{m['sum_T_over_sum_LB']:.9f}|")
        lines.append("")
    def comparison(name, values):
        lo, hi = values["ci95_saved_s"]
        lines.append(f"- {name}：平均节省 {values['mean_saved_s']:.6f} s，配对 bootstrap 95% CI [{lo:.6f}, {hi:.6f}]，胜/负/平 {values['wins']}/{values['losses']}/{values['ties']}。")
    table(f"固定控制合计：{result['unique_cases']} 个互不重复案例", result["metrics"])
    comparison("derived 相对 original", result["derived_vs_original"])
    comparison("derived 相对 RL", result["derived_vs_rl"])
    for pilot in result["pilot_results"]:
        lines.append("")
        table("独立开发批次：" + pilot["trial"], pilot["metrics"])
        for label, values in pilot["comparisons_vs_baseline"].items():
            comparison(label + " 相对 derived", values)
        for label in pilot["candidate_labels"]:
            comparison(label + " 相对 RL", pilot["comparisons_vs_rl"][label])
    lines.extend(["", "JSON 保存完整原始记录、源码和模型归档的输入 SHA；统计依赖也锁定 SHA。配对 CI 使用冻结的 10,000 次重采样，正值表示候选节省费用。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Exact replay check against an existing fixed-control index")
    args = parser.parse_args()
    destinations = [args.output.resolve(), args.report.resolve()]
    require(len(set(destinations)) == 2 and not any(p.exists() for p in destinations), "Refuse output overwrite")
    batches = [p.resolve() for p in args.batches]
    require(len(set(batches)) == len(batches), "Duplicate input batch")
    require(not any(p.is_relative_to(b) for p in destinations for b in batches), "Do not write inside frozen batches")
    runner, inputs = load_statistics(), Inputs()
    inputs.bind(__file__)
    for name, expected in LOCKED.items():
        inputs.bind(ROOT / "experiments" / name, expected)
    controls, bounds, identities, pilots, evidence = {p: [] for p in CONTROLS}, {}, None, [], []
    runtime, physics, case_hashes, trials = None, None, set(), set()
    for batch in batches:
        manifest, summary, rows, local_bounds, common = load_batch(batch, runner, inputs)
        require(manifest["trial"] not in trials and not (set(bounds) & set(local_bounds)), "Repeated trial or case seed")
        require(not (case_hashes & {b["case_sha256"] for b in local_bounds.values()}), "Duplicate case world across pilots")
        current = {p: runner.policy_identity(manifest["policies"][p]) for p in CONTROLS}
        environment = {k: manifest[k] for k in ("runtime_identity", "limits", "platform", "thread_environment", "python_executable_sha256")}
        require(identities is None or identities == current, "Fixed control identity changed across pilots")
        require(runtime is None or runtime == environment, "Cross-batch runtime changed")
        require(physics is None or physics == common, "Cross-batch physics/helpers changed")
        identities, runtime, physics = current, environment, common
        trials.add(manifest["trial"])
        case_hashes.update(b["case_sha256"] for b in local_bounds.values())
        bounds.update(local_bounds)
        for label in CONTROLS:
            controls[label].extend(rows[label])
        group_metrics = {label: metrics(group, local_bounds, runner) for label, group in rows.items()}
        for label, values in group_metrics.items():
            for key in ("runs", "successful", "failed_clears", "mean_s", "p95_s", "max_s", "mean_cpu_s", "mean_measurements"):
                require(summary["averages"][label][key] == values[key], "Summary metric differs from complete raw rows")
        pilots.append({"trial": manifest["trial"], "stage": "development/pilot", "candidate_labels": [p for p in rows if p not in CONTROLS],
            "unique_cases": len(local_bounds), "metrics": group_metrics,
            "comparisons_vs_baseline": {p: runner.compare_rows(rows["baseline"], r) for p, r in rows.items() if p != "baseline"},
            "comparisons_vs_rl": {p: runner.compare_rows(rows["rl"], r) for p, r in rows.items() if p != "rl"}})
        evidence.append({"batch": str(batch), **{k + "_sha256": sha(batch / (k + ".json")) for k in ("manifest", "summary", "audit")}})
    result = {"scope": SCOPE, "unique_cases": len(bounds), "frozen_control_identities": identities,
        "input_evidence": evidence, "metrics": {p: metrics(r, bounds, runner) for p, r in controls.items()},
        "derived_vs_original": runner.compare_rows(controls["original"], controls["baseline"]),
        "derived_vs_rl": runner.compare_rows(controls["rl"], controls["baseline"]),
        "rows": controls, "bounds_by_seed": bounds, "pilot_results": pilots,
        "runtime_identity": runtime, "common_physical_identity": physics,
        "cost_scope": "Complete penalized cost including every failed run; old posthoc LB reused unchanged."}
    if args.reference:
        inputs.bind(args.reference)
        reference = runner.read(args.reference)
        for key in ("unique_cases", "frozen_control_identities", "derived_vs_original", "derived_vs_rl", "rows", "bounds_by_seed"):
            require(result[key] == reference[key], "Exact reference replay mismatch: " + key)
        for label, values in reference["metrics"].items():
            for key, value in values.items():
                require(result["metrics"][label][key] == value, f"Exact reference replay mismatch: {label}/{key}")
        result["reference_replay"] = {"path": str(args.reference.resolve()), "sha256": sha(args.reference), "exact_all_existing_statistics_rows_identities_bounds": True}
    inputs.verify_unchanged()
    result["input_sha256"] = inputs.hashes
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    with args.report.open("x", encoding="utf-8") as stream:
        stream.write(render(result))
    print(json.dumps({"unique_cases": result["unique_cases"], "verified_inputs": len(inputs.hashes), "reference_replay": bool(args.reference), "output_sha256": sha(args.output), "report_sha256": sha(args.report)}))


if __name__ == "__main__":
    main()
