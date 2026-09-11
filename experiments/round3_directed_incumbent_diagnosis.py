"""Read-only mechanism audit of a FINISHED directed-incumbent batch.

No policy, simulator, scenario generator, SQLite or hidden-world evaluation.
Only public action/planning logs and the already completed physical audit are
used. The original pure fixed-route solver reconstructs the old incumbent.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import importlib
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.round3_posthoc_audit import digest, read, sha, verify_batch


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def close(actual, expected, label, tolerance=1e-7):
    require(math.isfinite(actual) and math.isfinite(expected)
            and abs(actual-expected) <= tolerance, label)


def price(log, order):
    tasks, matrix, initial = log["tasks"], log["travel_times_s"], log["initial_times_s"]
    n = len(tasks)
    require(len(order) == n and all(type(i) is int for i in order)
            and set(order) == set(range(n)), "invalid complete route permutation")
    require(len(matrix) == n and len(initial) == n and all(len(row) == n for row in matrix), "matrix dimensions")
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                for v in initial + [x for row in matrix for x in row]), "nonfinite/negative matrix")
    require(all(matrix[i][i] == 0 for i in range(n)), "matrix diagonal")
    require(all(t["kind"] in ("source", "cover") and math.isfinite(t["service_s"])
                and t["service_s"] >= 0 for t in tasks), "task service")
    scan = log["scan_source_s"]
    require(math.isfinite(scan) and scan >= 0, "scan service")
    total, last, sources = 0.0, None, sum(t["kind"] == "source" for t in tasks)
    for j in order:
        total += (initial[j] if last is None else matrix[last][j]) + tasks[j]["service_s"]
        if tasks[j]["kind"] == "source":
            sources -= 1
        else:
            total += scan * sources
        last = j
    return total


def action_key(action):
    return (action["action"], action["channel"], tuple(action["position"]),
            action["result"], action.get("bearing_deg"))


def physical_costs(actions):
    position, channel, total = (0.0, 0.0), 1, Counter()
    for action in actions:
        move = round(math.dist(position, action["position"]) / 5 * 1e6) / 1e6
        total["movement_s"] += move
        if action["action"] == "measure":
            total["detection_s"] += 5.0
            total["switching_s"] += channel != action["channel"]
            channel = action["channel"]
        else:
            require(action["action"] == "clear", "unsupported physical action")
            total["clear_s"] += 5.0 if action["result"] == "success" else 3.0
        # Clear does not tune the radio.
        position = tuple(action["position"])
        close(action["virtual_time_s"], sum(total.values()), "public-prefix virtual accounting", 1e-5)
    return {k: total[k] for k in ("movement_s", "detection_s", "switching_s", "clear_s")}


def verify_decision(log, actions, *, require_incumbent):
    require(log["expansions_after_parent"] == log["expansions_before_parent"] + log["parent_expansions"], "parent expansion accounting")
    require(log["expansions_after_decision"] == log["expansions_after_parent"] + log["directed_expansions"], "child expansion accounting")
    if log["status"] != "directed_selected":
        return {"status": log["status"], "reason": log.get("fallback_reason"), "complete": False}
    require(log["transition_audit"]["complete"], "selected incomplete model")
    cost = price(log, log["route"]["order"])
    close(cost, log["route"]["cost_s"], "returned route cost")
    require(log["route"]["lower_bound_s"] <= cost + 1e-7, "invalid proxy interval")
    require(log["route"]["expanded"] == log["directed_expansions"] <= log["route_budget"], "route expansion budget")
    first = log["tasks"][log["route"]["order"][0]]
    require(all(first[k] == log["selected"][k] for k in ("kind", "channel", "exit_representative")), "selected route/task mismatch")
    parent_cost = None
    if require_incumbent:
        require(log.get("incumbent_supplied") is True, "complete new model missing incumbent")
        parent_cost = price(log, log["incumbent_order"])
        close(parent_cost, log["parent_order_cost_new_model_s"], "supplied parent price")
        close(cost-parent_cost, log["candidate_minus_parent_order_same_model_s"], "same-model delta")
        require(log["incumbent_result_order"] == log["route"]["order"], "returned incumbent route binding")
        close(cost, log["incumbent_result_cost_s"], "returned incumbent price binding")
        require(cost <= parent_cost + 1e-9, "supplied incumbent contract violated")
    executed = log["execution_status"] == "first_physical_action_recorded"
    if executed:
        ordinal = log["first_actual_action_ordinal"]
        require(ordinal == log["after_actual_action_count"]+1 and 1 <= ordinal <= len(actions), "first action prefix ordinal")
        actual = actions[ordinal-1]
        require(actual == log["first_actual_action"] and log["selected_task_matches_actual"] is True, "actual action binding")
        if first["kind"] == "source":
            require(actual["channel"] == first["channel"], "selected source/channel mismatch")
        else:
            require(actual["phase"] == "coverage" and actual["position"] == first["exit_representative"], "selected cover/action mismatch")
        j = log["route"]["order"][0]
        if j in log["uncertain_target_indices"]:
            arc = next(a for a in log["transition_audit"]["arcs"]
                       if a["target_index"] == j and a["origin"] == log["current_position"])
            require(actual["action"] == "measure" and actual["position"] == arc["first_probe"], "predicted first probe not executed")
    else:
        require(log["execution_status"] in ("interrupted_before_physical_action", "ended_without_physical_action",
                "no_physical_action_before_next_decision"), "unknown execution state")
        require("first_actual_action" not in log, "nonexecution contains actual action claim")
    return {"status": log["status"], "complete": True, "executed": executed,
            "prefix": log["after_actual_action_count"], "changed": log["changed_first_task"],
            "exact": log["route"]["exact"], "gap_s": cost-log["route"]["lower_bound_s"],
            "cost_s": cost, "supplied_parent_cost_s": parent_cost,
            "matrix_sha256": digest({k:log[k] for k in ("tasks", "travel_times_s", "initial_times_s", "scan_source_s", "current_position")})}


def reconstruct_parent(log, relocation, helper, config):
    """Replay ONLY the deterministic frozen point solver, including variants."""
    require(log["after_actual_action_count"] == relocation["after_actual_action_count"], "relocation prefix")
    tasks = log["tasks"]
    covers = sum(t["kind"] == "cover" for t in tasks)
    before = [dict(t, exit_representative=relocation["remaining_before"][i]) if i < covers else dict(t)
              for i, t in enumerate(tasks)]
    total = log["expansions_before_parent"]

    def solve(items):
        nonlocal total
        finite = [helper.RouteTask(helper.Position.coerce(t["exit_representative"]), t["kind"] == "source", t["service_s"]) for t in items]
        result = helper.solve_state_route(finite, helper.Position.coerce(log["current_position"]),
            scan_source_s=log["scan_source_s"], max_expansions=max(0, min(config["max_expansions"], config["max_total_expansions"]-total)))
        total += result.expanded
        return result

    baseline = solve(before)
    close(baseline.cost_s, relocation["baseline_proxy_s"], "reconstructed parent baseline cost")
    best_cost, best_order, best_tasks = baseline.cost_s, baseline.order, before
    for variant in relocation["evaluated"]:
        alternate = [dict(t) for t in before]
        alternate[variant["site_index"]]["exit_representative"] = variant["position"]
        result = solve(alternate)
        fixed = baseline.cost_s-variant["fixed_order_gain_s"]
        cost, order = (fixed, baseline.order) if fixed < result.cost_s else (result.cost_s, result.order)
        close(cost, variant["proxy_cost_s"], "reconstructed parent variant cost")
        require(result.expanded == variant["route_expanded"], "reconstructed parent variant budget")
        if cost < best_cost-1.0:
            best_cost, best_order, best_tasks = cost, order, alternate
    require(total == log["expansions_after_parent"], "reconstructed total parent expansions")
    require(best_tasks == tasks, "reconstructed selected parent layout")
    close(best_cost, log["parent_planning"]["cost_s"], "reconstructed selected parent cost")
    require(all(best_tasks[best_order[0]][k] == log["parent_selected"][k]
                for k in ("kind", "channel", "exit_representative")), "reconstructed selected parent first task")
    return tuple(best_order)


def load_parent_helper(source_root, identity):
    # Require the full source snapshot to match the named old control first.
    for relative, expected in identity["source_sha256"].items():
        require(sha(source_root / relative) == expected, "parent source mismatch: " + relative)
    sys.path.insert(0, str(source_root / "src"))
    module = importlib.import_module("planning.state_route")
    require(Path(module.__file__).resolve() == (source_root / "src/planning/state_route.py").resolve(), "unexpected imported parent solver")
    # The only callable used is this pure finite task solver; no policy class.
    return module


def aggregate(rows):
    completed = [r for r in rows if r["complete"]]
    return {"decisions": len(rows), "complete_models": len(completed),
        "fallback_reasons": dict(Counter(r.get("reason") for r in rows if not r["complete"])),
        "executed_models": sum(r["executed"] for r in completed),
        "changed_first_tasks": sum(r["changed"] and r["executed"] for r in completed),
        "supplied_parent_routes": sum(r["supplied_parent_cost_s"] is not None for r in completed),
        "exact_models": sum(r["exact"] for r in completed),
        "worse_than_reconstructed_parent": sum(r["cost_s"] > r["reconstructed_parent_cost_s"]+1e-9 for r in completed),
        "max_proxy_regression_s": max((r["cost_s"]-r["reconstructed_parent_cost_s"] for r in completed), default=0.0),
        "mean_gap_s": statistics.mean(r["gap_s"] for r in completed) if completed else None}


def build(batch, audit_path, source_root):
    manifest, summary, _, _ = verify_batch(batch)
    require({"baseline", "old_directed", "candidate"} <= set(manifest["policies"]), "missing paired controls")
    audit = read(audit_path)
    evidence = {str(batch / f): sha(batch / f) for f in ("manifest.json", "summary.json")}
    evidence[str(audit_path)] = sha(audit_path)
    require(any(b["manifest_sha256"] == sha(batch/"manifest.json") and b["summary_sha256"] == sha(batch/"summary.json") for b in audit["batches"]), "physical audit bound to different batch")
    helper = load_parent_helper(source_root, manifest["policies"]["old_directed"])
    for relative, expected in summary["records_sha256"].items():
        require(sha(batch/relative) == expected, "changed raw archive")
        evidence[str(batch/relative)] = expected
    audit_rows = {(r["strategy"], r["seed"]):r for r in audit["rows"]}
    rows = {"candidate": [], "old_directed": []}
    case_rows, common_rows = [], []
    for seed in manifest["seeds"]:
        public, case = {}, {"seed": seed}
        for arm in ("baseline", "old_directed", "candidate"):
            path = batch/f"records/{arm}-{seed}.json.gz"
            raw = json.loads(gzip.decompress(path.read_bytes()))
            require(raw["frozen_manifest_sha256"] == digest(manifest) and raw["spec"] == manifest["policies"][arm]["spec"], "record source/spec binding")
            require(raw["round3_integrity"]["before_and_after_verified"] is True, "record source changed")
            report, row = raw["summary"], raw["row"]
            require(row["seed"] == seed and row["strategy"] == arm, "record role/case mismatch")
            costs = physical_costs(report["action_history"])
            close(sum(costs.values()), row["virtual_time_s"], "full physical cost", 1e-5)
            bound = audit_rows[(arm, seed)]
            require(bound["input_sha256"] == sha(path) and bound["case_sha256"] == row["case_sha256"], "lower bound/raw binding")
            close(bound["virtual_time_s"], row["virtual_time_s"], "lower bound time binding")
            eligible = bool(bound["audit_passed"] and bound["ratio_eligible"])
            lower = bound["physical_lower_s"] if eligible else None
            case[arm] = {"time_s": row["virtual_time_s"], "penalized_time_s": row["penalized_time_s"],
                "successful": row["successful"], "failed_clears": row["failed_clear_count"],
                "runtime_s": row["program_runtime_s"], "cpu_s": row["program_cpu_s"],
                "physical_lower_s": lower, "T_LB": row["virtual_time_s"]/lower if lower else None, **costs}
            public[arm] = report
            if arm == "baseline":
                continue
            logs = report["strategy_parameters"]["directed_localization_log"]
            relocations = report["strategy_parameters"]["relocation_log"]
            require(len(logs) == len(relocations), "unaligned parent/child logs")
            config = raw["spec"]["kwargs"]["config"]
            previous = 0
            for i, (log, relocation) in enumerate(zip(logs, relocations)):
                require(log["expansions_before_parent"] == previous, "expansion chain")
                previous = log["expansions_after_decision"]
                require(previous <= config["max_total_expansions"], "case expansion cap")
                item = {"seed": seed, "decision_index": i,
                        **verify_decision(log, report["action_history"], require_incumbent=arm == "candidate")}
                if item["complete"]:
                    parent = reconstruct_parent(log, relocation, helper, config)
                    parent_cost = price(log, parent)
                    item.update(reconstructed_parent_order=list(parent), reconstructed_parent_cost_s=parent_cost)
                    if arm == "candidate":
                        require(list(parent) == log["incumbent_order"], "supplied route differs from real parent")
                rows[arm].append(item)
        common = 0
        for a, b in zip(public["candidate"]["action_history"], public["old_directed"]["action_history"]):
            if action_key(a) != action_key(b):
                break
            common += 1
        case["common_actual_prefix_candidate_old"] = common
        case["candidate_minus_old_penalized_s"] = case["candidate"]["penalized_time_s"]-case["old_directed"]["penalized_time_s"]
        case["candidate_minus_baseline_penalized_s"] = case["candidate"]["penalized_time_s"]-case["baseline"]["penalized_time_s"]
        case_rows.append(case)
        # Matching only by decision index after trajectories diverge is invalid.
        for new in (r for r in rows["candidate"] if r["seed"] == seed and r["complete"] and r["prefix"] <= common):
            matches = [r for r in rows["old_directed"] if r["seed"] == seed and r["complete"]
                       and r["prefix"] == new["prefix"] and r["matrix_sha256"] == new["matrix_sha256"]]
            if len(matches) == 1:
                old = matches[0]
                common_rows.append({"seed": seed, "prefix": new["prefix"], "matrix_sha256": new["matrix_sha256"],
                    "old_cost_s": old["cost_s"], "candidate_cost_s": new["cost_s"],
                    "old_minus_candidate_proxy_s": old["cost_s"]-new["cost_s"]})
    performance = {}
    for arm in ("baseline", "old_directed", "candidate"):
        values = [c[arm] for c in case_rows]
        eligible = [v for v in values if v["T_LB"] is not None]
        performance[arm] = {"runs": len(values), "successful": sum(v["successful"] for v in values),
            "failed_clears": sum(v["failed_clears"] for v in values),
            "mean_time_s": statistics.mean(v["time_s"] for v in values),
            "mean_cpu_s": statistics.mean(v["cpu_s"] for v in values),
            "mean_T_LB": statistics.mean(v["T_LB"] for v in eligible) if eligible else None,
            "sum_T_over_sum_LB": sum(v["time_s"] for v in eligible)/sum(v["physical_lower_s"] for v in eligible) if eligible else None}
    deltas = [c["candidate_minus_old_penalized_s"] for c in case_rows]
    return {"version": "round3-directed-incumbent-mechanism-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sqlite_read": False, "new_cases": 0, "policy_calls": 0, "simulator_calls": 0,
        "script_sha256": sha(__file__), "verification_dependency_sha256": sha(ROOT/"experiments/round3_posthoc_audit.py"),
        "input_sha256": evidence, "source_commits": {k:v["commit"] for k,v in manifest["policies"].items()},
        "parent_solver_source_sha256": manifest["policies"]["old_directed"]["source_sha256"],
        "mechanism_checks_passed": True, "physical_audits_passed": audit["all_audits_passed"],
        "mechanism": {arm:aggregate(values) for arm,values in rows.items()}, "performance": performance,
        "candidate_vs_old": {"mean_saved_s": -statistics.mean(deltas), "wins": sum(x < -1e-6 for x in deltas),
            "losses": sum(x > 1e-6 for x in deltas), "ties": sum(abs(x) <= 1e-6 for x in deltas),
            "worst_regression_s": max(deltas), "inference": "descriptive paired results, not a new significance claim"},
        "same_actual_prefix_and_matrix": common_rows, "cases": case_rows, "decisions": rows,
        "limitation": "Returned proxy <= its same-matrix parent incumbent is a local finite-model guarantee, not a bound on whole-episode time. Different post-divergence matrices are not interchangeable.",
        "registered_candidate_vs_baseline": summary["comparisons"]["candidate"]}


def render(result):
    lines = ["# 有向定位 incumbent：冻结批机制复核", "", result["limitation"], "",
        "|角色|完整模型|实测执行|供应父路线|劣于同模型父路线|exact|", "|---|---:|---:|---:|---:|---:|"]
    for arm, r in result["mechanism"].items():
        lines.append(f"|{arm}|{r['complete_models']}|{r['executed_models']}|{r['supplied_parent_routes']}|{r['worse_than_reconstructed_parent']}|{r['exact_models']}|")
    lines += ["", "|角色|成功/局数|失败清除|均时 s|均 CPU s|均 T/LB|总 T/总 LB|", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, r in result["performance"].items():
        ratio = lambda x: f"{x:.6f}" if x is not None else "不可用"
        lines.append(f"|{arm}|{r['successful']}/{r['runs']}|{r['failed_clears']}|{r['mean_time_s']:.6f}|{r['mean_cpu_s']:.6f}|{ratio(r['mean_T_LB'])}|{ratio(r['sum_T_over_sum_LB'])}|")
    r = result["candidate_vs_old"]
    lines += ["", f"同局 candidate 相对 old_directed 平均节省 {r['mean_saved_s']:.6f} s；{r['wins']} 胜、{r['losses']} 负、{r['ties']} 平，最差配对退步 {r['worst_regression_s']:.6f} s。此为描述性结果，是否晋级仍按预注册基准比较判断。",
        "", f"已核验 {len(result['same_actual_prefix_and_matrix'])} 个相同实际历史前缀且同一矩阵的跨版本比较；轨迹分歧之后不将两组代理费用直接比较。", "",
        "完整输入 SHA、源码提交、供应/重建路线、实际首动作绑定、每局费用与旧物理下界见同名 JSON。此诊断不重新生成下界、不调用策略、模拟器或 SQLite。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--parent-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output stem; existing files are rejected")
    args = parser.parse_args()
    json_path, md_path = args.output.with_suffix(".json"), args.output.with_suffix(".md")
    require(not json_path.exists() and not md_path.exists(), "refusing to overwrite previous evidence")
    result = build(args.batch.resolve(), args.audit.resolve(), args.parent_source_root.resolve())
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    md_path.write_text(render(result), encoding="utf-8")
    print(json.dumps({"mechanism": result["mechanism"], "performance": result["performance"],
                      "candidate_vs_old": result["candidate_vs_old"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
