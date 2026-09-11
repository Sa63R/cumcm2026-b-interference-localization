"""Read finished three-arm archives; diagnose mechanism and actual divergence.

No policy/simulator import, new cases, SQLite or network. Full batches require
an independently generated original-LB audit tied to each exact record hash.
--smoke only checks parsing of named completed implementation-smoke archives;
it is deliberately not accepted as independent performance evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.round2_posthoc_audit import digest, read, sha, verify_batch


PHASE_COSTS = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")
EVENT_KEYS = ("action", "channel", "position", "result", "bearing_deg", "virtual_time_s")


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {"count": len(values), "min": values[0], "mean": statistics.fmean(values), "max": values[-1]}


def identity(task):
    if task is None:
        return None
    return (("source", task["channel"]) if task["kind"] == "source"
            else ("cover", *task["position"]))


def event(action):
    return {key: action.get(key) for key in EVENT_KEYS}


def checked_history(record):
    raw, summary = record["history"], record["summary"]
    if not raw or raw[0]["action"] != "/enter" or raw[-1]["action"] != "/exit":
        raise ValueError("archive lacks completed enter/exit history")
    if not all(item["response"].get("accepted") is True for item in raw):
        raise ValueError("non-accepted physical response requires separate protocol diagnosis")
    physical = [item for item in raw if item["action"] in ("/measure", "/clear")]
    logged = summary["action_history"]
    if len(physical) != len(logged) or len(raw) != len(physical)+2:
        raise ValueError("physical and policy history lengths differ")
    for action, actual in zip(logged, physical):
        response = actual["response"]
        observed = {"action": actual["action"][1:], "channel": actual["channel"],
                    "position": [actual["position"]["x"], actual["position"]["y"]],
                    "result": response.get("measure_result", response.get("clear_result")),
                    "bearing_deg": response.get("svd_deg"), "virtual_time_s": response["virtual_time_s"]}
        if event(action) != observed:
            raise ValueError("policy history does not match actual physical feedback")
    if summary["virtual_time_s"] != raw[-1]["response"]["virtual_time_s"]:
        raise ValueError("summary time differs from exit feedback")
    return logged


def pool_support(pool):
    channels = pool.get("per_channel", {})
    return {"draw_weight_ess": pool.get("draw_weight_ess"),
            "distinct_active_position_worlds": pool.get("distinct_active_position_worlds"),
            "position_pool_ess": distribution([x.get("position_ess") for x in channels.values()]),
            "position_pool_distinct": distribution([x.get("distinct_positions") for x in channels.values()]),
            "scope": "Pool proposal support and equal draw weights are not actual evaluated sample sizes"}


def planning_diagnosis(summary):
    params = summary.get("strategy_parameters", {})
    logs = params.get("observation_tree_log", [])
    depth = params.get("observation_tree", {}).get("depth")
    result = {"depth": depth, "planning_attempts": len(logs), "statuses": dict(Counter(x["status"] for x in logs)),
              "completed_comparisons": 0, "logged_changed": 0, "macro_identity_changed": 0,
              "mechanism_active": 0, "first_actions_with_multiple_effective_z": 0,
              "first_actions_with_observation_dependent_actions": 0,
              "tail_evaluations_started": 0, "completed_tails_retained_in_nodes": 0,
              "tail_completion_not_identifiable_from_retained_nodes": 0,
              "generated_actions": 0, "planning_runtime_s": 0., "fallback_reasons": {},
              "decisions": [], "errors": []}
    reasons = Counter()
    history = summary["action_history"]
    for ordinal, log in enumerate(logs):
        status = log["status"]
        complete = status in ("changed", "evaluated_baseline")
        boundary = log["after_actual_action_count"]
        if type(boundary) is not int or not 0 <= boundary <= len(history):
            raise ValueError("planning boundary outside physical action history")
        instant = history[boundary-1]["virtual_time_s"] if boundary else 0.
        if instant != log["virtual_time_s"]:
            raise ValueError("planning log is not at its declared actual boundary")
        changed = identity(log["baseline"]) != identity(log["selected"])
        if changed != (status == "changed"):
            result["errors"].append(f"decision {ordinal}: status and macro identity differ")
        decision = {"index": ordinal, "action_boundary": boundary, "status": status,
                    "baseline": log["baseline"], "selected": log["selected"],
                    "macro_identity_changed": changed,
                    "next_actual_action": event(history[boundary]) if boundary < len(history) else None,
                    "fallback_reason": log.get("reason"), "root_pool": pool_support(log.get("root_belief", {})),
                    "tail_evaluations_started": log.get("tail_evaluations", 0),
                    "generated_actions": log.get("generated_actions", 0),
                    "planning_runtime_s": log.get("planning_runtime_s", 0), "first_actions": []}
        retained_tails = retained_comparisons = 0
        for candidate in log.get("candidates", []):
            valid = [n for n in candidate.get("nodes", []) if n.get("status") == "conditional_action_evaluated"]
            distinct_z = len({n["observation_sha256"] for n in valid})
            distinct_actions = len({identity(n["selected"]) for n in valid})
            if (candidate.get("distinct_effective_observations", distinct_z) != distinct_z
                    or candidate.get("distinct_conditional_actions", distinct_actions) != distinct_actions):
                result["errors"].append(f"decision {ordinal}: candidate mechanism counters mismatch")
            first = {"task": {key: candidate[key] for key in ("kind", "channel", "position")},
                     "effective_distinct_z": distinct_z, "distinct_conditional_actions": distinct_actions,
                     "representatives_retained": len(candidate.get("nodes", [])),
                     "paired_gain_s": candidate.get("paired_gain_s"),
                     "heuristic_margin_s": candidate.get("heuristic_selection_margin_s"), "nodes": []}
            for node in candidate.get("nodes", []):
                kind = node.get("status")
                if kind == "conditional_action_evaluated":
                    retained_comparisons += 1
                    completed_tails = sum(len(values) for values in node["training_costs_s"]) + len(node["heldout_costs_s"])
                elif kind == "fixed_v1_tail":
                    completed_tails = len(node["costs_s"])
                elif kind == "terminal_or_no_alternative":
                    completed_tails = 1
                else:
                    raise ValueError(f"unknown retained node status: {kind}")
                if kind in ("conditional_action_evaluated", "fixed_v1_tail"):
                    train_n = node.get("actual_training_draws")
                    eval_n = node.get("actual_evaluation_draws")
                    expected_eval = len(node["heldout_costs_s"] if kind == "conditional_action_evaluated" else node["costs_s"])
                    if eval_n != expected_eval or (kind == "conditional_action_evaluated" and any(
                            len(values) != train_n for values in node["training_costs_s"])):
                        result["errors"].append(f"decision {ordinal}: evaluated sample counts disagree with cost arrays")
                    for label,n in (("training",train_n),("evaluation",eval_n)):
                        used = node.get(label+"_used_support")
                        if used and (used["draws"] != n or used["equal_draw_weight_ess"] != n
                                     or not 0 < used["distinct_active_position_worlds"] <= n):
                            result["errors"].append(f"decision {ordinal}: {label} used-support counters inconsistent")
                retained_tails += completed_tails
                belief = node.get("belief", {})
                first["nodes"].append({"status": kind, "observation_sha256": node["observation_sha256"],
                    "observation_count": node["observation_count"], "first_cost_s": node["first_cost_s"],
                    "inner_baseline": node.get("inner_baseline"), "selected": node.get("selected"),
                    "completed_tail_count_from_cost_arrays": completed_tails,
                    "actual_training_draws": node.get("actual_training_draws"),
                    "actual_evaluation_draws": node.get("actual_evaluation_draws"),
                    "training_used_support": node.get("training_used_support"),
                    "evaluation_used_support": node.get("evaluation_used_support"),
                    "training_pool": pool_support(belief.get("training_pool", {})),
                    "evaluation_pool": pool_support(belief.get("evaluation_pool", {})),
                    "unused_conditional_draws": node.get("unused_conditional_draws")})
            decision["first_actions"].append(first)
            result["first_actions_with_multiple_effective_z"] += distinct_z >= 2
            result["first_actions_with_observation_dependent_actions"] += distinct_z >= 2 and distinct_actions >= 2
        active = complete and depth == 2 and any(x["effective_distinct_z"] >= 2 for x in decision["first_actions"])
        if bool(log.get("mechanism_active", False)) != active:
            result["errors"].append(f"decision {ordinal}: mechanism_active differs from retained evidence")
        missing = log.get("tail_evaluations", 0)-retained_tails
        if missing < 0 or (complete and missing):
            result["errors"].append(f"decision {ordinal}: complete-tail count mismatch")
        decision.update(mechanism_active_recomputed=active, completed_tails_retained=retained_tails,
                        tails_not_identifiable_from_retained_nodes=missing,
                        second_layer_comparisons_logged=log.get("second_layer_comparisons", 0),
                        second_layer_comparisons_retained=retained_comparisons,
                        partial_candidate_logging_gap=log.get("second_layer_comparisons", 0)-retained_comparisons)
        result["completed_comparisons"] += complete
        result["logged_changed"] += status == "changed"
        result["macro_identity_changed"] += changed
        result["mechanism_active"] += active
        result["tail_evaluations_started"] += log.get("tail_evaluations", 0)
        result["completed_tails_retained_in_nodes"] += retained_tails
        result["tail_completion_not_identifiable_from_retained_nodes"] += missing
        result["generated_actions"] += log.get("generated_actions", 0)
        result["planning_runtime_s"] += log.get("planning_runtime_s", 0)
        if status == "fallback":
            reasons[log.get("reason", "missing_reason")] += 1
        result["decisions"].append(decision)
    result["fallback_reasons"] = dict(reasons)
    result["unlogged_skips"] = "Not observable: disabled, max_searches, fewer than two candidates and no remaining planning time can return before log creation"
    result["tail_count_scope"] = "Started != completed on fallback; partial current candidate nodes are not retained by strategy"
    return result


def divergence(base, candidate, diagnosis, base_diagnosis=None):
    bh, ch = base["summary"]["action_history"], candidate["summary"]["action_history"]
    index = next((i for i, (a,b) in enumerate(zip(bh,ch)) if event(a) != event(b)), min(len(bh),len(ch)))
    identical = len(bh) == len(ch) == index
    changed = [d for d in diagnosis["decisions"] if d["macro_identity_changed"] and d["action_boundary"] <= index]
    cause = changed[0] if changed else None
    shared = (all(event(a) == event(b) for a,b in zip(bh[:cause["action_boundary"]],ch[:cause["action_boundary"]]))
              if cause else None)
    other_changes = [d for d in (base_diagnosis or {}).get("decisions", [])
                     if d["macro_identity_changed"] and d["action_boundary"] <= index]
    other_cause = other_changes[0] if other_changes else None
    return {"physical_history_identical": identical, "first_difference_index_0based": None if identical else index,
            "baseline_event": event(bh[index]) if not identical and index < len(bh) else None,
            "candidate_event": event(ch[index]) if not identical and index < len(ch) else None,
            "first_logged_macro_change_boundary": cause["action_boundary"] if cause else None,
            "shared_prefix_at_first_macro_change": shared,
            "first_logged_baseline_task": cause["baseline"] if cause else None,
            "first_logged_selected_task": cause["selected"] if cause else None,
            "comparator_first_logged_macro_change": ({k:other_cause[k] for k in
                ("action_boundary","baseline","selected")} if other_cause else None),
            "attribution": ("identical actions" if identical else
                "first policy choice changed after a shared physical prefix; downstream time is a whole-episode observation, not a local causal saving proof"
                if cause and shared else "comparator also/alone changed its task; see both logged choices, cannot attribute pair difference solely to candidate"
                if other_cause else "not explained by a shared-prefix logged macro change; inspect real deadline/fallback/protocol")}


def build(batches, audits):
    bounds, audit_inputs = {}, []
    for path in audits:
        data = read(path)
        if data.get("version") != "q3-round2-posthoc-original-physical-v1":
            raise ValueError("requires original physical-LB auditor output")
        audit_inputs.append({"path":str(path.resolve()),"sha256":sha(path)})
        for row in data["rows"]:
            key = (row["batch"],row["strategy"],row["seed"],row["input_sha256"])
            if key in bounds and bounds[key] != row:
                raise ValueError("conflicting physical audit rows")
            bounds[key] = row
    rows, pairs, batch_evidence, comparisons = [], [], [], {}
    seen = set()
    for batch in batches:
        batch = batch.resolve(strict=True)
        manifest, summary, runner, _ = verify_batch(batch)
        if runner != "verified" or set(manifest["policies"]) != {"baseline","root_mc","observation_tree"}:
            raise ValueError("complete frozen three-arm experiment with archived runner required")
        label = f"{manifest['trial']}/{manifest['stage']}"
        if label in seen:
            raise ValueError("duplicate batch label")
        seen.add(label)
        batch_evidence.append({"batch":label,"path":str(batch),"manifest_sha256":sha(batch/"manifest.json"),
            "summary_sha256":sha(batch/"summary.json"),"runner_sha256":manifest["runner_sha256"],
            "all_source_archives_and_files_verified":True})
        comparisons[label] = {"versus_baseline":summary["comparisons"],
                              "tree_vs_root_mc":summary.get("observation_tree_vs_root_mc")}
        summarized = {(r["strategy"],r["seed"]):r for r in summary["rows"]}
        if len(summarized) != len(summary["records_sha256"]):
            raise ValueError("duplicate/incomplete summary rows")
        for seed in manifest["seeds"]:
            records, diagnoses = {}, {}
            for policy in ("baseline","root_mc","observation_tree"):
                relative = f"records/{policy}-{seed}.json.gz"
                path, expected = batch/relative, summary["records_sha256"][relative]
                if sha(path) != expected:
                    raise ValueError("record hash differs from finished summary")
                with gzip.open(path,"rt",encoding="utf-8") as stream:
                    record = json.load(stream)
                raw = record["row"]
                if (raw != summarized[(policy,seed)] or record["frozen_manifest_sha256"] != digest(manifest)
                        or record["spec"] != manifest["policies"][policy]["spec"]
                        or record.get("evaluation_phase") != "after_policy_termination"):
                    raise ValueError("record identity differs from frozen protocol")
                checked_history(record)
                bound = bounds.get((label,policy,seed,expected))
                if not bound or bound["case_sha256"] != raw["case_sha256"] or bound["virtual_time_s"] != raw["virtual_time_s"]:
                    raise ValueError("missing exact-record original-LB audit")
                diagnosis = planning_diagnosis(record["summary"])
                rows.append({"batch":label,"strategy":policy,"seed":seed,"case_sha256":raw["case_sha256"],
                    "input_sha256":expected,"successful":raw["successful"],"failed_clear_count":raw["failed_clear_count"],
                    "virtual_time_s":raw["virtual_time_s"],"program_cpu_s":raw.get("program_cpu_s"),
                    "program_runtime_s":raw["program_runtime_s"],"actual_action_count":raw["action_count"],
                    "actual_measurement_count":raw["measurement_count"],
                    "time_breakdown":{key:raw[key] for key in PHASE_COSTS},
                    "physical_lower_s":bound["physical_lower_s"],"time_over_lower":bound["time_over_physical_lower"],
                    "physical_audit_passed":bound["audit_passed"],"ratio_eligible":bound["ratio_eligible"],"planning":diagnosis})
                records[policy], diagnoses[policy] = record, diagnosis
            if len({r["row"]["case_sha256"] for r in records.values()}) != 1:
                raise ValueError("three arms do not share the same case")
            for left,right in (("baseline","root_mc"),("baseline","observation_tree"),("root_mc","observation_tree")):
                a,b = records[left]["row"],records[right]["row"]
                pairs.append({"batch":label,"seed":seed,"baseline":left,"candidate":right,
                    "saving_s":a["virtual_time_s"]-b["virtual_time_s"],
                    "time_breakdown_saving":{k:a[k]-b[k] for k in PHASE_COSTS},
                    "divergence":divergence(records[left],records[right],diagnoses[right],diagnoses[left])})
    return assemble(rows,pairs,batch_evidence,audit_inputs,comparisons)


def assemble(rows,pairs,batches,audits,comparisons):
    aggregates = {}
    for policy in sorted({r["strategy"] for r in rows}):
        group = [r for r in rows if r["strategy"] == policy]
        plans = [r["planning"] for r in group]
        reasons = Counter()
        for plan in plans:
            reasons.update(plan["fallback_reasons"])
        counts = ("planning_attempts","completed_comparisons","logged_changed","macro_identity_changed",
                  "mechanism_active","tail_evaluations_started","completed_tails_retained_in_nodes",
                  "tail_completion_not_identifiable_from_retained_nodes","generated_actions","planning_runtime_s",
                  "first_actions_with_multiple_effective_z","first_actions_with_observation_dependent_actions")
        aggregates[policy] = {"runs":len(group),"successful":sum(r["successful"] for r in group),
            "mean_virtual_s":statistics.fmean(r["virtual_time_s"] for r in group),
            "maximum_virtual_s":max(r["virtual_time_s"] for r in group),
            "mean_cpu_s":distribution([r["program_cpu_s"] for r in group])["mean"],
            "mean_wall_s":statistics.fmean(r["program_runtime_s"] for r in group),
            "mean_case_time_over_lower":distribution([r["time_over_lower"] for r in group])["mean"],
            "failed_clears":sum(r["failed_clear_count"] for r in group),
            **{key:sum(p[key] for p in plans) for key in counts},"fallback_reasons":dict(reasons)}
    return {"version":"round2-observation-tree-diagnosis-v1","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "script_sha256":sha(__file__),"batch_verifier_sha256":sha(Path(__file__).with_name("round2_posthoc_audit.py")),
        "sqlite_read":False,"network_access":False,"new_simulations":0,"batches":batches,
        "physical_audit_inputs":audits,"aggregates":aggregates,"rows":rows,"pairs":pairs,
        "frozen_batch_comparisons":comparisons,
        "all_log_checks_passed":all(not r["planning"]["errors"] for r in rows),
        "scope":"Mechanism activation is not evidence of improvement; model pools are not evaluated sample sizes"}


def smoke(paths):
    records = []
    for path in paths:
        with gzip.open(path,"rt",encoding="utf-8") as stream:
            data = json.load(stream)
        checked_history(data)
        records.append({"path":str(path.resolve()),"sha256":sha(path),"scope":data.get("scope"),
                        "source_unchanged_claim":data.get("source_unchanged_during_smoke"),
                        "planning":planning_diagnosis(data["summary"])})
    return {"version":"round2-observation-tree-smoke-parser-only","script_sha256":sha(__file__),
            "scope":"Parsing check only; unpaired/unfrozen development records are not independent performance evidence",
            "records":records,"all_log_checks_passed":all(not r["planning"]["errors"] for r in records)}


def markdown(result):
    if "aggregates" not in result:
        return "# 观测树日志解析烟测\n\n仅解析已完成开发记录，不作性能判断或独立确认。\n\n" + "\n".join(
            f"- {Path(r['path']).name}：尝试{r['planning']['planning_attempts']}，完成{r['planning']['completed_comparisons']}，"
            f"真实宏动作选择改变{r['planning']['macro_identity_changed']}，机制有效{r['planning']['mechanism_active']}，"
            f"核对错误={r['planning']['errors']}。" for r in result["records"]) + "\n"
    lines = ["# 三组观测树实验：机制、开销与真实分歧", "",
        "本报告只读已完成三组配对归档。manifest、runner、source zip及逐源文件、全部records哈希已核对；下界由指定独立旧口径审计提供，逐记录哈希绑定。",
        "机制激活不等于更快。池内数百个位置ESS、抽出的16个等权世界、实际用于选动作的2个世界和复核的2个世界是不同数量。",
        "tail_evaluations是开始次数；fallback时正在处理的候选节点可能没写入日志，未归档差额不能认定全部成功完成。程序开销与虚拟时间分别报告。", "",
        "|策略|完成/局|尝试/完成比较/改变|机制有效|tail开始/留存完成|假想动作|规划秒|平均虚拟秒|均T/LB|平均CPU秒|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def number(value):
        return "不适用" if value is None else f"{value:.6f}"
    for policy,g in result["aggregates"].items():
        lines.append(f"|{policy}|{g['successful']}/{g['runs']}|{g['planning_attempts']}/{g['completed_comparisons']}/{g['macro_identity_changed']}|"
            f"{g['mechanism_active']}|{g['tail_evaluations_started']}/{g['completed_tails_retained_in_nodes']}|{g['generated_actions']}|"
            f"{g['planning_runtime_s']:.3f}|{g['mean_virtual_s']:.6f}|{number(g['mean_case_time_over_lower'])}|{number(g['mean_cpu_s'])}|")
        lines += ["", f"{policy} fallback原因：`{json.dumps(g['fallback_reasons'],ensure_ascii=False)}`。", ""]
    lines += ["下界沿用先知物理松弛LB；T/LB不是相对可达在线最优策略的近似比。正节省表示右侧策略较快，不能由第一次局部选择推断整局节省。", "",
        "|批/案例|策略|虚拟秒|旧LB秒|T/LB|首次真实分歧（0起）|较基准节省秒|移动节省秒|检测节省秒|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["rows"]:
        pair = next((p for p in result["pairs"] if p["batch"]==row["batch"] and p["seed"]==row["seed"]
                     and p["baseline"]=="baseline" and p["candidate"]==row["strategy"]),None)
        diff = pair["divergence"]["first_difference_index_0based"] if pair else None
        lines.append(f"|{row['batch']}/{row['seed']}|{row['strategy']}|{row['virtual_time_s']:.6f}|{number(row['physical_lower_s'])}|"
            f"{number(row['time_over_lower'])}|{diff if diff is not None else '相同/基准'}|"
            f"{number(pair['saving_s'] if pair else None)}|{number(pair['time_breakdown_saving']['movement_s'] if pair else None)}|"
            f"{number(pair['time_breakdown_saving']['detection_s'] if pair else None)}|")
    lines += ["", "逐first-action的有效z、条件动作种类、实际train/eval支持与池ESS、每局完整时间分解、首次改变的基准/新任务及物理动作均见JSON。",
              "未写日志的跳过（开关、尝试上限、候选不足或时间不足）不可从记录反推次数。", "",
              "独立旧LB审计输入：" + "; ".join(f"{a['path']}（SHA256 {a['sha256']}）" for a in result["physical_audit_inputs"]), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--batches", nargs="+", type=Path)
    mode.add_argument("--smoke", nargs="+", type=Path)
    parser.add_argument("--audit", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.batches and not args.audit:
        parser.error("finished batch diagnosis requires --audit for the exact old-LB audit")
    if args.output.resolve() == args.report.resolve() or args.output.exists() or args.report.exists():
        parser.error("output/report must be distinct new paths; overwrite is forbidden")
    result = build(args.batches,args.audit) if args.batches else smoke(args.smoke)
    for path,content in ((args.output,json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"),
                         (args.report,markdown(result))):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("x",encoding="utf-8") as stream:
            stream.write(content)
    print(json.dumps({"all_log_checks_passed":result["all_log_checks_passed"],"output":str(args.output)},ensure_ascii=False))
    if not result["all_log_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
