"""Read finished paired archives; verify executable tail claims independently.

No SQLite, world reconstruction, simulator, policy import, or new evaluation.
Geometry/truth lower-bound checks belong to the separate posthoc auditor.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.round2_posthoc_audit import digest, read, sha, verify_batch


WALL_FIELDS = {"real_timestamp_ms", "remaining_real_duration_s"}


def canonical(value):
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items() if key not in WALL_FIELDS}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def micros(value):
    return round(value * 1_000_000)


def point(action):
    return [action["position"]["x"], action["position"]["y"]]


def visits(actions):
    if not all(action["action"] == "/clear" and action["response"].get("clear_result") == "success"
               and action["response"].get("accepted") is True for action in actions):
        return None
    return [[action["channel"], *point(action)] for action in actions]


def cost(start, route):
    """Independent physical arithmetic, no candidate planner imported."""
    total, previous = 0, start
    for _, x, y in route:
        total += round(math.hypot(x - previous[0], y - previous[1]) / 5 * 1_000_000) + 5_000_000
        previous = (x, y)
    return total


def checked_history(record):
    history = record["history"]
    if not history or history[0]["action"] != "/enter" or history[-1]["action"] != "/exit":
        raise ValueError("finished enter/exit archive required")
    if not all(action["response"].get("accepted") is True for action in history):
        raise ValueError("non-accepted historical action needs separate protocol diagnosis")
    physical = [action for action in history if action["action"] in {"/measure", "/clear"}]
    if len(history) != len(physical) + 2:
        raise ValueError("unexpected nonphysical actions in episode")
    logged = record["summary"]["action_history"]
    if len(physical) != len(logged):
        raise ValueError("policy action log is incomplete")
    for raw, action in zip(physical, logged):
        result_key = "clear_result" if raw["action"] == "/clear" else "measure_result"
        if (raw["action"].removeprefix("/") != action["action"] or point(raw) != action["position"]
                or raw["channel"] != action["channel"] or raw["response"][result_key] != action["result"]
                or raw["response"]["virtual_time_s"] != action["virtual_time_s"]):
            raise ValueError("policy log differs from physical feedback")
    if micros(history[-1]["response"]["virtual_time_s"]) != micros(record["row"]["virtual_time_s"]):
        raise ValueError("summary time differs from physical exit")
    return history, physical


def audit_pair(base, candidate):
    if (base["row"]["seed"], base["row"]["case_sha256"]) != (
            candidate["row"]["seed"], candidate["row"]["case_sha256"]):
        raise ValueError("paired cases differ")
    bh, bp = checked_history(base)
    ch, cp = checked_history(candidate)
    errors, examined = [], []
    logs = candidate["summary"]["strategy_parameters"]["certified_tail_log"]
    accepted = [entry for entry in logs if entry.get("accepted")]
    identical = canonical(bh) == canonical(ch)
    whole_gain = micros(base["row"]["virtual_time_s"]) - micros(candidate["row"]["virtual_time_s"])
    prior_invalidated = False
    for ordinal, entry in enumerate(accepted):
        local_errors = []
        n = entry["after_actual_action_count"]
        if type(n) is not int or not 0 <= n < len(cp):
            raise ValueError("accepted tail action boundary outside actual history")
        prefix_equal = canonical(bh[:n + 1]) == canonical(ch[:n + 1])
        if ordinal == 0 and not prefix_equal:
            local_errors.append("first accepted tail did not share the actual baseline prefix")
        start = point(cp[n - 1]) if n else [0., 0.]
        start_us = micros(cp[n - 1]["response"]["virtual_time_s"]) if n else 0
        if entry["position"] != start or entry["start_time_us"] != start_us:
            local_errors.append("tail start position/time mismatch")
        bv, cv = entry["baseline_visits"], entry["candidate_visits"]
        if (len(bv) != entry["source_count"] or len(cv) != len(bv)
                or sorted(v[0] for v in bv) != sorted(v[0] for v in cv)
                or len({v[0] for v in cv}) != len(cv)):
            local_errors.append("compared routes do not visit the same unique source set")
        baseline_calculated, candidate_calculated = cost(start, bv), cost(start, cv)
        if baseline_calculated != entry["baseline_cost_us"] or candidate_calculated != entry["candidate_cost_us"]:
            local_errors.append("logged compared cost differs from independent movement arithmetic")
        if entry["gain_us"] != baseline_calculated - candidate_calculated or entry["gain_us"] < 10000:
            local_errors.append("accepted full route did not meet declared gain threshold")
        baseline_suffix_matches = None
        if prefix_equal:
            baseline_suffix_matches = visits(bp[n:]) == bv
            if not baseline_suffix_matches:
                local_errors.append("predicted baseline visits differ from the complete actual baseline suffix")
            elif micros(base["row"]["virtual_time_s"]) - start_us != baseline_calculated:
                local_errors.append("predicted baseline cost differs from actual complete suffix")
        executed = entry.get("executed", [])
        actual_executed = cp[n:n + len(executed)]
        expected_execution = [[a["channel"], *point(a), a["response"]["virtual_time_s"]]
                              for a in actual_executed]
        if (executed != expected_execution or visits(actual_executed) is None
                or [v[:3] for v in executed] != cv[:len(executed)]):
            local_errors.append("executed log differs from committed physical clear prefix")
        completed = bool(entry.get("completed"))
        invalidated = bool(entry.get("cancelled") or entry.get("interrupted"))
        if completed:
            if invalidated or entry.get("dominance_status") != "complete_execution_matches_compared_route":
                local_errors.append("completed status conflicts with cancellation/interruption")
            if len(executed) != len(cv) or visits(cp[n:]) != cv:
                local_errors.append("completed candidate did not execute precisely the whole committed suffix")
            if entry.get("actual_cost_us") != candidate_calculated or micros(candidate["row"]["virtual_time_s"]) - start_us != candidate_calculated:
                local_errors.append("completed candidate cost not physically realized")
            if not prior_invalidated and prefix_equal and whole_gain != entry["gain_us"]:
                local_errors.append("whole episode saving does not equal committed full-tail saving")
        else:
            valid_failure_status = {"execution_preconditions_invalidated", "incomplete_execution_no_full_tail_claim"}
            if not invalidated or entry.get("dominance_status") not in valid_failure_status:
                local_errors.append("unfinished accepted tail falsely retains complete dominance")
        examined.append({"accepted_index": ordinal, "action_boundary": n,
                         "prefix_identical": prefix_equal,
                         "baseline_suffix_matches": baseline_suffix_matches,
                         "baseline_suffix_cost_us": baseline_calculated,
                         "candidate_suffix_cost_us": candidate_calculated,
                         "declared_gain_us": entry["gain_us"], "completed": completed,
                         "cancelled": entry.get("cancelled"), "interrupted": entry.get("interrupted"),
                         "source_count": len(cv), "executed_count": len(executed),
                         "baseline_channels": [v[0] for v in bv], "candidate_channels": [v[0] for v in cv],
                         "order_changed": [v[0] for v in bv] != [v[0] for v in cv],
                         "planning_runtime_s": entry.get("planning_runtime_s"),
                         "whole_episode_dominance_verified": bool(completed and prefix_equal and not prior_invalidated and not local_errors),
                         "errors": local_errors})
        errors.extend(f"tail {ordinal}: {error}" for error in local_errors)
        prior_invalidated |= invalidated or not completed
    if not accepted and not identical:
        errors.append("episode changed without any accepted tail")
    return {"seed": base["row"]["seed"], "case_sha256": base["row"]["case_sha256"],
            "baseline_time_s": base["row"]["virtual_time_s"],
            "candidate_time_s": candidate["row"]["virtual_time_s"],
            "whole_gain_s": whole_gain / 1e6, "physical_history_identical": identical,
            "baseline_success": base["row"]["successful"], "candidate_success": candidate["row"]["successful"],
            "baseline_failed_clears": base["row"]["failed_clear_count"],
            "candidate_failed_clears": candidate["row"]["failed_clear_count"],
            "baseline_cpu_s": base["row"].get("program_cpu_s"),
            "candidate_cpu_s": candidate["row"].get("program_cpu_s"),
            "accepted_tails": len(accepted), "tails": examined, "errors": errors,
            "audit_passed": not errors,
            "whole_episode_nonregression_verified": bool(not errors and (
                (not accepted and identical) or (not prior_invalidated and any(
                    tail["whole_episode_dominance_verified"] for tail in examined))))}


def build(batches, *, baseline, candidate, physical_audits=()):
    bounds, sources = {}, []
    for path in physical_audits:
        report = read(path)
        sources.append({"path": str(path), "sha256": sha(path)})
        for row in report["rows"]:
            if row["ratio_eligible"]:
                bounds[(row["case_sha256"], row["strategy"])] = row
    rows, evidence = [], []
    for batch in batches:
        batch = batch.resolve(strict=True)
        manifest, summary, runner, _ = verify_batch(batch)
        if set(manifest["policies"]) != {baseline, candidate}:
            raise ValueError("explicit baseline/candidate must cover the complete paired batch")
        evidence.append({"path": str(batch), "manifest_sha256": sha(batch / "manifest.json"),
                         "summary_sha256": sha(batch / "summary.json"), "runner_verified": runner == "verified"})
        summary_rows = {(r["strategy"], r["seed"]): r for r in summary["rows"]}
        for seed in manifest["seeds"]:
            records, hashes = [], {}
            for policy in (baseline, candidate):
                relative = f"records/{policy}-{seed}.json.gz"
                path = batch / relative
                hashes[relative] = sha(path)
                if hashes[relative] != summary["records_sha256"][relative]:
                    raise ValueError("record changed after batch completion")
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    record = json.load(stream)
                if (record["row"] != summary_rows[(policy, seed)]
                        or record["frozen_manifest_sha256"] != digest(manifest)
                        or record["spec"] != manifest["policies"][policy]["spec"]):
                    raise ValueError("record identity differs from frozen experiment")
                records.append(record)
            row = audit_pair(*records)
            row.update(batch=f"{manifest['trial']}/{manifest['stage']}", input_sha256=hashes)
            for label, policy in (("baseline", baseline), ("candidate", candidate)):
                bound = bounds.get((row["case_sha256"], policy))
                if bound:
                    if row[f"{label}_time_s"] != bound["virtual_time_s"]:
                        raise ValueError("physical audit refers to a different policy episode")
                    row[f"{label}_physical_lower_s"] = bound["physical_lower_s"]
                    row[f"{label}_time_over_lower"] = bound["time_over_physical_lower"]
            rows.append(row)
    return {"version": "round2-certified-tail-execution-diagnosis-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__),
            "batch_verifier_sha256": sha(Path(__file__).with_name("round2_posthoc_audit.py")),
            "batches": evidence, "physical_audit_inputs": sources,
            "sqlite_read": False, "simulator_calls": 0, "new_cases": 0,
            "geometry_scope": "Separate physical/causal auditor; this script verifies executable comparisons only",
            "pairs": len(rows), "all_audits_passed": all(r["audit_passed"] for r in rows),
            "all_whole_episode_nonregression_verified": all(r["whole_episode_nonregression_verified"] for r in rows),
            "wins": sum(r["whole_gain_s"] > 0 for r in rows),
            "ties": sum(r["whole_gain_s"] == 0 for r in rows),
            "losses": sum(r["whole_gain_s"] < 0 for r in rows),
            "mean_gain_s": statistics.mean(r["whole_gain_s"] for r in rows), "rows": rows}


def markdown(result):
    lines = ["# 持证清除尾段：完整实际轨迹独立核对", "",
             f"共 {result['pairs']} 对；全部执行核对通过：{result['all_audits_passed']}。"
             f"整局不退步核验：{result['all_whole_episode_nonregression_verified']}。",
             f"胜/平/负：{result['wins']}/{result['ties']}/{result['losses']}；平均节省 {result['mean_gain_s']:.6f} 秒。", "",
             "逐条比较真实动作与反馈，仅删除 real_timestamp_ms、remaining_real_duration_s。"
             "基准完整后缀直接取配对基准记录，不由候选自己的模型再次生成。"
             "接受前前缀、所有排队点、成功响应、每段费用、整局节省均须一致；未接受的局必须整条物理轨迹相同。", "",
             "取消/中断撤销整局支配；后来新尾段完成不能消除之前的前缀偏离。"
             "此处是事后执行核对，不证明全球最优；CPU另行统计。没有读取SQLite或新增仿真。", "",
             "下界列引用独立物理审计的旧先知下界；若未提供审计显示‘待独立物理审计’，不以尾段费用冒充整局下界。", "",
             "|批/案例|接受尾段|前缀及兑现通过|基准秒|候选秒|节省秒|基准T/LB|候选T/LB|",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in result["rows"]:
        ratio = lambda key: f"{row[key]:.6f}" if key in row else "待独立物理审计"
        lines.append(f"|{row['batch']}/{row['seed']}|{row['accepted_tails']}|{row['audit_passed']}|"
                     f"{row['baseline_time_s']:.6f}|{row['candidate_time_s']:.6f}|{row['whole_gain_s']:.6f}|"
                     f"{ratio('baseline_time_over_lower')}|{ratio('candidate_time_over_lower')}|")
    lines += ["", "## 实际发生替换的尾段", ""]
    for row in result["rows"]:
        for tail in row["tails"]:
            lines.append(f"- {row['batch']}/{row['seed']}：第 {tail['action_boundary']} 次物理动作后，"
                         f"{tail['source_count']} 源；原序列 {tail['baseline_channels']}，新序列 {tail['candidate_channels']}；"
                         f"原完整后缀 {tail['baseline_suffix_cost_us']/1e6:.6f} 秒，候选 {tail['candidate_suffix_cost_us']/1e6:.6f} 秒；"
                         f"完整兑现={tail['completed']}，前缀相同={tail['prefix_identical']}，"
                         f"原后缀逐点匹配={tail['baseline_suffix_matches']}，错误={tail['errors']}。")
    lines += ["", f"诊断脚本 SHA-256：`{result['script_sha256']}`。完整输入哈希及核对字段见同名JSON。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--candidate", default="certified_tail")
    parser.add_argument("--physical-audits", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.batches, baseline=args.baseline, candidate=args.candidate,
                   physical_audits=args.physical_audits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("pairs", "all_audits_passed", "all_whole_episode_nonregression_verified",
                                                  "wins", "ties", "losses", "mean_gain_s")}, ensure_ascii=False))
    if not result["all_audits_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
