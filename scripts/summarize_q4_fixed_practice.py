"""Validate and report one fixed-size R12 practice batch without simulator access."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import time

from summarize_q4_source_count_practice import ENTRY_SHA, checked_row, read, save, sha


def read_snapshot(path):
    # The collecting runner may be replacing its small summary while we refresh.
    for attempt in range(5):
        try:
            return read(path)
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(0.05)


def summarize(rows):
    actual = [r["actual_time_s"] for r in rows]
    per_source = [r["time_per_source_s"] for r in rows]
    bounds = [r["lower_bound_s"] for r in rows]
    bound_per_source = [r["lower_bound_per_source_s"] for r in rows]
    n_total = sum(r["source_count"] for r in rows)
    return {
        "runs": len(rows), "source_count_sum": n_total,
        "mean_actual_time_s": statistics.mean(actual) if rows else None,
        "mean_time_per_source_s": statistics.mean(per_source) if rows else None,
        "min_time_per_source_s": min(per_source) if rows else None,
        "max_time_per_source_s": max(per_source) if rows else None,
        "mean_lower_bound_per_source_s": statistics.mean(bound_per_source) if rows else None,
        "sum_time_over_sum_lower_bound": sum(actual) / sum(bounds) if rows else None,
        "actual_time_sum_s": sum(actual), "lower_bound_sum_s": sum(bounds),
        "weighted_time_per_source_s": sum(actual) / n_total if rows else None,
        "weighted_lower_bound_per_source_s": sum(bounds) / n_total if rows else None,
        "min_actual_time_s": min(actual) if rows else None,
        "max_actual_time_s": max(actual) if rows else None,
        "sample_std_actual_time_s": statistics.stdev(actual) if len(rows) > 1 else None,
        "sample_std_time_per_source_s": statistics.stdev(per_source) if len(rows) > 1 else None,
    }


def markdown(value):
    def fmt(number, digits=2):
        return "—" if number is None else f"{number:.{digits}f}"
    result = value["overall"]
    label = "已完成" if value["completed"] else "尚未完成"
    report = ["# 问题 4：独立固定局数演练", "",
        f"{label}：已验证 {len(value['rows'])}/{value['expected_runs']} 局；批次状态 `{value['batch_status']}`。本表仅使用这批新演练。", "",
        "|干扰源数|局数|平均整局用时（秒）|平均每源用时（秒）|每源最好～最坏（秒）|平均下界/源（秒）|用时÷下界|",
        "|---|---:|---:|---:|---:|---:|---:|",
        "|" + "|".join(["混合 10～16", str(result["runs"]), fmt(result["mean_actual_time_s"]),
            fmt(result["mean_time_per_source_s"]), fmt(result["min_time_per_source_s"]) + "～" + fmt(result["max_time_per_source_s"]),
            fmt(result["mean_lower_bound_per_source_s"]), fmt(result["sum_time_over_sum_lower_bound"], 3)]) + "|", "",
        "整局用时 T 包括从固定原点出发后的移动、探测、定位和清除。表中每源均值按各局等权计算：mean(T/N)；每源下界均值为 mean(LB/N)。最好与最坏为各局 T/N 的样本最小值和最大值。用时÷下界使用 ΣT/ΣLB，未取逐局比值的平均。", "",
        "下界沿用历史条件性全清理论下界。", "",
        f"另按总干扰源数加权：ΣT/ΣN = {fmt(result['weighted_time_per_source_s'])} 秒/源，ΣLB/ΣN = {fmt(result['weighted_lower_bound_per_source_s'])} 秒/源；总计 {result['source_count_sum']} 个源。", "",
        f"整局样本范围 {fmt(result['min_actual_time_s'])}～{fmt(result['max_actual_time_s'])} 秒；整局样本标准差 {fmt(result['sample_std_actual_time_s'])} 秒，每源样本标准差 {fmt(result['sample_std_time_per_source_s'])} 秒。", ""]
    if value["unreported_case_directories"]:
        report += [f"另有 {len(value['unreported_case_directories'])} 个场景目录尚未列入批次摘要，原始记录保留，不计入已验证局均值。", ""]
    if value["failures"]:
        report += [f"存在 {len(value['failures'])} 项失败或证据不完整记录，详见 summary.json；本批次未标记完成。", ""]
    per_run = ["# 问题 4：本批次逐局记录", "",
        "|局号|干扰源数 N|整局用时 T（秒）|理论下界 LB（秒）|T/N（秒）|LB/N（秒）|T/LB|测量数|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in value["rows"]:
        cells = [str(row["run"]), str(row["source_count"])]
        cells += [fmt(row[key]) for key in ("actual_time_s", "lower_bound_s", "time_per_source_s", "lower_bound_per_source_s")]
        cells += [fmt(row["time_over_lower_bound"], 3), str(row["measurement_count"])]
        per_run.append("|" + "|".join(cells) + "|")
    return "\n".join(report), "\n".join(per_run) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.expected_runs <= 200:
        raise ValueError("Invalid expected run count")
    batch, output = args.batch.resolve(strict=True), args.output.resolve()
    if output == batch or output.is_relative_to(batch):
        raise ValueError("Reports must use a separate output directory")
    protocol = read(batch / "protocol.json")
    if (protocol.get("problem") != 4 or protocol.get("mode") != "practice"
            or protocol.get("practice_entry_sha256") != ENTRY_SHA
            or protocol.get("strategy") != "compact_joint_continuation"
            or protocol.get("config") != "after_active_miss_optical"):
        raise ValueError("Batch is not the qualified R12 practice configuration")
    if (type(protocol.get("fixed_runs")) is not int or protocol["fixed_runs"] != args.expected_runs
            or type(protocol.get("maximum_runs")) is not int or protocol["maximum_runs"] != args.expected_runs
            or "minimum_per_source_count" not in protocol or protocol["minimum_per_source_count"] is not None):
        raise ValueError("Batch protocol must request exactly the expected fixed count without source-count stopping")
    snapshot = read_snapshot(batch / "summary.json") if (batch / "summary.json").exists() else {"rows": [], "status": "starting", "protocol": protocol}
    if snapshot.get("protocol") != protocol:
        raise ValueError("Batch protocol changed")
    rows, failures, seen = [], [], set()
    for descriptor in snapshot["rows"]:
        index = descriptor.get("run") if isinstance(descriptor, dict) else None
        if type(index) is not int or index < 1:
            failures.append({"kind": "invalid_run_descriptor"})
            continue
        if index in seen:
            failures.append({"run": index, "kind": "duplicate_run_descriptor"})
            continue
        seen.add(index)
        try:
            rows.append(checked_row(batch / f"case-{index:03d}", index, "fixed_batch", descriptor))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            failures.append({"run": index, "kind": "evidence_validation_failed", "error_type": type(exc).__name__})
    if seen != set(range(1, len(seen) + 1)):
        failures.append({"kind": "nonconsecutive_run_indices"})
    if len(seen) > args.expected_runs:
        failures.append({"kind": "more_than_expected_runs", "observed": len(seen)})
    pending = []
    for path in batch.glob("case-*"):
        if not path.is_dir():
            continue
        if not re.fullmatch(r"case-\d{3}", path.name):
            failures.append({"kind": "unexpected_case_directory"})
        elif int(path.name[5:]) not in seen:
            pending.append(path.name)
    pending.sort()
    if (batch / "failure.json").exists():
        failure = read_snapshot(batch / "failure.json")
        item = {"kind": "batch_failure", "evidence_sha256": sha(batch / "failure.json")}
        for key in ("run", "returncode"):
            if type(failure.get(key)) is int:
                item[key] = failure[key]
        failures.append(item)
    for _ in snapshot.get("failures", []):
        failures.append({"kind": "batch_summary_failure"})
    known_status = {"starting", "running", "fixed_runs_completed", "practice_entry_failed", "stop_requested", "maximum_runs", "error_stopped"}
    batch_status = snapshot.get("status") if snapshot.get("status") in known_status else "unknown"
    if batch_status in ("practice_entry_failed", "error_stopped") and not failures:
        failures.append({"kind": "batch_failed_without_detail"})
    completed = (len(rows) == args.expected_runs and seen == set(range(1, args.expected_runs + 1))
                 and batch_status == "fixed_runs_completed" and not pending and not failures)
    value = {"problem": 4, "mode": "practice", "strategy": "compact_joint_continuation",
        "config": "after_active_miss_optical", "practice_entry_sha256": ENTRY_SHA,
        "protocol_sha256": sha(batch / "protocol.json"), "expected_runs": args.expected_runs,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(), "batch_status": batch_status,
        "completed": completed, "unreported_case_directories": pending, "failures": failures,
        "definitions": {"mean_time_per_source_s": "mean(T/N), equal weight per run",
            "mean_lower_bound_per_source_s": "mean(LB/N), equal weight per run",
            "sum_time_over_sum_lower_bound": "sum(T)/sum(LB)",
            "weighted_time_per_source_s": "sum(T)/sum(N)",
            "weighted_lower_bound_per_source_s": "sum(LB)/sum(N)",
            "lower_bound_kind": "historical_conditional_all_clear_containing_region_bound"},
        "rows": rows, "overall": summarize(rows)}
    output.mkdir(parents=True, exist_ok=True)
    save(output / "summary.json", value)
    report, per_run = markdown(value)
    for name, content in (("REPORT.md", report), ("PER_RUN.md", per_run)):
        pending_path = output / (name + ".tmp")
        pending_path.write_text(content, encoding="utf-8")
        pending_path.replace(output / name)
    print(json.dumps({"validated_runs": len(rows), "expected_runs": args.expected_runs, "completed": completed,
        "pending_cases": len(pending), "failures": len(failures), "batch_status": batch_status}, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
