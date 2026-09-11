"""Read-only evidence validation and reporting for two frozen R12 practice batches."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ENTRY_SHA = "8c94a435fecfdafe3a60e8952c26a67b54e6c2138610d721a2eb51c92230d849"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def same(a, b):
    return (type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b) and abs(a - b) < 0.00001)


def checked_row(case, index, tag, expected=None):
    run = case / "run-001"
    summary_path, lower_path = run / "summary.json", run / "lower_bounds.json"
    summary, lower, registered = read(summary_path), read(lower_path), read(case / "result-001.json")
    n, actual = lower["official_source_total"], lower["actual_virtual_time_s"]
    bound = lower["conditional_guaranteed_all_clear_lower_s"]
    if (type(n) is not int or n not in range(10, 17)
            or not all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in (actual, bound))
            or registered.get("completed") is not True or registered.get("official_all_clear_verified") is not True
            or lower.get("official_all_clear_verified") is not True
            or registered.get("source_total") != n or registered.get("cleared_count") != n
            or lower.get("observed_cleared_sources") != n
            or summary.get("problem") != 4 or summary.get("declared_mode") != "practice"
            or summary.get("data_origin") != "simulator_http_session"
            or summary.get("method_metadata", {}).get("practice_entry_sha256") != ENTRY_SHA
            or not same(actual, registered.get("virtual_time_s"))
            or not same(actual, summary.get("search", {}).get("virtual_time_s"))
            or not same(bound, registered.get("conditional_guaranteed_all_clear_lower_s"))
            or not same(actual / bound, lower.get("time_to_conditional_lower_bound_ratio"))
            or not same(actual / bound, registered.get("time_to_conditional_lower_bound_ratio"))):
        raise ValueError(f"Invalid completed practice evidence: {tag}/{case.name}")
    summary_sha, lower_sha = sha(summary_path), sha(lower_path)
    if (lower.get("summary_sha256") != summary_sha
            or registered.get("lower_bounds_sha256") != lower_sha
            or lower.get("registration_sha256") != sha(run / "registration.json")):
        raise ValueError(f"Evidence hash mismatch: {tag}/{case.name}")
    if expected is not None and (
            expected.get("summary_sha256") != summary_sha or expected.get("lower_bounds_sha256") != lower_sha
            or expected.get("source_count") != n or expected.get("all_clear") is not True
            or not same(expected.get("actual_time_s"), actual) or not same(expected.get("lower_bound_s"), bound)
            or not same(expected.get("time_over_lower_bound"), actual / bound)):
        raise ValueError(f"Baseline aggregate differs from original evidence: {case.name}")
    search = summary["search"]
    return {"batch": tag, "run": index, "case_directory": case.name, "source_count": n,
            "actual_time_s": actual, "lower_bound_s": bound, "time_per_source_s": actual / n,
            "lower_bound_per_source_s": bound / n, "time_over_lower_bound": actual / bound,
            "all_clear": True, "measurement_count": search.get("measurement_count"),
            "accepted_actions": search.get("accepted_actions"), "clear_attempt_count": search.get("clear_attempt_count"),
            "program_wall_time_s": summary["program_wall_time_s"], "time_breakdown": search.get("time_breakdown"),
            "summary_sha256": summary_sha, "lower_bounds_sha256": lower_sha}


def baseline_rows(baseline):
    protocol, snapshot = read(baseline / "protocol.json"), read(baseline / "summary.json")
    if (protocol.get("problem") != 4 or protocol.get("mode") != "practice"
            or protocol.get("practice_entry_sha256") != ENTRY_SHA or snapshot.get("protocol") != protocol):
        raise ValueError("Baseline protocol is not the frozen R12 practice batch")
    old = snapshot["rows"]
    indices = [r["run"] for r in old]
    if (not indices or any(type(i) is not int or i < 1 for i in indices)
            or len(set(indices)) != len(indices)
            or {p.name for p in baseline.glob("case-*") if p.is_dir()} != {f"case-{i:03d}" for i in indices}):
        raise ValueError("Baseline has duplicate, missing, or unaccounted-for cases")
    return [checked_row(baseline / f"case-{r['run']:03d}", r["run"], "baseline", r) for r in old]


def aggregate(rows):
    groups = []
    for n in range(10, 17):
        selected = [r for r in rows if r["source_count"] == n]
        group = {"source_count": n, "runs": len(selected),
                 "baseline_runs": sum(r["batch"] == "baseline" for r in selected),
                 "new_runs": sum(r["batch"] == "new" for r in selected)}
        if selected:
            for key in ("actual_time_s", "time_per_source_s", "lower_bound_s", "lower_bound_per_source_s", "time_over_lower_bound"):
                values = [r[key] for r in selected]
                group.update({"mean_" + key: statistics.mean(values), "min_" + key: min(values),
                              "max_" + key: max(values), "sample_std_" + key: statistics.stdev(values) if len(values) > 1 else None})
            group["sum_time_over_sum_lower_bound"] = sum(r["actual_time_s"] for r in selected) / sum(r["lower_bound_s"] for r in selected)
        groups.append(group)
    return groups


def markdown(snapshot):
    def fmt(value):
        return "—" if value is None else f"{value:.2f}"
    lines = ["# 问题 4 模拟器演练：按干扰源数量统计", "",
             f"状态：`{snapshot['status']}`；完成 {len(snapshot['rows'])} 局。", "",
             "每源秒数 = 从原点出发至全部清除的模拟器计费用时 / 干扰源总数，包含移动、探测与清除；不是单次测量操作时间。",
             "理论下界沿用历史条件性全清下界；比值为组内总用时 / 总下界。最小与最大是本次样本观察范围，标准差使用 n−1。所有抽取场景均保留，数量只在演练结束后读取。", "",
             "|源数|局数（旧+新）|平均整局秒|平均秒/源|每源最好～最坏|平均下界秒/源|用时/下界|",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for g in snapshot["groups"]:
        count = f"{g['runs']}（{g['baseline_runs']}+{g['new_runs']}）"
        cells = [str(g["source_count"]), count, fmt(g.get("mean_actual_time_s")),
                 fmt(g.get("mean_time_per_source_s")), fmt(g.get("min_time_per_source_s")) + "～" + fmt(g.get("max_time_per_source_s")),
                 fmt(g.get("mean_lower_bound_per_source_s")), fmt(g.get("sum_time_over_sum_lower_bound"))]
        lines.append("|" + "|".join(cells) + "|")
    lines += ["", "## 整局范围和标准差", "", "|源数|整局最好～最坏秒|整局标准差秒|每源标准差秒|",
              "|---:|---:|---:|---:|"]
    for g in snapshot["groups"]:
        cells = [str(g["source_count"]), fmt(g.get("min_actual_time_s")) + "～" + fmt(g.get("max_actual_time_s")),
                 fmt(g.get("sample_std_actual_time_s")), fmt(g.get("sample_std_time_per_source_s"))]
        lines.append("|" + "|".join(cells) + "|")
    lines += ["", "## 逐局记录", "", "|批次|局号|源数|总秒|下界秒|秒/源|下界秒/源|用时/下界|测量数|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in snapshot["rows"]:
        cells = ["旧" if r["batch"] == "baseline" else "新", str(r["run"]), str(r["source_count"])]
        cells += [fmt(r[key]) for key in ("actual_time_s", "lower_bound_s", "time_per_source_s", "lower_bound_per_source_s", "time_over_lower_bound")]
        cells += [str(r["measurement_count"])]
        lines.append("|" + "|".join(cells) + "|")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--new-batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-per-count", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.minimum_per_count <= 10:
        raise ValueError("Invalid minimum sample count")
    baseline, new_batch = args.baseline.resolve(strict=True), args.new_batch.resolve(strict=True)
    rows = baseline_rows(baseline)
    protocol = read(new_batch / "protocol.json")
    if (protocol.get("problem") != 4 or protocol.get("mode") != "practice"
            or protocol.get("practice_entry_sha256") != ENTRY_SHA
            or protocol.get("strategy") != "compact_joint_continuation"
            or protocol.get("config") != "after_active_miss_optical"):
        raise ValueError("New batch is not the same qualified R12 practice configuration")
    if (new_batch / "summary.json").exists():
        current = read(new_batch / "summary.json")
        if current.get("protocol") != protocol:
            raise ValueError("New batch protocol changed")
    else:
        current = {"rows": [], "status": "starting"}
    indices = [r["run"] for r in current["rows"]]
    if (any(type(i) is not int or i < 1 for i in indices) or len(indices) != len(set(indices))):
        raise ValueError("New batch contains invalid or duplicated cases")
    for expected in current["rows"]:
        index = expected["run"]
        rows.append(checked_row(new_batch / f"case-{index:03d}", index, "new", expected))
    reported_cases = {f"case-{i:03d}" for i in indices}
    pending = sorted(p.name for p in new_batch.glob("case-*") if p.is_dir() and p.name not in reported_cases)
    groups = aggregate(rows)
    total_time, total_bound = sum(r["actual_time_s"] for r in rows), sum(r["lower_bound_s"] for r in rows)
    total_sources = sum(r["source_count"] for r in rows)
    value = {
        "protocol": {"problem": 4, "mode": "practice", "strategy": "compact_joint_continuation",
            "config": "after_active_miss_optical", "practice_entry_sha256": ENTRY_SHA,
            "minimum_per_source_count": args.minimum_per_count,
            "baseline_protocol_sha256": sha(baseline / "protocol.json"),
            "baseline_summary_sha256": sha(baseline / "summary.json"),
            "new_protocol_sha256": sha(new_batch / "protocol.json"),
            "lower_bound_kind": "historical_conditional_all_clear_containing_region_bound",
            "primary_ratio": "sum(actual_time_s)/sum(lower_bound_s)",
            "per_source_time_definition": "Full billed virtual time divided by official terminal source count.",
            "retention_rule": "All completed cases in both batch snapshots are included; unreported case directories are listed separately."},
        "status": ("source_counts_covered" if current.get("status") in {"stop_requested", "source_counts_covered"}
                   and all(g["runs"] >= args.minimum_per_count for g in groups) and not pending
                   and not (new_batch / "failure.json").exists() else current.get("status", "unknown")),
        "source_batch_status": current.get("status", "unknown"),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "groups_meet_minimum": all(g["runs"] >= args.minimum_per_count for g in groups),
        "unreported_new_case_directories": pending,
        "new_batch_failure_present": (new_batch / "failure.json").exists(),
        "rows": rows, "groups": groups,
        "overall": {"runs": len(rows), "source_count_sum": total_sources,
            "actual_time_sum_s": total_time, "lower_bound_sum_s": total_bound,
            "weighted_time_per_source_s": total_time / total_sources,
            "weighted_lower_bound_per_source_s": total_bound / total_sources,
            "sum_time_over_sum_lower_bound": total_time / total_bound,
            "mean_run_time_per_source_s": statistics.mean(r["time_per_source_s"] for r in rows)},
    }
    output = args.output.resolve()
    if output == baseline or output == new_batch or output.is_relative_to(baseline) or output.is_relative_to(new_batch):
        raise ValueError("Reports must use a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    save(output / "summary.json", value)
    report, per_run = markdown(value).split("## 逐局记录", 1)
    overall = value["overall"]
    report += (f"\n合计 {overall['runs']} 局、{total_sources} 个干扰源；按点加权平均 {overall['weighted_time_per_source_s']:.2f} 秒/源，"
               f"对应下界 {overall['weighted_lower_bound_per_source_s']:.2f} 秒/源，总用时/总下界 {overall['sum_time_over_sum_lower_bound']:.3f}。\n")
    if pending:
        report += f"\n另有 {len(pending)} 个新批次目录尚未进入完成摘要，保留在原目录，不计入完成局均值。\n"
    for name, content in (("REPORT.md", report), ("PER_RUN.md", "# 问题 4 演练逐局记录\n" + per_run)):
        temp = output / (name + ".tmp")
        temp.write_text(content, encoding="utf-8")
        temp.replace(output / name)
    print(json.dumps({"completed_runs": len(rows), "groups_meet_minimum": value["groups_meet_minimum"],
        "counts": {g["source_count"]: g["runs"] for g in groups}, "pending_cases": len(pending),
        "status": value["status"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
