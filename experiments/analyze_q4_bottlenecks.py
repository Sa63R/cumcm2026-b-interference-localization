"""Read archived Q4 triangular trajectories; no new scenarios or policy calls.

Only the geometry replay uses observations. Post-termination source truth is
read separately to label directional silence causes, never to generate actions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from localization import CandidateRegion


FEES = ("movement_s", "detection_s", "switching_s", "optical_s", "removal_s")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "mean": None, "median": None, "max": None}
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "max": max(values)}


def read_trace(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    report, evaluation = record["search"], record["evaluation"]
    if report["problem"] != 4 or report["variant"] != "triangular":
        raise ValueError("expected historical Q4 triangular search/evaluation schema")
    if not evaluation["all_cleared"] or report.get("error") or report.get("exit_error"):
        raise ValueError("this audit expects the completed successful historical archive")
    actions = report["action_history"]
    coverage_indices = [i for i, a in enumerate(actions) if a["phase"] == "coverage"]
    coverage_end = max(coverage_indices)
    end_time = actions[coverage_end]["virtual_time_s"]
    regions, positive_counts, direction_counts, first_ready, first_seen = {}, Counter(), Counter(), {}, {}
    ready_radius_at_end, positive_at_end, source_rows = {}, {}, {}
    phases = defaultdict(Counter)
    per_source = defaultdict(Counter)
    previous, old_channel, prior_time, station, previous_cover_position = (0., 0.), 1, 0., 0, None
    ledger_error, later_scans, count16_station = 0., 0, None
    # Archived truth is used only for these post-hoc categorical labels.
    truth = {s["channel"]: s for s in evaluation["ground_truth"]["sources"]}
    silence_reasons = Counter()
    for i, action in enumerate(actions):
        channel, position, phase = action["channel"], action["position"], action["phase"]
        fee, source_fee = phases[phase], per_source[channel]
        cost = {name: 0. for name in FEES}
        cost["movement_s"] = round(math.dist(previous, position)/5*1_000_000)/1_000_000
        fee["actions"] += 1
        if phase == "coverage" and position != previous_cover_position:
            station += 1
            previous_cover_position = position
        if action["action"] == "measure":
            cost["detection_s"] = 5.
            cost["switching_s"] = float(old_channel != channel)
            old_channel = channel
            fee[action["result"]] += 1
            if phase == "coverage" and channel in first_ready:
                later_scans += 1
            if action["result"] in ("direction", "near"):
                positive_counts[channel] += 1
                first_seen.setdefault(channel, {"index": i, "time_s": action["virtual_time_s"], "station": station})
            if action["result"] == "direction":
                direction_counts[channel] += 1
                regions.setdefault(channel, CandidateRegion()).observe(position, action["bearing_deg"])
                if not regions[channel].vertices:
                    raise ValueError(f"empty replay region {path.name}:{i}")
                circle = regions[channel].enclosing_disk()
                if circle.radius <= 19.9:
                    first_ready.setdefault(channel, {"index": i, "time_s": action["virtual_time_s"],
                        "station": station, "phase": phase, "kind": "enclosing_disk",
                        "positive_observations": positive_counts[channel], "direction_observations": direction_counts[channel],
                        "radius_m": circle.radius, "safe_clear_arrival_s": max(0.,
                            math.dist(position, circle.center)-(19.9-circle.radius))/5})
            elif action["result"] == "near":
                first_ready.setdefault(channel, {"index": i, "time_s": action["virtual_time_s"],
                    "station": station, "phase": phase, "kind": "near", "radius_m": None,
                    "positive_observations": positive_counts[channel], "direction_observations": direction_counts[channel],
                    "safe_clear_arrival_s": 0.})
            if phase == "active_localization":
                source_fee["active_measurements"] += 1
                source_fee["active_"+action["result"]] += 1
                if action["result"] == "no_signal":
                    source = truth[channel]
                    dx, dy = position[0]-source["x"], position[1]-source["y"]
                    distance = math.hypot(dx, dy)
                    outside_radius = distance > source["reception_radius_m"]
                    backwards = False
                    if source["orientation_deg"] is not None:
                        theta = math.radians(source["orientation_deg"])
                        backwards = math.cos(theta)*dx+math.sin(theta)*dy < -1e-12*max(1., distance)
                    reason = ("backside_and_out_of_radius" if backwards and outside_radius
                              else "directional_backside_only" if backwards
                              else "out_of_radius_only" if outside_radius else "unexplained")
                    silence_reasons[reason] += 1
                    source_fee[reason] += 1
        else:
            cost["optical_s"] = 3.
            cost["removal_s"] = 2. if action["result"] == "success" else 0.
            fee["failed_clear"] += action["result"] != "success"
            if phase == "guaranteed_clearance":
                source_fee["fallback_attempts"] += 1
                source_fee["fallback_failed"] += action["result"] != "success"
                source_fee["fallback_cost_s"] += sum(cost.values())
            if action["result"] == "success":
                source_rows[channel] = {"channel": channel, "clear_index": i,
                    "clear_time_s": action["virtual_time_s"], "clear_phase": phase,
                    "source_type_posthoc": "omni" if truth[channel]["orientation_deg"] is None else "directional"}
        for name, value in cost.items():
            fee[name] += value
            if phase != "coverage":
                source_fee[name] += value
        if count16_station is None and len(first_seen) == 16:
            count16_station = station
        ledger_error = max(ledger_error, abs(action["virtual_time_s"]-prior_time-sum(cost.values())))
        previous, prior_time = position, action["virtual_time_s"]
        if i == coverage_end:
            positive_at_end = dict(positive_counts)
            ready_radius_at_end = {c: r.enclosing_disk().radius for c, r in regions.items()}
    fee_totals = {name: sum(p[name] for p in phases.values()) for name in FEES}
    if max(abs(fee_totals[name]-evaluation["time_breakdown_s"][name]) for name in FEES) > 1e-4:
        raise ValueError(f"fee decomposition does not match original evaluation: {path.name}")
    for channel, region in regions.items():
        archived = report["source_estimates"][str(channel)]
        if (len(region.vertices) != len(archived["vertices"]) or
                any(math.dist(a, b) > 1e-7 for a, b in zip(region.vertices, archived["vertices"]))):
            raise ValueError(f"final replay polygon does not reproduce archive: {path.name}, channel {channel}")
    for channel, row in source_rows.items():
        ready = first_ready.get(channel)
        row.update(first_seen=first_seen.get(channel), first_ready=ready,
                   certified_at_coverage_end=bool(ready and ready["index"] <= coverage_end),
                   certified_strictly_before_coverage_end=bool(ready and ready["index"] < coverage_end),
                   coverage_positive_count=positive_at_end.get(channel, 0),
                   radius_at_coverage_end_m=ready_radius_at_end.get(channel),
                   actual_wait_ready_to_clear_s=(row["clear_time_s"]-ready["time_s"] if ready else None),
                   local_cost_after_coverage=dict(per_source[channel]))
    ready_count = sum(s["certified_at_coverage_end"] for s in source_rows.values())
    return {"case_id": evaluation["case_id"], "trace": path.name, "sha256": digest(path),
        "kind": "random" if "-random-" in evaluation["case_id"] else "hard",
        "source_total": evaluation["source_total"], "total_s": evaluation["virtual_time_s"],
        "coverage_stations": report["coverage_points_total"], "coverage_end_index": coverage_end,
        "coverage_end_time_s": end_time, "phases": {key: dict(value) for key, value in phases.items()},
        "ledger_max_prefix_error_s": ledger_error, "failed_clear_count": evaluation["failed_clear_count"],
        "ready_at_coverage_end": ready_count,
        "later_coverage_measurements_after_certificate": later_scans,
        "sixteen_distinct_sources_seen_at_station": count16_station,
        "active_silence_reasons_posthoc": dict(silence_reasons),
        "sources": [source_rows[c] for c in sorted(source_rows)]}


def aggregate(cases):
    source_rows = [s for case in cases for s in case["sources"]]
    ready = [s for s in source_rows if s["certified_at_coverage_end"]]
    phase_names = sorted({key for case in cases for key in case["phases"]})
    mean_phases = {phase: {key: sum(case["phases"].get(phase, {}).get(key, 0.) for case in cases)/len(cases)
        for key in FEES+("actions", "direction", "near", "no_signal", "failed_clear")}
        for phase in phase_names}
    return {"runs": len(cases), "source_total": len(source_rows),
        "mean_total_s": statistics.mean(c["total_s"] for c in cases), "mean_phases": mean_phases,
        "source_ready_by_coverage_end": len(ready), "source_ready_fraction": len(ready)/len(source_rows),
        "source_ready_strictly_before_coverage_end": sum(s["certified_strictly_before_coverage_end"] for s in source_rows),
        "cases_all_sources_ready_at_coverage_end": sum(c["ready_at_coverage_end"] == c["source_total"] for c in cases),
        "ready_first_certificate_kind": dict(Counter(s["first_ready"]["kind"] for s in ready)),
        "ready_positive_observation_counts": dict(Counter(s["first_ready"]["positive_observations"] for s in ready)),
        "ready_station": stats(s["first_ready"]["station"] for s in ready),
        "ready_by_station15": sum(s["first_ready"]["station"] <= 15 for s in ready),
        "actual_wait_ready_to_clear_s": stats(s["actual_wait_ready_to_clear_s"] for s in ready),
        "safe_clear_travel_at_first_certificate_s": stats(s["first_ready"]["safe_clear_arrival_s"] for s in ready),
        "coverage_positive_counts_when_unready": dict(Counter(s["coverage_positive_count"] for s in source_rows if not s["certified_at_coverage_end"])),
        "later_coverage_measurements_after_certificate": sum(c["later_coverage_measurements_after_certificate"] for c in cases),
        "mean_later_coverage_measurements_after_certificate": statistics.mean(c["later_coverage_measurements_after_certificate"] for c in cases),
        "sixteen_source_cases": [{"case_id": c["case_id"], "station": c["sixteen_distinct_sources_seen_at_station"]}
                                  for c in cases if c["source_total"] == 16],
        "active_silence_reasons_posthoc": dict(sum((Counter(c["active_silence_reasons_posthoc"]) for c in cases), Counter())),
        "unready_types": dict(Counter(s["source_type_posthoc"] for s in source_rows if not s["certified_at_coverage_end"])),
        "worst_cases": sorted(cases, key=lambda c: c["total_s"], reverse=True)[:3]}


def analyze(study):
    study = Path(study).resolve()
    manifest = json.loads((study/"manifest.json").read_text(encoding="utf-8"))
    checked = {}
    for name in ("src/geometry/__init__.py", "src/localization/__init__.py"):
        checked[name] = digest(ROOT/name)
        if checked[name] != manifest["source_sha256"][name]:
            raise ValueError(f"historical replay dependency mismatch: {name}")
    paths = sorted((study/"traces").glob("q4-*--triangular.json.gz"))
    if not paths:
        raise ValueError("no archived Q4 triangular traces found")
    cases = [read_trace(path) for path in paths]
    summary = json.loads((study/"summary.json").read_text(encoding="utf-8"))
    return {"scope": "read_only_archived_trajectories_not_new_policy_evaluation", "new_cases_run": 0,
        "manifest_sha256": digest(study/"manifest.json"), "summary_sha256": digest(study/"summary.json"),
        "script_sha256": digest(Path(__file__)), "replay_dependencies": checked,
        "all_trace_path_sha256": [{"path": c["trace"], "sha256": c["sha256"]} for c in cases],
        "original_random_comparison": [g for g in summary["groups"] if g["problem"] == 4 and g["case_kind"] == "random"],
        "random": aggregate([c for c in cases if c["kind"] == "random"]),
        "hard": aggregate([c for c in cases if c["kind"] == "hard"]),
        "hard_cases": [c for c in cases if c["kind"] == "hard"],
        "ledger_max_prefix_error_s": max(c["ledger_max_prefix_error_s"] for c in cases)}


def make_markdown(result):
    random = result["random"]
    lines = ["# Q4旧triangular策略：历史瓶颈诊断", "",
        "只读`results/study`中2026-09-10归档的100个随机局与7个困难局；没有运行新场景、模型或官方接口。"
        "几何回放仅使用真实历史正观测；`no_signal`不缩小Q4候选区域。真值只在终止后的无信号原因分类中使用。", "",
        "## 原随机集比较", "", "| 策略 | 平均虚拟秒 | 平均测量次数 | 平均失败光学清除次数 |",
        "|---|---:|---:|---:|"]
    for group in result["original_random_comparison"]:
        lines.append(f"| {group['strategy']} | {group['mean_virtual_time_s']:.3f} | {group['mean_measurements']:.2f} | {group['mean_failed_clears']:.2f} |")
    lines += ["", "这些策略使用同一批场景但覆盖点与路径不同；历史‘成功完成’允许有计费的光学试探失败，不能把它写成零失败清除。", "",
        "## triangular费用分解", "", "| 实际动作phase | 移动秒/局 | 测量秒/局 | 换频秒/局 | 光学秒/局 | 移除秒/局 | 总秒/局 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for phase, fee in random["mean_phases"].items():
        lines.append(f"| {phase} | " + " | ".join(f"{fee[key]:.3f}" for key in FEES) + f" | {sum(fee[key] for key in FEES):.3f} |")
    cover = random["mean_phases"]["coverage"]
    coverage_total = sum(cover[key] for key in FEES)
    lines += ["", f"固定31站×20频道=620次覆盖测量，覆盖总成本{coverage_total:.3f}秒，占{100*coverage_total/random['mean_total_s']:.2f}%。"
        "其间没有源被清除，因此也没有因清除而减少后续频道扫描；大量无信号本身是合法覆盖费用，不能凭事后真值直接删掉。", "",
        "## 覆盖结束前已具备的清除证书", "",
        f"本批共{random['source_total']}个源，其中{random['source_ready_by_coverage_end']}个"
        f"（{100*random['source_ready_fraction']:.2f}%）在覆盖结束时已经near或MEC半径≤19.9米；"
        f"严格早于最后覆盖动作就持证的有{random['source_ready_strictly_before_coverage_end']}个。"
        f"{random['cases_all_sources_ready_at_coverage_end']}/100局的全部源在覆盖结束时均已持证。", "",
        f"首次证书类型：`{json.dumps(random['ready_first_certificate_kind'],ensure_ascii=False)}`；"
        f"首次持证累计正观测数分布：`{json.dumps(random['ready_positive_observation_counts'],ensure_ascii=False)}`。"
        f"持证站序中位数{random['ready_station']['median']:.1f}、均值{random['ready_station']['mean']:.2f}；"
        f"其中{random['ready_by_station15']}个在第15站及以前已持证。", "",
        f"这些源从首次持证到真实清除平均等待{random['actual_wait_ready_to_clear_s']['mean']:.3f}秒，"
        f"中位数{random['actual_wait_ready_to_clear_s']['median']:.3f}秒。"
        f"在首次持证的实际位置前往最近19.9米安全圆内清除位置，几何移动平均{random['safe_clear_travel_at_first_certificate_s']['mean']:.3f}秒。"
        "前者不是可直接节省的时间，后者不含之后回到覆盖路线或插入其它源的机会成本。", "",
        f"已持证后又发生的同频道覆盖测量共{random['later_coverage_measurements_after_certificate']}次，"
        f"平均{random['mean_later_coverage_measurements_after_certificate']:.2f}次/局；"
        f"对应检测费平均{5*random['mean_later_coverage_measurements_after_certificate']:.2f}秒/局。"
        "这是可检验的跳过已持证频道机会，不是已经实现的净节省；跳过会影响换频、后续外包区域和清除路径。", "",
        f"覆盖结束仍未持证源的累计正观测数：`{json.dumps(random['coverage_positive_counts_when_unready'],ensure_ascii=False)}`；"
        f"事后源类型：`{json.dumps(random['unready_types'],ensure_ascii=False)}`。", "",
        f"另有{len(random['sixteen_source_cases'])}局最终含16个源，实际发现第16个不同频道发生在"
        f"第{min(c['station'] for c in random['sixteen_source_cases'])}—{max(c['station'] for c in random['sixteen_source_cases'])}站，旧策略仍扫满31站。"
        "当观测已经发现16个不同源时，公开源数上界可停止未发现频道搜索；仍须完成已发现源的清除，不能据此立即退出。", "",
        "## 主动探测无信号与尾部", "",
        f"100随机局active无信号的事后原因：`{json.dumps(random['active_silence_reasons_posthoc'],ensure_ascii=False)}`。"
        "分类依据终止后记录中的真实半径与朝向，只作解释；不能将朝向、半径送入在线策略。"
        "在圆形接收距离内也可能因背向而无信号，因此第三问的全向安全探测与负观测推断不能直接迁移。", "",
        "| 最坏随机案例 | 总秒 | 兜底总秒 | 兜底光学次数/失败次数 | active次数/无信号次数 |",
        "|---|---:|---:|---:|---:|"]
    for case in random["worst_cases"]:
        fallback = case["phases"].get("guaranteed_clearance", {})
        active = case["phases"].get("active_localization", {})
        lines.append(f"| {case['case_id']} | {case['total_s']:.3f} | {sum(fallback.get(k,0) for k in FEES):.3f} | "
                     f"{fallback.get('actions',0)}/{fallback.get('failed_clear',0)} | {active.get('actions',0)}/{active.get('no_signal',0)} |")
    lines += ["", "最坏随机局主要是在相同固定覆盖底座之上叠加背向探测、较长清除路线与光学兜底；"
        "普通局则以固定覆盖和覆盖结束后的串行清除旅行占大头。不能只凭平均光学时间小就忽略其尾部风险。", "",
        "尤其应区分‘覆盖只有一次正观测，仍为长方位带’和‘已有多次正观测，但交会角度与误差导致区域仍大’两类未定位源。"
        "以下具体源只按既有费用排序，不构成新策略试验：", ""]
    for case in random["worst_cases"]:
        difficult = sorted(case["sources"], key=lambda s:s["local_cost_after_coverage"].get("fallback_cost_s",0), reverse=True)[:2]
        for source in difficult:
            fee = source["local_cost_after_coverage"]
            if not fee.get("fallback_attempts"):
                continue
            lines.append(f"- `{case['case_id']}`频道{source['channel']}：覆盖正观测{source['coverage_positive_count']}次，"
                f"覆盖结束MEC半径{source['radius_at_coverage_end_m']:.3f}米；active {fee.get('active_measurements',0)}次，"
                f"其中无信号{fee.get('active_no_signal',0)}次；后续光学兜底{fee['fallback_attempts']}次、"
                f"失败{fee['fallback_failed']}次，兜底含移动成本{fee['fallback_cost_s']:.3f}秒。")
    lines += ["",
        "| 困难案例 | 总秒 | 覆盖结束持证源/总源 | active无信号次数 | 兜底失败次数 |",
        "|---|---:|---:|---:|---:|"]
    for case in result["hard_cases"]:
        lines.append(f"| {case['case_id']} | {case['total_s']:.3f} | {case['ready_at_coverage_end']}/{case['source_total']} | "
                     f"{case['phases'].get('active_localization',{}).get('no_signal',0)} | {case['failed_clear_count']} |")
    lines += ["", "## 优先原型与边界", "",
        "1. 保留三角覆盖的几何完备性及真实反馈，只跳过已有清除证书的频道，并把持证清除与剩余覆盖点共同排程。先验证它是否抵消新增绕路，不能把‘更早清除’当成总时更短。",
        "2. 对覆盖后仍未持证、尤其只有一次正观测的源，单独保留Q4可见性不确定与有界光学兜底；不要套用Q3的1000米内必接收、全向负观测半平面或强行预测无信号排除。",
        "3. 路线/扫描与尾部定位分开消融；同一场景比较，完整计入失败光学、额外移动和后续覆盖认证。历史adaptive的扫描虽少却移动和光学失败更多，已经反驳‘越早处理越好’的简单推断。", "",
        "## 可复现证据", "",
        "命令：`python -B experiments/analyze_q4_bottlenecks.py --study results/study --report research/q4_state_search/BASELINE_DIAGNOSTICS.md`。"
        "加`--json`把完整逐源持证、费用、轨迹哈希信息输出到stdout；脚本不生成世界或调用策略。", "",
        f"- 历史manifest SHA256：`{result['manifest_sha256']}`。",
        f"- 历史summary SHA256：`{result['summary_sha256']}`。",
        f"- 分析脚本SHA256：`{result['script_sha256']}`。",
        f"- 所有107条轨迹的最大单动作费用前缀残差：{result['ledger_max_prefix_error_s']:.9g}秒；逐项费用总和与原evaluation相符。",
        "- 回放所用geometry/localization源码SHA与历史manifest完全一致，未以新定位器重解释旧数据。", ""]
    lines.insert(-1, "- 回放结束的逐源外包多边形与归档`source_estimates.vertices`逐顶点一致（1e-7米容差）。")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT/"results/study")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--json", action="store_true", help="print full structured evidence to stdout")
    args = parser.parse_args()
    result = analyze(args.study)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(make_markdown(result), encoding="utf-8")
    print(json.dumps(result if args.json else {"random": {key: value for key, value in result["random"].items()
                    if key != "worst_cases"}, "hard_cases": [{"case": c["case_id"], "total_s": c["total_s"],
                    "ready": c["ready_at_coverage_end"], "source_total": c["source_total"],
                    "failed_clear_count": c["failed_clear_count"]} for c in result["hard_cases"]],
                    "ledger_max_prefix_error_s": result["ledger_max_prefix_error_s"]},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
