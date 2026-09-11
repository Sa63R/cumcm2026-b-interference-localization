"""Offline diagnosis of completed Q3 records, with input-file provenance.

No simulator, SQLite, network access, raw requests or account exports.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
import statistics

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def pct(values, q):
    values = sorted(values)
    at = (len(values)-1)*q
    i = int(at)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(at-i)

def details(report, n, t, physical_lb, label, split, stronger_lb=None):
    actions = report["action_history"]
    source_channels = set(report["cleared_channels"])
    known = set()
    position = (0.,0.)
    last_virtual = 0.
    phases = defaultdict(Counter)
    measures = Counter()
    first_detected = {}
    clear_times = []
    repeated = Counter()
    event_rows = []
    for action in actions:
        phase = action["phase"]
        p = action["position"]
        c = action["channel"]
        movement = math.dist(position, p)/5
        elapsed = action["virtual_time_s"]-last_virtual
        known_before = c in known
        physical_source = c in source_channels
        if action["action"] == "measure":
            measures["measurements"] += 1
            measures[action["result"]] += 1
            repeated[(c,tuple(p))] += 1
            if not physical_source:
                measures["empty_channel"] += 1
            elif known_before:
                measures["known_source_channel"] += 1
            else:
                measures["not_yet_detected_source_channel"] += 1
            if phase=="coverage" and known_before:
                measures["coverage_known_source_channel"] += 1
            if physical_source and action["result"]=="no_signal":
                measures["source_channel_no_signal"] += 1
            if action["result"] in {"direction","near"}:
                known.add(c)
                first_detected.setdefault(c,action["virtual_time_s"])
            phases[phase]["measurements"] += 1
        else:
            phases[phase]["clear_attempts"] += 1
            if action["result"]=="success":
                clear_times.append(action["virtual_time_s"])
                known.add(c)
                first_detected.setdefault(c,action["virtual_time_s"])
        phases[phase]["actions"] += 1
        phases[phase]["movement_s"] += movement
        phases[phase]["total_virtual_s"] += elapsed
        position,last_virtual = p,action["virtual_time_s"]
        event_rows.append((action["action"],c,action["virtual_time_s"],phase,movement,elapsed))
    all_discovered_t = max(first_detected.values())
    last_clear = max(clear_times)
    after_discovery = [a for a in event_rows if a[2] > all_discovered_t+1e-6]
    after_clear = [a for a in event_rows if a[2] > last_clear+1e-6]
    planning = report["strategy_parameters"]["planning_log"]
    probe = report["strategy_parameters"]["probe_search_log"]
    changes = Counter()
    for k in ["expanded","generated","bound_pruned","dominance_pruned"]:
        changes[k] = sum(a[k] for a in planning)
    changes["planning_calls"] = len(planning)
    changes["inexact_calls"] = sum(not a["exact"] for a in planning)
    changes["max_model_gap_s"] = max([a["model_gap_s"] for a in planning] or [0])
    changes["mean_model_gap_s"] = statistics.mean(a["model_gap_s"] for a in planning) if planning else 0
    changes["probe_calls"] = len(probe)
    changes["refined_probe_changed"] = sum(a.get("changed_point",False) for a in probe)
    changes["probe_total_surrogate_improvement_s"] = sum(a.get("score_improvement_s",0) for a in probe)
    return dict(label=label,split=split,n=n,t=t,per_source=t/n,physical_lb=physical_lb,
                physical_lb_per_source=physical_lb/n,physical_ratio=t/physical_lb,
                archived_stronger_lb=stronger_lb,
                archived_stronger_ratio=t/stronger_lb if stronger_lb else None,
                breakdown=report["time_breakdown"],phases=dict(phases),measures=dict(measures),
                repeated_identical_position_channel=sum(v-1 for v in repeated.values()),
                first_clear_s=min(clear_times),all_discovered_s=all_discovered_t,last_clear_s=last_clear,
                after_last_clear_s=t-last_clear,after_all_discovered_s=t-all_discovered_t,
                empty_measures_after_all_discovered=sum(a[0]=="measure" and a[1] not in source_channels for a in after_discovery),
                empty_measures_after_last_clear=sum(a[0]=="measure" and a[1] not in source_channels for a in after_clear),
                after_last_clear_movement_s=sum(a[4] for a in after_clear),
                after_all_discovered_coverage_s=sum(a[5] for a in after_discovery if a[3]=="coverage"),
                planning=dict(changes),target_200_saving_needed_s=max(0,t-200*n))

def aggregate(rows):
    avg=lambda key:statistics.mean(r[key] for r in rows)
    out={k:avg(k) for k in ["t","per_source","physical_lb","physical_lb_per_source","physical_ratio",
                           "first_clear_s","all_discovered_s","last_clear_s","after_last_clear_s",
                           "after_all_discovered_s","empty_measures_after_all_discovered",
                           "empty_measures_after_last_clear","after_last_clear_movement_s",
                           "after_all_discovered_coverage_s","target_200_saving_needed_s"]}
    out.update(case_count=len(rows),mean_source_count=avg("n"),
               median_per_source=statistics.median(r["per_source"] for r in rows),
               p95_per_source=pct([r["per_source"] for r in rows],.95),min_per_source=min(r["per_source"] for r in rows),
               max_per_source=max(r["per_source"] for r in rows),under_200_count=sum(r["per_source"]<200 for r in rows),
               pooled_time_per_source=sum(r["t"] for r in rows)/sum(r["n"] for r in rows),
               pooled_physical_ratio=sum(r["t"] for r in rows)/sum(r["physical_lb"] for r in rows),
               repeated_identical_position_channel=sum(r["repeated_identical_position_channel"] for r in rows))
    for name in ["breakdown","measures","planning"]:
        keys=set().union(*(r[name] for r in rows))
        out[name]={key:statistics.mean(r[name].get(key,0) for r in rows) for key in sorted(keys)}
    phases=set().union(*(r["phases"] for r in rows))
    out["phases"]={p:{k:statistics.mean(r["phases"].get(p,{}).get(k,0) for r in rows)
                      for k in ["actions","movement_s","total_virtual_s","measurements","clear_attempts"]}
                    for p in sorted(phases)}
    if all(r["archived_stronger_lb"] for r in rows):
        out["archived_stronger_lb"]=avg("archived_stronger_lb")
        out["archived_stronger_lb_per_source"]=statistics.mean(r["archived_stronger_lb"]/r["n"] for r in rows)
        out["archived_stronger_ratio"]=avg("archived_stronger_ratio")
    return out

def report_markdown(result):
    practice = result["practice_overall"]
    random = result["synthetic_partitions"]["independent_random"]
    overall = result["synthetic_overall"]
    table = []
    for name, group in [("真实演练", practice), ("新普通合成案例", random), ("全部合成案例", overall)]:
        table.append(f"| {name} | {group['case_count']} | {group['per_source']:.2f} | "
                     f"{group['physical_lb_per_source']:.2f} | {group['physical_ratio']:.3f} | "
                     f"{group['under_200_count']}/{group['case_count']} |")
    by_n = []
    for n, group in result["practice_by_n"].items():
        by_n.append(f"| {n} | {group['case_count']} | {group['per_source']:.2f} | "
                    f"{group['physical_lb_per_source']:.2f} | {group['physical_ratio']:.3f} |")
    movement_share = 100 * practice["breakdown"]["movement_s"] / practice["t"]
    exact_share = 100 * (1 - practice["planning"]["inexact_calls"] / practice["planning"]["planning_calls"])
    active = practice["phases"]["active_localization"]
    clear = practice["phases"]["certified_clear"]
    coverage = practice["phases"]["coverage"]
    return f"""# 问题3基准：每源用时与改进空间诊断

本报告只读取已完成的记录，不运行模拟器、训练或SQLite查询。40局真实演练来自状态搜索760af823；54个合成案例来自其运行效率优化版本。后者已验证与原版动作及虚拟用时完全一致，所以这里作为同一虚拟性能基准。54个案例包含8个已用开发案例、32个新普通案例、14个新压力案例；只取每案例一份候选记录，不把重复计时算作新案例。

## 指标与结果

N是**单局干扰源数量**，case_count是**案例局数**。T为包含发现、定位、移动、切频、检测和清除的整局计费虚拟时间。表中每源用时为各局T/N的算术平均，下界比值为各局T/LB的算术平均。

| 数据 | 案例局数 | 平均T/N（秒/源） | 平均物理LB/N（秒/源） | mean(T/LB) | T/N低于200的局数 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

40局演练的“合计T/合计N”为{practice['pooled_time_per_source']:.2f}秒/源，区别于平均T/N的{practice['per_source']:.2f}秒/源。两种汇总都保存在JSON中，不能交换使用。

| 每局源数N | 演练案例局数 | 平均T/N（秒/源） | 平均物理LB/N（秒/源） | mean(T/LB) |
|---|---:|---:|---:|---:|
{chr(10).join(by_n)}

比较他人的“低于200秒/源”时，需要相同的整局计时口径、N分布和案例难度。当前N较多时，每源指标明显较低；这不能单独说明策略更强。40局演练平均T/N若按同一比例下降至200秒/源，需要约{100*(1-200/practice['per_source']):.2f}%的下降，不能把该目标当成已经可实现的承诺。

## 下界口径

- **合成案例物理下界**：策略结束后使用实际源中心，构造20米清除圆盘之间的边长下界，以开放路线子集DP求最短距离图路线，再加每源5秒。无需定位和发现，通常不是可实现的在线最优时间。
- **演练物理观测下界**：不读取真实源坐标，用已保存测向和清除信息形成源包含区域/圆盘，再计算相应物理下界；读取每局observation_lower_bound_s。它与合成真值下界的可用信息不同，因此两组T/LB不是严格同口径的直接排名。
- **演练原批汇总的较强下界**：conditional_guaranteed_all_clear_lower_s还计入实际空频道的必要完整性认证动作成本，依赖完整清除的保证条件。其平均LB/N为{practice['archived_stronger_lb_per_source']:.2f}秒/源，mean(T/LB)为{practice['archived_stronger_ratio']:.3f}。这里单独保留，未混入上表物理下界。

压力案例中存在一个物理LB/N已达214.49秒/源的边界场景。因此“所有允许的案例均低于200秒/源”不成立；平均指标改善可以作为独立的实验目标。

## 慢在哪里

40局演练平均整局{practice['t']:.2f}秒，其中移动{practice['breakdown']['movement_s']:.2f}秒（{movement_share:.2f}%）、检测{practice['breakdown']['detection_s']:.2f}秒、切频{practice['breakdown']['switching_s']:.2f}秒、光学与清除合计{practice['breakdown']['optical_s']+practice['breakdown']['removal_s']:.2f}秒。

按动作目标阶段归因：主动定位阶段移动{active['movement_s']:.2f}秒，前往认证清除点移动{clear['movement_s']:.2f}秒，覆盖阶段移动{coverage['movement_s']:.2f}秒。**主动定位移动并非全可节省**：其中含到达目标附近所必需的运动。阶段归因只是定位改进方向，并未给出可省时间的上界或证明绕路。

平均最后一次成功清除后仍用{practice['after_last_clear_s']:.2f}秒，其中移动{practice['after_last_clear_movement_s']:.2f}秒；这些动作承担剩余完整性确认，不能直接删除。演练中最长尾段约502秒，聚集/窄条压力案例可超过1300秒。尾部值得检验，但其平均体量不足以单独达成整体目标。

有限冻结任务路线规划约{exact_share:.3f}%的调用已经在该代理模型内求得精确解。该事实不代表原部分可观测问题最优，但说明仅增加同一模型的搜索展开预算不应是首要假设。

没有重复的同位置同频道测量。覆盖阶段平均{practice['measures']['coverage_known_source_channel']:.2f}次测量发生在已发现源的频道；这些测量可能仍改善定位或提供必要反馈，不能一概视为无效。

## 可检验假设

1. 将实际“移动—测量—定位—清除”服务过程和后续任务位置纳入任务转移成本，检验冻结点模型偏差是否造成额外移动。
2. 在沿途位置替代部分未来测量或覆盖站测量，保持合法观测与覆盖证书；通过消融区分测量位置变化和调度变化，评价整局T/N而非局部代理评分。
3. 提前积累完整性确认所需的有效覆盖，减少末尾长途移动；完整清除及可靠终止必须与时间收益共同核验。

这里只提出假设，不宣称这些方向一定有效。测量缓存、CPU缓存等实现加速不会自行降低计费虚拟时间。

## 复现与证据

在此分支根目录执行，输出目录应使用新目录，避免覆盖证据：

~~~text
python experiments/diagnose_per_source.py --output-dir research/virtual_under_200/reproduction
~~~

默认读取相邻q3-runtime-equivalent目录中的最终54案例批次，以及主项目的40局已完成演练批次。可通过--synthetic-batch、--practice-batch明确传入迁移后的路径。

baseline_diagnosis.json保存每个输入summary、lower_bounds、record及批次说明文件的SHA256、脱敏后的逐局统计和分组统计；分析结束时再次核对输入未变化。未导出队号、账户、案例访问凭据或原始请求。真实演练只用run序号标识。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-batch", type=Path, default=ROOT.parent /
                        "q3-runtime-equivalent/results/runtime_equivalent/linux/6c49ed604f98cc8e/bench")
    parser.add_argument("--practice-batch", type=Path, default=ROOT.parent /
                        "cumcm2026-b-interference-localization/results/practice_batches/q3-source-count-derived-20260912")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "research/virtual_under_200")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Do not disable assertions during evidence checks")
    output_json = args.output_dir / "baseline_diagnosis.json"
    output_md = args.output_dir / "BASELINE_DIAGNOSIS.md"
    if output_json.exists() or output_md.exists():
        parser.error("Refuse overwriting existing diagnosis outputs")
    batch, practice_root = args.synthetic_batch.resolve(), args.practice_batch.resolve()
    inputs, input_paths = {}, {}

    def remember(path, kind, root):
        name = kind + "/" + path.relative_to(root).as_posix()
        inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        input_paths[name] = path

    for name in ("manifest.json", "cases.json", "runs_with_bounds.json", "summary.json", "lower_bounds.json"):
        remember(batch / name, "synthetic", batch)
    for name in ("protocol.json", "summary.json"):
        remember(practice_root / name, "practice", practice_root)
    synthetic = []
    manifest, batch_summary = read(batch / "manifest.json"), read(batch / "summary.json")
    assert batch_summary["all_pairs_exactly_identical"] and batch_summary["all_runs_successful"]
    rows = read(batch / "runs_with_bounds.json")
    for row in rows:
        if row["repeat"] != 0 or row["side"] != "candidate":
            continue
        path = (batch / row["record_file"]).resolve()
        if batch not in path.parents:
            raise ValueError("Record path leaves synthetic batch")
        remember(path, "synthetic", batch)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            raw = json.load(stream)
        assert row["successful"] and row["all_cleared"]
        assert math.isclose(sum(raw["summary"]["time_breakdown"].values()), row["virtual_time_s"], abs_tol=.001)
        synthetic.append(details(raw["summary"], row["source_total"], row["virtual_time_s"],
                                 row["physical_lower_bound_s"], row["case_id"], row["split"]))
    assert len(synthetic) == len({r["label"] for r in synthetic}) == manifest["case_count"]
    practice = []
    protocol, practice_summary = read(practice_root / "protocol.json"), read(practice_root / "summary.json")
    for row in practice_summary["rows"]:
        folder = practice_root / f"run-{row['run']:03d}"
        for name in ("summary.json", "lower_bounds.json"):
            remember(folder / name, "practice", practice_root)
        raw, lb = read(folder / "summary.json"), read(folder / "lower_bounds.json")
        assert row["all_clear"] and lb["official_all_clear_verified"]
        assert math.isclose(row["lower_bound_s"], lb["conditional_guaranteed_all_clear_lower_s"])
        assert math.isclose(sum(raw["search"]["time_breakdown"].values()), row["actual_time_s"], abs_tol=.001)
        assert hashlib.sha256((folder / "summary.json").read_bytes()).hexdigest() == row["summary_sha256"]
        assert hashlib.sha256((folder / "lower_bounds.json").read_bytes()).hexdigest() == row["lower_bounds_sha256"]
        practice.append(details(raw["search"], row["source_count"], row["actual_time_s"], lb["observation_lower_bound_s"],
                                f"practice-{row['run']:03d}", "official_practice", row["lower_bound_s"]))
    assert len(practice) == len({r["label"] for r in practice})
    assert len(synthetic) == 54 and len(practice) == 40, "This baseline diagnosis expects frozen 54/40 batches"
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Offline completed-run diagnostics only; source-channel membership is consulted after policy termination, never supplied to an online policy.",
        "metric_definitions": {"n": "source count within one case", "case_count": "number of distinct cases",
                               "mean_source_count": "arithmetic average of case source counts",
                               "per_source": "mean of case T/N values in aggregates; T/N for an individual row",
                               "pooled_time_per_source": "sum(T)/sum(N), separately labelled",
                               "physical_ratio": "mean of case T/LB values in aggregates",
                               "target_200_saving_needed_s": "mean(max(0,T-200*N)); descriptive per-case positive deficit, not an achievable savings claim"},
        "attribution_caveat": "Movement is attributed to its destination action phase. Active-localization movement contains necessary travel and is not wholly removable.",
        "lb_scope": "Synthetic: true-centre 20m disk graph physical oracle bound. Practice: observation-containing-region physical oracle bound. Archived stronger practice bound adds conditional empty-channel certificate action necessities. Different information sets must not be conflated.",
        "baseline_policy_commit": protocol["source_commit"],
        "synthetic_runtime_optimized_commit": manifest["candidate"]["git_head"],
        "practice_spec_sha256": protocol["spec_sha256"],
        "synthetic_spec_sha256": manifest["candidate"]["spec_sha256"],
        "input_sha256": inputs,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "synthetic_overall": aggregate(synthetic),
        "synthetic_partitions": {p: aggregate([r for r in synthetic if r["split"] == p]) for p in sorted({r["split"] for r in synthetic})},
        "practice_overall": aggregate(practice),
        "practice_by_n": {str(n): aggregate([r for r in practice if r["n"] == n]) for n in sorted({r["n"] for r in practice})},
        "synthetic_by_n": {str(n): aggregate([r for r in synthetic if r["n"] == n and r["split"] != "independent_stress"])
                           for n in range(10, 17) if any(r["n"] == n and r["split"] != "independent_stress" for r in synthetic)},
        "rows": synthetic + practice}
    result["inputs_unchanged_during_analysis"] = all(
        hashlib.sha256(path.read_bytes()).hexdigest() == inputs[name] for name, path in input_paths.items())
    assert result["inputs_unchanged_during_analysis"], "Input files changed during analysis"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_json.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with output_md.open("x", encoding="utf-8") as stream:
        stream.write(report_markdown(result))
    print(json.dumps({"synthetic_cases": len(synthetic), "practice_cases": len(practice),
                      "input_files_hashed": len(inputs), "inputs_unchanged": result["inputs_unchanged_during_analysis"],
                      "practice_mean_T_over_N": result["practice_overall"]["per_source"],
                      "synthetic_random_mean_T_over_N": result["synthetic_partitions"]["independent_random"]["per_source"]}))


if __name__ == "__main__":
    main()
