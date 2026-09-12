"""Validate paired result tables and write a Chinese comparison report."""
import argparse
import csv
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main(folder):
    comparison = read(folder / "comparison.json")
    manifest = read(folder / "manifest.json")
    arows = read(folder / "state_search/rows.json")
    brows = read(folder / "v3_origin20/rows.json")
    a, b = {r["seed"]: r for r in arows}, {r["seed"]: r for r in brows}
    expected = {c["seed"]: c for c in manifest["cases"]}
    if set(a) != set(b) or set(a) != set(expected) or len(a) != len(arows) or len(b) != len(brows):
        raise ValueError("Incomplete or duplicate paired table")
    for seed in expected:
        if a[seed]["case_sha256"] != b[seed]["case_sha256"] or a[seed]["case_sha256"] != expected[seed]["case_sha256"]:
            raise ValueError("Scenario identity mismatch")
        if a[seed]["source_total"] != b[seed]["source_total"] or a[seed]["source_total"] != expected[seed]["source_total"]:
            raise ValueError("Scenario count mismatch")
    with (folder / "paired.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(expected):
        raise ValueError("Paired CSV length mismatch")
    for row in csv_rows:
        seed = int(row["seed"])
        if (int(row["source_count"]) != a[seed]["source_total"] or
            not math.isclose(float(row["state_seconds"]), a[seed]["virtual_time_s"], abs_tol=1e-8) or
            not math.isclose(float(row["v3_seconds"]), b[seed]["virtual_time_s"], abs_tol=1e-8)):
            raise ValueError("CSV/JSON mismatch")
    sources = sum(r["source_total"] for r in arows)
    ta, tb = sum(r["virtual_time_s"] for r in arows), sum(r["virtual_time_s"] for r in brows)
    for actual, name in ((ta / sources, "state_pooled_seconds_per_source"), (tb / sources, "v3_pooled_seconds_per_source"),
                         ((tb-ta)/sources, "v3_minus_state_pooled_seconds_per_source")):
        if not math.isclose(actual, comparison[name], abs_tol=1e-9):
            raise ValueError("Aggregate mismatch")
    audits = {k: read(folder / k / "audit.json") for k in ("state_search", "v3_origin20")}
    if not all(v["passed"] for v in audits.values()):
        raise ValueError("A raw-action audit did not pass")
    sa = read(folder / "state_search/summary.json")
    sb = read(folder / "v3_origin20/summary.json")
    verified = {"passed": True, "paired_cases": len(a), "sources_per_method": sources,
                "audited_actions": sum(v["audited_actions"] for v in audits.values()),
                "audited_successful_clearances": sum(v["geometrically_verified_clearances"] for v in audits.values()),
                "same_scenarios_and_counts": True, "csv_json_agreement": True,
                "independent_aggregate_agreement": True, "both_trace_audits_passed": True}
    (folder / "pair-audit.json").write_text(json.dumps(verified, indent=2) + "\n", encoding="utf-8")
    ci = comparison["paired_pooled_saving_ci95_s"]
    lines = ["# 状态搜索与 v3 原点扫描：5000 场景配对比较", "",
             "两种方法在同一组固定场景、同一个本地物理与计费引擎上运行。算法及原有参数均未修改。", "",
             "## 主要结果", "", "| 指标 | 状态搜索 runtime-equivalent | v3 原点扫描 |", "|---|---:|---:|"]
    values = [("每源平均计费用时 / s", f"{ta/sources:.6f}", f"{tb/sources:.6f}"),
              ("每局平均计费用时 / s", f"{ta/len(a):.3f}", f"{tb/len(b):.3f}"),
              ("每局计费用时 P95 / s", f"{sa['p95_episode_virtual_time_s']:.3f}", f"{sb['p95_episode_virtual_time_s']:.3f}"),
              ("全清并正常退出", f"{sa['successful_cases']}/{len(a)}", f"{sb['successful_cases']}/{len(b)}"),
              ("清除源数", str(sa["cleared_sources"]), str(sb["cleared_sources"])),
              ("失败清除尝试", str(sa["failed_clear_attempts"]), str(sb["failed_clear_attempts"])),
              ("测量次数", str(sa["total_measurements"]), str(sb["total_measurements"]))]
    lines += [f"| {name} | {va} | {vb} |" for name, va, vb in values]
    lines += ["", "主指标为总任务计费用时除以总源数，包括发现、定位、换频、移动及清除，不是逐局 T/N 的等权平均。", "",
              f"状态搜索相对 v3 平均每源节省 **{comparison['v3_minus_state_pooled_seconds_per_source']:.6f} s**，平均计费用时减少 **{100*comparison['state_reduction_relative_to_v3']:.3f}%**。以整局配对重采样 4000 次，节省量的 bootstrap 95% 区间为 **[{ci[0]:.6f}, {ci[1]:.6f}] s/源**。",
              f"逐局比较：状态搜索更快 {comparison['state_faster_cases']} 局，v3 更快 {comparison['v3_faster_cases']} 局，平局 {comparison['tied_cases']} 局。状态搜索最差一局比 v3 多用 {comparison['state_worst_regression_case']['extra_seconds']:.3f} s，不能理解为每局必胜。", "",
              "结论仅适用于这批固定合成随机场景上的平均任务用时；不等价于官方比赛成绩，也不证明对任意源布局、误差模式都更优。", "",
              "## 按每局源数分组", "", "| 源数 | 局数 | 状态搜索 s/源 | v3 s/源 | v3减状态搜索 s/源 |", "|---:|---:|---:|---:|---:|"]
    for n in range(10,17):
        aa = [r for r in arows if r["source_total"] == n]
        bb = [r for r in brows if r["source_total"] == n]
        ma = sum(r["virtual_time_s"] for r in aa)/(n*len(aa))
        mb = sum(r["virtual_time_s"] for r in bb)/(n*len(bb))
        lines.append(f"| {n} | {len(aa)} | {ma:.3f} | {mb:.3f} | {mb-ma:+.3f} |")
    lines += ["", "## 费用差异", "", "| 项目 | 状态搜索 s/源 | v3 s/源 | v3减状态搜索 s/源 |", "|---|---:|---:|---:|"]
    labels = {"movement_s":"移动", "switching_s":"换频", "detection_s":"测向检测", "optical_s":"光学搜索", "removal_s":"清除"}
    for key, label in labels.items():
        va, vb = sa["time_components_total_s"][key]/sources, sb["time_components_total_s"][key]/sources
        lines.append(f"| {label} | {va:.6f} | {vb:.6f} | {vb-va:+.6f} |")
    lines += ["", "## 固定协议、版本与核验", "",
              f"- 每种方法 {len(a)} 局、{sources} 个源；种子 {min(a)}–{max(a)}，场景清单在运行前固定。",
              "- 沿用 random_scenario(3, seed)：源数在10–16间均匀抽取，1800 m圆内按面积均匀分布，接收半径在1000–1500 m间均匀分布。",
              "- 测角误差使用共同的场景—频道—坐标键；同一场景下相同频道和相同测点得到同一误差。算法选择不同测点时不强行使用同一误差数值。",
              "- 状态搜索使用原入口；v3原点扫描使用现有协议桥接，只适配本地客户端生命周期和共同动作预算。两种方法都仅访问合法观测。",
              f"- 状态搜索代码基准：{manifest['source']['state_commit']}；v3代码来自main：{manifest['source']['v3_commit']}。复制的v3源码与main逐字节相同。",
              "- 每局预算：1200 s实际时间、360000 s虚拟时间、10000个动作；所有失败保留，禁止按结果挑选或提前停止。",
              "- 原8核4060主机，6个低优先级CPU进程，不使用GPU；v3的Numba编译预热在计时外。实际程序运行时长不是本次优劣判断指标。",
              f"- 独立重算 {verified['audited_actions']} 条动作的计费，验证 {verified['audited_successful_clearances']} 次清除的20 m几何条件，两套全量审计均通过。",
              "", "## 文件及复现", "",
              "comparison.json 为配对效应、置信区间与胜负统计；paired.csv 为逐局配对；两方法子目录保留逐局表、汇总、全量审计及原始记录哈希。原始逐步压缩记录保存在原4060主机，具体位置见执行凭据。", "",
              "```text", "PYTHONPATH=.deps python3 experiments/run_q3_paired5000.py --output results/reproduction --cases 5000 --seed-start 2026092300 --workers 6",
              "python3 experiments/audit_q3_random_source_batch.py results/reproduction/state_search",
              "python3 experiments/audit_q3_random_source_batch.py results/reproduction/v3_origin20",
              "python3 experiments/summarize_q3_paired.py results/reproduction", "```", "",
              "复现时先将本目录的 source.tar.gz 解压到独立目录，再从该目录运行测试命令；其中包含 PAIRED_SOURCE_SNAPSHOT.json 及运行时的完整源码字节，避免 Git 换行转换影响字节校验。使用 numpy 与 numba；.deps 仅为本次隔离安装位置。汇总脚本见 Git 中 experiments/summarize_q3_paired.py。"]
    (folder / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(verified))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    main(parser.parse_args().folder)
