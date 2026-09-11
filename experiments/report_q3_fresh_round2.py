"""Render audited round-two statistics as portable Markdown tables."""
import argparse
import json
from pathlib import Path

LABELS = {"development_original_nominal": "原名义开发组", "development_original_pressure": "原压力开发组",
          "development_added_pressure": "新增压力开发组", "nominal": "冻结名义测试", "pressure": "冻结压力测试",
          "reconstruction": "新原始案例重建验证"}


def number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def table(lines, headers, rows):
    lines.extend(["", "| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)])
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines.append("")


def render(data):
    lines = ["# 第二轮完整结果表", "", "由独立审计后的 analysis.json 生成。所有时间单位为秒；比值使用组平均时间÷组平均下界。"
             "开发、名义测试、压力测试、重建验证分开。失败不计作更短的完成时间。",
             "", f"纳入 {data['unique_unconflicted_runs']} 次不冲突运行，审计 {data['audit']['audited_traces']} 条轨迹、"
             f"{data['audit']['audited_api_actions']} 个API动作；审计通过={data['audit']['passed']}。",
             f"已完成批次CPU累计 {data['cpu']['total_completed_batch_cpu_s']:.2f} 秒。冒烟和单独诊断不在此数中。"]
    order = ["development_original_nominal", "development_original_pressure", "development_added_pressure", "nominal", "pressure", "reconstruction"]
    for group in sorted(data["groups"], key=lambda g: order.index(g) if g in order else 99):
        value = data["groups"][group]
        lines += ["", "## " + LABELS.get(group, group)]
        stats = value["strategies"]
        names = [n for n in ("B", "C", "R", "CR", "RA", "CRA") if n in stats]
        rows = []
        for name in names:
            s = stats[name]; lb = s["lower_bounds"]
            rows.append([name, f"{s['successes']}/{s['cases']}", number(s['mean_s']), number(s['p95_s']),
                number(s['worst_s']), number(s['mean_time_per_source_s']),
                number(lb['physical']['mean_lower_bound_s']), number(lb['guarantee']['mean_lower_bound_s']),
                number(lb['physical']['ratio_of_mean_times'],3), number(lb['guarantee']['ratio_of_mean_times'],3)])
        table(lines, ["策略", "成功/例数", "平均T", "P95", "最慢", "平均T/N", "物理LB", "加强LB", "物理倍数", "加强倍数"], rows)
        comparisons = []
        for key, p in value['paired'].items():
            b, c = stats[p['baseline']], stats[p['candidate']]
            change = None if b['mean_s'] is None or c['mean_s'] is None else 100*(c['mean_s']/b['mean_s']-1)
            ci = p['paired_group_bootstrap_95_delta_s']
            comparisons.append([key, p['completed_pairs'], p['original_groups'], number(p['mean_group_delta_s']),
                number(change), "—" if ci is None else f"[{number(ci[0])}, {number(ci[1])}]",
                f"{p['case_wins']}/{p['case_losses']}/{p['case_ties']}"])
        table(lines, ["对照(后者−前者)", "完整配对", "原始组", "平均差", "均值变化%", "组bootstrap95%差区间", "胜/负/平"], comparisons)
        table(lines, ["策略", "移动", "测量", "切换", "清除", "失败清除次数", "真实后清拖尾", "平均/最大策略墙钟", "平均规划墙钟", "规划/改动作数"],
              [[n, number(stats[n]['mean_components']['movement_s']), number(stats[n]['mean_components']['detection_s']),
                number(stats[n]['mean_components']['switching_s']), number(stats[n]['mean_components']['removal_s']),
                stats[n]['failed_clears'], number(stats[n]['mean_confirmation_tail_s']),
                f"{number(stats[n]['mean_policy_wall_s'])}/{number(stats[n]['max_policy_wall_s'])}",
                number(stats[n]['mean_planning_wall_s']), f"{stats[n]['planner_calls']}/{stats[n]['planner_overrides']}"] for n in names])
        if any('observation_robust_physical' in stats[n]['lower_bounds'] for n in names):
            table(lines, ["策略", "不确定性稳健物理LB", "对应倍数", "不确定性稳健加强LB", "对应倍数"],
                  [[n, number(stats[n]['lower_bounds']['observation_robust_physical']['mean_lower_bound_s']),
                    number(stats[n]['lower_bounds']['observation_robust_physical']['ratio_of_mean_times'],3),
                    number(stats[n]['lower_bounds']['observation_robust_guarantee']['mean_lower_bound_s']),
                    number(stats[n]['lower_bounds']['observation_robust_guarantee']['ratio_of_mean_times'],3)] for n in names])
        lines += ["### 按源数分层", "", "每个分层仍属于同一组测试，不增加独立样本量；小分层不宜作总体推断。"]
        strata = []
        for n in sorted({int(n) for s in stats.values() for n in s['by_source_count']}):
            for name in names:
                s = stats[name]['by_source_count'].get(str(n))
                if s:
                    lb = s['lower_bounds']
                    strata.append([n, name, f"{s['successes']}/{s['cases']}", number(s['mean_s']), number(s['mean_time_per_source_s']),
                        number(lb['physical']['mean_lower_bound_s']), number(lb['guarantee']['mean_lower_bound_s']),
                        number(lb['physical']['ratio_of_mean_times'],3), number(lb['guarantee']['ratio_of_mean_times'],3)])
        table(lines, ["N", "策略", "成功/例数", "平均T", "平均T/N", "物理LB", "加强LB", "物理倍数", "加强倍数"], strata)
        for key, p in value['paired'].items():
            regressions = p['worst_regressions'][:3]
            if regressions:
                lines.append("### " + key + " 最大退步（最多3例）")
                table(lines, ["案例", "N", "基线T", "候选T", "增加T", "增加%", "物理LB", "加强LB", "候选物理/加强倍数"],
                      [[r['case_id'], r['source_total'], number(r['baseline_s']), number(r['candidate_s']), number(r['delta_s']),
                        number(r['relative_change_pct']), number(r['physical_lower_bound_s']), number(r['guarantee_lower_bound_s']),
                        f"{number(r['candidate_over_physical_lower_bound'],3)}/{number(r['candidate_over_guarantee_lower_bound'],3)}"] for r in regressions])
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(render(json.loads(a.analysis.read_text(encoding='utf-8'))), encoding='utf-8')
    print('rendered', a.out)


if __name__ == '__main__':
    main()
