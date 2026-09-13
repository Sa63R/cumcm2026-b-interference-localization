#!/usr/bin/env python3
"""Build paper-ready parameter and scenario sensitivity evidence.

This script does not compare algorithm variants.  It uses deterministic
geometry for Questions 2 and 3, and existing fixed-policy records for the
Question 3 and Question 4 scenario analyses.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
Q3_CSV = ROOT / "github-release-q3-q4/experiments/q3_selected/validation/local_paired400.csv"
Q4_CSV = ROOT / "output/q4_v6_lite_20260912/paired_records.csv"
Q4_PLAN = ROOT / "output/q4_v6_lite_20260912/plan.json"
RNG = np.random.default_rng(20260913)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(values: list[float], replicates: int = 20_000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    indices = RNG.integers(0, len(array), size=(replicates, len(array)))
    means = array[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def grouped_summary(rows: list[dict], group_key: str, value_key: str) -> list[dict]:
    result = []
    for group in sorted({row[group_key] for row in rows}):
        values = [float(row[value_key]) for row in rows if row[group_key] == group]
        ci_low, ci_high = bootstrap_mean_ci(values)
        result.append(
            {
                group_key: group,
                "cases": len(values),
                "mean_seconds_per_source": statistics.mean(values),
                "median_seconds_per_source": statistics.median(values),
                "p10_seconds_per_source": float(np.quantile(values, 0.10)),
                "p90_seconds_per_source": float(np.quantile(values, 0.90)),
                "mean_ci95_low": ci_low,
                "mean_ci95_high": ci_high,
            }
        )
    return result


def q2_parameter_sensitivity() -> tuple[list[dict], dict]:
    d1, d2 = 1000.0, 470.0
    gammas = np.linspace(8.0, 90.0, 329)
    epsilons = [0.8, 1.0, 1.2]
    rows = []
    for epsilon in epsilons:
        for gamma in gammas:
            diameter = 2.0 * (d1 + d2) * math.radians(epsilon) / math.sin(math.radians(gamma))
            rows.append(
                {
                    "bearing_half_error_deg": epsilon,
                    "crossing_angle_deg": float(gamma),
                    "estimated_diameter_m": diameter,
                }
            )
    selected = {}
    for epsilon in epsilons:
        selected[f"epsilon_{epsilon:.1f}_gamma_23"] = (
            2.0 * (d1 + d2) * math.radians(epsilon) / math.sin(math.radians(23.0))
        )
    for gamma in [20.0, 23.0, 26.0]:
        selected[f"epsilon_1.0_gamma_{gamma:.0f}"] = (
            2.0 * (d1 + d2) * math.radians(1.0) / math.sin(math.radians(gamma))
        )
    selected["local_elasticity_gamma_at_23"] = -math.radians(23.0) / math.tan(math.radians(23.0))
    selected["local_elasticity_epsilon"] = 1.0
    return rows, selected


def q3_geometric_sensitivity() -> tuple[list[dict], dict]:
    field_radius = 1800.0
    receive_radius = 1000.0
    half_sector = math.pi / 7.0
    radii = np.linspace(990.0, 1005.0, 301)
    rows = []
    for radius in radii:
        boundary_distance = math.sqrt(
            field_radius**2
            + radius**2
            - 2.0 * field_radius * radius * math.cos(half_sector)
        )
        worst_distance = max(radius, boundary_distance)
        rows.append(
            {
                "ring_radius_m": float(radius),
                "centre_distance_m": float(radius),
                "boundary_worst_distance_m": boundary_distance,
                "guaranteed_worst_distance_m": worst_distance,
                "coverage_margin_m": receive_radius - worst_distance,
                "coverage_guaranteed": bool(worst_distance <= receive_radius + 1e-12),
            }
        )
    optimum = field_radius / (2.0 * math.cos(half_sector))
    feasible_low = field_radius * math.cos(half_sector) - math.sqrt(
        receive_radius**2 - field_radius**2 * math.sin(half_sector) ** 2
    )
    chosen = 999.0
    chosen_boundary = math.sqrt(
        field_radius**2
        + chosen**2
        - 2.0 * field_radius * chosen * math.cos(half_sector)
    )
    summary = {
        "minimax_ring_radius_m": optimum,
        "feasible_ring_radius_low_m": feasible_low,
        "feasible_ring_radius_high_m": receive_radius,
        "chosen_ring_radius_m": chosen,
        "chosen_worst_distance_m": max(chosen, chosen_boundary),
        "chosen_coverage_margin_m": receive_radius - max(chosen, chosen_boundary),
    }
    return rows, summary


def load_q3_rows() -> list[dict]:
    rows = []
    with Q3_CSV.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["mode"] != "v3_origin20":
                continue
            rows.append(
                {
                    "source_count": int(row["true_sources"]),
                    "seconds_per_source": float(row["seconds_per_source"]),
                }
            )
    if len(rows) != 400:
        raise RuntimeError(f"Expected 400 Q3 records, found {len(rows)}")
    return rows


def load_q4_rows() -> list[dict]:
    plan = json.loads(Q4_PLAN.read_text(encoding="utf-8"))
    scenes = {
        case["key"]: case["scenario"]
        for case in plan["cases"]
        if case["group"] == "new_practice"
    }
    rows = []
    with Q4_CSV.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = row.get("case_key") or row.get("\ufeffcase_key")
            if row["group"] != "new_practice" or row["method"] != "lite":
                continue
            sources = scenes[key]["sources"]
            count = len(sources)
            radial_distances = [
                math.hypot(source["x_um"], source["y_um"]) / 1e6 for source in sources
            ]
            rows.append(
                {
                    "case_key": key,
                    "source_count": count,
                    "seconds_per_source": float(row["seconds_per_source"]),
                    "directional_share": sum(source["kind"] == "directional" for source in sources) / count,
                    "mean_receive_radius_m": statistics.mean(
                        source["max_receive_um"] / 1e6 for source in sources
                    ),
                    "mean_radial_distance_m": statistics.mean(radial_distances),
                    "edge_source_share": sum(distance >= 1500.0 for distance in radial_distances) / count,
                }
            )
    if len(rows) != 1000:
        raise RuntimeError(f"Expected 1000 Q4 Lite records, found {len(rows)}")
    return rows


def adjusted_quartile_summaries(rows: list[dict], parameters: list[str]) -> list[dict]:
    overall = statistics.mean(row["seconds_per_source"] for row in rows)
    count_means = {
        count: statistics.mean(
            row["seconds_per_source"] for row in rows if row["source_count"] == count
        )
        for count in sorted({row["source_count"] for row in rows})
    }
    adjusted = np.asarray(
        [
            row["seconds_per_source"] - count_means[row["source_count"]] + overall
            for row in rows
        ],
        dtype=float,
    )
    result = []
    for parameter in parameters:
        values = np.asarray([row[parameter] for row in rows], dtype=float)
        ordered = np.argsort(values, kind="stable")
        for quartile, indices in enumerate(np.array_split(ordered, 4), start=1):
            sample = adjusted[indices]
            ci_low, ci_high = bootstrap_mean_ci(sample.tolist())
            result.append(
                {
                    "parameter": parameter,
                    "quartile": quartile,
                    "cases": len(indices),
                    "parameter_min": float(values[indices].min()),
                    "parameter_max": float(values[indices].max()),
                    "parameter_mean": float(values[indices].mean()),
                    "source_count_adjusted_mean_seconds_per_source": float(sample.mean()),
                    "adjusted_mean_ci95_low": ci_low,
                    "adjusted_mean_ci95_high": ci_high,
                }
            )
    return result


def partial_sensitivity(rows: list[dict], replicates: int = 10_000) -> list[dict]:
    parameters = ["directional_share", "mean_receive_radius_m", "mean_radial_distance_m"]
    values = {name: np.asarray([row[name] for row in rows], dtype=float) for name in parameters}
    medians = {name: float(np.median(values[name])) for name in parameters}
    iqrs = {
        name: float(np.quantile(values[name], 0.75) - np.quantile(values[name], 0.25))
        for name in parameters
    }
    y = np.asarray([row["seconds_per_source"] for row in rows], dtype=float)
    source_counts = np.asarray([row["source_count"] for row in rows], dtype=int)

    def matrix(indices: np.ndarray) -> np.ndarray:
        columns = [np.ones(len(indices))]
        for count in range(11, 17):
            columns.append((source_counts[indices] == count).astype(float))
        for name in parameters:
            columns.append((values[name][indices] - medians[name]) / iqrs[name])
        return np.column_stack(columns)

    all_indices = np.arange(len(rows))
    coefficients = np.linalg.lstsq(matrix(all_indices), y, rcond=None)[0][-len(parameters) :]
    boot = np.empty((replicates, len(parameters)), dtype=float)
    for iteration in range(replicates):
        indices = RNG.integers(0, len(rows), size=len(rows))
        boot[iteration] = np.linalg.lstsq(matrix(indices), y[indices], rcond=None)[0][
            -len(parameters) :
        ]
    intervals = np.quantile(boot, [0.025, 0.975], axis=0)
    return [
        {
            "parameter": name,
            "observed_iqr": iqrs[name],
            "adjusted_change_seconds_per_source_per_iqr": float(coefficients[index]),
            "bootstrap_ci95_low": float(intervals[0, index]),
            "bootstrap_ci95_high": float(intervals[1, index]),
        }
        for index, name in enumerate(parameters)
    ]


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def plot_parameter_sensitivity(q2_rows: list[dict], q3_geometry: list[dict], q3_summary: dict) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.45))
    ax = axes[0]
    for epsilon, color in zip([0.8, 1.0, 1.2], ["#4c78a8", "#f58518", "#e45756"]):
        selected = [row for row in q2_rows if row["bearing_half_error_deg"] == epsilon]
        ax.plot(
            [row["crossing_angle_deg"] for row in selected],
            [row["estimated_diameter_m"] for row in selected],
            label=fr"$\varepsilon={epsilon:.1f}^\circ$",
            color=color,
            linewidth=1.8,
        )
    ax.axvline(23.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.scatter([23.0], [2 * 1470 * math.radians(1.0) / math.sin(math.radians(23.0))],
               color="#111111", zorder=4, s=24)
    ax.set_xlim(8, 90)
    ax.set_ylim(0, 650)
    ax.set_xlabel(r"Crossing angle $\gamma$ (deg)")
    ax.set_ylabel(r"Estimated diameter $D$ (m)")
    ax.set_title("(a) Q2 angular sensitivity")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)

    ax = axes[1]
    radii = np.asarray([row["ring_radius_m"] for row in q3_geometry])
    distances = np.asarray([row["guaranteed_worst_distance_m"] for row in q3_geometry])
    ax.plot(radii, distances, color="#4c78a8", linewidth=2.0)
    ax.axhline(1000.0, color="#e45756", linestyle="--", linewidth=1.2, label="1000 m limit")
    ax.axvspan(
        q3_summary["feasible_ring_radius_low_m"],
        q3_summary["feasible_ring_radius_high_m"],
        color="#54a24b",
        alpha=0.16,
        label="guaranteed interval",
    )
    ax.axvline(q3_summary["minimax_ring_radius_m"], color="#555555", linestyle=":", linewidth=1.1)
    ax.scatter([999.0], [999.0], color="#111111", zorder=4, s=28, label="chosen $a=999$ m")
    ax.set_xlim(990, 1005)
    ax.set_ylim(996.5, 1006)
    ax.set_xlabel(r"Ring radius $a$ (m)")
    ax.set_ylabel("Worst guaranteed distance (m)")
    ax.set_title("(b) Q3 coverage-radius sensitivity")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.18)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(HERE / "model_parameter_sensitivity.pdf")
    fig.savefig(HERE / "model_parameter_sensitivity.png")
    plt.close(fig)


def plot_scenario_sensitivity(
    q3_counts: list[dict],
    q4_counts: list[dict],
    quartiles: list[dict],
    partial: list[dict],
) -> None:
    configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    for ax, summary, title, color in [
        (axes[0, 0], q3_counts, "(a) Q3 sensitivity to source count", "#4c78a8"),
        (axes[0, 1], q4_counts, "(b) Q4 sensitivity to source count", "#f58518"),
    ]:
        x = np.asarray([row["source_count"] for row in summary])
        y = np.asarray([row["mean_seconds_per_source"] for row in summary])
        low = np.asarray([row["mean_ci95_low"] for row in summary])
        high = np.asarray([row["mean_ci95_high"] for row in summary])
        ax.errorbar(x, y, yerr=[y - low, high - y], marker="o", color=color,
                    capsize=3, linewidth=1.8)
        ax.set_xticks(x)
        ax.set_xlabel("Number of sources")
        ax.set_ylabel("Mean task time (s/source)")
        ax.set_title(title)
        ax.grid(alpha=0.18)

    ax = axes[1, 0]
    directional = [row for row in quartiles if row["parameter"] == "directional_share"]
    x = np.arange(1, 5)
    y = np.asarray([row["source_count_adjusted_mean_seconds_per_source"] for row in directional])
    low = np.asarray([row["adjusted_mean_ci95_low"] for row in directional])
    high = np.asarray([row["adjusted_mean_ci95_high"] for row in directional])
    labels = [
        f"{row['parameter_min']:.2f}–{row['parameter_max']:.2f}" for row in directional
    ]
    ax.errorbar(x, y, yerr=[y - low, high - y], marker="o", color="#e45756",
                capsize=3, linewidth=1.8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Directional-source share (quartile range)")
    ax.set_ylabel("Source-count-adjusted time (s/source)")
    ax.set_title("(c) Q4 directional-share sensitivity")
    ax.grid(alpha=0.18)

    ax = axes[1, 1]
    labels_map = {
        "directional_share": "Directional share",
        "mean_receive_radius_m": "Mean receive radius",
        "mean_radial_distance_m": "Mean radial position",
    }
    positions = np.arange(len(partial))
    coefficients = np.asarray([row["adjusted_change_seconds_per_source_per_iqr"] for row in partial])
    low = np.asarray([row["bootstrap_ci95_low"] for row in partial])
    high = np.asarray([row["bootstrap_ci95_high"] for row in partial])
    ax.errorbar(coefficients, positions, xerr=[coefficients - low, high - coefficients],
                fmt="o", color="#4c78a8", capsize=3)
    ax.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_yticks(positions, [labels_map[row["parameter"]] for row in partial])
    ax.set_xlabel("Adjusted change per observed IQR (s/source)")
    ax.set_title("(d) Q4 partial scenario sensitivity")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    fig.savefig(HERE / "scenario_sensitivity.pdf")
    fig.savefig(HERE / "scenario_sensitivity.png")
    plt.close(fig)


def make_report(summary: dict) -> str:
    q2 = summary["q2"]
    q3 = summary["q3"]
    q4 = summary["q4"]
    q3_counts = {row["source_count"]: row for row in q3["source_count"]}
    q4_counts = {row["source_count"]: row for row in q4["source_count"]}
    directional = [row for row in q4["quartiles"] if row["parameter"] == "directional_share"]
    partial = {row["parameter"]: row for row in q4["partial"]}
    q3_drop = 100 * (q3_counts[10]["mean_seconds_per_source"] - q3_counts[16]["mean_seconds_per_source"]) / q3_counts[10]["mean_seconds_per_source"]
    q4_drop = 100 * (q4_counts[10]["mean_seconds_per_source"] - q4_counts[16]["mean_seconds_per_source"]) / q4_counts[10]["mean_seconds_per_source"]
    direction_change = directional[-1]["source_count_adjusted_mean_seconds_per_source"] - directional[0]["source_count_adjusted_mean_seconds_per_source"]
    return f"""# 参数与场景敏感性分析

本分析不比较或删除算法组件。问题二、三采用正文公式做确定性参数扰动；问题三、四的场景敏感性只读取既有固定策略记录，不重新选择方法或调参。

## 问题二：交会角与测角误差

使用正文式 $D\\approx 2(d_1+d_2)\\varepsilon/\\sin\\gamma$，取算例中的 $d_1=1000$ m、$d_2=470$ m。在 $\\gamma=23^\\circ$ 时，$\\varepsilon=0.8^\\circ,1.0^\\circ,1.2^\\circ$ 对应的估计直径分别为 {q2['epsilon_0.8_gamma_23']:.2f}、{q2['epsilon_1.0_gamma_23']:.2f}、{q2['epsilon_1.2_gamma_23']:.2f} m。固定 $\\varepsilon=1^\\circ$，交会角从 $20^\\circ$、$23^\\circ$ 到 $26^\\circ$ 时，估计直径依次为 {q2['epsilon_1.0_gamma_20']:.2f}、{q2['epsilon_1.0_gamma_23']:.2f}、{q2['epsilon_1.0_gamma_26']:.2f} m。

在 $23^\\circ$ 附近，$D$ 对 $\\varepsilon$ 的弹性为 1，对 $\\gamma$ 的局部弹性为 {q2['local_elasticity_gamma_at_23']:.3f}。也就是说，小幅测角误差变化会近似等比例传递到定位直径，而提高交会角能够以近似相同比例抵消这种影响。曲线同时表明，交会角很小时 $1/\\sin\\gamma$ 急剧增大，支持正文采用横向偏移而非沿首测方向共线前进。

## 问题三：覆盖环半径

七个环上检测点的最坏保证距离为

$$d_{{\\max}}(a)=\\max\\left(a,\\sqrt{{1800^2+a^2-3600a\\cos(\\pi/7)}}\\right).$$

该函数在 $a^*={q3['geometry']['minimax_ring_radius_m']:.3f}$ m 处取得最小值。保证 $d_{{\\max}}\\le1000$ m 的半径区间为 [{q3['geometry']['feasible_ring_radius_low_m']:.3f}, {q3['geometry']['feasible_ring_radius_high_m']:.3f}] m。正文选择 $a=999$ m 时，最坏距离为 {q3['geometry']['chosen_worst_distance_m']:.3f} m，仍有 {q3['geometry']['chosen_coverage_margin_m']:.3f} m 的解析余量，且距理论极小点不足 0.1 m。因此 999 m 并非任意经验值，而是接近最小化最坏覆盖距离的稳定选择。

## 问题三、四：源数敏感性

问题三400个既有本地场景中，源数从10增至16时，平均源均耗时从 {q3_counts[10]['mean_seconds_per_source']:.2f} 降至 {q3_counts[16]['mean_seconds_per_source']:.2f} s/源，下降 {q3_drop:.2f}%。问题四1000个固定 Lite 本地场景中，相应数值从 {q4_counts[10]['mean_seconds_per_source']:.2f} 降至 {q4_counts[16]['mean_seconds_per_source']:.2f} s/源，下降 {q4_drop:.2f}%。这不是源越多任务越简单，而是完整任务含有覆盖搜索和结束确认等固定成本；源数增加后，这些成本被更多源分摊。因此比较不同批次的“秒/源”时必须同时报告源数分布。

## 问题四：场景结构敏感性

为避免源数混杂，先在每个源数组内减去该组均值，再加回总体均值。定向源比例最低四分位的调整后均值为 {directional[0]['source_count_adjusted_mean_seconds_per_source']:.2f} s/源，最高四分位为 {directional[-1]['source_count_adjusted_mean_seconds_per_source']:.2f} s/源，增加 {direction_change:.2f} s/源。

再以源数固定效应控制源数，对定向源比例、平均接收半径、平均径向位置同时作线性局部敏感性估计。参数跨越本批数据的一个四分位距时：

- 定向源比例：{partial['directional_share']['adjusted_change_seconds_per_source_per_iqr']:+.2f} s/源，95% bootstrap 区间 [{partial['directional_share']['bootstrap_ci95_low']:+.2f}, {partial['directional_share']['bootstrap_ci95_high']:+.2f}]；
- 平均接收半径：{partial['mean_receive_radius_m']['adjusted_change_seconds_per_source_per_iqr']:+.2f} s/源，区间 [{partial['mean_receive_radius_m']['bootstrap_ci95_low']:+.2f}, {partial['mean_receive_radius_m']['bootstrap_ci95_high']:+.2f}]；
- 平均径向位置：{partial['mean_radial_distance_m']['adjusted_change_seconds_per_source_per_iqr']:+.2f} s/源，区间 [{partial['mean_radial_distance_m']['bootstrap_ci95_low']:+.2f}, {partial['mean_radial_distance_m']['bootstrap_ci95_high']:+.2f}]。

在本批自建场景内，定向源比例是最明显的场景难度因素；平均接收半径的区间跨零，未观察到稳定的线性影响；源整体更靠近边界时耗时略升。以上是固定本地生成分布上的关联，不是因果效应，也不能替代官方正式测试。

## 输出

- `model_parameter_sensitivity.pdf/png`：问题二交会角与问题三环半径；
- `scenario_sensitivity.pdf/png`：三、四问源数和问题四场景结构；
- 四份 CSV：逐点参数曲线、按源数统计、四分位统计与局部敏感性系数；
- `summary.json`：全部正文数值及边界说明。
"""


def make_tex(summary: dict) -> str:
    q2 = summary["q2"]
    q3 = summary["q3"]
    q4 = summary["q4"]
    q3_counts = {row["source_count"]: row for row in q3["source_count"]}
    q4_counts = {row["source_count"]: row for row in q4["source_count"]}
    directional = [row for row in q4["quartiles"] if row["parameter"] == "directional_share"]
    partial = {row["parameter"]: row for row in q4["partial"]}
    return rf"""% 由 output/sensitivity_analysis_20260913/make_sensitivity_analysis.py 生成。
% 不含算法组件消融；只使用解析扰动和固定策略的既有场景记录。
\section{{模型的敏感性分析}}\label{{sec:sensitivity}}

\subsection{{几何参数敏感性}}

问题二中，定位区域直径近似满足式\eqref{{eq:q2-2}}。取算例的
$d_1=1000\,\mathrm{{m}}$、$d_2=470\,\mathrm{{m}}$，当交会角
$\gamma=23^\circ$ 时，测角误差半角从 $0.8^\circ$ 增至
$1.0^\circ$、$1.2^\circ$，估计直径依次为
{q2['epsilon_0.8_gamma_23']:.2f}、{q2['epsilon_1.0_gamma_23']:.2f}、
{q2['epsilon_1.2_gamma_23']:.2f}\,m；固定 $\varepsilon=1^\circ$，
$\gamma=20^\circ,23^\circ,26^\circ$ 时，估计直径依次为
{q2['epsilon_1.0_gamma_20']:.2f}、{q2['epsilon_1.0_gamma_23']:.2f}、
{q2['epsilon_1.0_gamma_26']:.2f}\,m。由此可见，小幅测角误差近似等比例传递到定位直径，
而提高交会角能够明显抑制误差放大；当交会角很小时，
$1/\sin\gamma$ 急剧增大，支持第二问采用横向偏移形成交会角。

第三问七个环上检测点的最坏保证距离为
\[
d_{{\max}}(a)=\max\left(a,\sqrt{{1800^2+a^2-3600a\cos(\pi/7)}}\right).
\]
该函数在 $a^*={q3['geometry']['minimax_ring_radius_m']:.3f}\,\mathrm{{m}}$ 处最小；
满足 $d_{{\max}}\le1000\,\mathrm{{m}}$ 的半径区间为
$[{q3['geometry']['feasible_ring_radius_low_m']:.3f},\,{q3['geometry']['feasible_ring_radius_high_m']:.3f}]\,\mathrm{{m}}$。
本文取 $a=999\,\mathrm{{m}}$ 时最坏距离为
{q3['geometry']['chosen_worst_distance_m']:.3f}\,m，保留
{q3['geometry']['chosen_coverage_margin_m']:.3f}\,m 的解析余量，且与理论极小点十分接近。
因此该参数并非任意经验取值。

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\linewidth,height=0.42\textheight,keepaspectratio]{{figures/model_parameter_sensitivity.pdf}}
\caption{{几何参数敏感性。左图为问题二估计定位直径随交会角和测角误差的变化；右图为问题三七点环半径变化时的最坏保证接收距离。}}
\label{{fig:model-parameter-sensitivity}}
\end{{figure}}

\subsection{{场景参数敏感性}}

在问题三400个既有本地场景中，源数由10增至16时，平均源均耗时由
{q3_counts[10]['mean_seconds_per_source']:.2f} 降至
{q3_counts[16]['mean_seconds_per_source']:.2f}\,s/源；问题四1000个固定
V6 Lite 本地场景中，相应数值由
{q4_counts[10]['mean_seconds_per_source']:.2f} 降至
{q4_counts[16]['mean_seconds_per_source']:.2f}\,s/源。这并不表示源越多越容易，
而是覆盖搜索和结束确认等固定成本被更多源分摊。因此跨批次比较源均耗时时，
必须同时报告源数分布。

进一步对问题四按源数作组内中心化，以排除源数的混杂影响。定向源比例最低和最高四分位的
调整后均值分别为
{directional[0]['source_count_adjusted_mean_seconds_per_source']:.2f} 和
{directional[-1]['source_count_adjusted_mean_seconds_per_source']:.2f}\,s/源。
在同时控制源数、定向比例、平均接收半径和平均径向位置的局部线性分析中，
定向源比例跨越一个样本四分位距对应耗时变化
{partial['directional_share']['adjusted_change_seconds_per_source_per_iqr']:+.2f}\,s/源，
95\% bootstrap 区间为
$[{partial['directional_share']['bootstrap_ci95_low']:+.2f},\,
{partial['directional_share']['bootstrap_ci95_high']:+.2f}]$；
平均接收半径对应变化为
{partial['mean_receive_radius_m']['adjusted_change_seconds_per_source_per_iqr']:+.2f}\,s/源，
区间跨越零；平均径向位置对应变化为
{partial['mean_radial_distance_m']['adjusted_change_seconds_per_source_per_iqr']:+.2f}\,s/源。
说明在本批场景中，定向源比例是最明显的难度因素，接收半径的平均变化影响较弱，
源整体更靠近边界时耗时略有上升。

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\linewidth,height=0.63\textheight,keepaspectratio]{{figures/scenario_sensitivity.pdf}}
\caption{{场景参数敏感性。误差线为按场景 bootstrap 得到的95\%区间；问题四定向比例分组结果已按源数校正，右下图为控制源数后的局部敏感性系数。}}
\label{{fig:scenario-sensitivity}}
\end{{figure}}

上述结论只描述固定算法在自建场景生成分布内的敏感性，不将相关关系解释为因果关系，
也不与官方不同地图的演练成绩合并。
"""


def main() -> None:
    q2_rows, q2_summary = q2_parameter_sensitivity()
    q3_geometry, q3_geometry_summary = q3_geometric_sensitivity()
    q3_rows = load_q3_rows()
    q4_rows = load_q4_rows()
    q3_counts = grouped_summary(q3_rows, "source_count", "seconds_per_source")
    q4_counts = grouped_summary(q4_rows, "source_count", "seconds_per_source")
    parameters = [
        "directional_share",
        "mean_receive_radius_m",
        "mean_radial_distance_m",
        "edge_source_share",
    ]
    quartiles = adjusted_quartile_summaries(q4_rows, parameters)
    partial = partial_sensitivity(q4_rows)
    summary = {
        "analysis_kind": "parameter_and_scenario_sensitivity_without_component_ablation",
        "evidence_boundary": {
            "q2_q3_geometry": "deterministic analysis of equations used in the paper",
            "q3_scenarios": "400 existing local reconstructed scenarios, v3_origin20 only",
            "q4_scenarios": "1000 existing new-practice local reconstructed scenarios, V6 Lite only",
            "official_claim": "none; local results are not official or formal tests",
        },
        "q2": q2_summary,
        "q3": {"geometry": q3_geometry_summary, "source_count": q3_counts},
        "q4": {"source_count": q4_counts, "quartiles": quartiles, "partial": partial},
    }

    write_csv(
        HERE / "q2_angle_sensitivity.csv",
        ["bearing_half_error_deg", "crossing_angle_deg", "estimated_diameter_m"],
        q2_rows,
    )
    write_csv(
        HERE / "q3_ring_radius_sensitivity.csv",
        [
            "ring_radius_m",
            "centre_distance_m",
            "boundary_worst_distance_m",
            "guaranteed_worst_distance_m",
            "coverage_margin_m",
            "coverage_guaranteed",
        ],
        q3_geometry,
    )
    count_fields = [
        "source_count",
        "cases",
        "mean_seconds_per_source",
        "median_seconds_per_source",
        "p10_seconds_per_source",
        "p90_seconds_per_source",
        "mean_ci95_low",
        "mean_ci95_high",
    ]
    write_csv(HERE / "q3_source_count_sensitivity.csv", count_fields, q3_counts)
    write_csv(HERE / "q4_source_count_sensitivity.csv", count_fields, q4_counts)
    write_csv(
        HERE / "q4_scenario_quartiles.csv",
        [
            "parameter",
            "quartile",
            "cases",
            "parameter_min",
            "parameter_max",
            "parameter_mean",
            "source_count_adjusted_mean_seconds_per_source",
            "adjusted_mean_ci95_low",
            "adjusted_mean_ci95_high",
        ],
        quartiles,
    )
    write_csv(
        HERE / "q4_partial_sensitivity.csv",
        [
            "parameter",
            "observed_iqr",
            "adjusted_change_seconds_per_source_per_iqr",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
        ],
        partial,
    )
    (HERE / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_parameter_sensitivity(q2_rows, q3_geometry, q3_geometry_summary)
    plot_scenario_sensitivity(q3_counts, q4_counts, quartiles, partial)
    (HERE / "敏感性分析报告.md").write_text(make_report(summary), encoding="utf-8")
    (HERE / "sensitivity_section.tex").write_text(make_tex(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
