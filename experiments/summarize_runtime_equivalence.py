"""Read completed runtime batches; resample cases, never repeated timings.

No policy, simulator, network, or lower-bound solver is executed. Multiple
--batch arguments produce separate host-specific analyses, not pooled clocks.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.runtime_equivalence_benchmark import (
    BASELINE_COMMIT, HARNESS_FILES, SPEC_PATH, SUMMARY_RUNTIME_KEYS,
    compare_pair, digest, percentile, summarize, without_keys,
)

PARTITIONS = ("development", "independent_random", "independent_stress")
SIDES = ("baseline", "candidate")
METRICS = ("program_runtime_s", "program_cpu_s", "virtual_time_s")
BOOTSTRAP_DRAWS = 2000
STATISTICAL_SEED = 112607  # Only posthoc bootstrap; never used by a policy.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stats(values):
    return {"mean": statistics.mean(values), "p95": percentile(values, .95), "max": max(values)}


def restore_summary_channel_keys(summary):
    # SearchReport.source_estimates is populated with integer channel keys.
    # The worker hashes it before JSON serialization; JSON changes keys to
    # strings, whose sorted order differs (1, 10, 2 versus 1, 2, 10).
    restored = dict(summary)
    estimates = summary["source_estimates"]
    require(all(str(int(channel)) == channel for channel in estimates), "Unexpected source_estimates key schema")
    restored["source_estimates"] = {int(channel): value for channel, value in estimates.items()}
    return restored


def bootstrap_case_savings(baseline, candidate, *, draws=BOOTSTRAP_DRAWS, seed=STATISTICAL_SEED):
    """Inputs must already be paired per-case means of all repeat timings."""
    require(len(baseline) == len(candidate) and bool(baseline), "Empty or unpaired case means")
    differences = [b - c for b, c in zip(baseline, candidate)]
    rng = random.Random(seed)
    samples = [statistics.mean(differences[rng.randrange(len(differences))]
                               for _ in differences) for _ in range(draws)]
    return {"independent_sample_count": len(differences), "bootstrap_draws": draws,
            "statistical_seed": seed, "mean_saved_s": statistics.mean(differences),
            "mean_saved_s_percentile_95ci": [percentile(samples, .025), percentile(samples, .975)],
            "candidate_over_baseline_sum": sum(candidate) / sum(baseline),
            "candidate_sum_change_percent": 100 * (sum(candidate) / sum(baseline) - 1),
            "candidate_slower_case_count": sum(d < 0 for d in differences),
            "candidate_faster_case_count": sum(d > 0 for d in differences),
            "equal_case_count": sum(d == 0 for d in differences),
            "max_candidate_increment_s": max(c - b for b, c in zip(baseline, candidate)),
            "single_case_ci_is_not_generalization_evidence": len(differences) == 1}


def collapse_cases(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["case_id"]][row["side"]].append(row)
    cases = []
    for case_id, sides in sorted(grouped.items()):
        first = sides["baseline"][0]
        case = {"case_id": case_id, "case_sha256": first["case_sha256"],
                "split": first["split"], "repeats": len(sides["baseline"]),
                "physical_lower_bound_s": first["physical_lower_bound_s"]}
        for side in SIDES:
            case[side] = {metric: statistics.mean(row[metric] for row in sides[side]) for metric in METRICS}
            case[side].update(
                all_repeats_successful=all(row["successful"] for row in sides[side]),
                all_repeats_all_cleared=all(row["all_cleared"] for row in sides[side]),
                failed_clear_count_per_episode=statistics.mean(row["failed_clear_count"] for row in sides[side]),
                time_over_physical_lower_bound=statistics.mean(row["time_over_physical_lower_bound"] for row in sides[side]),
            )
        cases.append(case)
    return cases


def group_statistics(rows, cases):
    if not cases:
        return {"case_count": 0, "not_present": True}
    result = {"case_count": len(cases), "run_count": len(rows),
              "mean_physical_lower_bound_s": statistics.mean(c["physical_lower_bound_s"] for c in cases),
              "case_mean_statistics": {}, "individual_run_statistics": {}, "paired_savings": {}}
    for side in SIDES:
        selected = [row for row in rows if row["side"] == side]
        result["individual_run_statistics"][side] = {metric: stats([r[metric] for r in selected]) for metric in METRICS}
        result["case_mean_statistics"][side] = {
            **{metric: stats([c[side][metric] for c in cases]) for metric in METRICS},
            "success_rate": statistics.mean(c[side]["all_repeats_successful"] for c in cases),
            "all_clear_success_rate": statistics.mean(c[side]["all_repeats_all_cleared"] for c in cases),
            "failed_clear_count_all_repeats": sum(r["failed_clear_count"] for r in selected),
            "mean_failed_clear_count_per_episode": statistics.mean(c[side]["failed_clear_count_per_episode"] for c in cases),
            "mean_time_over_physical_lower_bound": statistics.mean(c[side]["time_over_physical_lower_bound"] for c in cases),
            "sum_time_over_sum_physical_lower_bound": sum(c[side]["virtual_time_s"] for c in cases) / sum(c["physical_lower_bound_s"] for c in cases),
        }
    for metric in METRICS[:2]:
        result["paired_savings"][metric] = bootstrap_case_savings(
            [c["baseline"][metric] for c in cases], [c["candidate"][metric] for c in cases])
    return result


def analyze_batch(batch):
    batch = Path(batch).resolve()
    inputs = {}

    def capture(name):
        path = (batch / name).resolve()
        require(batch in path.parents, "Input path escapes batch")
        raw = path.read_bytes()
        inputs[name] = hashlib.sha256(raw).hexdigest()
        return raw

    manifest = json.loads(capture("manifest.json"))
    summary = json.loads(capture("summary.json"))
    rows = json.loads(capture("runs_with_bounds.json"))
    bounds = json.loads(capture("lower_bounds.json"))
    inventory = json.loads(capture("cases.json"))
    initial_rows = [json.loads(line) for line in capture("runs.jsonl").decode("utf-8-sig").splitlines() if line.strip()]
    require(not (batch / "worker_failure.json").exists(), "Batch contains worker_failure.json")
    n, repeats = manifest["case_count"], manifest["repeats"]
    require(isinstance(n, int) and n > 0 and isinstance(repeats, int) and repeats > 0, "Invalid declared counts")
    require(len(rows) == len(initial_rows) == n * repeats * 2, "Incomplete run count")
    require(len(inventory) == n and digest(inventory) == manifest["cases_sha256"], "Case inventory digest/count mismatch")
    require(summary["case_count"] == n and summary["repeats"] == repeats and
            summary["paired_runs"] == n * repeats, "Summary counts disagree with manifest")
    require(manifest["balanced_ab_ba"] == summary["balanced_ab_ba"] == (repeats % 2 == 0), "AB/BA flag mismatch")
    require(summary["completed_at_utc"], "Missing batch completion marker")
    for side in SIDES:
        identity = manifest[side]
        require(identity["aggregate_sha256"] == digest(identity["source_sha256"]), "Source manifest digest mismatch")
        require(identity["spec_sha256"] == digest(manifest["spec"]), "Configuration digest mismatch")
    require(manifest["baseline"]["git_head"] == BASELINE_COMMIT, "Incorrect baseline commit")
    for name in HARNESS_FILES + [SPEC_PATH] + [p for p in manifest["baseline"]["source_sha256"]
                                              if p.startswith(("src/simulation/", "src/simulator_client/"))]:
        require(manifest["baseline"]["source_sha256"][name] == manifest["candidate"]["source_sha256"].get(name),
                "Physics/client/harness/spec source differs: " + name)

    indexed = {}
    for row, original in zip(rows, initial_rows):
        require({k: v for k, v in row.items() if k not in ("physical_lower_bound_s", "time_over_physical_lower_bound")} == original,
                "runs.jsonl differs from bound-augmented row")
        key = (row["case_id"], row["repeat"], row["side"])
        require(key not in indexed and row["side"] in SIDES and row["repeat"] in range(repeats), "Duplicate/invalid run key")
        require(row["split"] in PARTITIONS, "Unknown partition")
        for metric in METRICS:
            require(math.isfinite(row[metric]) and row[metric] >= 0, "Nonfinite/negative timing")
        lb = bounds[row["case_id"]]["physical_lower_bound_s"]
        require(math.isfinite(lb) and lb > 0 and row["physical_lower_bound_s"] == lb, "Invalid or inconsistent old lower bound")
        require(row["time_over_physical_lower_bound"] == row["virtual_time_s"] / lb, "Incorrect T/LB")
        raw = json.loads(gzip.decompress(capture(row["record_file"])))
        metadata = {"side", "repeat", "split", "record_file", "physical_lower_bound_s", "time_over_physical_lower_bound"}
        require(raw["runtime_equivalence_metrics"] == {k: v for k, v in row.items() if k not in metadata}, "Raw record metrics mismatch")
        actions = raw["summary"]["action_history"]
        expected = {
            "action_history_sha256": digest(actions), "action_step_sha256": [digest(a) for a in actions],
            "summary_without_runtime_sha256": digest(without_keys(restore_summary_channel_keys(raw["summary"]), SUMMARY_RUNTIME_KEYS)),
            "evaluation_without_runtime_sha256": digest(without_keys(raw["evaluation"], {"wall_time_s"})),
            "observations_without_timestamp_sha256": digest(without_keys(raw["history"], {"real_timestamp_ms"})),
            "row_without_runtime_sha256": digest(without_keys(raw["row"], {"program_runtime_s"})),
        }
        require(all(row[k] == value for k, value in expected.items()), "Raw action/observation/summary digest mismatch")
        require(all(row[k] == value for k, value in raw["row"].items()), "Raw episode row mismatch")
        indexed[key] = row
    case_ids = sorted({key[0] for key in indexed})
    require(len(case_ids) == n and set(bounds) == set(case_ids), "Case/bound count mismatch")
    require({(item["scenario"]["case_id"], item["split"]) for item in inventory} ==
            {(row["case_id"], row["split"]) for row in rows}, "Case inventory partition mismatch")
    reconstructed_pairs = []
    for case_id in case_ids:
        reference = indexed[(case_id, 0, "baseline")]
        for repeat in range(repeats):
            pair = [indexed[(case_id, repeat, side)] for side in SIDES]
            require(all(r["split"] == reference["split"] and r["case_sha256"] == reference["case_sha256"] for r in pair),
                    "Case identity or partition changes across repeats")
            reconstructed_pairs.append({"repeat": repeat, "case_id": case_id, **compare_pair(*pair)})
            for side in SIDES:
                require(compare_pair(indexed[(case_id, 0, side)], indexed[(case_id, repeat, side)])["identical"],
                        "Physical trajectory changes across timing repeats")
    pair_key = lambda p: (p["case_id"], p["repeat"])
    require(sorted(reconstructed_pairs, key=pair_key) == sorted(summary["pairs"], key=pair_key), "Pair records mismatch")
    recomputed = summarize(rows, summary["pairs"])
    for name in ("groups", "runtime_reductions", "all_pairs_exactly_identical", "different_pairs", "all_runs_successful"):
        require(summary[name] == recomputed[name], "Summary aggregate mismatch: " + name)
    expected_valid = summary["source_unchanged"] and recomputed["all_pairs_exactly_identical"] and recomputed["all_runs_successful"]
    require(summary["valid_runtime_only_result"] == expected_valid, "Summary validity flag mismatch")
    cases = collapse_cases(rows)
    return {"batch": str(batch), "manifest": manifest, "evidence_sha256": inputs,
            "case_count": n, "repeats": repeats, "completed_run_count": len(rows),
            "standard_54_cases_two_repeats": n == 54 and repeats == 2,
            "count_formula": f"{n} cases × {repeats} repeats × 2 implementations = {len(rows)} runs",
            "checks": {"integrity_consistent": True, "source_unchanged_reported_by_runner": summary["source_unchanged"],
                       "all_physical_pairs_identical": recomputed["all_pairs_exactly_identical"],
                       "all_virtual_pairs_exactly_equal": all(indexed[(c, r, "baseline")]["virtual_time_s"] ==
                           indexed[(c, r, "candidate")]["virtual_time_s"] for c in case_ids for r in range(repeats)),
                       "all_runs_successful": recomputed["all_runs_successful"], "valid_runtime_only_result": expected_valid},
            "partitions": {split: group_statistics([r for r in rows if split == "overall" or r["split"] == split],
                           [c for c in cases if split == "overall" or c["split"] == split]) for split in (*PARTITIONS, "overall")},
            "cases": cases}


def render(result):
    lines = ["# 等价实现运行时间复核", "",
             "本报告只读已完成记录。每案例先平均全部重复计时，再按案例配对重采样 2000 次；固定随机种子仅用于事后统计。不同批次不合并墙钟或 CPU。",
             "主表 P95/max 是案例平均值的分位数/最大值；每次运行的统计另存 JSON。节省为基准减候选，变化百分比为 100×(候选总和/基准总和−1)。", ""]
    for batch in result["batches"]:
        lines += [f"## {Path(batch['batch']).name}", "", f"完成数量：{batch['count_formula']}。标准完整批：{batch['standard_54_cases_two_repeats']}。",
                  f"来源：{batch['manifest']['platform']}；基准 `{batch['manifest']['baseline']['git_head']}`；候选 `{batch['manifest']['candidate']['git_head']}`。",
                  f"原始记录及派生表一致：{batch['checks']['integrity_consistent']}；全部配对物理轨迹相同：{batch['checks']['all_physical_pairs_identical']}；虚拟用时逐对相等：{batch['checks']['all_virtual_pairs_exactly_equal']}。", "",
                  "| 分区 | n | 指标 | 基准 mean / P95 / max (s) | 候选 mean / P95 / max (s) | 平均节省 95% CI (s) | 总和变化 | 变慢案例 / 最大增量(s) |",
                  "|---|---:|---|---|---|---|---:|---|"]
        for split, group in batch["partitions"].items():
            if not group["case_count"]:
                continue
            for metric, label in [("program_runtime_s", "墙钟"), ("program_cpu_s", "CPU")]:
                sides = [group["case_mean_statistics"][s][metric] for s in SIDES]
                cells = [" / ".join(f"{v[k]:.6f}" for k in ("mean", "p95", "max")) for v in sides]
                saving = group["paired_savings"][metric]
                lo, hi = saving["mean_saved_s_percentile_95ci"]
                lines.append(f"| {split} | {group['case_count']} | {label} | {cells[0]} | {cells[1]} | {saving['mean_saved_s']:.6f} [{lo:.6f}, {hi:.6f}] | {saving['candidate_sum_change_percent']:.3f}% | {saving['candidate_slower_case_count']} / {saving['max_candidate_increment_s']:.6f} |")
        lines += ["", "| 分区 | 实现 | 虚拟 mean / P95 / max (s) | 全部重复全清率 | 失败清除次数（全部重复） | 平均旧 LB (s) | mean(T/LB) | sum(T)/sum(LB) |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for split, group in batch["partitions"].items():
            if not group["case_count"]:
                continue
            for side in SIDES:
                value = group["case_mean_statistics"][side]
                timing = " / ".join(f"{value['virtual_time_s'][k]:.6f}" for k in ("mean", "p95", "max"))
                lines.append(f"| {split} | {side} | {timing} | {value['all_clear_success_rate']:.1%} | {value['failed_clear_count_all_repeats']} | {group['mean_physical_lower_bound_s']:.6f} | {value['mean_time_over_physical_lower_bound']:.9f} | {value['sum_time_over_sum_physical_lower_bound']:.9f} |")
        lines += ["", "计时范围（原 manifest）：" + batch["manifest"]["timing_scope"], ""]
    lines += ["旧 LB 使用全知的 20m 清除圆盘间松弛距离：起点扣 20m、源间扣 40m（沿用原数值裕量），精确子集 DP 最短开放路长度 / 5，再加每源 5s 清除；不含发现、定位、调谐或空频道认证，不是在线可达最优。mean(T/LB) 与 sum(T)/sum(LB) 分别计算，不能混称。", "",
              "这些是无 HTTP 延迟的本地合成研究案例，不是官方演练/正式成绩。墙钟受机器负载影响；偶发计时退步不能称为策略变差。有限案例轨迹相同不等于对全部输入的证明。Bootstrap 区间描述当前案例集合的抽样变动，不消除共同机器负载、压力场景设计或样本量小的限制；单案例退化区间不能外推。", "",
              "JSON 附输入及原始压缩记录 SHA256、完整 manifest、逐案例均值与所有分区统计。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    require(len({p.resolve() for p in args.batch}) == len(args.batch), "Duplicate batch path")
    require(not args.output.exists() and not args.report.exists(), "Refuse to overwrite existing evidence")
    result = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "sampling_unit": "case; all within-case repeat timings averaged before paired bootstrap",
              "bootstrap_draws": BOOTSTRAP_DRAWS, "statistical_seed": STATISTICAL_SEED,
              "batches": [analyze_batch(batch) for batch in args.batch]}
    for path in (args.output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.report.write_text(render(result), encoding="utf-8")
    print(json.dumps({"batches": len(result["batches"]), "output": str(args.output),
                      "checks": [b["checks"] for b in result["batches"]]}, ensure_ascii=False))
    return 0 if all(b["checks"]["valid_runtime_only_result"] for b in result["batches"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
