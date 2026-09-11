"""Read completed evidence only; no policy calls, simulation, database or network.

Regenerate the handoff index while verifying raw-record and audit identities.
The first-version projection explicitly uses the same physical LB as round two,
not its separately published (stronger) conditional-certification bound.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/round2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    inputs, audited = {}, {}
    for path in sorted(OUT.glob("*audit.json")):
        doc = read(path)
        if not isinstance(doc, dict) or doc.get("version") != "q3-round2-posthoc-original-physical-v1":
            continue
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        for row in doc["rows"]:
            key = (row["batch"], row["strategy"], row["seed"])
            if key in audited and audited[key]["input_sha256"] != row["input_sha256"]:
                raise ValueError(f"Conflicting audit identity: {key}")
            audited[key] = row
    batches, cases, records = [], set(), 0
    for summary_path in sorted((ROOT / "results/round2").glob("*/*/summary.json")):
        summary = read(summary_path)
        if "comparisons" not in summary:
            continue
        directory = summary_path.parent
        manifest_path = directory / "manifest.json"
        manifest = read(manifest_path)
        label = f"{manifest['trial']}/{manifest['stage']}"
        inputs[summary_path.relative_to(ROOT).as_posix()] = sha(summary_path)
        inputs[manifest_path.relative_to(ROOT).as_posix()] = sha(manifest_path)
        assert len(summary["rows"]) == len(manifest["seeds"]) * len(manifest["policies"])
        by_policy = defaultdict(list)
        keys = set()
        for row in summary["rows"]:
            policy, seed = row["strategy"], row["seed"]
            key = (label, policy, seed)
            assert key not in keys
            keys.add(key)
            audit = audited[key]
            relative = f"records/{policy}-{seed}.json.gz"
            expected = summary["records_sha256"][relative]
            assert sha(directory / relative) == expected == audit["input_sha256"]
            assert row["case_sha256"] == audit["case_sha256"]
            assert row["virtual_time_s"] == audit["virtual_time_s"]
            assert audit["audit_passed"] and audit["ratio_eligible"]
            by_policy[policy].append(audit)
            cases.add((row["case_id"], row["case_sha256"]))
            records += 1
        groups = {}
        for policy, rows in by_policy.items():
            groups[policy] = {
                **summary["averages"][policy],
                "mean_physical_lower_s": mean(r["physical_lower_s"] for r in rows),
                "mean_case_time_over_lower": mean(r["time_over_physical_lower"] for r in rows),
                "sum_time_over_sum_lower": sum(r["virtual_time_s"] for r in rows)
                                          / sum(r["physical_lower_s"] for r in rows),
                "source_commit": manifest["policies"][policy]["commit"],
                "spec": manifest["policies"][policy]["spec"],
            }
        batches.append({"batch": label, "cases": len(manifest["seeds"]),
                        "records": len(summary["rows"]), "groups": groups,
                        "comparisons": summary["comparisons"],
                        "directory": directory.relative_to(ROOT).as_posix()})
    first_path = ROOT.parent / "q3-v1-artifacts/final-physical-bounds-audit.json"
    first = read(first_path)
    assert first["audit_passed"] and first["records"] == 1136
    first_groups = defaultdict(list)
    first_rows = []
    for row in first["rows"]:
        assert row["audit_passed"] and row["successful"]
        lb = row["physical_bounds"]["physical_clairvoyant_lower_s"]
        projected = {k: row[k] for k in ("label", "seed", "case_id", "case_sha256", "input_sha256", "virtual_time_s")}
        projected.update(physical_lower_s=lb, time_over_physical_lower=row["virtual_time_s"] / lb)
        first_rows.append(projected)
        first_groups[row["label"]].append(projected)
    original = {}
    for label, rows in first_groups.items():
        original[label] = {
            "runs": len(rows), "mean_time_s": mean(r["virtual_time_s"] for r in rows),
            "mean_physical_lower_s": mean(r["physical_lower_s"] for r in rows),
            "mean_case_time_over_lower": mean(r["time_over_physical_lower"] for r in rows),
            "sum_time_over_sum_lower": sum(r["virtual_time_s"] for r in rows) / sum(r["physical_lower_s"] for r in rows),
        }
    return {"kind": "state-search-research-handoff-index-v1",
            "new_simulations": 0, "official_requests": 0, "sqlite_read": False,
            "physical_lower_formula": "LB = open source-disk edge-DP length / 5 + 5*N; epsilon=1e-6 m",
            "round2_records_verified": records, "round2_unique_cases": len(cases),
            "round2_batches": batches, "input_sha256": inputs,
            "script_sha256": sha(Path(__file__)),
            "first_version": {"source": "../q3-v1-artifacts/final-physical-bounds-audit.json",
                              "source_sha256": sha(first_path), "groups": original, "rows": first_rows},
            "scope": "Historical paired evidence only. Stage counts are separate; records are not independent cases. No new claim about current RL."}


def render(result):
    lines = ["# 状态搜索研究：完整量化索引", "",
             f"第二轮共 {len(result['round2_batches'])} 个完整批次、{result['round2_records_verified']} 条实际运行记录、"
             f"{result['round2_unique_cases']} 个按案例标识与哈希去重的场景。已逐条核验原始gzip、summary和独立审计的哈希及用时。",
             "这些记录含同案例多方法及不同阶段，不应合并成一个总体估计；初步开发、压力与最终随机的证据等级不同。", "",
             "本页统一用原物理先知下界 LB；均比是逐局T/LB平均，总量比是ΣT/ΣLB。所有表格单位秒。", "",
             "| 批次 | 方法 | 全清/局数 | 失败清除 | 均T | P95 | max | 均LB | 均比 | 总量比 | CPU均值 | 墙钟均值 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for batch in result["round2_batches"]:
        for policy, g in batch["groups"].items():
            cpu = f"{g['mean_cpu_s']:.6f}" if "mean_cpu_s" in g else "未单列"
            lines.append(f"| {batch['batch']} | {policy} | {g['successful']}/{g['runs']} | {g['failed_clears']} | "
                         f"{g['mean_s']:.6f} | {g['p95_s']:.6f} | {g['max_s']:.6f} | {g['mean_physical_lower_s']:.6f} | "
                         f"{g['mean_case_time_over_lower']:.6f} | {g['sum_time_over_sum_lower']:.6f} | {cpu} | {g['mean_runtime_s']:.6f} |")
    lines += ["", "## 每个批次自身的配对结果", "", "正节省表示候选更快；不同案例集之间不能按均值直接排名。", "",
              "| 批次/候选 | 对数 | 平均节省 | 95%配对区间 | 胜/平/负 | 最坏配对退化 | 扩样门槛 |",
              "| --- | ---: | ---: | --- | --- | ---: | --- |"]
    for batch in result["round2_batches"]:
        for policy, c in batch["comparisons"].items():
            lo, hi = c["ci95_saved_s"]
            lines.append(f"| {batch['batch']}/{policy} | {c['pairs']} | {c['mean_saved_s']:.6f} | [{lo:.6f}, {hi:.6f}] | "
                         f"{c['wins']}/{c['ties']}/{c['losses']} | {c['worst_paired_regression_s']:.6f} | {c['pilot_expansion_gate']} |")
    lines += ["", "最终是否推荐还要看阶段顺序、独立确认与最终复核，不能只看此处通用计算的扩样布尔值。", "",
              "## 首版共同最终评测，另列，统一回旧物理下界", "",
              "从首版已完成物理审计提取，不重新跑案例。首版报告中的1.5920等使用额外认证费用的更强下界，与下表分母不同。", "",
              "| 分组/方法 | 局数 | 均T | 均LB | 均比 | 总量比 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, g in result["first_version"]["groups"].items():
        lines.append(f"| {label} | {g['runs']} | {g['mean_time_s']:.6f} | {g['mean_physical_lower_s']:.6f} | "
                     f"{g['mean_case_time_over_lower']:.6f} | {g['sum_time_over_sum_lower']:.6f} |")
    lines += ["", "首版历史RL是GAE=.95/u512冻结版本，不能代表当前正在改进的强化学习版本。", "",
              "复现：在q3-round2运行 `python -m experiments.summarize_state_search_round2`。仅重算索引，不启动策略。完整逐局投影、分组配置、输入和脚本哈希见同名JSON。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    result = build()
    (OUT / "STATE_SEARCH_EXPERIMENT_INDEX.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "STATE_SEARCH_EXPERIMENT_INDEX.md").write_text(render(result), encoding="utf-8")
    print(json.dumps({"batches": len(result["round2_batches"]), "records": result["round2_records_verified"],
                      "unique_cases": result["round2_unique_cases"], "first_version": result["first_version"]["groups"]}, ensure_ascii=True))
