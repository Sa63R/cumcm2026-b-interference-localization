"""Post-termination audit and the original clairvoyant physical lower bound.

Reads explicitly named, FINISHED round2 batches only. No SQLite, HTTP, policy
execution, simulator construction or scenario generation. Existing independent
physical and prefix-certificate auditors are reused with dependency hashes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def load_helpers():
    theory = WORKSPACE / "q3-state-search/research/theory_v1"
    geometry_root = WORKSPACE / "q3-geometric"
    sys.path[:0] = [str(theory), str(geometry_root), str(geometry_root / "src")]
    loaded = []
    for name, path in (("_round2_physical_auditor", theory / "audit_eval_bounds.py"),
                       ("_round2_observation_auditor", geometry_root / "experiments/research_v1_physical_audit.py")):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded.append(module)
    paths = [theory / "audit_eval_bounds.py", theory / "certify_bounds.py",
             geometry_root / "experiments/research_v1_physical_audit.py"]
    for name in ("geometry", "localization", "localization.omni", "planning.disk_cover"):
        paths.append(Path(sys.modules[name].__file__))
    return *loaded, {str(path): sha(path) for path in paths}


def aggregate(rows):
    eligible = [row for row in rows if row["ratio_eligible"]]
    return {"runs": len(rows), "successful": sum(row["successful"] for row in rows),
            "audit_passed": sum(row["audit_passed"] for row in rows),
            "failed_clears": sum(row["failed_clear_count"] for row in rows),
            "ratio_eligible_runs": len(eligible),
            "mean_time_s": statistics.mean(row["virtual_time_s"] for row in rows),
            "mean_physical_lower_s": statistics.mean(row["physical_lower_s"] for row in eligible) if eligible else None,
            "mean_case_time_over_lower": statistics.mean(row["time_over_physical_lower"] for row in eligible) if eligible else None,
            "sum_time_over_sum_lower": (sum(row["virtual_time_s"] for row in eligible) /
                                       sum(row["physical_lower_s"] for row in eligible)) if eligible else None,
            "maximum_case_time_over_lower": max((row["time_over_physical_lower"] for row in eligible), default=None)}


def verify_batch(batch):
    manifest, summary = read(batch / "manifest.json"), read(batch / "summary.json")
    expected = {f"records/{policy}-{seed}.json.gz" for policy in manifest["policies"]
                for seed in manifest["seeds"]}
    if expected != set(summary["records_sha256"]):
        raise ValueError("Summary archive set is not the complete declared experiment")
    actual = {path.relative_to(batch).as_posix() for path in (batch / "records").glob("*.json.gz")}
    if actual != expected:
        raise ValueError("Extra or missing completed archive")
    runner_status, runner_path = "unavailable", None
    if (batch / "runner.py").is_file():
        if sha(batch / "runner.py") != manifest["runner_sha256"]:
            raise ValueError("Archived runner differs from frozen manifest")
        runner_status = "verified"
        runner_path = batch / "runner.py"
    else:
        # Early pilots archived the common runner once outside individual
        # batches. Only an exact content-hash match can supply that evidence.
        for path in sorted((ROOT / "research/round2/frozen_harnesses").glob("*.py")):
            if sha(path) == manifest["runner_sha256"]:
                runner_status, runner_path = "verified", path
                break
    for policy, identity in manifest["policies"].items():
        if digest(identity["spec"]) != identity["spec_sha256"]:
            raise ValueError("Frozen spec identity mismatch")
        archive = batch / f"{policy}-source.zip"
        if sha(archive) != identity["source_archive_sha256"]:
            raise ValueError("Frozen source archive identity mismatch")
        with zipfile.ZipFile(archive) as source:
            actual_sources = {name for name in source.namelist() if name.startswith("src/") and name.endswith(".py")}
            if actual_sources != set(identity["source_sha256"]):
                raise ValueError("Frozen source set is incomplete")
            for name, expected_hash in identity["source_sha256"].items():
                if hashlib.sha256(source.read(name)).hexdigest() != expected_hash:
                    raise ValueError("Frozen source file hash mismatch")
            for name, expected_hash in identity.get("evaluation_helper_sha256", {}).items():
                if hashlib.sha256(source.read(name)).hexdigest() != expected_hash:
                    raise ValueError("Frozen evaluation interface hash mismatch")
    return manifest, summary, runner_status, runner_path


def build(batches, seed_caches=()):
    physical, observations, dependency_hashes = load_helpers()
    observation_extensions, extension_hashes = {}, {}
    oracle_checks = physical.self_check()  # n=1..7, all permutations versus subset DP.
    # The two aggregate ratios are different estimands, not interchangeable.
    example = [{"ratio_eligible": True, "successful": True, "audit_passed": True,
                "failed_clear_count": 0, "virtual_time_s": t,
                "physical_lower_s": lower, "time_over_physical_lower": t / lower}
               for t, lower in ((2., 1.), (12., 3.))]
    assert aggregate(example)["mean_case_time_over_lower"] == 3.
    assert aggregate(example)["sum_time_over_sum_lower"] == 3.5
    started = time.perf_counter()
    rows, geometry_cache, cover_cache, case_identities, bounds = [], {}, {}, {}, {}
    cache_inputs = []
    for cache_path in seed_caches:
        cache = read(cache_path)
        if (cache.get("version") != "q3-round2-posthoc-original-physical-v1"
                or digest(cache["physical_cache"]) != cache["physical_cache_sha256"]):
            raise ValueError("Physical cache format/integrity mismatch")
        for dependency, dependency_hash in dependency_hashes.items():
            if cache["dependency_sha256"].get(dependency) != dependency_hash:
                raise ValueError("Physical cache was produced by different audit dependencies")
        for key, value in cache["physical_cache"].items():
            if key in geometry_cache and geometry_cache[key] != value:
                raise ValueError("Conflicting cached physical bounds")
            geometry_cache[key] = value
        cache_inputs.append({"path": str(cache_path), "sha256": sha(cache_path)})
    initial_cache_size = len(geometry_cache)
    batch_evidence = []
    for batch in batches:
        batch = batch.resolve(strict=True)
        manifest, summary, runner_status, runner_path = verify_batch(batch)
        if manifest["trial"] == "derived_silence" and "derived_silence" not in observation_extensions:
            # Keep this independent proof extension out of the physical-bound
            # cache identity: the original physical oracle is unchanged.
            extension_path = ROOT / "experiments/round2_derived_silence_audit.py"
            extension_spec = importlib.util.spec_from_file_location("_round2_derived_silence_auditor", extension_path)
            extension = importlib.util.module_from_spec(extension_spec)
            sys.modules[extension_spec.name] = extension
            extension_hashes[str(extension_path)] = sha(extension_path)
            extension_spec.loader.exec_module(extension)
            observation_extensions["derived_silence"] = extension
        if manifest["trial"] == "current_probe" and "current_probe" not in observation_extensions:
            extension_path = ROOT / "experiments/current_probe_prefix_audit.py"
            extension_spec = importlib.util.spec_from_file_location("_round2_current_probe_auditor", extension_path)
            extension = importlib.util.module_from_spec(extension_spec)
            sys.modules[extension_spec.name] = extension
            extension_hashes[str(extension_path)] = sha(extension_path)
            extension_spec.loader.exec_module(extension)
            observation_extensions["current_probe"] = extension
        label = f"{manifest['trial']}/{manifest['stage']}"
        summary_rows = {(row["strategy"], row["seed"]): row for row in summary["rows"]}
        if len(summary_rows) != len(summary["records_sha256"]):
            raise ValueError("Duplicate or incomplete summary rows")
        batch_evidence.append({"batch": label, "path": str(batch),
                               "manifest_sha256": sha(batch / "manifest.json"),
                               "summary_sha256": sha(batch / "summary.json"),
                               "runner_archive_status": runner_status,
                               "runner_archive_path": str(runner_path) if runner_path else None,
                               "frozen_runner_sha256": manifest["runner_sha256"]})
        for relative, expected_hash in sorted(summary["records_sha256"].items()):
            path = batch / relative
            if sha(path) != expected_hash:
                raise ValueError("Completed record changed after batch summary")
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                record = json.load(stream)
            raw = record["row"]
            policy, seed = raw["strategy"], raw["seed"]
            if (record["frozen_manifest_sha256"] != digest(manifest)
                    or raw != summary_rows[(policy, seed)]
                    or record["spec"] != manifest["policies"][policy]["spec"]):
                raise ValueError("Record identity differs from frozen experiment")
            identity = (raw["case_id"], raw["case_sha256"])
            if seed in case_identities and case_identities[seed] != identity:
                raise ValueError("Same evaluation seed does not refer to the same case")
            case_identities[seed] = identity
            errors = []
            observation_result = None
            physical_result = None
            movement_actions = None
            try:
                sources, movement_actions = physical.audit_record(record)
                if not raw["accepted_exit"] or record["history"][-1]["action"] != "/exit":
                    raise ValueError("Archive has not completed an explicit exit")
                geometry = [[c, sources[c]["x"], sources[c]["y"]] for c in sorted(sources)]
                geometry_key = digest(geometry)
                if geometry_key not in geometry_cache:
                    geometry_cache[geometry_key] = physical.physical_bounds(sources.values())
                physical_result = geometry_cache[geometry_key]
                bounds.setdefault(str(seed), {"case_id": raw["case_id"],
                    "case_sha256": raw["case_sha256"], "source_total": len(sources),
                    "geometry_key": geometry_key, **physical_result})
            except Exception as error:
                errors.append(f"physical: {type(error).__name__}: {error}")
            try:
                # This auditor reads only summary/history and checks the
                # maximum vertex distance from the actual clear point. It
                # therefore accepts valid lens points outside the MEC subdisk.
                extension = observation_extensions.get("derived_silence") if manifest["trial"] == "derived_silence" else None
                observation_result = (extension.observation_audit(record, cover_cache, observations)
                                      if extension else observations.observation_audit(record, cover_cache))
                if manifest["trial"] == "current_probe":
                    probe_audit = observation_extensions["current_probe"].audit_current_probe(
                        record, observation_result, legacy=observations)
                    for dependency, dependency_hash in probe_audit["dependency_sha256"].items():
                        if dependency in extension_hashes and extension_hashes[dependency] != dependency_hash:
                            raise ValueError("Current-probe proof dependency changed across records")
                        extension_hashes[dependency] = dependency_hash
                    # The extension returns the original observations object;
                    # remove the duplicate before nesting to avoid a cycle.
                    probe_audit.pop("observations", None)
                    observation_result["current_probe_prefix_audit"] = probe_audit
            except Exception as error:
                errors.append(f"causal_certificate: {type(error).__name__}: {error}")
            lower = physical_result["physical_clairvoyant_lower_s"] if physical_result else None
            rounding = (0.5e-6 * movement_actions + 1e-6) if movement_actions is not None else None
            machine_lower = max(0., lower - rounding) if lower is not None else None
            if raw["successful"] and machine_lower is not None and raw["virtual_time_s"] + 1e-8 < machine_lower:
                errors.append("Successful route is below its physical lower bound")
            eligible = bool(raw["successful"] and not errors and lower is not None)
            rows.append({"batch": label, "strategy": policy, "seed": seed,
                         "case_id": raw["case_id"], "case_sha256": raw["case_sha256"],
                         "source_total": raw["source_total"], "successful": raw["successful"],
                         "failed_clear_count": raw["failed_clear_count"],
                         "virtual_time_s": raw["virtual_time_s"],
                         "program_runtime_s": raw["program_runtime_s"],
                         "physical_lower_s": lower, "ratio_eligible": eligible,
                         "time_over_physical_lower": raw["virtual_time_s"] / lower if eligible else None,
                         "movement_rounding_allowance_s": rounding,
                         "machine_physical_lower_s": machine_lower,
                         "time_over_machine_physical_lower": raw["virtual_time_s"] / machine_lower if eligible else None,
                         "audit_passed": not errors, "errors": errors,
                         "observation_certificate_audit": observation_result,
                         "input_path": str(path), "input_sha256": expected_hash})
        print(json.dumps({"audited_batch": label, "records": len(summary["rows"])}), flush=True)
    groups = {f"{batch}:{policy}": aggregate([row for row in rows if row["batch"] == batch and row["strategy"] == policy])
              for batch, policy in sorted({(row["batch"], row["strategy"]) for row in rows})}
    for path, expected_hash in dependency_hashes.items():
        if sha(path) != expected_hash:
            raise ValueError("Audit dependency changed during execution")
    for path, expected_hash in extension_hashes.items():
        if sha(path) != expected_hash:
            raise ValueError("Observation proof extension changed during execution")
    return {"version": "q3-round2-posthoc-original-physical-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "truth_scope": "post-policy-termination archives only",
            "sqlite_read": False, "simulator_calls": 0, "new_simulations": 0,
            "exhaustive_small_graph_oracles": oracle_checks,
            "ratio_aggregation_oracle": True, "dependency_sha256": dependency_hashes,
            "observation_extension_sha256": extension_hashes,
            "script_sha256": sha(__file__), "batches": batch_evidence,
            "records": len(rows), "unique_cases": len(case_identities),
            "independent_physical_bound_calculations": len(geometry_cache) - initial_cache_size,
            "seed_cache_inputs": cache_inputs,
            "physical_cache": geometry_cache, "physical_cache_sha256": digest(geometry_cache),
            "all_runner_archives_available_and_verified": all(item["runner_archive_status"] == "verified" for item in batch_evidence),
            "all_audits_passed": all(row["audit_passed"] for row in rows),
            "wall_s": time.perf_counter() - started,
            "formula": "LB = min_permutation (max(0,|s_first|-20-eps) + sum max(0,|s_i-s_j|-40-eps))/5 + 5*N; eps=1e-6 m",
            "lower_scope": "Clairvoyant physical relaxation; not attainable online minimum or continuous disk-route exact optimum",
            "groups": groups, "bounds_by_seed": bounds, "rows": rows}


def markdown(result):
    lines = ["# 问题3第二轮：已完成试验的独立审核与旧口径下界", "",
             f"本次只读{result['records']}份已完成归档，对应{result['unique_cases']}个不同场景。"
             "不同试验如使用同一组场景，基准重复运行不能当成新增独立场景。未读取验证SQLite，未启动新仿真或官方测试。", "",
             f"全部物理与因果证书审核通过：{result['all_audits_passed']}。下界新计算{result['independent_physical_bound_calculations']}组，供同局不同方案复用。", "",
             f"历史runner文件全部可核验：{result['all_runner_archives_available_and_verified']}。若为False，说明旧批次只留runner哈希而没有原文件；不能宣称这项归档核验已完成，源代码zip、spec和实际轨迹仍单独核验。", "",
             "## 下界口径", "",
             "沿用物理先知下界。事后知道N个源中心s_i，清除点只需进入各源半径20米圆盘；初始位置为原点，不要求返回原点。", "",
             "- 初始边：a_i=max(0,‖s_i‖−20−ε)。", "- 源间边：d_ij=max(0,‖s_i−s_j‖−40−ε)，ε=10⁻⁶米。",
             "- 用子集动态规划精确求这些边构成的最短开放Hamilton路径L。",
             "- LB=L/5+5N秒；移动速度5米/秒，每源必须检查3秒、成功移除2秒。", "",
             "每段真实移动长度不少于对应圆盘间最短距离，所以图路径给出下界。不同边允许选择不一致的圆盘端点，故图DP精确不意味着连续圆盘访问问题已经精确求解。它还省略了在线发现、测向、排除遗漏和换频道，不能把T/LB直接解释为相对可达最优策略的近似比。", "",
             "主列保持旧连续口径；JSON另给出减去0.5微秒×移动动作数+1微秒后的机器计费下界，以覆盖逐步四舍五入。两者只有微秒量级差异。源中心路线给出的先知上界也存于JSON，但不是在线可行上界。", "",
             "## 汇总", "", "逐局均比值=(1/n)Σ(T_i/LB_i)，给每局相同权重；总量比=ΣT_i/ΣLB_i，相当于按LB_i加权的逐局比值。两者分别报告，不能互换。", "",
             "| 批次/策略 | 完成/记录 | 审核通过 | 平均T/秒 | 平均LB/秒 | 逐局均T/LB | 总T/总LB |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, group in result["groups"].items():
        values = [group[key] for key in ("mean_time_s", "mean_physical_lower_s", "mean_case_time_over_lower", "sum_time_over_sum_lower")]
        cells = [f"{value:.6f}" if value is not None else "不适用" for value in values]
        lines.append(f"| {name} | {group['successful']}/{group['runs']} | {group['audit_passed']} | " + " | ".join(cells) + " |")
    columns = sorted({(row["batch"], row["strategy"]) for row in result["rows"]},
                     key=lambda item: (item[0], item[1] != "baseline", item[1]))
    lines.extend(["", "## 每局复用同一下界", "",
                  "仅列出本次实际审核的批次和策略；不同批次的重复基准分别保留。同一场景复用同一个下界。每个结果格依次为T（秒）/T÷LB，未运行的组合明确标记。", "",
                  "| 场景seed | N | L/米 | LB/秒 | " + " | ".join(f"{batch}/{policy} T / 比值" for batch, policy in columns) + " |",
                  "| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in columns) + " |"])
    for seed, bound in sorted(result["bounds_by_seed"].items(), key=lambda item: int(item[0])):
        candidates = []
        for batch, policy in columns:
            choices = [row for row in result["rows"] if row["seed"] == int(seed)
                       and row["strategy"] == policy and row["batch"] == batch]
            if len(choices) > 1:
                raise ValueError("Duplicate case/batch/policy while rendering")
            row = choices[0] if choices else None
            candidates.append(f"{row['virtual_time_s']:.6f} / {row['time_over_physical_lower']:.6f}"
                              if row and row["ratio_eligible"] else "未通过审核或不可比" if row else "未运行")
        lines.append(f"| {seed} | {bound['source_total']} | {bound['source_route_lower_m']:.6f} | {bound['physical_clairvoyant_lower_s']:.6f} | " + " | ".join(candidates) + " |")
    lines.extend(["", "## 独立审核内容与限制", "",
                  "- 校验已存在的归档runner、源代码zip及逐文件哈希、冻结spec、manifest、完整案例×策略集合、summary与各原始记录一致性；缺失旧runner单独披露。",
                  "- 复用audit_record：事后场景哈希、物理测量反馈、误差包络、清除距离、每步微秒账本和汇总计数。",
                  "- 复用observation_audit：逐步只用当时观测重建区域，核验推断静默的1500米距离证书、真实测量覆盖、最终无遗漏证书与16源上限终止。",
                  "- 清除证书直接检查实际清除点到所有候选顶点的最大距离，并结合既往near的5米圆。lens的清除点可以位于最小包围圆内部安全圆之外，只要直接顶点距离证书成立。",
                  "- 图DP通过1至7节点的全部排列对照；两种比值汇总通过人工算例检查。",
                  "- 本次批次及阶段以表内名称为准；pilot与开发诊断不能作为独立确认或最优性证明。全部记录保留，失败或审计不通过的记录不能伪装成有效完整清除T/LB。", "",
                  "## 渲染来源", "",
                  f"原始JSON中的审计脚本SHA-256：`{result['script_sha256']}`。",
                  f"本次Markdown渲染脚本SHA-256：`{sha(__file__)}`。",
                  "若二者不同，表示仅用更新后的显示模板重新渲染已有JSON；没有重算实验或下界，也没有修改原JSON的审计哈希。", ""])
    if any("round2_derived_silence_audit.py" in path for path in result.get("observation_extension_sha256", {})) and any(row["batch"].startswith("derived_silence/") for row in result["rows"]):
        lines.extend(["", "本次还加载了单独记录哈希的冗余测量证书审核器：以精确有理数复核相对距离不等式、此前同频道真实无信号反馈、扫描过程中的16源上限及实际覆盖计数；删除新增推断后，原审核器仍须独立通过清除与终止证书。物理下界及其缓存依赖没有改变。", ""])
    if any(row["batch"].startswith("current_probe/") for row in result["rows"]):
        lines.extend(["", "本次逐条重建当前位置探测的实际观测前缀，核验目标频道去重、接收证书、原候选集合及评分、所选位置和下一实际动作收费。独立实现及纯几何依赖哈希另行记录；原清除、终止及物理下界审核保持有效。", ""])
    errors = [row for row in result["rows"] if row["errors"]]
    if errors:
        lines += ["审核错误：", ""] + [f"- {row['batch']}/{row['strategy']}/{row['seed']}: {row['errors']}" for row in errors]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=Path, default=[ROOT / "results/round2/feedback/pilot", ROOT / "results/round2/clear_lens/pilot"])
    parser.add_argument("--output", type=Path, default=ROOT / "research/round2/pilot_audit.json")
    parser.add_argument("--report", type=Path, default=ROOT / "research/round2/pilot_audit.md")
    parser.add_argument("--seed-cache", nargs="*", type=Path, default=[], help="Read verified prior audit physical_cache entries without rewriting them")
    args = parser.parse_args(argv)
    if args.output.exists() or args.report.exists():
        raise ValueError("Use new output paths; do not overwrite an earlier audit")
    result = build(args.batches, args.seed_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"records": result["records"], "unique_cases": result["unique_cases"],
                      "all_audits_passed": result["all_audits_passed"], "groups": result["groups"]}))
    return int(not result["all_audits_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
