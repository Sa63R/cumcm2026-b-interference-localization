"""Audit saved local traces and create the user-facing experiment report."""

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from workflow.evidence import audit_findings
from experiments.run_study import source_manifest
from experiments.build_paper import table


def main():
    validation=ROOT/"results/validation"
    validation.mkdir(parents=True,exist_ok=True)
    raw=(validation/"pytest.txt").read_bytes()
    pytest=raw.decode("utf-16" if raw.startswith((b"\xff\xfe",b"\xfe\xff")) else "utf-8-sig")
    (validation/"pytest.txt").write_text(pytest,encoding="utf-8")
    count=re.search(r"(\d+) passed",pytest)
    if not count or re.search(r"\d+ failed|\d+ error",pytest):raise ValueError("Passing test evidence required")
    study=ROOT/"results/study"
    manifest=json.loads((study/"manifest.json").read_text(encoding="utf-8"))
    if manifest["source_changed_during_run"]:raise ValueError("Study source changed during execution")
    if manifest["source_sha256"]!=source_manifest():raise ValueError("Current solver differs from study provenance")
    with (study/"runs.csv").open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
    audited=0
    for row in rows:
        with gzip.open(study/row["trace"],"rt",encoding="utf-8") as stream:trace=json.load(stream)
        ev=trace["evaluation"];search=trace["search"]
        if ev["source_total"]!=int(row["source_total"]) or ev["cleared_total"]!=int(row["cleared_total"]):
            raise ValueError("CSV trace count mismatch")
        if not ev["all_cleared"] or not search["completion_certified_under_model"]:
            raise ValueError("Unsuccessful study case")
        if abs(ev["virtual_time_s"]-float(row["virtual_time_s"]))>1e-6:
            raise ValueError("CSV trace time mismatch")
        if abs(sum(ev["time_breakdown_s"].values())-ev["virtual_time_s"])>1e-6:
            raise ValueError("Time decomposition mismatch")
        truth={s["channel"]:s for s in ev["ground_truth"]["sources"]}
        for action in search["action_history"]:
            if action["action"]=="clear" and action["result"]=="success":
                source=truth[action["channel"]]
                if math.dist(action["position"],(source["x"],source["y"]))>20+1e-7:
                    raise ValueError("Clear action farther than 20 m from truth")
        audited+=1
    _,formal_findings=audit_findings(ROOT/"支撑材料/正式测试登记.csv")
    result={"passed":int(count.group(1)),"local_study_runs":len(rows),"trace_audits_passed":audited,
            "all_local_runs_cleared":True,"study_source_matches_current":True,
            "official_formal_ready":not formal_findings,"official_missing":formal_findings}
    (validation/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    summary=json.loads((study/"summary.json").read_text(encoding="utf-8"))["groups"]
    content=["# T06 本地实验报告", "",
        "本报告只记录依据题面自建物理引擎的研究实验。官方演练待界面与账号就绪后实测，六次正式测试不能由这里的数据替代。", "",
        f"每问100个随机案例（种子1000—1099）与7个困难案例，问题3比较3种策略、问题4比较4种策略，共{len(rows)}次运行；全部全清，逐局CSV与压缩轨迹均通过独立材料核对。", "",
        f"当前回归测试：{count.group(1)} passed。运行时求解源文件未发生变化，SHA-256与当前文件完全一致。", "",
        "## 随机案例", "",
        table(["问题","策略","局数/全清","总虚拟时间均值s","各局T/K均值s","合并ΣT/ΣK s","基准均值降幅"],
            [[g["problem"],g["strategy"],f"{g['runs']}/{g['all_clear_runs']}",f"{g['mean_virtual_time_s']:.3f}",
              f"{g['mean_case_time_per_cleared_s']:.3f}",f"{g['pooled_time_per_cleared_s']:.3f}",
              f"{g['relative_mean_time_reduction_pct']:.3f}%"] for g in summary if g["case_kind"]=="random"]), "",
        "默认问题3采用adaptive_center，问题4采用triangular。三角方案把覆盖点从45减到31，主要节省检测和频道切换；点更少不意味着覆盖路线更短。部分边界/最小接收半径案例仍比方格慢，选择依据为本地随机均值及覆盖可靠性。", "",
        "问题2的minimax选点减少了问题3的检测次数，但增加了移动，总虚拟时间未优于中心策略。这是保留对照而不作为默认策略的依据。", "",
        "## 困难案例", "",
        table(["问题","策略","全清/局数","各局T/K均值s","最大动作数"],
            [[g["problem"],g["strategy"],f"{g['all_clear_runs']}/{g['runs']}",
              f"{g['mean_case_time_per_cleared_s']:.3f}",g["max_actions"]] for g in summary if g["case_kind"]=="hard"]), "",
        "涵盖边界朝外、1000m接收半径、正/负极限误差、地点交替极限误差、不同频道密集源、近乎平行方向。困难例是压力测试，不按随机抽样解释。", "",
        "## 可追溯材料", "",
        "- `results/study/manifest.json`：随机规则、版本与源文件SHA-256。\n- `results/study/runs.csv`：全部逐局结果。\n- `results/study/summary.json`：分组指标、耗时分解、2000次成对自助抽样95%区间。\n- `results/study/traces/*.json.gz`：每局观测动作及结束后真值。\n- `results/validation/pytest.txt`：自动测试原始输出。\n- `results/validation/summary.json`：记录审计与正式缺项。\n- `论文/figures/`：直接从上述记录生成的五幅图。", "",
        "本地现实耗时是内存引擎加策略时间，不含官方网络延迟，不能填入正式表。总体生成分布为研究假设，不代表官方场景分布；100%本地成功不等于官方测试成功。", ""]
    (ROOT/"资料汇总/T06本地实验报告.md").write_text("\n".join(content),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":main()
