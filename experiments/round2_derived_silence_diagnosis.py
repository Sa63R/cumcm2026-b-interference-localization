"""Read completed four-arm archives and their exact-record physical/prefix audit.

No strategy, simulator, scenario generation, SQLite or network is used. A trace
deletion certificate describes only the supplied paired actions, not a general
policy dominance theorem. All outputs are exclusive new files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.round2_posthoc_audit import digest, read, sha, verify_batch

POLICIES = ("baseline","relative_only","cap_only","derived_silence")
COSTS = ("movement_s","switching_s","detection_s","optical_s","removal_s")
ROOT = Path(__file__).resolve().parents[1]


def percentile(values,p):
    values = sorted(values)
    x = (len(values)-1)*p
    lo = int(x)
    return values[lo]+(x-lo)*(values[min(lo+1,len(values)-1)]-values[lo])


def signature(action):
    return (action["action"],action["channel"],tuple(action["position"]),
            action["result"],action.get("bearing_deg"))


def query_key(action):
    return (action["channel"],tuple(action["position"]))


def ledger(actions, *, check_times=False):
    """Independent integer-microsecond billing; only measure retunes receiver."""
    totals = dict.fromkeys(COSTS,0)
    position,channel = (0.,0.),1
    rows = []
    for index,action in enumerate(actions):
        target = action["position"]
        if (action["action"] not in ("measure","clear") or type(action["channel"]) is not int
                or not 1 <= action["channel"] <= 20 or len(target) != 2
                or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in target)):
            raise ValueError("Invalid physical action")
        costs = dict.fromkeys(COSTS,0)
        costs["movement_s"] = round(math.dist(position,target)/5*1_000_000)
        if action["action"] == "measure":
            if action["result"] not in ("no_signal","near","direction"):
                raise ValueError("Invalid measurement outcome")
            costs["switching_s"] = int(action["channel"] != channel)*1_000_000
            costs["detection_s"] = 5_000_000
            channel = action["channel"]
        else:
            if action["result"] not in ("success","no_target_in_range"):
                raise ValueError("Invalid clear outcome")
            costs["optical_s"] = 3_000_000
            costs["removal_s"] = int(action["result"] == "success")*2_000_000
        for name,value in costs.items():
            totals[name] += value
        time_s = sum(totals.values())/1_000_000
        if check_times and action["virtual_time_s"] != time_s:
            raise ValueError(f"Physical billing mismatch at action {index}")
        rows.append({"index_0based":index,**{k:v/1_000_000 for k,v in costs.items()},
                     "virtual_time_s":time_s,"receiver_channel_after":channel})
        position = target
    return {"virtual_time_s":sum(totals.values())/1_000_000,
            "time_breakdown":{k:v/1_000_000 for k,v in totals.items()},"actions":rows,
            "measurements":sum(a["action"] == "measure" for a in actions),
            "clear_attempts":sum(a["action"] == "clear" for a in actions),
            "failed_clears":sum(a["action"] == "clear" and a["result"] != "success" for a in actions),
            "receiver_channel_after":channel}


def physical_record(record):
    raw,summary = record["history"],record["summary"]
    if (not summary or len(raw)<2 or raw[0]["action"] != "/enter" or raw[-1]["action"] != "/exit"
            or raw[0]["response"]["virtual_time_s"] != 0
            or any(a["response"].get("accepted") is not True for a in raw)):
        raise ValueError("Only completed accepted enter/action/exit histories are supported")
    actions = summary["action_history"]
    if len(raw) != len(actions)+2:
        raise ValueError("Physical versus policy history length mismatch")
    for actual,reported in zip(raw[1:-1],actions):
        response = actual["response"]
        observed = {"action":actual["action"].removeprefix("/"),"channel":actual["channel"],
            "position":[actual["position"]["x"],actual["position"]["y"]],
            "result":response.get("measure_result",response.get("clear_result")),
            "bearing_deg":response.get("svd_deg")}
        if (signature(observed) != signature(reported) or response["virtual_time_s"] != reported["virtual_time_s"]
                or actual.get("physical_measurement",True) is not True or reported.get("physical_measurement",True) is not True):
            raise ValueError("Actual physical feedback differs from policy log or is fabricated inference")
    billed = ledger(actions,check_times=True)
    if billed["virtual_time_s"] != summary["virtual_time_s"] or billed["virtual_time_s"] != raw[-1]["response"]["virtual_time_s"]:
        raise ValueError("Exit/summary time differs from complete physical ledger")
    for name in COSTS:
        # Client movement accumulates unrounded distances, while engine bills each segment.
        tolerance = len(actions)*0.5e-6+1e-6 if name == "movement_s" else 1e-9
        if abs(summary["time_breakdown"][name]-billed["time_breakdown"][name])>tolerance:
            raise ValueError("Summary time breakdown differs from complete physical ledger")
    if (summary["measurement_count"] != billed["measurements"] or summary["clear_attempt_count"] != billed["clear_attempts"]
            or summary["accepted_actions"] != len(raw)):
        raise ValueError("Summary action counters differ from complete physical ledger")
    return actions,billed


def certified_omissions(summary,observation_audit):
    """Use exact-record, passed independent certificates; never infer from truth."""
    if (observation_audit.get("derived_audit_version") != "q3-derived-silence-prefix-audit-v1"
            or observation_audit.get("legacy_without_relative_constraints_passed") is not True
            or observation_audit.get("relative_inferred_coverage_credits") != 0
            or observation_audit.get("inferred_coverage_credits") != 0):
        raise ValueError("Missing unweakened derived-silence prefix audit")
    params = summary.get("strategy_parameters",{})
    events = params.get("inferred_no_signal_constraints",[])
    scans = params.get("derived_scan_audit",[])
    relative = [e for e in events if e.get("method") == "relative_actual_negative"]
    checks = observation_audit.get("relative_certificates",[])
    expected = Counter((e["after_actual_action_count"],e["channel"],e["witness_action_ordinal"]) for e in relative)
    actual = Counter((e["after_actual_action_count"],e["channel"],e["witness_action_ordinal"]) for e in checks
                     if e.get("exact_binary64_vertex_certificate") is True and e.get("coverage_credits") == 0)
    if (expected != actual or len(relative) != observation_audit["relative_inferred_silence_verified"]
            or len(events)-len(relative) != observation_audit["legacy_inferred_silence_verified"]):
        raise ValueError("Inference events differ from the bound independent audit")
    proofs = defaultdict(list)
    for ordinal,e in enumerate(events):
        if e.get("physical_measurement") is not False:
            raise ValueError("Inference claims physical action")
        proofs[e["after_actual_action_count"]].append({"channel":e["channel"],"position":e["position"],
            "kind":"relative" if e in relative else "legacy_silence","event_ordinal":ordinal,
            "certificate_sha256":digest(e)})
    count_skips = 0
    for scan_index,scan in enumerate(scans):
        for event in scan["count_skipped"]:
            count_skips += 1
            proofs[event["after_actual_action_count"]].append({"channel":event["channel"],"position":scan["position"],
                "kind":"source_count_cap","scan_index":scan_index,"certificate_sha256":digest(event)})
    totals = observation_audit["derived_scan_totals"]
    if (totals["relative_silence_skips"] != len(relative) or totals["count_cap_skips"] != count_skips
            or len(scans) != len(observation_audit["derived_scan_checks"])):
        raise ValueError("Skip counts differ from independently checked scan schedule")
    for prefix in proofs:
        if type(prefix) is not int or not 0 <= prefix <= len(summary["action_history"]):
            raise ValueError("Omission certificate is outside the actual prefix")
    counts = {**totals,"legacy_inferred_skips":len(events)-len(relative),
        "reported_clear_certified_skips":params.get("skipped_certified_measurements",0),
        "scan_clear_certified_skips":sum(len(s["clear_certified_skipped"]) for s in scans),
        "new_mechanism_omissions":len(relative)+count_skips}
    return dict(proofs),counts


def trace_diagnosis(base,candidate,proofs):
    """Exact subsequence and certified-deletion alignment, including repeats."""
    b,c = list(map(signature,base)),list(map(signature,candidate))
    first = next((i for i,(x,y) in enumerate(zip(b,c)) if x!=y),min(len(b),len(c)))
    first = None if first==len(b)==len(c) else first
    cursor,embedding,non_subsequence = 0,[],None
    for j,token in enumerate(c):
        while cursor<len(b) and b[cursor]!=token:
            cursor+=1
        if cursor==len(b):
            non_subsequence={"candidate_index_0based":j,"candidate_action":candidate[j],
                "last_matched_baseline_index_0based":embedding[-1] if embedding else None,
                "reason":"No matching physical action/feedback remains in the baseline suffix"}
            break
        embedding.append(cursor)
        cursor+=1
    # At each candidate prefix, every allowed deletion consumes one separately
    # recorded certificate at that exact prefix. DP avoids false negatives from
    # a greedy match of repeated actions. States reset certificate use on match.
    states,parents = {0},{}
    failure = None
    for j in range(len(c)+1):
        allowed = Counter(query_key(p) for p in proofs.get(j,[]))
        following = set()
        for start in sorted(states):
            consumed = Counter()
            i=start
            while True:
                if j==len(c) and i==len(b):
                    parents[(j+1,i)]=(start,i)
                    following.add(i)
                elif j<len(c) and i<len(b) and b[i]==c[j]:
                    if i+1 not in following:
                        parents[(j+1,i+1)]=(start,i)
                        following.add(i+1)
                if i==len(b):
                    break
                a=base[i];key=query_key(a)
                if a["action"]!="measure" or a["result"]!="no_signal" or consumed[key]>=allowed[key]:
                    break
                consumed[key]+=1
                i+=1
        if not following:
            failure={"candidate_prefix_count":j,"candidate_action":candidate[j] if j<len(c) else None,
                "reachable_baseline_next_indices":sorted(states),
                "baseline_next_actions":[{"index_0based":i,"action":base[i] if i<len(b) else None} for i in sorted(states)],
                "available_prefix_certificates":proofs.get(j,[]),
                "reason":"No alignment extends this prefix using only same-prefix certified no_signal query deletions; missing certificates are unknown, not assumed"}
            break
        states=following
    certified=failure is None
    deleted,matched=[],[]
    if certified:
        end=len(b)
        for level in range(len(c)+1,0,-1):
            start,match=parents[(level,end)]
            prefix=level-1
            available=defaultdict(list)
            for proof in proofs.get(prefix,[]):
                available[query_key(proof)].append(proof)
            for index in range(start,match):
                deleted.append({"baseline_index_0based":index,"candidate_prefix_count":prefix,
                                "action":base[index],"proof":available[query_key(base[index])].pop(0)})
            if prefix<len(c):
                matched.append({"baseline_index_0based":match,"candidate_index_0based":prefix})
            end=start
        deleted.sort(key=lambda x:x["baseline_index_0based"])
        matched.sort(key=lambda x:x["candidate_index_0based"])
    return {"identical_physical_actions_and_feedback":b==c,"first_actual_difference_index_0based":first,
        "first_baseline_action":base[first] if first is not None and first<len(base) else None,
        "first_candidate_action":candidate[first] if first is not None and first<len(candidate) else None,
        "physical_subsequence":non_subsequence is None,"first_non_subsequence_prefix":non_subsequence,
        "only_certified_negative_query_deletions":certified,"first_uncertified_deletion_prefix":failure,
        "deleted_queries":deleted,"matched_actions":matched,
        "scope":"Exact supplied action/position/feedback sequence ignoring changed timestamps; failed proof alignment is not permission to invent missing inference logs"}


def bind_audit(data,batch,policy,seed,record_sha,raw):
    if data.get("all_audits_passed") is not True:
        raise ValueError("Refusing failed or incomplete posthoc audit")
    rows=[r for r in data["rows"] if (r["batch"],r["strategy"],r["seed"])==(batch,policy,seed)]
    if len(rows)!=1:
        raise ValueError("Missing or duplicate exact-record audit row")
    row=rows[0]
    if (row["input_sha256"]!=record_sha or row["case_sha256"]!=raw["case_sha256"]
            or row["virtual_time_s"]!=raw["virtual_time_s"] or row.get("audit_passed") is not True or row.get("errors")):
        raise ValueError("Exact-record audit identity mismatch or audit failed")
    if (not isinstance(row.get("physical_lower_s"),(int,float)) or not math.isfinite(row["physical_lower_s"])
            or row["physical_lower_s"]<=0 or row["ratio_eligible"]!=raw["successful"]):
        raise ValueError("Invalid bound/ratio eligibility")
    if row["ratio_eligible"] and row["time_over_physical_lower"]!=raw["virtual_time_s"]/row["physical_lower_s"]:
        raise ValueError("Audited ratio differs from its exact case bound")
    return row


def aggregates(rows):
    output={}
    for batch,policy in sorted({(r["batch"],r["strategy"]) for r in rows}):
        group=[r for r in rows if (r["batch"],r["strategy"])==(batch,policy)]
        valid=[r for r in group if r["ratio_eligible"]]
        times=[r["virtual_time_s"] for r in group]
        cpu=[r["program_cpu_s"] for r in group]
        output[f"{batch}:{policy}"]={"batch":batch,"strategy":policy,"runs":len(group),
            "successful":sum(r["successful"] for r in group),"failed_clears":sum(r["ledger"]["failed_clears"] for r in group),
            "mean_virtual_s":statistics.fmean(times),"p95_virtual_s":percentile(times,.95),"max_virtual_s":max(times),
            "mean_cpu_s":statistics.fmean(cpu) if all(v is not None for v in cpu) else None,
            "mean_wall_s":statistics.fmean(r["program_runtime_s"] for r in group),
            "ratio_eligible_runs":len(valid),
            "mean_case_time_over_lower":statistics.fmean(r["virtual_time_s"]/r["physical_lower_s"] for r in valid) if valid else None,
            "sum_time_over_sum_lower":sum(r["virtual_time_s"] for r in valid)/sum(r["physical_lower_s"] for r in valid) if valid else None,
            "mean_measurements":statistics.fmean(r["ledger"]["measurements"] for r in group),
            "mean_clear_attempts":statistics.fmean(r["ledger"]["clear_attempts"] for r in group),
            "mean_ledger":{k:statistics.fmean(r["ledger"]["time_breakdown"][k] for r in group) for k in COSTS},
            "total_omissions":{k:sum(r["omission_counts"][k] for r in group) for k in group[0]["omission_counts"]}}
    return output


def build(batches,audit_paths):
    audits=[];inputs={str(Path(__file__).resolve()):sha(__file__)}
    for path in audit_paths:
        path=path.resolve(strict=True)
        if path.suffix.lower()!=".json":
            raise ValueError("Audit must be an explicit JSON file, never a database")
        data=read(path)
        if data.get("version")!="q3-round2-posthoc-original-physical-v1" or data.get("all_audits_passed") is not True:
            raise ValueError("Requires completed passed original physical/post-prefix audit")
        ext=ROOT/"experiments/round2_derived_silence_audit.py"
        if data.get("observation_extension_sha256",{}).get(str(ext))!=sha(ext):
            raise ValueError("Missing/mismatched independent derived-silence auditor source")
        inputs[str(path)]=sha(path);inputs[str(ext)]=sha(ext);audits.append(data)
    verifier=Path(__file__).with_name("round2_posthoc_audit.py")
    inputs[str(verifier.resolve())]=sha(verifier)
    rows,pairs,evidence=[],[],[];seen=set()
    for batch in batches:
        batch=batch.resolve(strict=True)
        manifest,summary,runner,runner_path=verify_batch(batch)
        if set(manifest["policies"])!=set(POLICIES) or manifest["trial"]!="derived_silence" or runner!="verified":
            raise ValueError("Requires complete frozen derived_silence four-arm batch with verified runner")
        label=f"{manifest['trial']}/{manifest['stage']}"
        if label in seen:
            raise ValueError("Duplicate batch label")
        seen.add(label)
        matched_audits=[d for d in audits if any(b["batch"]==label for b in d["batches"])]
        if len(matched_audits)!=1:
            raise ValueError("Require exactly one matching audit per batch")
        audit=matched_audits[0]
        a_batch=next(b for b in audit["batches"] if b["batch"]==label)
        if a_batch["manifest_sha256"]!=sha(batch/"manifest.json") or a_batch["summary_sha256"]!=sha(batch/"summary.json"):
            raise ValueError("Audit batch manifest/summary hash mismatch")
        for path in [batch/"manifest.json",batch/"summary.json",runner_path,*[batch/f"{p}-source.zip" for p in POLICIES]]:
            inputs[str(path.resolve())]=sha(path)
        indexed={(r["strategy"],r["seed"]):r for r in summary["rows"]}
        if len(indexed)!=len(summary["records_sha256"]):
            raise ValueError("Duplicate/incomplete summary rows")
        evidence.append({"batch":label,"path":str(batch),"all_frozen_sources_verified":True,
            "policy_source_files_sha256":{p:manifest["policies"][p]["source_sha256"] for p in POLICIES},
            "frozen_comparisons":summary["comparisons"]})
        for seed in manifest["seeds"]:
            by_policy={}
            for policy in POLICIES:
                path=batch/f"records/{policy}-{seed}.json.gz";record_sha=sha(path)
                if record_sha!=summary["records_sha256"][path.relative_to(batch).as_posix()]:
                    raise ValueError("Record hash mismatch")
                with gzip.open(path,"rt",encoding="utf-8") as stream: record=json.load(stream)
                raw=record["row"]
                if (raw!=indexed[(policy,seed)] or raw["strategy"]!=policy or raw["seed"]!=seed
                        or record["frozen_manifest_sha256"]!=digest(manifest)
                        or record["spec"]!=manifest["policies"][policy]["spec"]
                        or record.get("evaluation_phase")!="after_policy_termination"
                        or record.get("audit",{}).get("errors") or raw.get("errors")):
                    raise ValueError("Record provenance, post-termination phase or local audit failed")
                bound=bind_audit(audit,label,policy,seed,record_sha,raw)
                actions,billed=physical_record(record)
                if (billed["virtual_time_s"]!=raw["virtual_time_s"] or billed["measurements"]!=raw["measurement_count"]
                        or billed["failed_clears"]!=raw["failed_clear_count"] or len(actions)+2!=raw["action_count"]
                        or any(billed["time_breakdown"][k]!=raw[k] for k in COSTS)):
                    raise ValueError("Frozen row disagrees with independently reconstructed charges")
                proofs,counts=certified_omissions(record["summary"],bound["observation_certificate_audit"])
                inputs[str(path)]=record_sha
                row={"batch":label,"strategy":policy,"seed":seed,"input_sha256":record_sha,"case_sha256":raw["case_sha256"],
                    "successful":raw["successful"],"virtual_time_s":raw["virtual_time_s"],"penalized_time_s":raw["penalized_time_s"],
                    "program_cpu_s":raw.get("program_cpu_s"),"program_runtime_s":raw["program_runtime_s"],
                    "physical_lower_s":bound["physical_lower_s"],"ratio_eligible":bound["ratio_eligible"],
                    "time_over_lower":bound["time_over_physical_lower"],"ledger":billed,"omission_counts":counts,
                    "prefix_omission_certificates":proofs}
                rows.append(row);by_policy[policy]=(row,actions,proofs)
            if len({x[0]["case_sha256"] for x in by_policy.values()})!=1:
                raise ValueError("Four arms do not share the same case identity")
            base,base_actions,_=by_policy["baseline"]
            for policy in POLICIES[1:]:
                row,actions,proofs=by_policy[policy]
                trace=trace_diagnosis(base_actions,actions,proofs)
                actual_saving=base["virtual_time_s"]-row["virtual_time_s"]
                deleted_only=None
                if trace["only_certified_negative_query_deletions"]:
                    removed={d["baseline_index_0based"] for d in trace["deleted_queries"]}
                    fixed=ledger([a for i,a in enumerate(base_actions) if i not in removed])
                    if fixed["time_breakdown"]!=row["ledger"]["time_breakdown"]:
                        raise ValueError("Certified deletion reconstruction differs from actual candidate billing")
                    deleted_only={"deleted_queries":len(removed),"reconstructed_candidate_virtual_s":fixed["virtual_time_s"],
                        "fixed_trace_saving_s":base["virtual_time_s"]-fixed["virtual_time_s"],
                        "actual_saving_matches_fixed_trace":fixed["virtual_time_s"]==row["virtual_time_s"]}
                pairs.append({"batch":label,"seed":seed,"baseline":"baseline","candidate":policy,
                    "actual_whole_episode_saving_s":actual_saving,
                    "actual_time_breakdown_saving":{k:base["ledger"]["time_breakdown"][k]-row["ledger"]["time_breakdown"][k] for k in COSTS},
                    "actual_measurement_reduction":base["ledger"]["measurements"]-row["ledger"]["measurements"],
                    "trace":trace,"certified_deletion_bill":deleted_only,
                    "nominal_omitted_detection_charge_s":5*row["omission_counts"]["new_mechanism_omissions"],
                    "nominal_scope":"5 seconds per omitted query is a local detection-charge tally, not whole-episode savings; receiver channel, geometry and subsequent scheduling can differ"})
    for path,value in inputs.items():
        if sha(path)!=value: raise ValueError("Input changed during diagnosis")
    return {"version":"round2-derived-silence-diagnosis-v1","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "input_sha256":inputs,"sqlite_read":False,"network_access":False,"new_simulations":0,"strategy_calls":0,
        "batches":evidence,"aggregates":aggregates(rows),"rows":rows,"pairs":pairs,
        "scope":"Original clairvoyant physical LB; mean(T/LB) and sum(T)/sum(LB) use the same eligible runs. Omission potential is not actual benefit. Exact archive identities and passed independent audits are prerequisites."}


def markdown(result):
    n=lambda x:"不适用" if x is None else f"{x:.6f}"
    lines=["# 推断静默四组实验诊断","","仅分析明确指定的完整归档；已核对manifest、runner、源zip/逐文件、记录哈希、独立物理与前缀证书审核。未启动仿真或读取数据库。",
        "旧LB是先知物理松弛，不是可达在线最优值。逐局均比与总量比使用相同审核合格局，分别列出。", "",
        "|批次/策略|全清/局|失败清除|平均T|P95 T|最大T|平均CPU|均T/LB|总T/总LB|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,g in result["aggregates"].items():
        lines.append(f"|{name}|{g['successful']}/{g['runs']}|{g['failed_clears']}|"+"|".join(n(g[k]) for k in
            ("mean_virtual_s","p95_virtual_s","max_virtual_s","mean_cpu_s","mean_case_time_over_lower","sum_time_over_sum_lower"))+"|")
    lines += ["","|批次/策略|相对静默省略|数量上限省略|旧静默省略|平均实际测量|平均换频秒|平均移动秒|平均检测秒|平均光学秒|平均移除秒|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,g in result["aggregates"].items():
        t=g["total_omissions"];b=g["mean_ledger"]
        lines.append(f"|{name}|{t['relative_silence_skips']}|{t['count_cap_skips']}|{t['legacy_inferred_skips']}|{n(g['mean_measurements'])}|"+
                     "|".join(n(b[k]) for k in ("switching_s","movement_s","detection_s","optical_s","removal_s"))+"|")
    lines += ["","clear不换接收频道；移动按每段微秒取整重建。省略数×5秒只是局部检测收费，不代替整局节省。", "",
        "|批/案例|候选|实际整局节省秒|实际少测|仅删除有证负查询|物理子序列|首个不能延续的候选索引|", "|---|---|---:|---:|---|---|---:|"]
    for p in result["pairs"]:
        t=p["trace"];failure=t["first_non_subsequence_prefix"] or t["first_uncertified_deletion_prefix"]
        index=(failure.get("candidate_index_0based",failure.get("candidate_prefix_count")) if failure else "无")
        lines.append(f"|{p['batch']}/{p['seed']}|{p['candidate']}|{n(p['actual_whole_episode_saving_s'])}|{p['actual_measurement_reduction']}|"
                     f"{t['only_certified_negative_query_deletions']}|{t['physical_subsequence']}|{index}|")
    lines += ["","索引从0起。物理子序列比较不要求累计时间相同，但要求动作、坐标、频道及实际反馈一致。若仅证书对齐失败，记为无法按现有前缀日志证明，不能当作策略违法。",
        "JSON保留首个动作分歧、最初不能延续的子序列/证书前缀、逐项删除见证、逐动作收费和整局分项差额；改变后的全局路线不得用局部查询潜力解释为必然收益。",
        "", "全部输入与依赖SHA256见JSON input_sha256，所有输出以新文件保存。", ""]
    return "\n".join(lines)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--batches",nargs="+",type=Path,required=True)
    cli.add_argument("--audit",nargs="+",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--report",type=Path,required=True)
    args=cli.parse_args()
    if args.output.resolve()==args.report.resolve() or args.output.exists() or args.report.exists():
        cli.error("output/report must be distinct new paths; overwrite is forbidden")
    result=build(args.batches,args.audit)
    for path,content in ((args.output,json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"),(args.report,markdown(result))):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("x",encoding="utf-8") as stream: stream.write(content)
    print(json.dumps({"rows":len(result["rows"]),"output":str(args.output),"all_checks_passed":True}))


if __name__=="__main__":
    main()
