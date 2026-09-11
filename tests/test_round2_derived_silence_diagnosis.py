"""Pure archive/billing/alignment contracts; no simulated scenarios or policies."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from experiments.round2_derived_silence_diagnosis import (
    ROOT,POLICIES,COSTS,aggregates,bind_audit,build,certified_omissions,digest,ledger,
    physical_record,sha,trace_diagnosis,
)


def action(channel=1,position=(0.,0.),result="no_signal",kind="measure"):
    return {"action":kind,"channel":channel,"position":list(position),"result":result,"phase":"fixture"}


def complete_record(actions):
    actions=deepcopy(actions)
    bill=ledger(actions)
    for item,cost in zip(actions,bill["actions"]): item["virtual_time_s"]=cost["virtual_time_s"]
    raw=[{"action":"/enter","channel":None,"position":{"x":0.,"y":0.},
          "response":{"accepted":True,"virtual_time_s":0.}}]
    for item in actions:
        raw.append({"action":"/"+item["action"],"channel":item["channel"],
                    "position":dict(zip(("x","y"),item["position"])),
                    "response":{"accepted":True,"virtual_time_s":item["virtual_time_s"],
                        "measure_result" if item["action"]=="measure" else "clear_result":item["result"]}})
    raw.append({"action":"/exit","response":{"accepted":True,"virtual_time_s":bill["virtual_time_s"]}})
    return {"history":raw,"summary":{"action_history":actions,"virtual_time_s":bill["virtual_time_s"],
        "time_breakdown":bill["time_breakdown"],"measurement_count":bill["measurements"],
        "clear_attempt_count":bill["clear_attempts"],"accepted_actions":len(raw),"strategy_parameters":{}}}


def proof(query):
    return {"channel":query["channel"],"position":query["position"],"kind":"fixture_certified_negative"}


def prefix_audit():
    return {"derived_audit_version":"q3-derived-silence-prefix-audit-v1",
        "legacy_without_relative_constraints_passed":True,"relative_inferred_coverage_credits":0,
        "inferred_coverage_credits":0,"relative_certificates":[],"relative_inferred_silence_verified":0,
        "legacy_inferred_silence_verified":0,"derived_scan_checks":[],
        "derived_scan_totals":{"relative_silence_skips":0,"count_cap_skips":0,"completed_geometric_scans":0,
            "count_certified_scans":0,"no_physical_action_scans":0}}


def test_billing_clear_does_not_retune_and_failed_clear_only_pays_optical():
    actions=[action(5),action(1,result="success",kind="clear"),action(5),
             action(2,(3.,4.),"no_target_in_range","clear")]
    result=ledger(actions)
    assert result["time_breakdown"]==dict(movement_s=1.,switching_s=1.,detection_s=10.,optical_s=6.,removal_s=2.)
    assert result["receiver_channel_after"]==5
    assert result["virtual_time_s"]==20.


def test_each_movement_segment_is_rounded_separately():
    actions=[action(1,(.0000015,0.)),action(1,(.000003,0.))]
    assert ledger(actions)["time_breakdown"]["movement_s"]==0.
    assert round((.000003/5)*1_000_000)==1


def test_delete_query_recomputes_following_receiver_not_fixed_six_seconds():
    original=[action(2),action(3),action(2)]
    kept=[original[0],original[2]]
    a,b=ledger(original),ledger(kept)
    assert a["virtual_time_s"]-b["virtual_time_s"]==7.
    assert a["time_breakdown"]["switching_s"]-b["time_breakdown"]["switching_s"]==2.


def test_certified_alignment_handles_duplicate_action_without_greedy_false_negative():
    a,b=action(1),action(2,result="near")
    result=trace_diagnosis([a,a,b],[a,b],{0:[proof(a)]})
    assert result["physical_subsequence"] and result["only_certified_negative_query_deletions"]
    assert [d["baseline_index_0based"] for d in result["deleted_queries"]]==[0]
    assert result["matched_actions"][0]["baseline_index_0based"]==1


def test_future_certificate_cannot_justify_deletion_at_earlier_prefix():
    a,b=action(1),action(2,result="near")
    result=trace_diagnosis([a,b],[b],{1:[proof(a)]})
    assert result["physical_subsequence"]
    assert not result["only_certified_negative_query_deletions"]
    assert result["first_uncertified_deletion_prefix"]["candidate_prefix_count"]==0


@pytest.mark.parametrize("bad",[action(1,result="near"),action(1,result="success",kind="clear")])
def test_certificate_never_authorizes_deleting_positive_query_or_clear(bad):
    result=trace_diagnosis([bad],[],{0:[proof(bad)]})
    assert result["physical_subsequence"] and not result["only_certified_negative_query_deletions"]


def test_each_deletion_needs_a_separate_same_position_certificate():
    a=action(1)
    assert not trace_diagnosis([a,a],[],{0:[proof(a)]})["only_certified_negative_query_deletions"]
    assert not trace_diagnosis([a],[],{0:[proof(action(1,(1.,0.)))]})["only_certified_negative_query_deletions"]


def test_reports_exact_first_non_subsequence_prefix_not_just_first_difference():
    a,b=action(1),action(2)
    result=trace_diagnosis([a,b],[b,a],{})
    assert result["first_actual_difference_index_0based"]==0
    assert not result["physical_subsequence"]
    assert result["first_non_subsequence_prefix"]["candidate_index_0based"]==1


def test_public_checks_never_traverse_hidden_payload():
    class Hidden:
        def __getitem__(self,key): raise AssertionError("hidden payload accessed")
        def __deepcopy__(self,memo): raise AssertionError("hidden payload copied")
    record=complete_record([action()]);record["evaluation"]=Hidden()
    actions,bill=physical_record(record)
    assert bill["measurements"]==1
    assert trace_diagnosis(actions,actions,{})["only_certified_negative_query_deletions"]


@pytest.mark.parametrize("mutation",["billing","fake_physical","feedback","no_exit"])
def test_physical_tampering_or_incomplete_run_is_rejected(mutation):
    record=complete_record([action()])
    if mutation=="billing":
        record["history"][1]["response"]["virtual_time_s"]+=1
        record["summary"]["action_history"][0]["virtual_time_s"]+=1
    elif mutation=="fake_physical": record["history"][1]["physical_measurement"]=False
    elif mutation=="feedback": record["summary"]["action_history"][0]["result"]="near"
    else: record["history"].pop()
    with pytest.raises(ValueError): physical_record(record)


def test_missing_independent_prefix_proof_is_not_accepted_by_stats_claim():
    summary=complete_record([action()])["summary"]
    summary["strategy_parameters"]["inferred_no_signal_constraints"]=[{
        "after_actual_action_count":1,"channel":1,"position":[0.,0.],"physical_measurement":False,
        "method":"relative_actual_negative","witness_action_ordinal":1}]
    with pytest.raises(ValueError,match="Inference events differ"):
        certified_omissions(summary,prefix_audit())
    audit=prefix_audit();audit["relative_inferred_coverage_credits"]=1
    with pytest.raises(ValueError,match="unweakened"):
        certified_omissions(summary,audit)


def test_two_lower_bound_aggregates_are_not_interchanged():
    rows=[]
    for t,lb in ((2.,1.),(12.,3.)):
        rows.append({"batch":"fixture","strategy":"baseline","successful":True,"virtual_time_s":t,
            "program_cpu_s":1.,"program_runtime_s":1.,"physical_lower_s":lb,"ratio_eligible":True,
            "ledger":{"failed_clears":0,"measurements":0,"clear_attempts":0,"time_breakdown":dict.fromkeys(COSTS,0.)},
            "omission_counts":{"new_mechanism_omissions":0}})
    result=aggregates(rows)["fixture:baseline"]
    assert result["mean_case_time_over_lower"]==3.
    assert result["sum_time_over_sum_lower"]==3.5
    assert result["p95_virtual_s"]==11.5


@pytest.fixture
def archive(tmp_path):
    """Synthetic storage contract, not purported real physical proof output."""
    batch=tmp_path/"batch";batch.mkdir();(batch/"records").mkdir()
    runner=batch/"runner.py";runner.write_text("# inert fixture, never executed\n",encoding="utf8")
    manifest={"trial":"derived_silence","stage":"fixture","seeds":[1],"policies":{},"runner_sha256":sha(runner)}
    for policy in POLICIES:
        source=b"# inert source, never imported\n"
        with zipfile.ZipFile(batch/f"{policy}-source.zip","w") as z: z.writestr("src/fixture.py",source)
        import hashlib
        spec={"entrypoint":"never_import:this_is_only_a_fixture","kwargs":{}}
        manifest["policies"][policy]={"spec":spec,"spec_sha256":digest(spec),
            "source_sha256":{"src/fixture.py":hashlib.sha256(source).hexdigest()},
            "source_archive_sha256":sha(batch/f"{policy}-source.zip")}
    (batch/"manifest.json").write_text(json.dumps(manifest),encoding="utf8")
    summary={"rows":[],"records_sha256":{},"comparisons":{}}
    audit_rows=[]
    for policy in POLICIES:
        record=complete_record([action()]);billed=ledger(record["summary"]["action_history"])
        row={"strategy":policy,"seed":1,"case_sha256":"a"*64,"successful":False,"virtual_time_s":5.,
            "penalized_time_s":360000.,"program_cpu_s":1.,"program_runtime_s":1.,"measurement_count":1,
            "failed_clear_count":0,"action_count":3,"errors":[],**billed["time_breakdown"]}
        record.update(row=row,spec=manifest["policies"][policy]["spec"],frozen_manifest_sha256=digest(manifest),
                      evaluation_phase="after_policy_termination",audit={"errors":[]})
        path=batch/f"records/{policy}-1.json.gz"
        with gzip.open(path,"wt",encoding="utf8") as stream: json.dump(record,stream)
        summary["records_sha256"][path.relative_to(batch).as_posix()]=sha(path);summary["rows"].append(row)
        audit_rows.append({"batch":"derived_silence/fixture","strategy":policy,"seed":1,"input_sha256":sha(path),
            "case_sha256":row["case_sha256"],"virtual_time_s":5.,"audit_passed":True,"errors":[],
            "physical_lower_s":1.,"ratio_eligible":False,"time_over_physical_lower":None,
            "observation_certificate_audit":prefix_audit()})
    (batch/"summary.json").write_text(json.dumps(summary),encoding="utf8")
    extension=ROOT/"experiments/round2_derived_silence_audit.py"
    audit={"version":"q3-round2-posthoc-original-physical-v1","all_audits_passed":True,"rows":audit_rows,
        "observation_extension_sha256":{str(extension):sha(extension)},
        "batches":[{"batch":"derived_silence/fixture","manifest_sha256":sha(batch/"manifest.json"),
                    "summary_sha256":sha(batch/"summary.json")} ]}
    path=tmp_path/"audit.json";path.write_text(json.dumps(audit),encoding="utf8")
    return batch,path


def test_complete_frozen_storage_contract_parses_without_importing_entrypoint(archive):
    batch,audit=archive
    result=build([batch],[audit])
    assert len(result["rows"])==4 and len(result["pairs"])==3
    assert all(p["trace"]["only_certified_negative_query_deletions"] for p in result["pairs"])
    assert result["strategy_calls"]==result["new_simulations"]==0
    assert "never_import" not in sys.modules


@pytest.mark.parametrize("mutation",["unfinished","record_hash","source_hash","failed_audit","audit_row_hash","audit_batch_hash","auditor_source"])
def test_unfinished_hash_mismatch_and_failed_audits_are_rejected(archive,mutation):
    batch,path=archive
    if mutation=="unfinished": (batch/"records/cap_only-1.json.gz").unlink()
    elif mutation=="record_hash":
        with (batch/"records/baseline-1.json.gz").open("ab") as stream: stream.write(b"changed")
    elif mutation=="source_hash":
        with (batch/"baseline-source.zip").open("ab") as stream: stream.write(b"changed")
    else:
        data=json.loads(path.read_text())
        if mutation=="failed_audit": data["all_audits_passed"]=False
        elif mutation=="audit_row_hash": data["rows"][0]["input_sha256"]="b"*64
        elif mutation=="audit_batch_hash": data["batches"][0]["manifest_sha256"]="b"*64
        else: data["observation_extension_sha256"]={}
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError,FileNotFoundError)): build([batch],[path])


def test_cli_refuses_overwrite_before_opening_batch(tmp_path):
    output=tmp_path/"keep.json";output.write_text("keep")
    result=subprocess.run([sys.executable,"-B","-m","experiments.round2_derived_silence_diagnosis",
        "--batches","does-not-exist","--audit","forbidden.sqlite3","--output",str(output),"--report",str(tmp_path/"new.md")],
        cwd=ROOT,capture_output=True,text=True)
    assert result.returncode!=0 and "overwrite is forbidden" in result.stderr
    assert output.read_text()=="keep" and not (tmp_path/"new.md").exists()
