from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from experiments.q4_rl_evaluate import (
    completion_audit, report_rows, run_one, validate_specs, DEFAULT_SPECS,
)
from q4_rl.scenarios import build_case, case_requests, split_of_seed


def sample_rows():
    result = []
    for seed in (8100000,8100001):
        for label,elapsed in (("r8",1000.),("heuristic",900.)):
            result.append(dict(case_id=str(seed),seed=seed,case_sha256=str(seed),strategy=label,
                family="random",source_mode="mixed",stage="development",successful=True,
                virtual_time_s=elapsed,penalized_time_s=elapsed,common_lower_bound_s=200.,
                time_over_lower_bound=elapsed/200.,penalized_time_over_lower_bound=elapsed/200.,
                failed_clear_count=1,program_runtime_s=.1,process_cpu_s=.08,audit_runtime_s=.01,
                measurement_count=20,source_total=10,cleared_total=10,stop_reason="exited",errors=[],
                movement_s=elapsed-101,switching_s=1.,detection_s=50.,optical_s=30.,removal_s=20.))
    return result


def test_fresh_namespaces_and_whole_range_validation():
    assert split_of_seed(8000000)=="train"
    assert split_of_seed(8200000)=="confirmation"
    with pytest.raises(ValueError):
        build_case(610001)
    with pytest.raises(ValueError):
        build_case(8100000,split="train")
    with pytest.raises(ValueError):
        case_requests("development",start=8199999,count=2)


@pytest.mark.parametrize("mode",["mixed","all_directional"])
@pytest.mark.parametrize("family",["random","boundary_outward","narrow_strip"])
def test_synthetic_cases_are_reproducible_and_legal(mode,family):
    first = build_case(8100001,split="development",source_mode=mode,family=family)
    assert first==build_case(8100001,split="development",source_mode=mode,family=family)
    assert 10<=len(first.sources)<=16
    assert len({s.channel for s in first.sources})==len(first.sources)
    assert all(math.hypot(s.x,s.y)<=1800 and 1000<=s.reception_radius_m<=1500 for s in first.sources)
    assert any(s.orientation_deg is not None for s in first.sources)
    assert all(s.orientation_deg is not None for s in first.sources) == (mode=="all_directional")


def test_frozen_controls_and_learned_model_requirements():
    assert validate_specs({"r8":deepcopy(DEFAULT_SPECS["r8"])})
    changed=deepcopy(DEFAULT_SPECS["r8"])
    changed["kwargs"]["max_expansions"]=100
    with pytest.raises(ValueError):
        validate_specs({"r8":changed})
    with pytest.raises(ValueError):
        validate_specs({"learned":{"entrypoint":"q4_rl.controller:run_q4_rl"}})


def test_full_paired_statistics_and_ratio_definitions():
    report=report_rows(sample_rows(),samples=40)
    candidate=report["summaries"]["heuristic"]
    assert candidate["ratio_of_sums"]==4.5
    assert candidate["mean_individual_penalized_ratio"]==4.5
    assert candidate["failed_clear_total"]==2
    assert candidate["all_clear_rate_wilson_ci95"][0]<1
    assert report["paired"]["heuristic"]["mean_saving_ci95_s"]==[100.,100.]
    assert report["paired"]["heuristic"]["p95_saving_ci95_s"]==[100.,100.]
    assert "mixed/random" in report["strata"]


def test_one_independent_seed_does_not_claim_a_confidence_interval():
    report=report_rows(sample_rows()[:2],samples=20)
    assert report["paired"]["heuristic"]["mean_saving_ci95_s"] is None
    assert report["paired"]["heuristic"]["uncertainty_status"]=="insufficient_independent_seed_clusters"


@pytest.mark.parametrize("damage",["missing","duplicate","truth","lower"])
def test_pair_corruption_is_rejected(damage):
    rows=sample_rows()
    if damage=="missing": rows.pop()
    elif damage=="duplicate": rows.append(deepcopy(rows[0]))
    elif damage=="truth": rows[-1]["case_sha256"]="different"
    elif damage=="lower": rows[-1]["common_lower_bound_s"]=199.
    with pytest.raises(ValueError):
        report_rows(rows,samples=20)


def test_failed_fast_exit_never_improves_score_or_gets_all_clear_ratio():
    rows=sample_rows()
    rows[-1].update(successful=False,virtual_time_s=1.,penalized_time_s=360000.,
        time_over_lower_bound=None,penalized_time_over_lower_bound=1800.,cleared_total=0)
    report=report_rows(rows,samples=30)
    assert report["summaries"]["heuristic"]["mean_penalized_time_s"]==180450.
    assert report["paired"]["heuristic"]["mean_saved_s"]<0
    assert len(report["summaries"]["heuristic"]["failure_records"])==1
    rows[-1]["time_over_lower_bound"]=.005
    with pytest.raises(ValueError):
        report_rows(rows,samples=20)


def test_completion_requires_evidence_not_claimed_source_count():
    assert completion_audit([])["passed"] is False
    history=[dict(action="/clear",channel=c,position={"x":0,"y":0},
              response={"accepted":True,"clear_result":"success"}) for c in range(1,17)]
    assert completion_audit(history)["reason"]=="sixteen_actual_clears"
    assert completion_audit(history[:10])["passed"] is False


def test_run_one_retains_failed_empty_episode_without_truth_in_policy(monkeypatch):
    import experiments.q4_rl_evaluate as module
    seen={}
    def early_exit(client,**kwargs):
        with pytest.raises(AssertionError):
            getattr(client,"scenario")
        seen.update(kwargs)
        client.enter()
        client.exit()
        return SimpleNamespace(error=None,exit_error=None,completion_certified_under_model=False,
            cleared_count=0,virtual_time_s=0.,as_dict=lambda:{"completion_reason":"early_exit"})
    monkeypatch.setattr(module,"entrypoint",lambda _:early_exit)
    monkeypatch.setattr(module,"common_bound",lambda _:dict(common_lower_bound_s=100.,common_lower_bound_rounded_s=99.999999))
    request=dict(seed=8100001,split="development",family="random",source_mode="mixed")
    record=run_one(request,"unit_early_exit",{"entrypoint":"unit:early_exit"})
    assert not record["row"]["successful"]
    assert record["row"]["penalized_time_s"]==360000.
    assert record["row"]["time_over_lower_bound"] is None
    assert record["evaluation_phase"]=="after_policy_termination"
    assert len(record["history"])==2
    assert "seed" not in seen and "case_id" not in seen


@pytest.mark.parametrize("stage",["confirmation","final"])
def test_independent_stage_cannot_open_without_selection(monkeypatch,tmp_path,stage):
    import experiments.q4_rl_evaluate as module
    output=tmp_path/"forbidden"
    monkeypatch.setattr("sys.argv",["q4_rl_evaluate","--output",str(output),"--stage",stage,
        "--strategies","r8","--count","1","--freeze-only"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code==2
    assert not output.exists()
