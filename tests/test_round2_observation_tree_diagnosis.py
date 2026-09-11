"""Read-only parser checks on already completed implementation smoke records."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.round2_observation_tree_diagnosis import checked_history, planning_diagnosis, policy_arms, markdown

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "results/round2/observation_tree/model_smoke/depth2-90s.json.gz"


@pytest.fixture
def completed():
    with gzip.open(SMOKE,"rt",encoding="utf-8") as stream:
        return json.load(stream)


def test_real_smoke_started_complete_and_actual_support_are_distinct(completed):
    checked_history(completed)
    result = planning_diagnosis(completed["summary"])
    assert not result["errors"]
    assert result["tail_evaluations_started"] == result["completed_tails_retained_in_nodes"] == 48
    assert result["mechanism_active"] == 1
    nodes = [n for d in result["decisions"] for a in d["first_actions"] for n in a["nodes"]]
    assert any(n["training_used_support"]["draws"] == 2
               and n["training_pool"]["position_pool_ess"]["max"] > 2 for n in nodes)


def test_fallback_retention_gap_is_unknown_not_completed(completed):
    summary = deepcopy(completed["summary"])
    log = summary["strategy_parameters"]["observation_tree_log"][0]
    log.update(status="fallback",reason="PlanningRefused: compute_deadline",mechanism_active=False,
               selected=deepcopy(log["baseline"]),candidates=[])
    result = planning_diagnosis(summary)
    assert not result["errors"]
    assert result["tail_evaluations_started"] == 48
    assert result["completed_tails_retained_in_nodes"] == 0
    assert result["tail_completion_not_identifiable_from_retained_nodes"] == 48


def test_mechanism_and_support_tampering_are_detected(completed):
    summary = deepcopy(completed["summary"])
    log = summary["strategy_parameters"]["observation_tree_log"][0]
    log["mechanism_active"] = False
    log["candidates"][0]["nodes"][0]["actual_training_draws"] = 256
    errors = planning_diagnosis(summary)["errors"]
    assert any("mechanism_active" in e for e in errors)
    assert any("sample counts" in e for e in errors)


def test_physical_feedback_is_checked_before_diagnosis(completed):
    completed["summary"]["action_history"][0]["virtual_time_s"] += 1
    with pytest.raises(ValueError,match="physical feedback"):
        checked_history(completed)


def test_cli_refuses_overwrite_before_reading_inputs(tmp_path):
    output = tmp_path/"preserved.json"
    output.write_text("keep",encoding="utf-8")
    result = subprocess.run([sys.executable,"-m","experiments.round2_observation_tree_diagnosis",
        "--smoke",str(SMOKE),"--output",str(output),"--report",str(tmp_path/"new.md")],
        cwd=ROOT,capture_output=True,text=True)
    assert result.returncode != 0 and "overwrite is forbidden" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path/"new.md").exists()


@pytest.fixture
def coupled():
    path = ROOT/"results/round2/coupled_tree/model_smoke/depth2.json.gz"
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        return json.load(stream)


def partial_coupled(completed, freeze_choice):
    summary = deepcopy(completed["summary"])
    log = summary["strategy_parameters"]["observation_tree_log"][0]
    candidate = log["candidates"][0]
    node = candidate["nodes"][0]
    log["candidates"] = [candidate]
    candidate["nodes"] = [node]
    candidate.update(status="in_progress",predictive_costs_s=[])
    for key in ("mean_cost_s","distinct_effective_observations","distinct_conditional_actions"):
        candidate.pop(key,None)
    log.update(status="fallback",reason="PlanningRefused: incomplete_simulated_tail:compute_deadline",
               selected=deepcopy(log["baseline"]),mechanism_active=False)
    node.update(status="training_choice_frozen" if freeze_choice else "in_progress",actual_evaluation_draws=0)
    node["evaluation_used_support"]["evaluated_worlds"] = 0
    node.pop("coupled_costs_s")
    node.pop("value_s")
    if not freeze_choice:
        node["actual_training_draws"] = 0
        for key in ("selected","training_costs_s","training_used_support"):
            node.pop(key)
    count = 7 if freeze_choice else 3
    log["tail_completion_log"] = log["tail_completion_log"][:count]
    log["tail_completion_log"][-1]["status"] = "not_completed"
    log["tail_completion_log"][-1].pop("cost_s")
    log.update(tail_evaluations=count,tail_evaluations_completed=count-1,
               second_layer_comparisons=int(freeze_choice))
    return summary


def test_coupled_completed_smoke_has_one_reused_evaluation_and_no_eval_ess(coupled):
    checked_history(coupled)
    result = planning_diagnosis(coupled["summary"])
    assert not result["errors"]
    assert result["tail_evaluations_started"] == result["tail_completions_verified_by_ledger"] == 42
    assert result["completed_tails_retained_in_nodes"] == 42
    assert result["mechanism_active"] == 1
    nodes = [n for d in result["decisions"] for a in d["first_actions"] for n in a["nodes"]]
    assert len(nodes) == 6
    assert all(n["actual_evaluation_draws"] == 1 and n["fresh_conditional_evaluation_draws"] == 0 for n in nodes)
    assert all(n["evaluation_used_support"]["evaluated_worlds"] == 1 and not n["evaluation_is_independent_draw"] for n in nodes)
    assert all(n["conditional_training_pool_draws"] == 8 and n["actual_training_draws"] == 2 for n in nodes)


def test_coupled_depth1_has_no_conditional_training():
    with gzip.open(ROOT/"results/round2/coupled_tree/model_smoke/depth1.json.gz","rt",encoding="utf-8") as stream:
        record = json.load(stream)
    result = planning_diagnosis(record["summary"])
    assert not result["errors"]
    assert result["tail_completions_verified_by_ledger"] == 6
    assert result["mechanism_active"] == 0
    nodes = [n for d in result["decisions"] for a in d["first_actions"] for n in a["nodes"]]
    assert all(n["actual_training_draws"] == 0 and n["actual_evaluation_draws"] == 1 for n in nodes)


def test_coupled_partial_training_retains_completed_tails_without_inventing_selection(coupled):
    result = planning_diagnosis(partial_coupled(coupled,False))
    assert not result["errors"]
    assert result["tail_evaluations_started"] == 3
    assert result["tail_completions_verified_by_ledger"] == 2
    assert result["completed_tails_retained_in_nodes"] == 0
    assert result["completed_tails_missing_cost_arrays"] == 2
    assert result["tail_not_completed_explicit"] == 1
    assert result["tail_completion_unknown"] == 0
    node = result["decisions"][0]["first_actions"][0]["nodes"][0]
    assert node["actual_training_draws"] == 0 and node["completed_training_tails"] == 2
    assert node["selected"] is None and result["mechanism_active"] == 0


def test_coupled_eval_failure_retains_frozen_common_action_without_claiming_evaluation(coupled):
    result = planning_diagnosis(partial_coupled(coupled,True))
    assert not result["errors"]
    assert result["tail_evaluations_started"] == 7
    assert result["tail_completions_verified_by_ledger"] == result["completed_tails_retained_in_nodes"] == 6
    assert result["tail_not_completed_explicit"] == 1 and result["tail_completion_unknown"] == 0
    node = result["decisions"][0]["first_actions"][0]["nodes"][0]
    assert node["selected"] and node["actual_training_draws"] == 2 and node["actual_evaluation_draws"] == 0
    assert result["decisions"][0]["partial_candidate_logging_gap"] == 0


def test_coupled_ledger_and_false_fresh_or_ess_claims_are_detected(coupled):
    summary = deepcopy(coupled["summary"])
    log = summary["strategy_parameters"]["observation_tree_log"][0]
    log["tail_evaluations_completed"] -= 1
    node = log["candidates"][0]["nodes"][0]
    node["evaluation_is_independent_draw"] = True
    node["evaluation_used_support"]["equal_draw_weight_ess"] = 1
    errors = planning_diagnosis(summary)["errors"]
    assert any("ledger counters" in e for e in errors)
    assert any("without a fresh draw" in e for e in errors)
    assert any("ESS claim" in e for e in errors)


def test_new_manifest_arms_and_reject_mixed_experiment():
    assert policy_arms({"baseline":{},"coupled_root":{},"coupled_tree":{}}) == ("baseline","coupled_root","coupled_tree")
    assert policy_arms(["baseline","root_mc","observation_tree"]) == ("baseline","root_mc","observation_tree")
    with pytest.raises(ValueError,match="three-arm"):
        policy_arms(["baseline","root_mc","coupled_tree"])


def test_markdown_first_table_remains_contiguous_and_preserves_existing_values():
    data = json.loads((ROOT/"research/round2/diagnoses/observation_tree_pilot_execution.json").read_text(encoding="utf-8"))
    lines = markdown(data).splitlines()
    start = next(i for i,line in enumerate(lines) if line.startswith("|策略|"))
    assert all(line.startswith("|") for line in lines[start:start+2+len(data["aggregates"])])
    assert "3142.907360|1.729053" in "\n".join(lines)
