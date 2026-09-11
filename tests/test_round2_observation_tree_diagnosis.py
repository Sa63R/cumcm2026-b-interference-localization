"""Read-only parser checks on already completed implementation smoke records."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.round2_observation_tree_diagnosis import checked_history, planning_diagnosis

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
