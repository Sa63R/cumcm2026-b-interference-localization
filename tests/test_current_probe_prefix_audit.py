"""Mutation checks on the explicitly already-opened implementation smoke only."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from experiments.current_probe_prefix_audit import audit_current_probe
from experiments.round2_posthoc_audit import load_helpers


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def legacy():
    return load_helpers()[1]


@pytest.fixture
def enabled():
    with gzip.open(ROOT/"results/round2/current_probe/model_smoke/enabled.json.gz", "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    return {"summary": record["summary"], "history": record["history"],
            "evaluation": {"time_breakdown_s": record["evaluation"]["time_breakdown_s"]}}


def test_all_public_prefixes_and_actual_current_selections_replay(enabled, legacy):
    original = deepcopy(enabled)
    result = audit_current_probe(enabled, legacy.observation_audit(enabled, {}))
    assert result["passed"] and result["localization_records"] == 24
    assert result["eligible"] == 5 and result["new_current_selected"] == 3
    assert result["dependency_unchanged_during_audit"] and result["dependency_sha256"]
    selected = [r for r in result["records"] if r["new_current_selected"]]
    assert [r["channel"] for r in selected] == [11, 18, 3]
    assert all(r["exact_input_vertex_reception_verified"] and r["real_movement_s"] == 0
               and r["actual_probe_bill_s"] == 6 for r in selected)
    assert enabled == original
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mutation,error", [
    ("future", "immediate actual public prefix"),
    ("current", "real current position"),
    ("fresh", "real freshness"),
    ("eligibility", "gate eligibility"),
    ("baseline_score", "original score_s"),
    ("original_9_as_23", "baseline_proposed_candidates"),
    ("false_activation", "new current activation"),
    ("free_measurement", "billing"),
])
def test_tampered_decisions_are_rejected(enabled, legacy, mutation, error):
    observations = legacy.observation_audit(enabled, {})
    logs = enabled["summary"]["strategy_parameters"]["current_probe_log"]
    row = logs[7]
    if mutation == "future":
        row["after_actual_action_count"] += 1
    elif mutation == "current":
        row["current_position"][0] += 1
    elif mutation == "fresh":
        logs[0]["fresh_for_channel"] = True
    elif mutation == "eligibility":
        row["eligible"] = False
    elif mutation == "baseline_score":
        row["baseline_probe_log"]["score_s"] -= 1
    elif mutation == "original_9_as_23":
        row["baseline_proposed_candidates"] = 9
    elif mutation == "false_activation":
        row["new_current_selected"] = False
    else:
        ordinal = row["actual_action_ordinal"]
        enabled["history"][ordinal]["response"]["virtual_time_s"] -= 5
        enabled["summary"]["action_history"][ordinal-1]["virtual_time_s"] -= 5
        row["actual_virtual_time_s"] -= 5
    with pytest.raises(ValueError, match=error):
        audit_current_probe(enabled, observations)


def test_baseline_without_any_new_fields_uses_legacy_and_actual_ledger(legacy):
    path = ROOT/"results/round2/feedback/pilot/records/baseline-200114.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    result = audit_current_probe(record, legacy.observation_audit(record, {}))
    assert result["passed"] and not result["records"]
    assert result["observations"]["terminal_certified"]


def test_unmatched_pending_probe_is_explicitly_rejected_not_called_executed(enabled, legacy):
    observations = legacy.observation_audit(enabled, {})
    pending = deepcopy(enabled["summary"]["strategy_parameters"]["current_probe_log"][-1])
    pending.update(execution_status="planned", after_actual_action_count=len(enabled["summary"]["action_history"]))
    enabled["summary"]["strategy_parameters"]["current_probe_log"].append(pending)
    with pytest.raises(ValueError, match="does not cover actual calls"):
        audit_current_probe(enabled, observations)
