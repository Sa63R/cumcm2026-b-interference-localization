"""Persistent audit regressions using a fresh local observation-only record."""

from copy import deepcopy

import pytest

from experiments.audit_q3_rollout import audit_record
from experiments.run_q3_rollout_comparison import BASELINE, run_case
from simulation import difficult_scenarios


@pytest.fixture(scope="module")
def baseline_record():
    case = difficult_scenarios(3)[0]
    record = run_case(case, BASELINE, None, max_actions=20_000)
    return record, {case.case_id: case.evaluation_config()}, {BASELINE: None}, 20_000


def test_fresh_local_baseline_record_passes_independent_audit(baseline_record):
    result = audit_record(*baseline_record)
    assert result["audit_passed"]
    assert result["run_successful"]
    assert result["all_cleared"] and result["certified"]
    assert result["source_count"] == result["cleared_count"] == 12
    assert result["completed_coverage_scans"] == 7
    assert result["max_action_time_error_s"] == 0


@pytest.mark.parametrize("tampering,expected_error", [
    ("time", "microsecond ledger"),
    ("clear_distance", "Physical clear mismatch"),
])
def test_physical_or_time_tampering_is_rejected(baseline_record, tampering, expected_error):
    original, cases, configurations, max_actions = baseline_record
    record = deepcopy(original)
    if tampering == "time":
        # Alter both copies so simple policy/engine agreement cannot expose
        # the corruption; the independent physical ledger must reject it.
        record["history"][1]["response"]["virtual_time_s"] += 1
        record["summary"]["action_history"][0]["virtual_time_s"] += 1
    else:
        index = next(i for i, action in enumerate(record["summary"]["action_history"])
                     if action["action"] == "clear" and action["result"] == "success")
        action = record["summary"]["action_history"][index]
        source = next(source for source in record["evaluation"]["ground_truth"]["sources"]
                      if source["channel"] == action["channel"])
        position = [source["x"] + 20.1, source["y"]]
        action["position"] = position
        record["history"][index + 1]["position"] = dict(x=position[0], y=position[1])
    with pytest.raises(ValueError, match=expected_error):
        audit_record(record, cases, configurations, max_actions)


def test_missing_explicit_exit_is_rejected(baseline_record):
    original, cases, configurations, max_actions = baseline_record
    record = deepcopy(original)
    record["history"].pop()
    with pytest.raises(ValueError, match="Missing explicit final /exit"):
        audit_record(record, cases, configurations, max_actions)


def test_incomplete_channel_scan_cannot_support_full_cover_claim(baseline_record):
    original, cases, configurations, max_actions = baseline_record
    record = deepcopy(original)
    occupied = {source["channel"] for source in record["evaluation"]["ground_truth"]["sources"]}
    absent = next(channel for channel in range(1, 21) if channel not in occupied)
    action = next(action for action in record["summary"]["action_history"]
                  if action["phase"] == "coverage" and action["channel"] == absent)
    action["phase"] = "active_localization"
    with pytest.raises(ValueError, match="Reported completed scans differ from trace"):
        audit_record(record, cases, configurations, max_actions)
