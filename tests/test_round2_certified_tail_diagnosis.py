"""Artificial archive fixtures only; never run an environment or real dataset."""

from copy import deepcopy

from experiments.round2_certified_tail_diagnosis import audit_pair


def archive(candidate=False):
    route = [(1, 95, 0, 29.), (2, 195, 0, 54.)] if candidate else [(1, 100, 0, 30.), (2, 200, 0, 55.)]
    history = [{"index": 0, "action": "/enter", "position": {"x": 0, "y": 0}, "channel": None,
                "response": {"accepted": True, "virtual_time_s": 0., "real_timestamp_ms": 1}}]
    summary = []
    for index, (action, channel, x, y, t, result) in enumerate(
            [("measure", 1, 0, 0, 5., "direction")] +
            [("clear", c, x, y, t, "success") for c, x, y, t in route], 1):
        response = {"accepted": True, "virtual_time_s": t,
                    "measure_result" if action == "measure" else "clear_result": result,
                    "real_timestamp_ms": 2 if candidate else 1}
        history.append({"index": index, "action": "/" + action, "position": {"x": x, "y": y},
                        "channel": channel, "response": response})
        summary.append({"action": action, "position": [x, y], "channel": channel,
                        "virtual_time_s": t, "result": result})
    history.append({"index": 4, "action": "/exit", "position": {"x": route[-1][1], "y": 0}, "channel": None,
                    "response": {"accepted": True, "virtual_time_s": route[-1][3]}})
    log = {"accepted": True, "after_actual_action_count": 1, "source_count": 2,
           "position": [0, 0], "start_time_us": 5_000_000,
           "baseline_visits": [[1, 100, 0], [2, 200, 0]], "baseline_cost_us": 50_000_000,
           "candidate_visits": [[1, 95, 0], [2, 195, 0]], "candidate_cost_us": 49_000_000,
           "gain_us": 1_000_000, "executed": [[1, 95, 0, 29.], [2, 195, 0, 54.]],
           "completed": True, "actual_cost_us": 49_000_000,
           "dominance_status": "complete_execution_matches_compared_route"}
    return {"row": {"seed": 1, "case_sha256": "artificial", "successful": True,
                    "failed_clear_count": 0, "virtual_time_s": route[-1][3]},
            "summary": {"action_history": summary,
                        "strategy_parameters": {"certified_tail_log": [log] if candidate else []}},
            "history": history}


def test_complete_commit_checks_independent_physical_suffix():
    result = audit_pair(archive(), archive(True))
    assert result["audit_passed"] and result["whole_episode_nonregression_verified"]
    assert result["whole_gain_s"] == 1.
    assert result["tails"][0]["baseline_suffix_matches"]


def test_unchanged_no_acceptance_is_verified():
    result = audit_pair(archive(), archive())
    assert result["physical_history_identical"] and result["whole_episode_nonregression_verified"]


def test_equal_total_cost_does_not_hide_changed_preacceptance_action():
    changed = archive(True)
    changed["history"][1]["channel"] = 2
    changed["summary"]["action_history"][0]["channel"] = 2
    result = audit_pair(archive(), changed)
    assert not result["audit_passed"]
    assert any("actual baseline prefix" in error for error in result["errors"])


def test_planner_self_claim_does_not_override_real_baseline_suffix():
    changed = archive(True)
    changed["summary"]["strategy_parameters"]["certified_tail_log"][0]["baseline_visits"][0][1] = 99
    result = audit_pair(archive(), changed)
    assert not result["audit_passed"]
    assert any("complete actual baseline suffix" in error for error in result["errors"])


def test_cancellation_cannot_keep_completed_dominance():
    changed = archive(True)
    changed["summary"]["strategy_parameters"]["certified_tail_log"][0]["cancelled"] = "state_changed"
    result = audit_pair(archive(), changed)
    assert not result["audit_passed"] and not result["whole_episode_nonregression_verified"]


def test_cancelled_local_record_does_not_claim_whole_episode_nonregression():
    changed = archive(True)
    entry = changed["summary"]["strategy_parameters"]["certified_tail_log"][0]
    entry.update(completed=False, cancelled="state_changed", dominance_status="execution_preconditions_invalidated")
    result = audit_pair(archive(), changed)
    assert result["audit_passed"] and not result["whole_episode_nonregression_verified"]


def test_missing_execution_cannot_be_called_completed():
    changed = archive(True)
    changed["summary"]["strategy_parameters"]["certified_tail_log"][0]["executed"].pop()
    result = audit_pair(archive(), changed)
    assert not result["audit_passed"]
    assert any("whole committed suffix" in error for error in result["errors"])


def test_no_accepted_tail_cannot_explain_a_changed_physical_episode():
    changed = archive(True)
    changed["summary"]["strategy_parameters"]["certified_tail_log"] = []
    result = audit_pair(archive(), changed)
    assert not result["audit_passed"]
    assert result["errors"] == ["episode changed without any accepted tail"]
