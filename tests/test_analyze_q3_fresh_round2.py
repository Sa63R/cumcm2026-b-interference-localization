"""Independent postprocessing tests with constructed public traces."""

import copy
import gzip
import json
import math

import pytest

from experiments.analyze_q3_fresh_round2 import (
    analyze, audit_trace, deduplicate, fingerprint, load_batches,
    paired_comparison, summarize_strategy, write_outputs,
)


def trace_fixture():
    sources = [dict(channel=c, x=100.0, y=0.0, reception_radius_m=1000.0, orientation_deg=None)
               for c in range(1, 11)]
    scenario = dict(case_id="unit", problem=3, seed=1, sources=sources, error_mode="zero", description="unit")
    observations = [dict(index=0, action="/enter", position=dict(x=0, y=0), channel=None,
                         response=dict(accepted=True, virtual_time_s=0.0))]
    history, position, current_channel = [], (0.0, 0.0), 1
    costs = dict(movement_s=0.0, switching_s=0.0, detection_s=0.0, optical_s=0.0, removal_s=0.0)

    def action(kind, p, channel, result, bearing=None):
        nonlocal position, current_channel
        costs["movement_s"] += round(math.dist(position, p)/5*1_000_000)/1_000_000
        position = p
        if kind == "measure":
            costs["detection_s"] += 5
            costs["switching_s"] += channel != current_channel
            current_channel = channel
        else:
            costs["optical_s"] += 3
            costs["removal_s"] += 2 if result == "success" else 0
        t = round(sum(costs.values()), 6)
        response = dict(accepted=True, virtual_time_s=t)
        response["measure_result" if kind == "measure" else "clear_result"] = result
        row = dict(action=kind, position=list(p), channel=channel, result=result, virtual_time_s=t, phase="fixture")
        if bearing is not None:
            response["svd_deg"] = bearing
            row["bearing_deg"] = bearing
        observations.append(dict(index=len(observations), action="/"+kind,
                                 position=dict(x=p[0], y=p[1]), channel=channel, response=response))
        history.append(row)

    action("measure", (0, 0), 1, "direction", 0.0)
    action("measure", (-0.0, 0), 1, "direction", 0.0)
    for channel in range(1, 11):
        action("clear", (100, 0), channel, "success")
    cover = ((0, 0),)+tuple((1200*math.cos(k*math.pi/3), 1200*math.sin(k*math.pi/3)) for k in range(6))
    for p in cover:
        for channel in range(11, 21):
            action("measure", p, channel, "no_signal")
    total = round(sum(costs.values()), 6)
    observations.append(dict(index=len(observations), action="/exit", position=dict(x=position[0], y=position[1]),
                             channel=None, response=dict(accepted=True, virtual_time_s=total, exit_reason="user_exit")))
    evaluation = dict(case_id="unit", source_total=10, cleared_total=10, cleared_channels=list(range(1, 11)),
                      remaining_channels=[], all_cleared=True, virtual_time_s=total, time_breakdown_s=costs,
                      measurement_count=72, failed_clear_count=0, action_count=len(observations), simulator_stop_reason="exited")
    row = dict(**evaluation, strategy="B", suite_group="unit_group", original_case_group="unit",
               scenario_sha256=fingerprint(scenario), error=None, success=True, completion_certified=True,
               correct_exit=True, incorrect_exit=False, time_per_source_s=total/10, policy_cpu_s=.001,
               policy_wall_s=.002, planner_stats={}, confirmation_tail_s=total-60)
    for label, bound in (("physical", 66.0), ("certified", 366.0), ("guarantee", 1235.12)):
        row[f"{label}_lower_bound_s"] = bound
        row[f"time_over_{label}_lower_bound"] = total/bound
    return dict(scenario=scenario, observations=observations, evaluation=evaluation, row=row, planner_stats={},
                search=dict(action_history=history, completion_certified=True,
                            channels={str(c): "cleared" if c <= 10 else "absent" for c in range(1, 21)}))


def row_fixture(case="unit", strategy="B", time=100, group=None):
    trace = trace_fixture()
    row = copy.deepcopy(trace["row"])
    row.update(case_id=case, strategy=strategy, virtual_time_s=time,
               original_case_group=group or case, audit=dict(passed=True, errors=[]), analysis_success=True,
               batch="test", trace_file="unit.json.gz")
    return row


def save_batch(path, trace, *, status="completed", configs=None):
    path.mkdir()
    configs = configs or [trace["row"]["strategy"]]
    manifest = dict(status=status, cases=1, configs=configs, case_ids=["unit"],
                    completed_runs=1, completed_cases=1, total_cpu_s=.02)
    (path/"results.json").write_text(json.dumps(dict(manifest=manifest, rows=[trace["row"]])), encoding="utf-8")
    with gzip.open(path/f"trace-000-{trace['row']['strategy']}.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(trace, handle)


def test_independent_cost_audit_ignores_phases_and_clear_keeps_measuring_channel():
    trace = trace_fixture()
    assert audit_trace(trace)["passed"]
    for row in trace["search"]["action_history"]:
        row["phase"] = "arbitrary_relabel"
    assert audit_trace(trace)["passed"]


def test_tampered_cost_and_truth_are_detected():
    trace = trace_fixture()
    trace["observations"][2]["response"]["virtual_time_s"] += 1
    assert "action_2:cumulative_cost_mismatch" in audit_trace(trace)["errors"]
    trace = trace_fixture()
    trace["scenario"]["sources"][0]["x"] = 130
    trace["row"]["scenario_sha256"] = fingerprint(trace["scenario"])
    assert any("clear_contradicts_truth" in e for e in audit_trace(trace)["errors"])


def test_repeated_fixed_position_feedback_is_checked_separately_from_error_band():
    trace = trace_fixture()
    trace["observations"][2]["response"]["svd_deg"] = .5
    trace["search"]["action_history"][1]["bearing_deg"] = .5
    audit = audit_trace(trace)
    assert "action_2:fixed_coordinate_feedback_changed" in audit["errors"]
    assert not any("bearing_exceeds" in e for e in audit["errors"])


def test_planning_budget_and_complete_decision_sample_count_are_checked():
    trace = trace_fixture()
    stats = dict(planning_wall_s=60.5, planning_calls=1, decisions=[dict(completed=True,
                 paired_delta_s=[0.0]*11, mean_delta_s=0, baseline_mean_remaining_s=100,
                 candidate_mean_remaining_s=100)])
    trace["planner_stats"] = stats
    trace["row"].update(planner_stats=stats, policy_wall_s=61)
    audit = audit_trace(trace)
    assert "planning_wall_budget_exceeded" in audit["errors"]
    assert "decision_0:not_twelve_paired_deltas" in audit["errors"]


def test_conflicting_duplicate_is_listed_and_never_chooses_faster_replicate():
    a, b = row_fixture(), row_fixture(time=99)
    b["batch"] = "replicate"
    unique, conflicts, _ = deduplicate([a, b])
    assert unique == []
    assert conflicts[0]["conflicting_fields"] == ["virtual_time_s"]
    assert len(conflicts[0]["occurrences"]) == 2
    assert a["deduplication_status"] == b["deduplication_status"] == "conflict_excluded"
    b["virtual_time_s"] = a["virtual_time_s"]
    unique, conflicts, duplicates = deduplicate([a, b])
    assert len(unique) == 1 and not conflicts and len(duplicates) == 1


def test_bootstrap_uses_original_case_groups_and_failed_runs_cannot_win():
    rows = []
    for case, group, delta in (("a-radius-1", "a", 1), ("a-radius-2", "a", 3), ("b", "b", 10)):
        rows += [row_fixture(case, "B", 100, group), row_fixture(case, "C", 100+delta, group)]
    rows += [row_fixture("failed", "B", 100), row_fixture("failed", "C", 1)]
    rows[-1].update(success=False, analysis_success=False, error="partial")
    paired = paired_comparison(rows, "B", "C", bootstrap_draws=100)
    assert paired["original_groups"] == 2
    assert paired["mean_group_delta_s"] == 6
    assert paired["mean_completed_delta_s"] == pytest.approx(14/3)
    assert paired["case_wins"] == 0 and len(paired["excluded_pairs"]) == 1
    summary = summarize_strategy([r for r in rows if r["strategy"] == "C"])
    assert summary["mean_s"] is None and summary["p95_s"] is None
    assert summary["failures"] == 1


def test_only_complete_batches_are_loaded_and_pilot_is_separate(tmp_path):
    trace = trace_fixture()
    save_batch(tmp_path/"complete", trace)
    save_batch(tmp_path/"running", trace, status="running")
    save_batch(tmp_path/"incomplete_count", trace, configs=["B", "C"])
    save_batch(tmp_path/"rollout_smoke", trace)
    rows, batches, skipped = load_batches([tmp_path/name for name in
                                           ("complete", "running", "incomplete_count", "rollout_smoke")])
    assert len(rows) == len(batches) == 2
    assert len(skipped) == 2
    assert rows[1]["suite_group"] == "pilot_rollout_smoke/unit_group"
    assert all(r["audit"]["passed"] for r in rows)


def test_exports_keep_raw_replicates_and_cpu_consumed(tmp_path):
    trace = trace_fixture()
    save_batch(tmp_path/"first", trace)
    save_batch(tmp_path/"repeat", trace)
    report, rows = analyze([tmp_path/"first", tmp_path/"repeat"], bootstrap_draws=10)
    assert report["raw_run_occurrences"] == 2
    assert report["unique_unconflicted_runs"] == 1
    assert report["cpu"]["total_completed_batch_cpu_s"] == .04
    assert report["cpu"]["total_raw_policy_cpu_s"] == .002
    assert report["cpu"]["total_unique_policy_cpu_s"] == .001
    out = tmp_path/"analysis"/"analysis.json"
    write_outputs(report, rows, out)
    assert out.is_file()
    assert len((out.parent/"all_runs.csv").read_text(encoding="utf-8-sig").splitlines()) == 3
