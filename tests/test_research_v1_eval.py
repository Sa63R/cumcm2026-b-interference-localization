import pytest
from experiments.research_v1_eval import (PROTOCOL, make_case, paired_comparison,
                                         read_json, run_case, summarize)


def row(case_id, time, success=True):
    return {"case_id": case_id, "case_sha256": case_id, "virtual_time_s": time,
            "penalized_time_s": time if success else 360000, "successful": success,
            "failed_clear_count": 0, "time_per_source_s": time/10, "program_runtime_s": .1}


def test_failure_cannot_win_by_quitting_early():
    protocol = read_json(PROTOCOL)
    base = [row(str(i), 3000) for i in range(8)]
    candidate = [row(str(i), 2700) for i in range(8)]
    candidate[0] = row("0", 10, False)
    result = paired_comparison(base, candidate, protocol)
    assert not result["performance_target_met_on_supplied_cases"]
    assert result["mean_reduction_fraction"] < 0


def test_missing_or_changed_truth_rejected():
    protocol = read_json(PROTOCOL)
    with pytest.raises(ValueError, match="identical"):
        paired_comparison([row("a", 100)], [row("b", 90)], protocol)
    modified = row("a", 90)
    modified["case_sha256"] = "different"
    with pytest.raises(ValueError, match="truth"):
        paired_comparison([row("a", 100)], [modified], protocol)
    assert not summarize([row("a", 100)], 2)["complete"]


def test_partitions_do_not_overlap():
    protocol = read_json(PROTOCOL)
    parts = list(protocol["partitions"].values())
    for i, p in enumerate(parts):
        for q in parts[i+1:]:
            assert p["seed_stop_exclusive"] <= q["seed_start"] or q["seed_stop_exclusive"] <= p["seed_start"]


def test_real_policy_retains_terminal_truth_and_legal_ledger():
    protocol = read_json(PROTOCOL)
    case = make_case("validation", 6000, protocol)
    spec = {"name": "efficient", "entrypoint": "strategies:run_search", "kwargs": {"variant": "efficient"}}
    record = run_case(case, spec, protocol)
    assert record["row"]["successful"]
    assert record["row"]["cleared_total"] == len(case.sources)
    assert record["evaluation_phase"] == "after_policy_termination"
    assert record["history"][-1]["action"] == "/exit"
