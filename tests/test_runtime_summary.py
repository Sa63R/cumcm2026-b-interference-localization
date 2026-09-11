"""Synthetic paired timings verify that repeats are not independent cases."""
from copy import deepcopy
import json

import pytest

from experiments.summarize_runtime_equivalence import (
    bootstrap_case_savings, collapse_cases, digest, group_statistics, restore_summary_channel_keys,
)


def fixture_rows():
    rows = []
    for case_id, lb, values in [
        ("a", 50., {"baseline": [1., 9.], "candidate": [2., 4.]}),
        ("b", 100., {"baseline": [10., 30.], "candidate": [9., 11.]}),
    ]:
        for side, timings in values.items():
            for repeat, timing in enumerate(timings):
                rows.append({"case_id": case_id, "case_sha256": case_id, "split": "development",
                             "side": side, "repeat": repeat, "physical_lower_bound_s": lb,
                             "program_runtime_s": timing, "program_cpu_s": timing / 2,
                             "virtual_time_s": 200., "time_over_physical_lower_bound": 200. / lb,
                             "successful": True, "all_cleared": True, "failed_clear_count": 0})
    return rows


def test_repeat_means_before_case_bootstrap_and_distinct_lower_bound_ratios():
    rows = fixture_rows()
    cases = collapse_cases(rows)
    result = group_statistics(rows, cases)
    assert [c["baseline"]["program_runtime_s"] for c in cases] == [5., 20.]
    assert [c["candidate"]["program_runtime_s"] for c in cases] == [3., 10.]
    wall = result["paired_savings"]["program_runtime_s"]
    assert wall["independent_sample_count"] == 2  # Not four paired repeats.
    assert wall["mean_saved_s"] == 6.
    assert wall["mean_saved_s_percentile_95ci"] == [2., 10.]
    assert wall["candidate_over_baseline_sum"] == 13. / 25.
    assert wall["candidate_slower_case_count"] == 0  # One individual timing was slower.
    assert result["case_mean_statistics"]["baseline"]["program_runtime_s"]["max"] == 20.
    assert result["individual_run_statistics"]["baseline"]["program_runtime_s"]["max"] == 30.
    assert result["mean_physical_lower_bound_s"] == 75.
    assert result["case_mean_statistics"]["baseline"]["mean_time_over_physical_lower_bound"] == 3.
    assert result["case_mean_statistics"]["baseline"]["sum_time_over_sum_physical_lower_bound"] == 400. / 150.


def test_duplicating_repeats_does_not_shrink_case_bootstrap_interval():
    original = fixture_rows()
    duplicates = deepcopy(original)
    for row in duplicates:
        row["repeat"] += 2
    first = group_statistics(original, collapse_cases(original))
    second = group_statistics(original + duplicates, collapse_cases(original + duplicates))
    assert first["paired_savings"] == second["paired_savings"]
    assert second["run_count"] == first["run_count"] * 2


def test_single_case_interval_cannot_be_generalized():
    value = bootstrap_case_savings([4.], [3.])
    assert value["mean_saved_s_percentile_95ci"] == [1., 1.]
    assert value["single_case_ci_is_not_generalization_evidence"]


@pytest.mark.parametrize("baseline,candidate", [([], []), ([1.], []), ([1., 2.], [1.])])
def test_bootstrap_rejects_unpaired_or_empty_cases(baseline, candidate):
    with pytest.raises(ValueError):
        bootstrap_case_savings(baseline, candidate)


def test_missing_partition_is_explicit():
    assert group_statistics([], []) == {"case_count": 0, "not_present": True}


def test_pre_serialization_summary_digest_restores_only_declared_integer_channel_keys():
    original = {"source_estimates": {1: {}, 2: {}, 10: {}}, "unrelated": {"2": 1, "10": 2}}
    parsed = json.loads(json.dumps(original))
    assert digest(parsed) != digest(original)
    before = deepcopy(parsed)
    assert digest(restore_summary_channel_keys(parsed)) == digest(original)
    assert parsed == before
    assert restore_summary_channel_keys(parsed)["unrelated"] == original["unrelated"]
