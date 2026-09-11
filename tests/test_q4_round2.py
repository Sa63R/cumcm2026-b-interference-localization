from copy import deepcopy
import pytest
from experiments.run_q4_round2 import report_rows


def rows():
    return [dict(seed=seed, case_sha256=str(seed), strategy=label, successful=True,
        virtual_time_s=time, penalized_time_s=time, common_lower_bound_s=100.,
        penalized_time_over_lower_bound=time/100., program_runtime_s=.1,
        movement_s=time-10, switching_s=1., detection_s=5., optical_s=3., removal_s=1.)
        for seed in (1, 2) for label,time in (("compact_baseline", 500.),("compact_candidate", 400.))]


def test_uses_actual_compact_baseline_and_common_bound():
    report = report_rows(rows())
    assert report["paired_vs_compact_baseline"]["compact_candidate"]["mean_saved_s"] == 100
    assert report["summaries"]["compact_candidate"]["mean_time_over_mean_lower_bound"] == 4


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "different_truth", "different_bound"])
def test_rejects_unpaired_comparisons(corruption):
    records = rows()
    if corruption == "missing":
        records.pop()
    elif corruption == "duplicate":
        records.append(deepcopy(records[0]))
    elif corruption == "different_truth":
        records[-1]["case_sha256"] = "changed"
    else:
        records[-1]["common_lower_bound_s"] = 99
    with pytest.raises(ValueError):
        report_rows(records)


def test_failures_are_penalized_not_dropped():
    records = rows()
    records[-1].update(successful=False, penalized_time_s=360000., penalized_time_over_lower_bound=3600.)
    report = report_rows(records)
    assert report["summaries"]["compact_candidate"]["runs"] == 2
    assert report["summaries"]["compact_candidate"]["mean_time_s"] == 180200
    assert report["paired_vs_compact_baseline"]["compact_candidate"]["all_complete"] is False
