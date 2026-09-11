import copy
import json

import pytest

from experiments.run_q3_fresh import suite as previous_suite
from experiments.run_q3_fresh_round2 import (
    _load_cases, all_lower_bounds, build_suite, pressure_suite, run_case, summarize,
)


def test_development_preserves_original_24_and_adds_twelve():
    cases, metadata = build_suite("development")
    assert cases[:24] == previous_suite(910100, 12, True)
    assert len(cases) == 36
    assert len(set(metadata["suite_groups"].values())) == 3
    nominal, _ = build_suite("nominal")
    pressure, _ = build_suite("pressure")
    assert len(nominal) == 64 and len(pressure) == 32
    assert not {c.seed for c in cases} & {c.seed for c in nominal + pressure}
    custom, _ = build_suite("nominal", start=991123, count=2)
    assert [c.seed for c in custom] == [991123, 991124]


def test_pressure_count_pairs_hold_layout_fixed():
    cases = pressure_suite(960000, 12)
    for i in (0, 2, 4):
        assert len(cases[i].sources) == 14
        assert len(cases[i+1].sources) == 16
        assert cases[i].sources == cases[i+1].sources[:14]
        assert cases[i].seed == cases[i+1].seed
    assert cases[-1].description == "smooth_spatial_error"
    assert cases == pressure_suite(960000, 12)


def test_guarantee_lower_bound_adds_no_confirmation_requirement_for_sixteen():
    c14, c16 = pressure_suite(960000, 2)
    b14, b16 = all_lower_bounds(c14), all_lower_bounds(c16)
    assert b14["guarantee_lower_bound_s"] == pytest.approx(1135.12)
    assert b14["guarantee_lower_bound_s"] >= b14["certified_lower_bound_s"]
    assert b16["confirmation_length_lower_bound_m"] == 0
    assert b16["guarantee_lower_bound_s"] == pytest.approx(b16["physical_lower_bound_s"])


def test_runner_records_complete_traces_ratios_and_failure_visibility(monkeypatch):
    case = pressure_suite(960000, 1)[0]
    # Mock only the exponential evaluator DP, preserving real policy and
    # coverage executions, so this integration check stays lightweight.
    monkeypatch.setattr("experiments.run_q3_fresh_round2.lower_bound", lambda *_: {
        "physical_lower_bound_s": 250.0, "certified_lower_bound_s": 430.0})
    rows, traces = run_case(case, ["B", "C"], group="development_added_pressure")
    assert all(r["success"] for r in rows)
    assert all(r["time_per_source_s"] == r["virtual_time_s"]/14 for r in rows)
    assert all(r["policy_cpu_s"] >= 0 and r["policy_wall_s"] >= 0 for r in rows)
    assert all(traces[name]["observations"][-1]["action"] == "/exit" for name in traces)
    failed = copy.deepcopy(rows[1])
    failed.update(case_id="deliberate-failure", success=False, error="injected", virtual_time_s=1.0)
    report = summarize(rows + [failed])["development_added_pressure"]
    assert report["C"]["failures"] == 1
    assert report["C"]["mean_s"] is None
    assert report["C"]["p95_s"] is None
    assert report["C"]["failure_case_ids"] == ["deliberate-failure"]
    assert report["B"]["mean_s"] == rows[0]["virtual_time_s"]


def test_external_cases_reject_duplicate_ids(tmp_path):
    case = pressure_suite(960000, 1)[0].evaluation_config()
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [case, case]}), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        _load_cases(path)
