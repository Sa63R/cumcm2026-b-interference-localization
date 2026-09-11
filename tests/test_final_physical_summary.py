import importlib.util
from pathlib import Path

import pytest


_path = Path(__file__).resolve().parents[1] / 'research/final_physical_audit_v1/summarize_audit.py'
_spec = importlib.util.spec_from_file_location('physical_summary', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def _audit(rows):
    return dict(rows=rows, records=len(rows), audit_passed=True, failed_audits=0,
                cache_hits=0, cache_misses=0, wall_s=0)


def _row(time, lower, success=True):
    return dict(label='synthetic', audit_passed=True, successful=success,
                eligible_for_full_clear_ratio=success, virtual_time_s=time,
                penalized_time_s=time if success else 360000., original_lower_s=lower)


def test_ratio_of_means_is_not_mean_of_ratios():
    result = _module.summarize(_audit([_row(10, 1), _row(30, 10)]), 'synthetic')
    bound = result['groups']['synthetic']['bounds']['original']
    assert bound['ratio_of_means'] == pytest.approx(40 / 11)
    assert bound['mean_per_case_ratio'] == 6.5
    assert bound['median_ratio'] == 6.5
    assert bound['p95_ratio'] == pytest.approx(9.65)


def test_failed_short_record_keeps_penalty_and_is_not_a_success_ratio():
    result = _module.summarize(_audit([_row(10, 1), _row(1, 1, False)]), 'synthetic')
    group = result['groups']['synthetic']
    assert group['runs'] == 2 and group['ratio_eligible'] == 1
    assert group['mean_penalized_time_s'] == 180005
    assert group['bounds']['original']['runs'] == 1
    assert group['bounds']['original']['mean_per_case_ratio'] == 10


def test_empty_audit_rejected():
    with pytest.raises(ValueError, match='Empty'):
        _module.summarize(_audit([]), 'synthetic')
