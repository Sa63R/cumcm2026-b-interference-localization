from fractions import Fraction as F

from six_disk_certificate import certificate, empty_channel_relaxation, rational_arc_certificate


def test_rational_arc_inequalities_have_exact_certificates():
    record = rational_arc_certificate()
    assert all(value is True for key, value in record.items() if key.endswith('_verified'))


def test_mixed_integer_certificate_and_scaling_gap():
    result = certificate()
    assert F(result['rational_gap']) == F(55, 161901)
    initial = empty_channel_relaxation(True)
    other = empty_channel_relaxation(False)
    assert [r['failed_clears'] for r in initial['candidates']] == [283, 230, 177, 124, 71, 18, 1, 0]
    assert initial['minimum_s'] == 33
    assert other['minimum_s'] == 34
    assert result['full_original_proof_read'] is False
    assert result['does_not_modify_frozen_benchmark_bounds'] is True


def test_zero_failure_bound_is_distinct_from_mixed_bound():
    initial = empty_channel_relaxation(True)
    other = empty_channel_relaxation(False)
    assert min(r['cost_s'] for r in initial['candidates'] if r['failed_clears'] == 0) == 35
    assert min(r['cost_s'] for r in other['candidates'] if r['failed_clears'] == 0) == 36
