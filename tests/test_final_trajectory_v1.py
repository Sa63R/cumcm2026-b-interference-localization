import pytest

from research.final_trajectory_v1.plot_trajectories import load_matched, select_case


def test_decimal_median_tie_uses_smaller_seed_without_float_roundoff():
    # float(.3)-float(.2) < float(.2)-float(.1), although decimal gaps equal.
    rows = [dict(seed=6001, virtual_time_s=.1), dict(seed=6002, virtual_time_s=.3)]
    seed, selection = select_case(rows)
    assert seed == 6001 and selection['baseline_median_s'] == .2
    assert select_case(list(reversed(rows)))[0] == seed


def test_baseline_selection_does_not_silently_exclude_failed_runs():
    rows = [dict(seed=6001, virtual_time_s=1., successful=True),
            dict(seed=6002, virtual_time_s=10., successful=False),
            dict(seed=6003, virtual_time_s=100., successful=True)]
    assert select_case(rows)[0] == 6002


def test_explicit_case_requires_reason_and_final_gate_precedes_payload_access():
    rows = [dict(seed=6000, virtual_time_s=12.)]
    with pytest.raises(ValueError, match='selection-reason'):
        select_case(rows, 6000)
    assert select_case(rows, 6000, '预先指定的负例')[1]['kind'] == 'explicit'
    with pytest.raises(ValueError, match='not in'):
        select_case(rows, 6001, 'out of partition')
    with pytest.raises(ValueError, match='allow-heldout'):
        load_matched({}, partition='final_random', protocol_path='does-not-exist.json')
