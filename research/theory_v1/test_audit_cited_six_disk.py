import copy

import pytest

from audit_cited_six_disk import BASE_VERSION, LABEL, enhance_batch, enhance_row
from certify_bounds import improved_bound


def row(n=10, *, initial_empty=False, successful=True):
    channels = list(range(2 if initial_empty else 1, n+(2 if initial_empty else 1)))
    old = improved_bound(10000., channels)
    return dict(case_id='synthetic-audit-facts', case_sha256='fixture', seed=6000,
                strategy='fixture', source_total=n, source_route_order_channels=channels,
                **old, movement_rounding_allowance_s=0.000071,
                conditional_machine_lower_s=old['lower_bound_continuous_s']-0.000071,
                virtual_time_s=4000. if successful else 1., successful=successful,
                eligible_for_full_clear_ratio=successful, failed_clear_count=0)


def batch(*rows):
    return dict(version=BASE_VERSION, simulator_requests_sent=False,
                truth_access='post-termination records only', records=len(rows),
                scenarios=len({r['case_sha256'] for r in rows}), rows=list(rows))


@pytest.mark.parametrize('n,increment', [(10, 30), (15, 15), (16, 0)])
@pytest.mark.parametrize('initial_empty', [False, True])
def test_exact_increment_initial_channel_and_n16(n, increment, initial_empty):
    original = row(n, initial_empty=initial_empty)
    result = enhance_row(original)
    added = result[LABEL]
    assert added['increment_continuous_s'] == increment
    assert added['increment_machine_s'] == pytest.approx(increment)
    assert added['movement_rounding_allowance_s'] == original['movement_rounding_allowance_s']
    assert added['empty_action_and_entry_switch_lower_s'] == (
        34*(20-n)-int(initial_empty) if n < 16 else 0)
    assert added['active'] is (n < 16)
    assert all(result[k] == v for k, v in original.items())


def test_observed_zero_failure_is_never_upgraded_to_zero_failure_policy_class():
    original = row()
    result = enhance_row(original)
    original['failed_clear_count'] = 20
    assert enhance_row(original)[LABEL] == result[LABEL]
    assert result[LABEL]['increment_continuous_s'] == 30  # Not 50.


def test_failed_lucky_trajectory_has_no_full_clear_ratio():
    result = enhance_row(row(successful=False))[LABEL]
    assert result['time_over_conditional_lower'] is None
    assert result['nonnegative_gap_s'] is None


def test_no_mutation_no_dp_and_no_double_enhancement(monkeypatch):
    import audit_eval_bounds
    monkeypatch.setattr(audit_eval_bounds, 'exact_open_graph', lambda *args: pytest.fail('DP must not run'))
    original = batch(row())
    snapshot = copy.deepcopy(original)
    result = enhance_batch(original)
    assert original == snapshot
    assert result['dp_calls'] == result['eval_gz_files_opened'] == 0
    with pytest.raises(ValueError, match='original audit_eval_bounds'):
        enhance_batch(result)


@pytest.mark.parametrize('key', ['lower_bound_continuous_s', 'conditional_machine_lower_s',
                                'empty_action_and_entry_switch_lower_s'])
def test_corrupted_old_formula_rejected(key):
    original = row()
    original[key] += 1
    with pytest.raises(ValueError, match='inconsistent'):
        enhance_row(original)


def test_default_scope_and_success_violation():
    original = row()
    original['seed'] = 8000  # Synthetic metadata only; no held-out file exists/read.
    with pytest.raises(ValueError, match='validation'):
        enhance_batch(batch(original))
    original = row()
    original['virtual_time_s'] = original['conditional_machine_lower_s']+1
    with pytest.raises(ValueError, match='violates cited conditional'):
        enhance_row(original)
