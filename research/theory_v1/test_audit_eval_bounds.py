"""Independent corruption, scope and lower/upper certificate checks."""
import math

import pytest

from audit_eval_bounds import audit_record, digest, exact_open_graph, physical_bounds, self_check
from certify_bounds import improved_bound


def lucky_record(n=10):
    """Legally guesses n source locations; does not claim empty certification."""
    sources = [dict(channel=c, x=0., y=0., reception_radius_m=1000., orientation_deg=None)
               for c in range(1, n+1)]
    truth = dict(case_id='q3-validation-lucky', problem=3, seed=6000, sources=sources,
                 error_mode='zero', description='Audit fixture')
    history = [dict(index=0, action='/enter', position=dict(x=0., y=0.), channel=None,
                    response=dict(accepted=True, virtual_time_s=0.))]
    for c in range(1, n+1):
        history.append(dict(index=c, action='/clear', position=dict(x=0., y=0.), channel=c,
                            response=dict(accepted=True, virtual_time_s=5.*c, clear_result='success')))
    history.append(dict(index=n+1, action='/exit', position=dict(x=0., y=0.), channel=None,
                        response=dict(accepted=True, virtual_time_s=5.*n)))
    components = dict(movement_s=0., switching_s=0., detection_s=0., optical_s=3.*n, removal_s=2.*n)
    row = dict(case_id=truth['case_id'], case_sha256=digest(truth), seed=6000,
               strategy='lucky_fixture', action_count=n+2, source_total=n, cleared_total=n,
               measurement_count=0, failed_clear_count=0, virtual_time_s=5.*n,
               all_cleared=True, completion_certified=False, successful=False,
               accepted_exit=True, errors=[], **components)
    evaluation = dict(kind='local_research_only', ground_truth=truth, action_count=n+2,
                      source_total=n, cleared_total=n, measurement_count=0,
                      failed_clear_count=0, time_breakdown_s=components)
    return dict(row=row, evaluation=evaluation, history=history,
                evaluation_phase='after_policy_termination')


def test_flat_dp_matches_seven_independent_exhaustive_nonmetric_oracles():
    assert self_check() == 7
    assert exact_open_graph([], []) == (0., [])


def test_lucky_all_clear_is_not_used_as_guarantee_evidence():
    record = lucky_record()
    sources, moves = audit_record(record)
    assert moves == 10
    assert improved_bound(0, sources)['lower_bound_continuous_s'] > record['row']['virtual_time_s']
    assert not record['row']['successful']  # Full-clear ratio must be null for this row.


def test_n16_disables_information_certification_terms():
    sources, _ = audit_record(lucky_record(16))
    result = improved_bound(0, sources)
    assert result['lower_bound_continuous_s'] == 80
    assert result['spatial_certification_lower_m'] == 0
    assert result['empty_action_and_entry_switch_lower_s'] == 0


def test_failure_before_enter_is_retained_without_full_clear_eligibility():
    record = lucky_record()
    record['history'] = []
    row, evaluation = record['row'], record['evaluation']
    for k in ('action_count', 'cleared_total', 'measurement_count', 'failed_clear_count'):
        row[k] = evaluation[k] = 0
    for k in evaluation['time_breakdown_s']:
        evaluation['time_breakdown_s'][k] = row[k] = 0.
    row.update(virtual_time_s=0., all_cleared=False, accepted_exit=False,
               errors=['Model failed before enter'])
    sources, moves = audit_record(record)
    assert len(sources) == 10 and moves == 0 and not row['successful']


@pytest.mark.parametrize('corruption', ('phase', 'source_hash', 'time', 'clear_feedback', 'ordering'))
def test_audit_rejects_corrupted_or_non_posthoc_records(corruption):
    record = lucky_record()
    if corruption == 'phase':
        record['evaluation_phase'] = 'before_policy_termination'
    elif corruption == 'source_hash':
        record['evaluation']['ground_truth']['sources'][0]['x'] = 1
    elif corruption == 'time':
        record['history'][5]['response']['virtual_time_s'] += 1
    elif corruption == 'clear_feedback':
        record['history'][1]['response']['clear_result'] = 'no_target_in_range'
    else:
        record['history'][5]['action'] = '/exit'
    with pytest.raises(ValueError):
        audit_record(record)


def test_physical_interval_has_independent_additive_certificate():
    for n in (10, 12, 16):
        sources = [dict(channel=i+1, x=1600*math.cos(i*2*math.pi/n),
                        y=1600*math.sin(i*2*math.pi/n)) for i in range(n)]
        result = physical_bounds(sources)
        assert result['physical_clairvoyant_lower_s'] <= result['feasible_clairvoyant_upper_s']
        assert result['physical_bracket_width_s'] <= 8*n-4+1e-4
        assert sorted(result['source_route_order_channels']) == list(range(1, n+1))
