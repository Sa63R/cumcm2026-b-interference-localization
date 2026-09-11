import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

import pytest

from experiments.research_v1_physical_audit import (
    ROOT, audit_paths, load_inputs, mixed_disk_cover, observation_audit, polygon_distance)
from planning.disk_cover import disk_cover_radius


def observation_record(actions, *, inferences=(), complete=False, reason='unfinished'):
    history, reported = [], []
    for index, (kind, channel, q, result, value) in enumerate(actions, start=1):
        response = {'accepted': True, 'virtual_time_s': 0.,
                    'measure_result' if kind == 'measure' else 'clear_result': result}
        report = dict(action=kind, channel=channel, position=list(q), result=result, virtual_time_s=0.)
        if kind == 'measure' and result == 'direction':
            response['svd_deg'] = report['bearing_deg'] = value
        if kind == 'clear':
            report['phase'] = value
        history.append(dict(index=index, action='/'+kind, channel=channel,
                            position=dict(x=q[0], y=q[1]), response=response))
        reported.append(report)
    return dict(history=history, summary=dict(action_history=reported,
        strategy_parameters={'inferred_no_signal_constraints': list(inferences)},
        completion_certified_under_model=complete, completion_reason=reason))


def test_power_diagram_mixed_radii_cover_and_interior_hole():
    assert mixed_disk_cover([(-.5, 0., 1.2), (.5, 0., 1.25)], arena=1.)['certified']
    assert mixed_disk_cover([(0., 0., .2), (0., 0., 1.1)], arena=1.)['certified']
    assert not mixed_disk_cover([(0., 0., 1.)], arena=1.)['certified']  # positive margin
    ring = [(math.cos(k*math.pi/3), math.sin(k*math.pi/3), .6) for k in range(6)]
    # Boundary covered, centre excluded: checking only the circumference is wrong.
    assert all(min(math.dist((math.cos(t/100), math.sin(t/100)), p[:2]) for p in ring) < .6
               for t in range(629))
    assert not mixed_disk_cover(ring, arena=1.)['certified']


def test_power_witness_matches_existing_equal_radius_voronoi():
    points = [(0., 0.)]+[(1150*math.cos(k*math.pi/3), 1150*math.sin(k*math.pi/3)) for k in range(6)]
    for radius in (900., 999., 1020.):
        result = mixed_disk_cover([(x, y, radius) for x, y in points])
        exact = disk_cover_radius(points)**2-(radius-1e-5)**2
        assert result['worst_power'] == pytest.approx(exact, abs=1e-7)


def test_prefix_clear_proof_uses_observations_and_rejects_false_claim():
    actions = [('measure', 1, (0., 500.), 'direction', 0.),
               ('measure', 1, (500., 0.), 'direction', 90.),
               ('clear', 1, (500., 500.), 'success', 'rl_certified_clear')]
    result = observation_audit(observation_record(actions))
    assert result['clear_attempts'][0]['prefix_certified']
    actions[-1] = ('clear', 1, (600., 500.), 'success', 'rl_certified_clear')
    with pytest.raises(ValueError, match='Claimed clear certificate'):
        observation_audit(observation_record(actions))
    actions[-1] = ('clear', 1, (600., 500.), 'success', 'optical_attempt')
    result = observation_audit(observation_record(actions))
    assert result['successful_clear_without_prefix_certificate'] == 1


def test_sixteen_detections_do_not_replace_sixteen_actual_clears():
    detected = [('measure', c, (0., 0.), 'near', None) for c in range(1, 17)]
    with pytest.raises(ValueError, match='16 clears'):
        observation_audit(observation_record(detected, reason='source_count_upper_bound_reached'))
    actions = detected+[('clear', c, (0., 0.), 'success', 'near_clear') for c in range(1, 17)]
    result = observation_audit(observation_record(actions, complete=True, reason='source_count_upper_bound_reached'))
    assert result['source_cap_actual_clears'] and result['terminal_certified']
    assert all(c['real_negative_measurements'] == 0 for c in result['absence'])


def test_each_empty_channel_requires_its_own_real_cover_not_aggregate_sites():
    points = [(0., 0.)]+[(1150*math.cos(k*math.pi/3), 1150*math.sin(k*math.pi/3)) for k in range(6)]
    actions = [('measure', c, (0., 0.), 'near', None) for c in range(1, 11)]
    actions += [('clear', c, (0., 0.), 'success', 'near_clear') for c in range(1, 11)]
    actions += [('measure', c, q, 'no_signal', None) for c in range(11, 21) for q in points]
    assert observation_audit(observation_record(actions, complete=True))['terminal_certified']
    with pytest.raises(ValueError, match='absence proof'):
        observation_audit(observation_record(actions[:-1], complete=True))


def test_inferred_silence_checked_at_prefix_and_never_credited_as_real_coverage():
    actions = [('measure', 1, (0., 0.), 'direction', 0.)]
    event = dict(after_actual_action_count=1, channel=1, position=[-1700., 0.], physical_measurement=False)
    result = observation_audit(observation_record(actions, inferences=[event]))
    assert result['inferred_silence_verified'] == 1 and result['inferred_coverage_credits'] == 0
    assert next(c for c in result['absence'] if c['channel'] == 1)['real_negative_measurements'] == 0
    event['position'] = [100., 0.]
    with pytest.raises(ValueError, match='1500m'):
        observation_audit(observation_record(actions, inferences=[event]))
    assert polygon_distance(((0., 0.), (2., 0.), (2., 2.), (0., 2.)), (1., 1.)) == 0.


def test_scope_guard_rejects_heldout_without_opening_payload(tmp_path):
    index = tmp_path/'inputs.json'
    index.write_text(json.dumps([dict(label='x', path='case-6048.json.gz', sha256='unused')]))
    with pytest.raises(ValueError, match='outside opened'):
        load_inputs(index)
    with pytest.raises(ValueError, match='independent final'):
        load_inputs(index, allow_heldout=True)
    index.write_text(json.dumps(dict(kind='final_identity_archive_audit')))
    with pytest.raises(ValueError, match='explicit'):
        load_inputs(index)


def test_short_failure_stays_in_audit_and_has_no_success_ratio(tmp_path):
    # Previously opened development case; no simulator or fresh truth generation.
    original = ROOT.parent/'q3-v1-artifacts/baseline-validation/validation-rollout/case-6000.json.gz'
    record = json.load(gzip.open(original, 'rt'))
    record = copy.deepcopy(record)
    record['history'] = record['history'][:1]
    parts = dict(movement_s=0., switching_s=0., detection_s=0., optical_s=0., removal_s=0.)
    record['row'].update(successful=False, all_cleared=False, completion_certified=False, accepted_exit=False,
        cleared_total=0, measurement_count=0, failed_clear_count=0, action_count=1,
        virtual_time_s=0., penalized_time_s=360000., errors=['synthetic early stop'], **parts)
    record['evaluation'].update(cleared_total=0, all_cleared=False, measurement_count=0,
        failed_clear_count=0, action_count=1, virtual_time_s=0., time_breakdown_s=parts)
    record['summary'] = observation_record([])['summary']
    path = tmp_path/'failure.json.gz'
    with gzip.open(path, 'wt') as stream:
        json.dump(record, stream)
    items = [dict(label='synthetic-failure', path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())]
    theory = ROOT.parent/'q3-state-search/research/theory_v1'
    result = audit_paths(items, theory_dir=theory, cache_path=tmp_path/'cache.json',
                         seed_cache=theory/'results/validation_geometry_cache.json', cited_six_disk=True)
    assert result['audit_passed'] and result['records'] == 1
    assert result['cache_hits'] == 1 and result['cache_misses'] == 0
    row = result['rows'][0]
    assert not row['successful'] and row['penalized_time_s'] == 360000.
    assert row['time_over_original_lower'] is None
    assert row['cited_six_disk']['time_over_conditional_lower'] is None


def test_empty_batch_cannot_pass_and_missing_archive_is_retained(tmp_path):
    theory = ROOT.parent/'q3-state-search/research/theory_v1'
    with pytest.raises(ValueError, match='empty batch'):
        audit_paths([], theory_dir=theory, cache_path=tmp_path/'empty.json')
    result = audit_paths([dict(label='missing', path=str(tmp_path/'absent.json.gz'), sha256='unused')],
                         theory_dir=theory, cache_path=tmp_path/'missing-cache.json')
    assert result['records'] == 1 and result['failed_audits'] == 1 and not result['audit_passed']
