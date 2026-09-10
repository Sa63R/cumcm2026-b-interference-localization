"""Report integration checks using synthetic metadata only, never real cases."""
import copy
import sys

import pytest

from experiments import research_v1_eval as harness
from experiments import research_v1_report as reporting


@pytest.fixture
def protocol(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Reporting tests must not construct or run simulator cases')
    monkeypatch.setattr(harness, 'make_case', forbidden)
    monkeypatch.setattr(harness, 'run_case', forbidden)
    return dict(schema_version=1, name='synthetic-report-test-only',
        partitions={name: dict(seed_start=start, seed_stop_exclusive=start+2)
                    for name, start in [('validation', 1), ('final_random', 11),
                                        ('final_stress', 21), ('future_reserved', 31)]},
        limits=dict(virtual_seconds_per_case=360000),
        acceptance=dict(minimum_mean_total_time_reduction_fraction=.05,
            paired_saving_bootstrap_ci95_lower_seconds_strictly_above=0,
            maximum_p95_total_time_ratio=1.05, bootstrap_resamples=128, bootstrap_seed=77))


def evaluation(root, protocol, split='validation', method='candidate', cost=900.):
    """No hidden coordinates, generated scenario, or checkpoint file is needed."""
    directory = root / f'{split}-{method}'
    part = protocol['partitions'][split]
    seeds = list(range(part['seed_start'], part['seed_stop_exclusive']))
    identity = dict(source_sha256={'synthetic_source.py': 'frozen-source'},
                    spec_sha256=f'frozen-{method}', checkpoint_sha256={'checkpoint': f'frozen-{method}'},
                    protocol_sha256=harness.digest(protocol))
    manifest = dict(split=split, seeds=seeds, strategy=method, identity=identity)
    rows = [dict(case_id=f'synthetic-{split}-{seed}', case_sha256=f'synthetic-truth-{seed}',
        seed=seed, strategy=method, successful=True, all_cleared=True,
        completion_certified=True, accepted_exit=True, cleared_total=10, source_total=10,
        virtual_time_s=cost, penalized_time_s=cost, time_per_source_s=cost/10,
        program_runtime_s=.01, measurement_count=12, failed_clear_count=0,
        movement_s=cost-50, switching_s=0., detection_s=0., optical_s=30., removal_s=20.,
        errors=[]) for seed in seeds]
    harness.write_json(directory / 'manifest.json', manifest)
    harness.write_json(directory / 'rows.json', rows)
    return directory


def alter_rows(directory, change):
    rows = harness.read_json(directory / 'rows.json')
    change(rows)
    harness.write_json(directory / 'rows.json', rows)


def fail_first(rows):
    rows[0].update(successful=False, all_cleared=False, completion_certified=False,
                   cleared_total=0, virtual_time_s=1., penalized_time_s=360000,
                   time_per_source_s=.1, movement_s=1., optical_s=0., removal_s=0.,
                   errors=['synthetic early failure'])


def comparison(root, protocol, split='validation', candidate_cost=900.):
    baseline = evaluation(root, protocol, split, 'baseline', 1000.)
    candidate = evaluation(root, protocol, split, 'candidate', candidate_cost)
    return reporting.compare_directories(baseline, {'candidate': candidate}, protocol)


@pytest.mark.parametrize('corruption,match', [
    ('missing', 'Missing'), ('duplicate_seed', 'duplicate'), ('duplicate_id', 'Duplicate case id'),
    ('manifest_subset', 'complete declared'), ('protocol', 'protocol differs'),
    ('strategy', 'strategy'), ('false_success', 'Success flag'), ('wrong_penalty', 'time penalty')])
def test_rejects_incomplete_or_inconsistent_directory(tmp_path, protocol, corruption, match):
    directory = evaluation(tmp_path, protocol)
    if corruption in ('manifest_subset', 'protocol'):
        manifest = harness.read_json(directory / 'manifest.json')
        if corruption == 'manifest_subset':
            manifest['seeds'] = manifest['seeds'][:1]
        else:
            manifest['identity']['protocol_sha256'] = 'different'
        harness.write_json(directory / 'manifest.json', manifest)
    else:
        rows = harness.read_json(directory / 'rows.json')
        if corruption == 'missing':
            rows.pop()
        elif corruption == 'duplicate_seed':
            rows[1]['seed'] = rows[0]['seed']
        elif corruption == 'duplicate_id':
            rows[1]['case_id'] = rows[0]['case_id']
        elif corruption == 'strategy':
            rows[0]['strategy'] = 'different'
        elif corruption == 'false_success':
            rows[0]['accepted_exit'] = False
        else:
            fail_first(rows)
            rows[0]['penalized_time_s'] = 1.
        harness.write_json(directory / 'rows.json', rows)
    with pytest.raises(ValueError, match=match):
        reporting.load_evaluation(directory, protocol)


def test_failed_short_run_remains_penalized_in_report_and_markdown(tmp_path, protocol):
    baseline = evaluation(tmp_path, protocol, method='baseline', cost=1000.)
    candidate = evaluation(tmp_path, protocol)
    alter_rows(candidate, fail_first)
    report = reporting.compare_directories(baseline, {'candidate': candidate}, protocol)
    method, contrast = report['methods']['candidate'], report['comparisons']['candidate']
    assert method['runs'] == 2 and method['successful_runs'] == 1
    assert method['raw_mean_total_time_s'] == 450.5
    assert method['penalized_mean_total_time_s'] == 180450.
    assert method['penalized_p95_total_time_s'] > 300000
    assert contrast['paired_cases'][0]['candidate_s'] == 360000
    assert contrast['mean_reduction_fraction'] < 0
    assert not contrast['performance_target_met_on_supplied_cases']
    text = reporting.markdown(report)
    assert '180450.00' in text and 'predeclared penalty' in text
    assert report['final_suite_acceptance'] is None


@pytest.mark.parametrize('setting,value', [
    ('minimum_mean_total_time_reduction_fraction', .2),
    ('paired_saving_bootstrap_ci95_lower_seconds_strictly_above', 100.),
    ('maximum_p95_total_time_ratio', .85)])
def test_report_honors_each_protocol_acceptance_threshold(tmp_path, protocol, setting, value):
    protocol['acceptance'][setting] = value
    report = comparison(tmp_path, protocol)
    assert not report['comparisons']['candidate']['performance_target_met_on_supplied_cases']


def test_complete_matching_frozen_final_partitions_pass(tmp_path, protocol):
    random = comparison(tmp_path, protocol, 'final_random')
    stress = comparison(tmp_path, protocol, 'final_stress')
    result = reporting.final_acceptance(random, stress, protocol)['acceptance']['candidate']
    assert result['first_version_practical_target_met']
    assert result['baseline_complete_and_successful']


@pytest.mark.parametrize('identity_field', ['source_sha256', 'spec_sha256', 'checkpoint_sha256'])
@pytest.mark.parametrize('method', ['baseline', 'candidate'])
def test_final_rejects_changed_frozen_identity(tmp_path, protocol, identity_field, method):
    random = comparison(tmp_path, protocol, 'final_random')
    stress = comparison(tmp_path, protocol, 'final_stress')
    stress['inputs'][method]['manifest']['identity'][identity_field] = 'changed'
    with pytest.raises(ValueError, match='changed between final partitions'):
        reporting.final_acceptance(random, stress, protocol)


@pytest.mark.parametrize('split', ['final_random', 'final_stress'])
@pytest.mark.parametrize('method', ['baseline', 'candidate'])
def test_any_failed_final_case_blocks_acceptance(tmp_path, protocol, split, method):
    random = comparison(tmp_path, protocol, 'final_random')
    stress = comparison(tmp_path, protocol, 'final_stress')
    directory = tmp_path / f'{split}-{method}'
    alter_rows(directory, fail_first)
    changed = reporting.compare_directories(tmp_path / f'{split}-baseline',
                                            {'candidate': tmp_path / f'{split}-candidate'}, protocol)
    if split == 'final_random':
        random = changed
    else:
        stress = changed
    result = reporting.final_acceptance(random, stress, protocol)['acceptance']['candidate']
    assert not result['first_version_practical_target_met']
    if method == 'baseline':
        assert not result['baseline_complete_and_successful']


@pytest.mark.parametrize('reason', ['incomplete', 'candidate_failed_clear', 'stress_tail'])
def test_final_completion_reliability_and_separate_stress_guard(tmp_path, protocol, reason):
    random = comparison(tmp_path, protocol, 'final_random')
    stress = comparison(tmp_path, protocol, 'final_stress', candidate_cost=1500. if reason == 'stress_tail' else 900.)
    if reason == 'incomplete':
        stress['methods']['candidate']['complete'] = False
    elif reason == 'candidate_failed_clear':
        stress['methods']['candidate']['failed_clear_count'] = 1
    assert not reporting.final_acceptance(random, stress, protocol)['acceptance']['candidate']['first_version_practical_target_met']


def test_final_rejects_wrong_partition_protocol_or_methods(tmp_path, protocol):
    random = comparison(tmp_path, protocol, 'final_random')
    stress = comparison(tmp_path, protocol, 'final_stress')
    with pytest.raises(ValueError, match='random and stress'):
        reporting.final_acceptance(stress, random, protocol)
    changed = copy.deepcopy(stress)
    changed['protocol_sha256'] = 'other'
    with pytest.raises(ValueError, match='protocol mismatch'):
        reporting.final_acceptance(random, changed, protocol)
    changed = copy.deepcopy(stress)
    del changed['inputs']['candidate']
    with pytest.raises(ValueError, match='different methods'):
        reporting.final_acceptance(random, changed, protocol)


def test_final_cli_uses_both_reports_without_reading_real_protocol(tmp_path, protocol, monkeypatch):
    random = comparison(tmp_path, protocol, 'final_random')
    stress = comparison(tmp_path, protocol, 'final_stress')
    for name, contents in [('random', random), ('stress', stress), ('protocol', protocol)]:
        harness.write_json(tmp_path / f'{name}.json', contents)
    monkeypatch.setattr(reporting, 'PROTOCOL', tmp_path / 'protocol.json')
    output = tmp_path / 'acceptance'
    monkeypatch.setattr(sys, 'argv', ['report', '--final-random-report', str(tmp_path / 'random.json'),
        '--final-stress-report', str(tmp_path / 'stress.json'), '--output', str(output)])
    reporting.main()
    result = harness.read_json(output / 'final_acceptance.json')
    assert result['acceptance']['candidate']['first_version_practical_target_met']
