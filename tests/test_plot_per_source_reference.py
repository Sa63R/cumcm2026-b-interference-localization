"""Pure synthetic summaries; no benchmark, physical case, or hidden truth."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import pytest


SCRIPT = Path(__file__).resolve().parents[1]/'research/q4_round2/plot_per_source_reference.py'
SPEC = importlib.util.spec_from_file_location('per_source_reference_plot', SCRIPT)
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def fixture_summary(kind='random'):
    rows = []
    for n in range(10, 17):
        for j in range(2):
            parts = dict(movement_s=(150+n+10*j)*n, detection_s=(30+j)*n,
                         switching_s=(1+.1*n)*n, optical_s=(15+2*j)*n, removal_s=2*n)
            t = sum(parts.values())
            rows.append(dict(seed=1000+2*n+j, strategy='compact_reference', source_total=n,
                cleared_total=n, successful=True, stage='stress' if kind == 'stress' else 'confirmation',
                virtual_time_s=t, penalized_time_s=t, **parts))
    values = [r['virtual_time_s']/r['source_total'] for r in rows]
    result = dict(complete=True, source_unchanged=True, infrastructure_errors=[], failure_rows=[],
        all_clear=True, primary_metric='equal-run mean(T_i/N_i), failed T_i=360000',
        failure_penalty_s=360000., runs=len(rows), successful=len(rows), rows=rows,
        mean_time_per_source_s=statistics.mean(values), mean_actual_time_per_source_s=statistics.mean(values),
        p95_time_per_source_s=statistics.quantiles(values, n=20, method='inclusive')[-1],
        mean_time_s=statistics.mean(r['virtual_time_s'] for r in rows),
        total_time_s=sum(r['virtual_time_s'] for r in rows), total_sources=sum(r['source_total'] for r in rows),
        pooled_time_per_source_s=sum(r['virtual_time_s'] for r in rows)/sum(r['source_total'] for r in rows),
        mean_components_s={key: statistics.mean(r[key] for r in rows) for key, _, _ in plot.COMPONENTS})
    result['by_source_count'] = {}
    for n in range(10, 17):
        group = sorted(r['virtual_time_s']/n for r in rows if r['source_total'] == n)
        result['by_source_count'][str(n)] = dict(runs=2, successful=2, all_clear=True,
            mean_time_per_source_s=(group[0]+group[1])/2,
            p95_time_per_source_s=.05*group[0]+.95*group[1])
    return result


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def test_primary_stack_divides_each_case_before_averaging_and_p95_is_descriptive():
    summary = fixture_summary()
    original = copy.deepcopy(summary)
    data = plot.validate_summary(summary, 'random')
    assert summary == original
    # N balanced over 10..16, j over 0..1: E(move/N)=150+13+5.
    assert data['mean_components_per_source_s']['movement_s'] == pytest.approx(168.)
    pooled_move = sum(r['movement_s'] for r in summary['rows'])/sum(r['source_total'] for r in summary['rows'])
    assert pooled_move != pytest.approx(168.)
    assert sum(data['mean_components_per_source_s'].values()) == pytest.approx(data['mean_time_per_source_s'])
    for entry in data['by_source_count']:
        group = [r['virtual_time_s']/r['source_total'] for r in summary['rows'] if r['source_total'] == entry['source_count']]
        assert entry['p95_s'] == pytest.approx(min(group)+.95*(max(group)-min(group)))


@pytest.mark.parametrize('field,value', [('complete', False), ('source_unchanged', False),
    ('all_clear', False), ('infrastructure_errors', [{'error': 'unknown case'}]),
    ('failure_rows', [{'successful': False}]), ('runs', 13), ('successful', 13)])
def test_incomplete_unknown_or_failed_summary_rejected(field, value):
    summary = fixture_summary()
    summary[field] = value
    with pytest.raises(ValueError):
        plot.validate_summary(summary, 'random')


@pytest.mark.parametrize('kind', ['failed', 'unknown_time', 'missing_fee', 'fee_sum', 'nonfinite',
    'duplicate', 'missing_stratum', 'unbalanced', 'wrong_method', 'wrong_stage', 'uncleared'])
def test_invalid_rows_are_never_silently_filtered(kind):
    summary = fixture_summary()
    row = summary['rows'][0]
    if kind == 'failed': row['successful'] = False
    elif kind == 'unknown_time': row['virtual_time_s'] = None
    elif kind == 'missing_fee': del row['movement_s']
    elif kind == 'fee_sum': row['optical_s'] += 1
    elif kind == 'nonfinite': row['movement_s'] = float('inf')
    elif kind == 'duplicate': row['seed'] = summary['rows'][1]['seed']
    elif kind == 'missing_stratum': summary['rows'] = [r for r in summary['rows'] if r['source_total'] != 10]
    elif kind == 'unbalanced': summary['rows'].pop()
    elif kind == 'wrong_method': row['strategy'] = 'another_method'
    elif kind == 'wrong_stage': row['stage'] = 'stress'
    elif kind == 'uncleared': row['cleared_total'] -= 1
    summary['runs'] = summary['successful'] = len(summary['rows'])
    with pytest.raises(ValueError):
        plot.validate_summary(summary, 'random')


@pytest.mark.parametrize('kind', ['primary', 'per_n_p95', 'raw_components', 'pooled_metric'])
def test_published_aggregates_must_match_recomputed_all_rows(kind):
    summary = fixture_summary()
    if kind == 'primary': summary['mean_time_per_source_s'] += 1
    elif kind == 'per_n_p95': summary['by_source_count']['10']['p95_time_per_source_s'] += 1
    elif kind == 'raw_components': summary['mean_components_s']['movement_s'] += 1
    else: summary['pooled_time_per_source_s'] += 1
    with pytest.raises(ValueError):
        plot.validate_summary(summary, 'random')


def test_rejection_happens_before_creating_any_figure_directory(tmp_path):
    random_path, stress_path = tmp_path/'random.json', tmp_path/'stress.json'
    save(random_path, fixture_summary())
    value = fixture_summary('stress')
    value['rows'][0]['virtual_time_s'] = None
    save(stress_path, value)
    with pytest.raises(ValueError):
        plot.render(random_path, stress_path, tmp_path/'figures')
    assert not (tmp_path/'figures').exists()


def test_two_inputs_must_refer_to_same_reference_method(tmp_path):
    random_path, stress_path = tmp_path/'random.json', tmp_path/'stress.json'
    save(random_path, fixture_summary())
    value = fixture_summary('stress')
    for row in value['rows']: row['strategy'] = 'another_reference'
    save(stress_path, value)
    with pytest.raises(ValueError, match='same strategy'):
        plot.render(random_path, stress_path, tmp_path/'figures')


def test_render_four_standalone_artifacts_with_source_hashes_and_no_overwrite(tmp_path):
    random_path, stress_path = tmp_path/'random.json', tmp_path/'stress.json'
    save(random_path, fixture_summary())
    save(stress_path, fixture_summary('stress'))
    output = tmp_path/'figures'
    metadata = plot.render(random_path, stress_path, output, title='纯构造格式QA（不是实验结果）')
    assert {p.name for p in output.iterdir()} == set(plot.OUTPUTS)
    for name in plot.OUTPUTS[:-1]:
        data = (output/name).read_bytes()
        assert data.startswith(b'\x89PNG' if name.endswith('.png') else b'%PDF')
        assert len(data) > 1000
        assert metadata['output_sha256'][name] == hashlib.sha256(data).hexdigest()
    assert metadata['inputs']['random']['sha256'] == hashlib.sha256(random_path.read_bytes()).hexdigest()
    assert metadata['inputs']['stress']['sha256'] == hashlib.sha256(stress_path.read_bytes()).hexdigest()
    assert metadata['target_s'] == 460 and 'no per-N threshold' in metadata['target_scope']
    assert 'not a CI' in metadata['percentile']
    saved = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match='Preserve existing'):
        plot.render(random_path, stress_path, output)
    assert saved == {p.name: p.read_bytes() for p in output.iterdir()}
