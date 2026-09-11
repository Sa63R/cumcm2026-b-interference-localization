"""Plot two completed single-arm, source-count-stratified Q4 summaries.

No strategy, simulator, raw scene, or hidden truth is read. All plotted values
are independently recomputed from every summary row. This validates arithmetic
and declared completeness, not the separate physical/qualification audit.

Example (only after the benchmark and its audits have completed):
  python -B research/q4_round2/plot_per_source_reference.py \
    --random-summary PATH/random/summary.json --stress-summary PATH/stress/summary.json \
    --output-dir PATH/figures
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


TARGET_S = 460.
COMPONENTS = (
    ('movement_s', '移动', '#426F95'),
    ('detection_s', '测量', '#36968B'),
    ('switching_s', '换频', '#D6AB47'),
    ('optical_s', '光学检查', '#BB765D'),
    ('removal_s', '移除', '#887B9C'),
)
OUTPUTS = ('per_source_by_count.png', 'per_source_by_count.pdf',
           'per_source_components.png', 'per_source_components.pdf',
           'per_source_reference_inputs.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f'Missing, nonfinite or negative {name}')
    return float(value)


def match(value, expected, name):
    value = number(value, name)
    require(math.isclose(value, expected, rel_tol=1e-11, abs_tol=1e-6), f'Summary mismatch: {name}')


def percentile(values, q=.95):
    values = sorted(values)
    at = q * (len(values)-1)
    low, high = math.floor(at), math.ceil(at)
    return values[low] * (1-(at-low)) + values[high] * (at-low)


def validate_summary(summary, kind):
    require(kind in ('random', 'stress'), 'Unknown dataset kind')
    require(summary.get('complete') is True and summary.get('source_unchanged') is True,
            'Refuse a complete-performance plot for an incomplete or source-changed batch')
    require(summary.get('infrastructure_errors') == [] and summary.get('failure_rows') == [],
            'Infrastructure errors or failed cases may not be silently dropped')
    require(summary.get('all_clear') is True, 'Only declared all-clear batches may use this plot')
    require(summary.get('primary_metric') == 'equal-run mean(T_i/N_i), failed T_i=360000'
            and summary.get('failure_penalty_s') == 360000., 'Unexpected primary metric/penalty convention')
    rows = summary.get('rows')
    require(isinstance(rows, list) and bool(rows), 'Need every original summary row')
    require(type(summary.get('runs')) is int and summary['runs'] == len(rows)
            and summary.get('successful') == len(rows), 'Batch size/success count differs from rows')
    seeds, methods, groups = set(), set(), {n: [] for n in range(10, 17)}
    for row in rows:
        require(isinstance(row, dict), 'Invalid row')
        seed, n = row.get('seed'), row.get('source_total')
        require(type(seed) is int and seed not in seeds, 'Missing/duplicate seed')
        seeds.add(seed)
        require(type(n) is int and 10 <= n <= 16, 'Invalid source-count stratum')
        require(row.get('successful') is True and row.get('infrastructure_error') is not True,
                'Every row must be successful; unknown/failed rows cannot be filtered')
        require(type(row.get('cleared_total')) is int and row['cleared_total'] == n,
                'Row does not declare all its sources cleared')
        stage = row.get('stage')
        require(stage == 'stress' if kind == 'stress' else stage in ('pilot', 'confirmation'),
                'Random/stress row stages differ from the supplied input role')
        method = row.get('strategy')
        require(isinstance(method, str) and method, 'Missing strategy label')
        methods.add(method)
        t = number(row.get('virtual_time_s'), 'observed virtual time')
        match(row.get('penalized_time_s'), t, 'successful row penalty')
        parts = {key: number(row.get(key), key) for key, _, _ in COMPONENTS}
        require(math.isclose(math.fsum(parts.values()), t, rel_tol=1e-11, abs_tol=1e-5),
                'Actual component fees do not sum to the actual total')
        groups[n].append((t, parts))
    require(len(methods) == 1, 'Single-arm summaries cannot mix strategies')
    counts = [len(groups[n]) for n in groups]
    require(all(counts) and len(set(counts)) == 1, 'Need the complete balanced N=10..16 strata')
    declared = summary.get('by_source_count')
    require(isinstance(declared, dict) and set(declared) == {str(n) for n in groups},
            'Missing or extra source-count summary strata')
    by_count = []
    for n, entries in groups.items():
        values = [t/n for t, _ in entries]
        saved = declared[str(n)]
        require(saved.get('runs') == len(entries) and saved.get('successful') == len(entries)
                and saved.get('all_clear') is True, 'Per-stratum completeness mismatch')
        average, p95 = statistics.mean(values), percentile(values)
        match(saved.get('mean_time_per_source_s'), average, f'N={n} mean T/N')
        match(saved.get('p95_time_per_source_s'), p95, f'N={n} P95 T/N')
        by_count.append(dict(source_count=n, runs=len(entries), mean_s=average, p95_s=p95))
    totals = [number(row['virtual_time_s'], 'time') for row in rows]
    per_source = [row['virtual_time_s']/row['source_total'] for row in rows]
    average = statistics.mean(per_source)
    for key, expected in dict(mean_time_per_source_s=average,
        mean_actual_time_per_source_s=average, p95_time_per_source_s=percentile(per_source),
        mean_time_s=statistics.mean(totals), total_time_s=math.fsum(totals),
        total_sources=sum(row['source_total'] for row in rows),
        pooled_time_per_source_s=math.fsum(totals)/sum(row['source_total'] for row in rows)).items():
        match(summary.get(key), expected, key)
    saved_parts = summary.get('mean_components_s')
    require(isinstance(saved_parts, dict), 'Missing raw aggregate component summary')
    parts_per_source = {}
    for key, _, _ in COMPONENTS:
        match(saved_parts.get(key), statistics.mean(row[key] for row in rows), f'raw mean {key}')
        # Divide EACH case first; mean(component)/mean(N) has another weighting.
        parts_per_source[key] = statistics.mean(row[key]/row['source_total'] for row in rows)
    require(math.isclose(math.fsum(parts_per_source.values()), average, abs_tol=1e-6),
            'Per-source components do not sum to the primary mean')
    return dict(kind=kind, method=next(iter(methods)), runs=len(rows),
                by_source_count=by_count, mean_time_per_source_s=average,
                p95_time_per_source_s=percentile(per_source),
                mean_components_per_source_s=parts_per_source)


def read_dataset(path, kind):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return validate_summary(json.loads(raw), kind), dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def render(random_summary, stress_summary, output_dir, *, title='Q4 可靠参考策略'):
    random_data, random_input = read_dataset(random_summary, 'random')
    stress_data, stress_input = read_dataset(stress_summary, 'stress')
    require(random_data['method'] == stress_data['method'], 'The two sets must use the same strategy label')
    output = Path(output_dir).resolve()
    require(not any((output/name).exists() for name in OUTPUTS), 'Preserve existing figure or provenance output')
    # Validation is complete before creating any output or importing plotting.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((font for font in ('Microsoft YaHei', 'Noto Sans CJK SC', 'SimHei') if font in fonts), 'DejaVu Sans')
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.family': selected_font, 'font.size': 10,
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.titleweight': 'bold', 'savefig.facecolor': 'white'}):
        figure, axis = plt.subplots(figsize=(8.8, 5.4))
        figure.subplots_adjust(left=.11, right=.97, bottom=.19, top=.82)
        for data, name, color in ((random_data, '随机', '#27668C'), (stress_data, '压力', '#B96C31')):
            x = [entry['source_count'] for entry in data['by_source_count']]
            axis.plot(x, [entry['mean_s'] for entry in data['by_source_count']],
                color=color, marker='o', linewidth=1.8, label=f'{name}：均值')
            axis.plot(x, [entry['p95_s'] for entry in data['by_source_count']],
                color=color, marker='s', markersize=4, linestyle='--', linewidth=1.2, label=f'{name}：P95')
        axis.set(xlabel='每局真实源数 N', ylabel='每源虚拟用时 T/N（秒）', xticks=list(range(10, 17)))
        axis.set_ylim(bottom=0)
        axis.grid(axis='y', color='#DDDDDD', linewidth=.6)
        axis.set_axisbelow(True)
        figure.suptitle(f'{title}：按源数分层', x=.11, ha='left', fontsize=14)
        axis.legend(frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(0., 1.20))
        figure.text(.11, .055, 'P95 是各层逐局分布的分位数，不是置信区间。\n'
            f'随机每层 {random_data["runs"]//7} 局；压力每层 {stress_data["runs"]//7} 局。', fontsize=9, color='#555555')
        for suffix in ('png', 'pdf'):
            figure.savefig(output/f'per_source_by_count.{suffix}', dpi=200)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8.2, 5.8))
        figure.subplots_adjust(left=.12, right=.97, bottom=.28, top=.87)
        datasets, bottoms = (random_data, stress_data), [0., 0.]
        for key, label, color in COMPONENTS:
            values = [data['mean_components_per_source_s'][key] for data in datasets]
            axis.bar([0, 1], values, bottom=bottoms, width=.48, label=label, color=color,
                     edgecolor='white', linewidth=.6)
            bottoms = [base+value for base, value in zip(bottoms, values)]
        axis.axhline(TARGET_S, color='#333333', linestyle='--', linewidth=1.2,
                     label='总体均值目标：460 秒/源')
        for index, data in enumerate(datasets):
            axis.annotate(f'{data["mean_time_per_source_s"]:.2f} 秒/源',
                (index, data['mean_time_per_source_s']), xytext=(0, 8), textcoords='offset points',
                ha='center', va='bottom', fontweight='bold')
        axis.set(xticks=[0, 1], xticklabels=[f'随机（{random_data["runs"]}局）', f'压力（{stress_data["runs"]}局）'],
                 ylabel='逐局费用/N 后的样本均值（秒/源）', xlim=(-.7, 1.7))
        axis.set_ylim(0, max(TARGET_S, *bottoms)*1.17)
        axis.grid(axis='y', color='#DDDDDD', linewidth=.6)
        axis.set_axisbelow(True)
        axis.set_title(f'{title}：总体每源费用构成', loc='left', pad=14)
        axis.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.13), fontsize=9)
        figure.text(.12, .035, '各项先逐局除以 N，再等权平均；总柱高等于 mean(T/N)。\n'
            '460 虚线仅表示总体均值目标，不要求每个源数层分别达标。', fontsize=9, color='#555555')
        for suffix in ('png', 'pdf'):
            figure.savefig(output/f'per_source_components.{suffix}', dpi=200)
        plt.close(figure)
    manifest = dict(schema='q4-per-source-reference-figures-v1',
        inputs={'random': random_input, 'stress': stress_input},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        title=title, font=selected_font, target_s=TARGET_S,
        target_scope='Overall equal-run mean(T/N) only; no per-N threshold on figure 1',
        percentile='Linear interpolation at 0.95*(sample_count-1); descriptive P95, not a CI',
        validation_scope='Complete summary rows and arithmetic only; not a replacement for physical or qualification audits',
        datasets={'random': random_data, 'stress': stress_data},
        output_sha256={name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in OUTPUTS[:-1]})
    with (output/OUTPUTS[-1]).open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random-summary', type=Path, required=True)
    parser.add_argument('--stress-summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--title', default='Q4 可靠参考策略')
    args = parser.parse_args(argv)
    value = render(args.random_summary, args.stress_summary, args.output_dir, title=args.title)
    print(json.dumps({'output_dir': str(args.output_dir.resolve()), 'inputs': value['inputs']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
