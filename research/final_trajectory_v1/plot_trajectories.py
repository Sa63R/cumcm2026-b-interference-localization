"""Draw saved, matched Q3 trajectories without policy or simulator imports."""
import argparse
from decimal import Decimal
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
ROLES = ('baseline', 'state', 'rl', 'geo')
LABELS = ('原 rollout 基线', '状态搜索', '强化学习', '几何方法')


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def select_case(baseline_rows, seed=None, reason=None):
    """All baseline runs count; never select using another method's result."""
    if not baseline_rows:
        raise ValueError('Cannot select a case from an empty partition')
    # Ledger times are decimal microseconds; binary float subtraction can
    # spuriously break the exact tie between the two central observations.
    middle = statistics.median(Decimal(str(row['virtual_time_s'])) for row in baseline_rows)
    if seed is not None:
        if not reason or not reason.strip():
            raise ValueError('Explicit --seed requires --selection-reason')
        if seed not in {row['seed'] for row in baseline_rows}:
            raise ValueError('Requested seed is not in the complete supplied partition')
        return seed, dict(kind='explicit', explanation=reason.strip(), baseline_median_s=float(middle))
    if reason:
        raise ValueError('--selection-reason applies only to an explicit --seed')
    chosen = min(baseline_rows, key=lambda row:(abs(Decimal(str(row['virtual_time_s']))-middle), row['seed']))
    return chosen['seed'], dict(kind='baseline_median', baseline_median_s=float(middle),
        comparison_arithmetic='Decimal of the recorded virtual_time_s, exact decimal ties',
        baseline_selected_time_s=chosen['virtual_time_s'],
        explanation='基线虚拟耗时最接近全样本中位数；等距按 seed 较小者；不按我方表现挑选')


def load_matched(directories, *, partition='validation', seed=None, reason=None,
                 allow_heldout=False, protocol_path=ROOT/'research/v1_protocol.json'):
    if partition not in ('validation', 'final_random', 'final_stress'):
        raise ValueError('Only opened development or later final partitions are supported')
    if partition != 'validation' and not allow_heldout:
        raise ValueError('Final archives require explicit --allow-heldout after freezing')
    if set(directories) != set(ROLES):
        raise ValueError('Exactly baseline, state, rl and geo directories are required')
    protocol = read(protocol_path)
    protocol_sha = digest(protocol)
    p = protocol['partitions'][partition]
    seeds = list(range(p['seed_start'], p['seed_stop_exclusive']))
    prepared = {}
    # Check complete directory inventories and metadata BEFORE opening payloads.
    for role in ROLES:
        directory = Path(directories[role]).resolve()
        manifest, rows = read(directory/'manifest.json'), read(directory/'rows.json')
        if (manifest['split'] != partition or manifest['seeds'] != seeds
                or manifest['identity']['protocol_sha256'] != protocol_sha):
            raise ValueError(f'{role}: partition/protocol mismatch')
        if ({path.name for path in directory.glob('case-*.json.gz')} != {f'case-{s}.json.gz' for s in seeds}
                or len(rows) != len(seeds) or {r['seed'] for r in rows} != set(seeds)):
            raise ValueError(f'{role}: missing, duplicate, extra or out-of-partition cases')
        if any(not math.isfinite(r['virtual_time_s']) or r['virtual_time_s'] < 0 for r in rows):
            raise ValueError('Invalid actual elapsed time')
        prepared[role] = (directory, manifest, {r['seed']:r for r in rows})
    chosen, selection = select_case(list(prepared['baseline'][2].values()), seed, reason)
    reference = prepared['baseline'][2]
    records, provenance = {}, {}
    for role, (directory, manifest, rows) in prepared.items():
        archives = {}
        for case_seed in seeds:
            row = rows[case_seed]
            if row['case_sha256'] != reference[case_seed]['case_sha256']:
                raise ValueError(f'{role}: case SHA differs from baseline at seed {case_seed}')
            path = directory/f'case-{case_seed}.json.gz'
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                record = json.load(stream)
            truth = record['evaluation']['ground_truth']
            if (record['row'] != row or record.get('evaluation_phase') != 'after_policy_termination'
                    or record['evaluation']['kind'] != 'local_research_only'
                    or row['strategy'] != manifest['strategy'] or truth['problem'] != 3
                    or truth['seed'] != case_seed or truth['case_id'] != row['case_id']
                    or digest(truth) != row['case_sha256']
                    or digest(record['spec']) != manifest['identity']['spec_sha256']):
                raise ValueError(f'{role}: archive/row/spec/post-run truth mismatch at seed {case_seed}')
            archives[path.name] = file_sha(path)
            if case_seed == chosen:
                records[role] = record
        provenance[role] = dict(directory=str(directory), manifest_sha256=file_sha(directory/'manifest.json'),
            rows_sha256=file_sha(directory/'rows.json'), identity=manifest['identity'],
            strategy=manifest['strategy'], archives_sha256=archives)
    return records, dict(partition=partition, partition_size=len(seeds), seed=chosen,
        case_sha256=reference[chosen]['case_sha256'], selection=selection,
        source_sha256=file_sha(__file__), protocol_sha256=protocol_sha, inputs=provenance,
        truth_use='Post-termination display only; no online truth access or new simulation.',
        baseline_sample_includes_failed_runs=True)


def trajectory(record):
    position = (0., 0.)
    route, measures, clears, failed = [position], [], [], []
    movement_us = 0
    for action in record['history']:
        if action['action'] not in ('/measure', '/clear'):
            continue
        if not action['response'].get('accepted'):
            raise ValueError('Trajectory contains a rejected physical action')
        q = (action['position']['x'], action['position']['y'])
        movement_us += round(math.dist(position, q)/5*1e6)
        if q != position:
            route.append(q)
        position = q
        if action['action'] == '/measure':
            measures.append(q)
        elif action['response']['clear_result'] == 'success':
            clears.append(dict(channel=action['channel'], position=q))
        else:
            failed.append(q)
    row = record['row']
    if (abs(movement_us/1e6-row['movement_s']) > 2e-6 or len(measures) != row['measurement_count']
            or len(clears) != row['cleared_total'] or len(failed) != row['failed_clear_count']
            or abs(record['history'][-1]['response']['virtual_time_s']-row['virtual_time_s']) > 2e-6):
        raise ValueError('Plotted actions disagree with elapsed/movement/count ledger')
    return dict(route=route, measures=sorted(set(measures)), clears=clears, failed=failed,
                actual_measurement_count=len(measures), unique_measurement_positions=len(set(measures)))


def render(records, metadata, prefix, *, dpi=200):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    import matplotlib.patheffects as pe
    from matplotlib import font_manager
    font = font_manager.findfont('Microsoft YaHei', fallback_to_default=False)
    plt.rcParams.update({'font.family':'Microsoft YaHei', 'axes.unicode_minus':False,
        'font.size':10, 'svg.fonttype':'path', 'svg.hashsalt':'q3-saved-trajectories-v1',
        'savefig.facecolor':'white', 'axes.spines.top':False, 'axes.spines.right':False})
    plots = {role:trajectory(record) for role, record in records.items()}
    limit = max(2000., math.ceil((max(abs(v) for data in plots.values() for p in data['route'] for v in p)+150)/250)*250)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 13.))
    fig.subplots_adjust(left=.073, right=.975, bottom=.145, top=.837, wspace=.18, hspace=.40)
    title = '开发案例预检' if metadata['partition'] == 'validation' else '最终冻结案例'
    if metadata['selection']['kind'] == 'baseline_median':
        choice = f"基线中位数规则选例：seed {metadata['seed']}，基线样本中位数 {metadata['selection']['baseline_median_s']:.1f} s"
    else:
        choice = f"显式指定 seed {metadata['seed']}；选择原因：{metadata['selection']['explanation']}"
    fig.suptitle(f'{title}：四种方法的真实行进轨迹', fontsize=18, fontweight='bold', y=.975)
    fig.text(.5, .941, choice, ha='center', fontsize=11)
    fig.text(.5, .916, '同一案例、同一坐标尺度；真值仅用于策略终止后的展示', ha='center', color='#535c66', fontsize=10)
    truth = records['baseline']['evaluation']['ground_truth']['sources']
    blue, red, green = '#235789', '#bc4749', '#25806f'
    source_offsets = {}
    for ax, role, label, letter in zip(axes.flat, ROLES, LABELS, 'abcd'):
        row, data = records[role]['row'], plots[role]
        route = data['route']
        ax.add_patch(Circle((0., 0.), 1800., fill=False, color='#78828d', lw=.9, alpha=.8))
        ax.plot([p[0] for p in route], [p[1] for p in route], color=blue, lw=1.05, alpha=.68, zorder=2)
        for a, b in zip(route, route[1:]):
            if math.dist(a, b) >= 220.:
                start = (a[0]+.48*(b[0]-a[0]), a[1]+.48*(b[1]-a[1]))
                end = (a[0]+.60*(b[0]-a[0]), a[1]+.60*(b[1]-a[1]))
                ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='-|>', mutation_scale=8,
                    color=blue, alpha=.75, lw=.8), zorder=3)
        if data['measures']:
            ax.scatter(*zip(*data['measures']), s=12, facecolor='white', edgecolor=blue, linewidth=.6, zorder=4)
        if data['clears']:
            ax.scatter(*zip(*(c['position'] for c in data['clears'])), s=72, facecolor='none',
                       edgecolor=green, linewidth=1.35, zorder=5)
        if data['failed']:
            ax.scatter(*zip(*data['failed']), s=32, marker='+', color='#9a641c', zorder=5)
        ax.scatter([s['x'] for s in truth], [s['y'] for s in truth], s=26, marker='x', color=red, linewidth=1.1, zorder=6)
        ax.scatter(0., 0., marker='D', s=28, color='#303943', zorder=7)
        ax.scatter(*route[-1], marker='s', s=24, facecolor='white', edgecolor='#303943', zorder=7)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel='x / m', ylabel='y / m')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(range(-int(limit//500)*500, int(limit//500)*500+1, 1000))
        ax.set_yticks(range(-int(limit//500)*500, int(limit//500)*500+1, 1000))
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.15, lw=.6)
        status = f"清除 {row['cleared_total']}/{row['source_total']}，失败清除 {row['failed_clear_count']}"
        if not row['successful']:
            status += f"；未成功，罚后 {row['penalized_time_s']:.0f} s"
        ax.set_title(f"({letter}) {label}\n总时 {row['virtual_time_s']:.1f} s · 移动 {row['movement_s']:.1f} s · 测量 {row['measurement_count']} 次\n{status}",
                     fontsize=10.2, linespacing=1.5, pad=9)
        order = ' → '.join(str(c['channel']) for c in data['clears']) or '无成功清除'
        ax.text(.5, -.17, f'清除顺序（频道号）：{order}', transform=ax.transAxes, ha='center', fontsize=7.9)
    fig.canvas.draw()
    # Use identical label offsets in all panels; greedy collision handling uses
    # only source positions, never a method's route or performance.
    ax = axes.flat[0]
    boxes = []
    scale = fig.dpi/72
    for source in sorted(truth, key=lambda s:s['channel']):
        x, y = ax.transData.transform((source['x'], source['y']))
        width, height = (7+6*len(str(source['channel'])))*scale, 11*scale
        choices = [(sx*distance, sy*distance*.65) for distance in (8, 18, 28, 40)
                   for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1))]
        def box(offset):
            u, v = x+offset[0]*scale, y+offset[1]*scale
            return (u, v-height/2, u+width, v+height/2)
        def overlap(candidate):
            return sum(max(0., min(candidate[2], b[2])-max(candidate[0], b[0]))*
                       max(0., min(candidate[3], b[3])-max(candidate[1], b[1])) for b in boxes)
        offset = min(choices, key=lambda q:(overlap(box(q)), math.hypot(*q)))
        source_offsets[source['channel']] = offset
        boxes.append(box(offset))
    for ax in axes.flat:
        for source in truth:
            ax.annotate(str(source['channel']), (source['x'], source['y']),
                xytext=source_offsets[source['channel']], textcoords='offset points',
                fontsize=8.5, color=red, va='center', zorder=8,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
    handles = [Line2D([], [], color=blue, lw=1.2, label='真实行进路线 / 方向箭头'),
        Line2D([], [], marker='o', color=blue, markerfacecolor='white', linestyle='', markersize=4, label='测量点（同坐标合并）'),
        Line2D([], [], marker='x', color=red, linestyle='', markersize=6, label='源真值 / 频道编号'),
        Line2D([], [], marker='o', color=green, markerfacecolor='none', linestyle='', markersize=8, label='实际成功清除点'),
        Line2D([], [], marker='D', color='#303943', linestyle='', markersize=4, label='起点'),
        Line2D([], [], marker='s', color='#303943', markerfacecolor='white', linestyle='', markersize=4, label='终点')]
    if any(data['failed'] for data in plots.values()):
        handles.append(Line2D([], [], marker='+', color='#9a641c', linestyle='', label='失败清除点'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .059), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .035, '绿色清除点与红色源真值可能近乎重合；编号为频道号，清除先后另列于每幅图下。',
             ha='center', color='#535c66', fontsize=9)
    fig.text(.5, .015, '同坐标显示合并不改变真实费用或测量次数；保留全部重复与跨频道动作。',
             ha='center', color='#535c66', fontsize=9)
    prefix = Path(prefix)
    paths = [prefix.with_suffix(suffix) for suffix in ('.png', '.svg', '.json')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Output prefix already exists; preserve the earlier figure')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=dpi, metadata={'Software':'Q3 saved trajectory renderer'})
    fig.savefig(paths[1], metadata={'Date':None, 'Creator':'Q3 saved trajectory renderer'})
    paths[1].write_text('\n'.join(line.rstrip() for line in paths[1].read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)
    metadata.update(font_path=font, font_sha256=file_sha(font), matplotlib_version=matplotlib.__version__, dpi=dpi,
        same_axis_limit_m=limit, source_annotation_offsets_points=source_offsets,
        plotted={role:dict(strategy=records[role]['row']['strategy'], row=records[role]['row'],
                          **data) for role, data in plots.items()})
    paths[2].write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ROLES:
        parser.add_argument('--'+role, type=Path, required=True)
    parser.add_argument('--partition', choices=('validation', 'final_random', 'final_stress'), default='validation')
    parser.add_argument('--allow-heldout', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--selection-reason')
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    records, metadata = load_matched({role:getattr(args, role) for role in ROLES}, partition=args.partition,
        seed=args.seed, reason=args.selection_reason, allow_heldout=args.allow_heldout)
    paths = render(records, metadata, args.output_prefix)
    print(json.dumps(dict(seed=metadata['seed'], selection=metadata['selection'], outputs=list(map(str, paths))), ensure_ascii=False))


if __name__ == '__main__':
    main()
