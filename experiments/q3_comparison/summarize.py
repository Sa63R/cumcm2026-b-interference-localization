"""Build a reproducible Chinese report from completed local benchmark files."""
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
LABELS = {'phased': '分阶段搜索与清除', 'joint': '初版联合策略', 'v2': 'v2 完整组合',
          'v3': 'v3 默认', 'v3_origin20': 'v3 保留原点扫描', 'optical': 'v3 有限提前光学尝试',
          'scenario': '新：多目标场景预测', 'future_cover': '新：搜索与清除位置联合设计',
          'scenario_future': '新：两方向组合'}


def read(name):
    path = RESULTS / name
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig')))
    for row in rows:
        for k in ['seed', 'repeat', 'true_sources', 'cleared', 'detects', 'switches', 'optical_attempts', 'failed_clears',
                  'scenario_decisions', 'changed_first_actions', 'accepted_future_plans']:
            if k in row:
                row[k] = int(row[k]) if row[k] else 0
        for k in ['virtual_seconds', 'seconds_per_source', 'movement_metres', 'local_runtime_seconds']:
            if k in row:
                row[k] = float(row[k])
        if 'success' in row:
            row['success'] = row['success'] == 'True'
    return rows


def summary(rows):
    out = {}
    for m in dict.fromkeys(x['mode'] for x in rows):
        rr = [x for x in rows if x['mode'] == m]
        out[m] = dict(cases=len(rr), successes=sum(x['success'] for x in rr),
                      total_sources=sum(x['true_sources'] for x in rr), cleared=sum(x['cleared'] for x in rr),
                      failed_clears=sum(x['failed_clears'] for x in rr),
                      p95_seconds_per_source=float(np.quantile([x['seconds_per_source'] for x in rr], .95)),
                      median_runtime=float(np.median([x['local_runtime_seconds'] for x in rr])),
                      p95_runtime=float(np.quantile([x['local_runtime_seconds'] for x in rr], .95)),
                      by_source_count={n: float(np.mean([x['seconds_per_source'] for x in rr if x['true_sources'] == n]))
                                       for n in range(10, 17) if any(x['true_sources'] == n for x in rr)})
        for k in ['virtual_seconds', 'seconds_per_source', 'movement_metres', 'detects', 'switches', 'local_runtime_seconds']:
            out[m][k] = float(np.mean([x[k] for x in rr]))
        out[m]['mean_cost_per_source'] = {k: float(np.mean([fn(x) / x['true_sources'] for x in rr])) for k, fn in {
            'movement': lambda x: x['movement_metres'] / 5,
            'detect': lambda x: x['detects'] * 5,
            'switch': lambda x: x['switches'],
            'optical_and_clear': lambda x: x['optical_attempts'] * 3 + x['cleared'] * 2}.items()}
    return out


def paired(rows, baseline='v3'):
    keys = lambda x: (x['kind'], x['seed'], x['noise'], x['repeat'])
    base = {keys(x): x for x in rows if x['mode'] == baseline}
    out = {}
    for m in dict.fromkeys(x['mode'] for x in rows):
        rr = [x for x in rows if x['mode'] == m and keys(x) in base]
        delta = np.array([base[keys(x)]['seconds_per_source'] - x['seconds_per_source'] for x in rr])
        rng = np.random.default_rng(20260912)
        boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
        out[m] = dict(mean_saved_per_source=float(delta.mean()),
                      bootstrap95=np.quantile(boot, [.025, .975]).tolist(),
                      faster=int((delta > 1e-7).sum()), slower=int((delta < -1e-7).sum()),
                      max_slowdown_per_source=float(-delta.min()))
    return out


def main():
    datasets = {n: read(n + '.csv') for n in ['original400', 'annex400', 'runtime50', 'stress350']}
    datasets['annex400'] += read('new400.csv')
    allrows = datasets['annex400']
    fields = list(dict.fromkeys(k for row in allrows for k in row))
    with (RESULTS / 'annex_all400.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(allrows)
    results = {n: summary(rr) for n, rr in datasets.items() if rr}
    results['paired_annex400'] = paired(datasets['annex400'])
    original = datasets['original400']
    oldpath = HERE / 'vendor/B_Q3_optimized_v3/results/holdout400.csv'
    old = {(int(x['seed']), x['mode']): x for x in csv.DictReader(oldpath.open(encoding='utf-8-sig'))}
    reproduction = {}
    for m in ['v2', 'v3']:
        rr = [x for x in original if x['mode'] == m]
        diff = np.array([x['virtual_seconds'] - float(old[x['seed'], m]['virtual_seconds']) for x in rr])
        reproduction[m] = dict(matched_within_1e6=int((abs(diff) < 1e-6).sum()), cases=len(rr),
                               max_abs_difference=float(abs(diff).max()), mean_difference=float(diff.mean()),
                               archived_mean_per_source=float(np.mean([float(x['seconds_per_source']) for x in old.values() if x['mode'] == m])))
    results['original_reproduction'] = reproduction
    (RESULTS / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    ann, runtime = results['annex400'], results.get('runtime50', {})
    lines = ['# 第三问本地方法对比（2026-09-12）', '',
        '主表来自本机实际执行的 400 组配对自建案例（种子 5000–5399），九种方法使用完全相同的源布局和误差函数。'
        '采用各案例先计算 T/N、再对案例等权平均的口径。这里是按附件规则修正的自建仿真，不是官方模拟器成绩。', '',
        '## 主结果', '', '| 方法 | 平均秒/源 | 平均总秒/例 | 平均路程 m | 检测次/例 | 串行计算秒/例 | 全清案例 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m in LABELS:
        x = ann[m]
        rt = runtime.get(m, {}).get('local_runtime_seconds', float('nan'))
        lines.append(f"| {LABELS[m]} | {x['seconds_per_source']:.3f} | {x['virtual_seconds']:.2f} | {x['movement_metres']:.2f} | {x['detects']:.2f} | {rt:.4f} | {x['successes']}/{x['cases']} |")
    winner = min(ann, key=lambda m: ann[m]['seconds_per_source'])
    lines += ['', f"这批案例平均最快的是 **{LABELS[winner]}：{ann[winner]['seconds_per_source']:.3f} 秒/源**。"
              f"距离平均 200 秒/源尚差 {ann[winner]['seconds_per_source'] - 200:.3f} 秒/源。", '',
              '计算时间为预热后的单进程串行测试：同一批 50 案例 × 2 次，每例轮换方法次序、第二遍反转次序。'
              '包含策略构造和执行，不包含环境构造、第一次 JIT 编译、HTTP 或官方程序启动。批量并行运行时的耗时不用于主表比较。', '',
              '## 相对 v3 的配对差异', '',
              '正数表示比 v3 更快；95% 区间为按案例配对 bootstrap（10000 次、固定种子）。区间只反映本自建分布的抽样不确定性。', '',
              '| 方法 | 节省秒/源 | 95% 区间 | 更快/更慢案例 |', '|---|---:|---:|---:|']
    for m in LABELS:
        p = results['paired_annex400'][m]
        lines.append(f"| {LABELS[m]} | {p['mean_saved_per_source']:.3f} | [{p['bootstrap95'][0]:.3f}, {p['bootstrap95'][1]:.3f}] | {p['faster']}/{p['slower']} |")
    movement_increase = ann['scenario']['mean_cost_per_source']['movement'] - ann['v3']['mean_cost_per_source']['movement']
    lines += ['', f"场景预测版的移动成本比 v3 增加约 {movement_increase:.3f} 秒/源，说明本版估价选出的停靠顺序增加了绕行。"
              '联合设计的平均改善区间包含零，没有看到可靠提速；v3 本来就包含搜索点移动和顺路补扫，新增未来覆盖计划的适用机会有限。'
              '这些结果评价的是当前一步预测与粗略尾部估价的实现，不能证明更强的全任务预测没有潜力。']
    lines += ['', '## 按源数量分组', '', '| 源数 | ' + ' | '.join(LABELS.values()) + ' |',
              '|---|' + '---:|' * len(LABELS)]
    for n in range(10, 17):
        lines.append(f'| {n} | ' + ' | '.join(f"{ann[m]['by_source_count'][n]:.2f}" for m in LABELS) + ' |')
    lines += ['', '## 计费与实现核查', '',
              '- 原代码在清除时调用 `_tune`，会切频并改变后续测向机状态。附件 1、2 明确清除不切频，因此新增 `AnnexSimulator.clear` 修正这一行为。',
              '- 原代码直接返回弧度全精度示向度。修正版按附件保留 0.01°，可行域误差半角扩大至 1.005°；物理测向误差仍限制在 ±1°。',
              '- 光学失败计 3 秒、成功计 5 秒，保持与附件一致。提前光学版本的失败尝试全部计入耗时，没有删除失败案例。',
              '- 下载包原文件保留在 vendor；复测入口以适配器和误差常量配置完成修改，没有覆盖原算法文件。',
              '- 新方法只有在真实观测形成完整覆盖证书，或发现 16 个不同频道时，才认定其他频道不存在。策略接口未提供真实源位置、源数、接收半径或种子。', '',
              '## 历史结果复测', '',
              '| 原包方法 | 历史秒/源 | 本机原规则秒/源 | 逐例吻合（总时间误差 < 1e-6 秒） | 最大总时间差 |',
              '|---|---:|---:|---:|---:|']
    for m in ['v2', 'v3']:
        x = reproduction[m]
        lines.append(f"| {m} | {x['archived_mean_per_source']:.3f} | {results['original400'][m]['seconds_per_source']:.3f} | {x['matched_within_1e6']}/{x['cases']} | {x['max_abs_difference']:.3f} s |")
    lines += ['', '原包结果若未逐例一致，不能宣称精确重现。源坐标和源码来自同一包；当前平台/数值库与历史环境不同，'
              '原包使用坐标六位小数作为误差哈希输入，极小坐标差也可能改变后续读数和路线。这里只报告观测到的差异，未把该机制当作逐例根因证明。', '',
              '## 压力与正确性检查', '',
              '原包 28 项测试、新增 7 项附件语义/真值包含性/新功能消融检查均通过。压力集为 250 组边界源及最小接收半径场景，'
              '加 100 组重合、原点、共线、簇状场景；覆盖 hash、smooth、固定 ±1°、交替 ±1° 五种误差。', '',
              '| 方法 | 全清案例 | 光学失败尝试总数 | 压力集平均秒/源 |', '|---|---:|---:|---:|']
    for m, x in results.get('stress350', {}).items():
        lines.append(f"| {LABELS.get(m, m)} | {x['successes']}/{x['cases']} | {x['failed_clears']} | {x['seconds_per_source']:.3f} |")
    stress = results.get('stress350', {})
    if stress:
        sw = min(stress, key=lambda m: stress[m]['seconds_per_source'])
        lines += ['', f"压力集最快的是 {LABELS[sw]}，说明随机分布上的最优平均版本并非所有布局都占优。"]
    cold_path = RESULTS / 'cold_start.json'
    if cold_path.exists():
        cold = json.loads(cold_path.read_text())
        lines += ['', f"另测了一次全新 Numba 缓存的启动：导入约 {cold['imports_s']:.3f} 秒，全部数值核首次编译/预热约 "
                  f"{cold['warmup_compile_s']:.3f} 秒，含进程启动总计 {cold['process_wall_s']:.3f} 秒。它没有计入每例预热计算时间，单次测量仅供启动开销参考。"]
    lines += ['', '## 来源与复现', '',
              '源文件目录：`/Users/zephyrr/竞赛/26国赛/gpt_download/q3`。压缩包 SHA-256 见 `source_manifest.json`。'
              '案例源数均匀取 10–16，位置按圆域面积均匀取样，接收半径均匀取 1000–1500 m；这些均是原包的自建分布假设。', '',
              '```bash', 'cd /Users/zephyrr/竞赛/26国赛/数模/code/experiments/q3_comparison',
              '.venv/bin/python -m unittest -v test_local',
              '.venv/bin/python run_comparison.py --profile original --cases 400 --workers 2 --modes phased joint v2 v3 v3_origin20 optical --out results/original400.csv',
              '.venv/bin/python run_comparison.py --profile annex --cases 400 --workers 2 --modes phased joint v2 v3 v3_origin20 optical --out results/annex400.csv',
              '.venv/bin/python run_comparison.py --profile annex --cases 400 --workers 2 --modes scenario future_cover scenario_future --out results/new400.csv',
              '.venv/bin/python run_comparison.py --profile annex --suite stress --workers 2 --out results/stress350.csv',
              '.venv/bin/python run_comparison.py --profile annex --suite runtime --cases 50 --out results/runtime50.csv',
              '.venv/bin/python summarize.py', '```', '',
              '每批的 `.meta.json` 记录命令、Python/NumPy/Numba 版本、预热和总耗时；CSV 为逐例结果；JSONL 为增量记录；首例 trace 保存完整动作。', '',
              '新方向实现详见 `新方法设计说明.md` 与 `new_methods.py`。下载包没有这两项的可执行版本，因此它们是本次新写的实验原型；旧版本则使用原包算法。'
              '新原型先在开发种子 0–19 上检查运行情况，再冻结该实现测试 5000–5399；没有依据这 400 组结果回调参数。'
              '这个配对集合曾用于原包历史验证，不能把它描述为整项研究从未见过的最终盲测。']
    (HERE / '本地速度对比.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({m: {'avg': x['seconds_per_source'], 'runtime': runtime.get(m, {}).get('local_runtime_seconds'), 'cleared': x['successes']} for m, x in ann.items()}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
