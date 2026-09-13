"""Rebuild the Chinese report from preserved results; never runs a simulator."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

PROJECT = Path(__file__).resolve().parents[3]
OUT = PROJECT / 'output/q4_speedup'
GROUPS = ['all_omni', 'mixed25', 'mixed50', 'mixed75', 'all_directional',
          'boundary_directional', 'boundary_mixed', 'plus1', 'minus1', 'clustered']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def percentile(xs, p):
    xs = sorted(xs)
    position = (len(xs) - 1) * p
    lo = int(position)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (position - lo)


def paired_result(rows, method, repeats, seed):
    baseline = {r['seed']: r for r in rows if r['method'] == 'v4'}
    candidate = sorted((r for r in rows if r['method'] == method), key=lambda r: r['seed'])
    by_group = defaultdict(list)
    pairs = []
    for row in candidate:
        base = baseline[row['seed']]
        assert base['scenario'] == row['scenario'] and base['targets'] == row['targets']
        delta = base['virtual_seconds'] / base['targets'] - row['virtual_seconds'] / row['targets']
        pair = dict(seed=row['seed'], scenario=row['scenario'], targets=row['targets'],
                    baseline_seconds_per_source=base['virtual_seconds'] / base['targets'],
                    candidate_seconds_per_source=row['virtual_seconds'] / row['targets'],
                    saved_seconds_per_source=delta,
                    relative_regression=row['virtual_seconds'] / base['virtual_seconds'] - 1)
        pairs.append(pair)
        by_group[row['scenario']].append(delta)
    assert set(by_group) == set(GROUPS)
    rng = random.Random(seed)
    samples = []
    n = len(pairs)
    for _ in range(repeats):
        total = 0.
        for group in GROUPS:
            values = by_group[group]
            total += sum(rng.choices(values, k=len(values)))
        samples.append(total / n)
    interval = [percentile(samples, .025), percentile(samples, .975)]
    old = statistics.mean(p['baseline_seconds_per_source'] for p in pairs)
    new = statistics.mean(p['candidate_seconds_per_source'] for p in pairs)
    worst = max(pairs, key=lambda p: p['relative_regression'])
    return dict(method=method, cases=n, successes=sum(r['success'] for r in candidate),
                baseline_mean_seconds_per_source=old, mean_seconds_per_source=new,
                mean_paired_saved_seconds_per_source=statistics.mean(p['saved_seconds_per_source'] for p in pairs),
                reduction_of_mean=(old-new)/old,
                stratified_paired_bootstrap95_saved_seconds_per_source=interval,
                bootstrap_includes_zero=interval[0] <= 0 <= interval[1],
                bootstrap=dict(repetitions=repeats, seed=seed, groups=GROUPS,
                               group_sizes={g:len(by_group[g]) for g in GROUPS},
                               algorithm='Within each fixed scenario group, sample complete paired differences with replacement at original group size; pool the mean T/N difference; percentile interval'),
                faster=sum(p['saved_seconds_per_source'] > 1e-6 for p in pairs),
                slower=sum(p['saved_seconds_per_source'] < -1e-6 for p in pairs),
                tied=sum(abs(p['saved_seconds_per_source']) <= 1e-6 for p in pairs),
                worst=worst,
                mean_wall_seconds=statistics.mean(r['wall_seconds'] for r in candidate),
                baseline_mean_wall_seconds=statistics.mean(baseline[r['seed']]['wall_seconds'] for r in candidate),
                groups={g:dict(cases=len(by_group[g]),mean_paired_saved_seconds_per_source=statistics.mean(by_group[g])) for g in GROUPS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap-repeats', type=int, default=50000)
    parser.add_argument('--bootstrap-seed', type=int, default=2026091202)
    parser.add_argument('--windows-evidence', type=Path,
                        default=OUT/'windows_adapter_evidence/Q4Practice/output/q4_speedup/schedule_dev/adapter_speedup/results.json')
    args = parser.parse_args()
    batches = {}
    all_rows = []
    source_paths = []
    for name, expected in [('geometry_holdout100',300),('prior_holdout100',200),('cells_confirm200',400)]:
        path = OUT/name/'runs.jsonl'
        rows = read_rows(path)
        assert len(rows) == expected
        assert all(r['success'] and r['targets'] == r['cleared'] for r in rows)
        assert all(r['coverage_verified'] is True or r['cleared'] == 16 for r in rows)
        batches[name] = rows
        all_rows.extend(rows)
        source_paths.append(path)
    scenario_targets = {}
    for row in all_rows:
        key = (row['scenario'],row['seed'])
        if key in scenario_targets:
            assert scenario_targets[key] == row['targets']
        scenario_targets[key] = row['targets']
    assert len(scenario_targets) == 300 and len(all_rows) == 900
    old_baselines = {r['seed']:r for r in batches['geometry_holdout100'] if r['method']=='v4'}
    for row in batches['prior_holdout100']:
        if row['method']=='v4':
            assert row['virtual_seconds'] == old_baselines[row['seed']]['virtual_seconds']
    comparisons = {}
    for index,(label,batch,method) in enumerate([
        ('prior100','prior_holdout100','prior_fixed'),
        ('combined100','geometry_holdout100','combined'),
        ('cells_screen100','geometry_holdout100','cells_flex'),
        ('cells_confirm200','cells_confirm200','cells_flex')]):
        comparisons[label] = paired_result(batches[batch],method,args.bootstrap_repeats,args.bootstrap_seed+index)
    compute_path = OUT/'schedule_dev/compute100/summary.json'
    coverage_path = OUT/'schedule_dev/coverage_compute100/summary.json'
    compute = read_json(compute_path)
    coverage = read_json(coverage_path)
    assert compute['cases']==100 and compute['all_actions_identical'] and compute['all_reports_identical']
    assert coverage['inputs']==100 and coverage['all_fields_except_wall_identical']
    source_paths += [compute_path,coverage_path]
    evidence = read_json(args.windows_evidence)
    windows_rows = evidence['runs']
    assert all(r['success'] and r['cleared']==r['targets'] for r in windows_rows)
    versions = Counter(r['report'].get('version','baseline') for r in windows_rows)
    v2 = any(k.endswith('-v2') for k in versions if k!='baseline')
    nonbase_versions = [k for k in versions if k!='baseline']
    if v2:
        assert all(k.endswith('-v2') for k in nonbase_versions)
    source_paths.append(args.windows_evidence)
    source_paths += [OUT/'frozen_candidates/freeze.json',OUT/'cells_confirm_protocol.json']
    source_paths.append(PROJECT/'code/experiments/q4_official_practice/run_speedup.py')
    preservation_path=OUT/'original_source_preservation.json'
    preservation=read_json(preservation_path)
    assert preservation['all_original_files_unchanged'] and preservation['changed_original_files']==[]
    source_paths.append(preservation_path)
    windows = dict(evidence=str(args.windows_evidence.relative_to(PROJECT)),
                   kind=evidence['kind'],runs=len(windows_rows),cleared_sources=sum(r['cleared'] for r in windows_rows),
                   versions=dict(versions),v2_verified=v2,
                   note='Windows offline synthetic transport through real protocol validation; not official practice')
    if v2:
        audit_path=OUT/'windows_adapter_evidence/guest_source_audit.json'
        deployment_path=OUT/'deployment_manifest_v2.json'
        console_path=OUT/'windows_adapter_evidence/q4_speedup_adapter_console.txt'
        audit=read_json(audit_path)
        assert audit==read_json(deployment_path)
        console=console_path.read_text(encoding='utf-8-sig')
        tests=re.search(r'Ran (\d+) tests in ([\d.]+)s',console)
        assert tests and console.rstrip().endswith('OK')
        windows.update(deployed_files_sha256_verified=len(audit),passed_test_methods=int(tests.group(1)),
                       test_suite_wall_seconds=float(tests.group(2)),
                       timing_note='Single Windows v2 suite timing; not a paired speed comparison with v1')
        source_paths += [audit_path,deployment_path,console_path]
    development = {
        'coverage':dict(static_layout_candidates=3040,full_task_runs=1182,
                        note='Exploratory reused historical development cases; original precision; not final holdout'),
        'scheduling':dict(full_task_runs=sum(len(read_rows(p)) for p in (OUT/'schedule_dev').glob('*/runs.jsonl')),
                          note='Shared sensing, delayed clearing, service routing and station-order development; includes separate 60-case screens, not primary holdout'),
        'opportunistic_clears':dict(full_task_runs=len(read_json(PROJECT/'code/experiments/q4_speedup/local_opportunities_allstops_dev80.json')['rows']),
                                    cases=80,methods=7),
        'unified_development':dict(full_task_runs=sum(len(read_rows(OUT/d/'runs.jsonl')) for d in ['combined_dev40','domain_dev80','sectors_dev30','prior_dev40']),
                                    datasets=['combined_dev40','domain_dev80','sectors_dev30','prior_dev40'])}
    incumbent_path=OUT/'incumbent_dev40/summary.json'
    incumbent_rows_path=OUT/'incumbent_dev40/runs.jsonl'
    incumbent_replay_path=OUT/'incumbent_dev40/known_worst_replay.json'
    original_cells_dev_path=OUT/'combined_dev40/summary.json'
    incumbent_summary=read_json(incumbent_path)['overall']
    incumbent_rows=read_rows(incumbent_rows_path)
    incumbent_replays=read_json(incumbent_replay_path)['rows']
    assert len(incumbent_rows)==160 and all(r['success'] and r['targets']==r['cleared'] for r in incumbent_rows)
    assert len({r['seed'] for r in incumbent_rows})==40
    original_cells_dev=read_json(original_cells_dev_path)['overall']['cells_flex']
    incumbent_worst=next(r for r in incumbent_replays if r['method']=='cells_flex_incumbent')
    development['incumbent_route']=dict(
        full_task_runs=len(incumbent_rows),scenarios=40,methods=4,all_cleared=True,
        summary=incumbent_summary,original_cells_flex_same_development=original_cells_dev,
        known_worst_replay={
            'kind':'Known-case diagnostic, not new holdout and excluded from the 40 development scenarios',
            'runs':len(incumbent_replays),
            'rows':[{k:r[k] for k in ['method','seed','scenario','success','targets','cleared','virtual_seconds']} for r in incumbent_replays]},
        promoted_to_new_holdout=False,deployed_to_default_or_official_entry=False,
        decision='Archive as a negative development result: preserving a shorter route through estimated centers does not ensure earlier source discovery.')
    source_paths += [incumbent_path,incumbent_rows_path,incumbent_replay_path,original_cells_dev_path]
    result = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  recommendation='v4_fast, q4-speedup-practice-20260912-v2; equivalent computation optimization; preserve V4 mission decisions',
                  primary_metric='Equal-weight mean of per-scenario virtual task time T_i / source count N_i; positive paired delta means improvement',
                  primary_counts=dict(independent_scenarios=len(scenario_targets),full_solver_runs=len(all_rows),
                                      successful_solver_runs=sum(r['success'] for r in all_rows),
                                      cleared_source_instances=sum(r['cleared'] for r in all_rows),
                                      sources_once_per_independent_scenario=sum(scenario_targets.values()),
                                      duplicate_baseline_runs_between_first_two_batches=100,
                                      note='CPU-equivalence, development and Windows-offline runs are separate and excluded from these counts'),
                  batches={name:dict(rows=len(rows),scenarios=len({r['seed'] for r in rows}),
                                     method_counts=dict(Counter(r['method'] for r in rows)),
                                     cleared_source_instances=sum(r['cleared'] for r in rows)) for name,rows in batches.items()},
                  mission_comparisons=comparisons,compute_only=compute,coverage_compute_only=coverage,
                  windows_offline=windows,
                  original_source_preservation=preservation,
                  official=dict(new_practice_runs_this_round=0,new_formal_runs_this_round=0,
                                previous_practice_runs=15,previous_results_unchanged=True,
                                blocker='Mac is locked. User must manually unlock, then Q4 practice mode must be freshly verified before entering.',
                                no_claim='The previous 15 practice runs do not evaluate this version.'),
                  development=development,
                  input_sha256={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    (OUT/'最终统计.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    a=comparisons['prior100'];c=comparisons['combined100'];s=comparisons['cells_screen100'];n=comparisons['cells_confirm200']
    ci=lambda d:'['+', '.join(f'{x:.3f}' for x in d['stratified_paired_bootstrap95_saved_seconds_per_source'])+']'
    table=[]
    for label,d in [('修正先验前瞻',a),('三项几何组合 combined',c),('cells_flex 首批筛选',s),('cells_flex 独立再确认',n)]:
        table.append(f"| {label} | {d['cases']} | {d['baseline_mean_seconds_per_source']:.3f} → {d['mean_seconds_per_source']:.3f} | {d['reduction_of_mean']*100:+.4f}% | {ci(d)} | {d['faster']}/{d['slower']}/{d['tied']} |")
    win_status=(f"v2 已在 Windows 离线协议适配测试中通过 {windows['passed_test_methods']} 项测试、{windows['runs']} 次完整运行、{windows['cleared_sources']} 次源清除；{windows['deployed_files_sha256_verified']} 个部署文件的SHA-256与清单逐项一致。该次测试套件墙钟为 {windows['test_suite_wall_seconds']:.3f} 秒，不能与非配对的v1运行直接计算加速比。"
                if v2 else f"Windows 离线协议适配测试已确认 v1 的 {windows['runs']} 次完整运行、{windows['cleared_sources']} 次源清除；v2 已完成本机适配检查，Windows 同步验证尚待最终证据。")
    report=rf'''# 第四问第二轮提速实测报告

**建议采用 `v4_fast` v2 的等价计算优化，继续保留 V4 的任务决策。** 主策略计算 CPU 耗时减少 **{compute['cpu_reduction']*100:.2f}%**，覆盖证明计算 CPU 耗时减少 **{coverage['cpu_reduction']*100:.2f}%**；两项均保持相应输出一致。改变任务决策的候选没有在独立再确认中证明稳定提速，不能据少量均值优势替换基线。

## 1. 本轮实际验证范围

最终任务比较覆盖 **300 个独立合成场景、900 次完整求解，全部成功，共 {result['primary_counts']['cleared_source_instances']:,} 次源清除**；按场景去重为 {result['primary_counts']['sources_once_per_independent_scenario']:,} 个源。首批几何100场与先验100场使用同一批场景，其中100次V4基线重复运行；独立再确认200场使用新种子。计算优化验证、开发筛选和 Windows 离线检查另列，不混入上述数量。

| 保存批次 | 独立场景 | 完整求解 | 清除源次数 |
|---|---:|---:|---:|
| geometry_holdout100 | 100 | 300 | 3924 |
| prior_holdout100，与上一行同场景 | 100 | 200 | 2616 |
| cells_confirm200，新场景 | 200 | 400 | 5220 |

每批包含全向、25/50/75%定向、全定向、边界朝外、固定正负极端误差及聚簇等10类场景。已修正旧生成器会把0/1比例夹为“两类至少各一个”的问题，当前全向/全定向场景确实只有对应类型；物理误差保持±1°，反馈按0.01°舍入，策略使用±1.005°保守界。它们是明确的合成压力分布，不代表已知的官方场景分布。

## 2. 改变任务决策：独立再确认未证明提速

主指标为每场 \(T_i/N_i\) 的等权平均，配对差为“V4秒/源−候选秒/源”；正值表示节省。下表区间采用**10组内配对 bootstrap**：每组保持样本数、对完整配对差有放回抽样，重复 {args.bootstrap_repeats:,} 次，取2.5%与97.5%分位数。组权重固定，没有把每个源误当作独立样本。

| 候选 | 场数 | V4 → 候选，秒/源 | 均值变化，正为节省 | 节省秒/源95%区间 | 快/慢/平 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

三项组合是预先冻结的主几何候选；首批100场中次级 `cells_flex` 看似更好，因此另取200个新场景确认，期间不再调参。再确认仅平均节省 **{n['mean_paired_saved_seconds_per_source']:.3f}秒/源（{n['reduction_of_mean']*100:.4f}%）**，但 bootstrap 区间 **{ci(n)} 包含0**。161场较快仍不能证明整体稳定收益，最坏一场反而慢 **{n['worst']['relative_regression']*100:.2f}%**。本轮不推荐替换任务决策。

最坏样本 `mixed25 / 663020263` 中，两版都清除16源，但一次8.27米合法微移触发不同路线，扫描站数由8增至16，总任务时间从3714.700秒增至5113.385秒。多出的1398.685秒主要来自移动、检测及切频；覆盖正确不等于时间更短。详见 [最坏案例审计](worst_case_audit.md)。独立再确认中，未启用两层计算缓存的任务候选比较，其平均本机墙钟时间由 {n['baseline_mean_wall_seconds']:.3f} 秒增至 {n['mean_wall_seconds']:.3f} 秒，反映了候选的额外计算开销；这组时间不是当前v2的Windows运行时间。

## 3. 可采用的成果：保持输出的计算优化

| 验证对象 | 工作量与一致性证据 | 原CPU秒 → 新CPU秒 | CPU耗时减少 |
|---|---|---:|---:|
| 主策略 `ExactComputeCache` | 100场、{compute['compared_actions']:,}条动作；动作、反馈、虚拟时间和完成报告全部一致；1299源全清 | {compute['baseline_cpu_s']:.6f} → {compute['accelerated_cpu_s']:.6f} | {compute['cpu_reduction']*100:.2f}% |
| `CoverageComputeCache` | 100个输入：4通过、19有反例、77未决；除 wall 外的所有返回字段一致，含叶盒记录检查 | {coverage['baseline_cpu_s']:.6f} → {coverage['accelerated_cpu_s']:.6f} | {coverage['cpu_reduction']*100:.2f}% |

主策略主要复用同一不可变多边形的包围圆计算，并保留原浮点运算、路线遍历和平局处理次序；覆盖证明复用重复的活跃点集凸包。缓存有上限，退出或异常均清理并恢复函数；原先已测试的{preservation['original_tested_files']}个Python文件SHA-256全部保持一致，原 vendor 文件未修改。当前 `run_speedup.py` 默认 `v4_fast`，版本 `q4-speedup-practice-20260912-v2`，同时启用两层计算优化。

**32.36%与51.76%不能相加，也不能当作虚拟任务或官方HTTP端到端提速。** 前者是主求解循环的配对计算测量，后者是覆盖证明子程序的独立输入测量；机器非独占，两版交替先后。主策略平均每场仅节省约 {compute['mean_cpu_saved_s']*1000:.2f} 毫秒CPU，动作相同意味着任务虚拟时间完全不变。带墙钟截止条件的前瞻候选不享有这项逐动作等价承诺。

## 4. 保证和开发筛选

`cells_flex` 将可行多边形分成完整光学覆盖条带，每次检查所选清除点距条带全部顶点小于20米；移站则必须证明“实际已扫点＋剩余站”的连续覆盖。`combined` 进一步把**原版已有的外切64边形**细化为128个源域半平面，并非补上原来不存在的圆域约束。

无信号保留目标及最后正观测；清除未命中不计成功，全部覆盖尝试失败即停止报错。正常结束须清除16源，或对实际 `scanpoints` 再做连续覆盖证明并核对接口计数。接口结果未知时保留原请求，不发送替代动作。公式与故障审查见 [算法构造与保证](算法构造与保证.md)。

以下仅列关键开发批次，均不充当最终留出效果：

| 开发分支 | 记录工作量 | 筛选结论 |
|---|---:|---|
| 覆盖布局、旋转、换站、冗余站 | 3040组静态减站搜索；另1182次完整任务 | 减站无可接受证书；额外扫描和冗余布局常更慢 |
| 调度、共享测向、延迟清除、站序 | {development['scheduling']['full_task_runs']}次任务，含各自60场后续筛选 | 未选出可替代V4的可靠策略 |
| 机会清除 | 80场×7组＝560次任务 | 收益接近零，降低阈值会增加失败尝试 |
| 统一几何、源域、扇区及先验开发 | 四批共600次任务 | 据此冻结候选后才进入上述留出阶段 |
| 保留已有较短路线 incumbent | 40场×4组＝160次完整任务，全清 | 未修复慢例，作为开发负结果归档 |

多个开发批次复用种子和基线；静态证书输入不是完整任务。早期独立覆盖筛选未使用后来统一的舍入及全类型修正，保留其开发身份。

最坏例审计后又尝试保留已有的较短目标中心路线：`incumbent` 在40场开发集上比V4慢 {abs(incumbent_summary['incumbent']['reduction_of_mean'])*100:.3f}%，`flex_incumbent` 省 {incumbent_summary['flex_incumbent']['reduction_of_mean']*100:.3f}%，`cells_flex_incumbent` 省 {incumbent_summary['cells_flex_incumbent']['reduction_of_mean']*100:.3f}%，但最后一项开发正态95%区间仍跨0，且不及原 `cells_flex` 在同一开发集上的 {original_cells_dev['reduction_of_mean']*100:.3f}%。已知慢例另行诊断重放得到 {incumbent_worst['virtual_seconds']:.3f} 秒，未修复原组合的5113.385秒退步。**保留较短的中心路线仍不能保证更早发现源。** 该分支未晋级新的留出测试，也未部署到默认 `v4_fast` 或官方入口；160次开发运行与已知样本诊断均不并入主验证的300场景/900次求解。数据见 `incumbent_dev40/summary.json` 和 `incumbent_dev40/known_worst_replay.json`。

## 5. 官方演练状态与复现

**本轮新增官方演练0场、正式测试0场。** 当前 Mac 锁屏，需要用户手动解锁，随后重新确认模拟器处于“第四问演练”才能启动。上一轮15场官方结果保持原样，不能用作本版成绩。

{win_status} 这类检查通过真实客户端的协议校验，输入仍来自本地合成传输，不能称作官方演练。

从项目根目录使用 Python 3.13 复现任务比较（每组场数由 `--cases` 指定）：

```sh
python3.13 code/experiments/q4_speedup/bench.py --methods v4 cells_flex combined --cases 10 --seed-base 551920260 --out output/q4_speedup/reproduce_geometry100
python3.13 code/experiments/q4_speedup/bench.py --methods v4 prior_fixed --cases 10 --seed-base 551920260 --out output/q4_speedup/reproduce_prior100
python3.13 code/experiments/q4_speedup/bench.py --methods v4 cells_flex --cases 20 --seed-base 662920260 --out output/q4_speedup/reproduce_cells200
python3.13 code/experiments/q4_speedup/make_speedup_report.py
```

计算等价脚本为 `compute_validate.py`、`coverage_compute_validate.py`，后者读取保留的 `schedule_dev/coverage_compute100/inputs.json`；运行会重建各自结果目录。离线接入检查脚本为 `test_speedup_adapter.py`。官方入口为 `code/experiments/q4_official_practice/run_speedup.py`，先在Windows界面确认Q4演练，再使用 `--method v4_fast --robot-id <队伍账号> --practice-confirmed`；该标志是人工确认，API本身不能识别演练/正式模式。

机器可读结果及bootstrap种子见 [最终统计](最终统计.json)，冻结候选与输入校验值随原始数据保存。
'''
    (OUT/'第四问_第二轮提速实测报告.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(primary_counts=result['primary_counts'],
                          cells_confirm_bootstrap95=n['stratified_paired_bootstrap95_saved_seconds_per_source'],
                          windows=windows),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
