"""Audit saved paired runs and report uncertainty without selecting cases."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from runner import HERE,METHODS,LABELS,Scenario,check_inputs

def audit(folder):
    status=json.loads((folder/'status.json').read_text())
    meta=json.loads((folder/'metadata.json').read_text())
    rows=[json.loads(line) for line in (folder/'results.jsonl').read_text().splitlines()]
    expected=meta['cases']*len(METHODS)*meta['repeats']
    assert status['status']=='completed' and len(rows)==expected==status['completed']
    assert len({(r['index'],r['method'],r['repeat']) for r in rows})==expected
    assert len(list((folder/'runs').glob('*.json.gz')))==expected
    group=defaultdict(list);scenes={};actions=0;failed=0;max_roundoff=0.
    for i,row in enumerate(rows):
        packed=(folder/row['evidence']).read_bytes()
        assert hashlib.sha256(packed).hexdigest()==row['evidence_sha256']
        data=json.loads(gzip.decompress(packed));state=data['state']
        scene=Scenario.from_dict(state['scenario'])
        assert scene.problem_no==3 and scene.profile=='practice'
        assert scene.case_id==row['case_id'] and scene.label==row['seed']
        assert row['seed']==meta['seed_prefix']+str(row['index'])
        canonical=json.dumps(scene.as_dict(),sort_keys=True)
        if row['index'] in scenes:assert scenes[row['index']]==canonical
        else:scenes[row['index']]=canonical
        group[(row['index'],row['repeat'])].append(row['method'])
        assert all(row[k]==v for k,v in data['evaluation'].items())
        assert row['replay_matched'] and row['replay_actions']==len(data['history'])
        assert row['virtual_us']==state['virtual_time_us']
        assert row['virtual_s']==state['virtual_time_s']==row['virtual_us']/1e6
        assert row['cleared']==state['cleared_count'] and row['source_total']==len(scene.sources)
        assert row['all_cleared']==(row['cleared']==row['source_total'])
        assert row['seconds_per_source']==(row['virtual_s']/row['cleared'] if row['cleared'] else None)
        costs=dict(movement=0,switching=0,detection=0,optical=0,removal=0)
        position=(0.,0.);channel=1;cleared=set();clear_attempts=0;measurements=0;switches=0
        sources={s.channel:s for s in scene.sources}
        for n,action in enumerate(data['history'],1):
            assert action['index']==n and action['response']['accepted']
            request=action['request'];path=action['path']
            if path in ('/measure','/clear'):
                point=(request['position']['x'],request['position']['y']);c=request['channel']
                distance=math.hypot(point[0]-position[0],point[1]-position[1])
                # Reproduce the documented per-segment integer-microsecond rounding.
                costs['movement']+=math.floor(distance*1e12/5_000_000+0.5)
                position=point
                if path=='/measure':
                    switches+=int(c!=channel);costs['switching']+=int(c!=channel)*1_000_000
                    channel=c;measurements+=1;costs['detection']+=5_000_000
                else:
                    clear_attempts+=1;costs['optical']+=3_000_000
                    source=sources.get(c)
                    expected_success=source is not None and c not in cleared and math.hypot(source.x-point[0],source.y-point[1])<=20
                    success=action['response']['clear_result']=='success'
                    assert success==expected_success
                    if success:cleared.add(c);costs['removal']+=2_000_000
                error=abs(sum(costs.values())/1e6-action['response']['virtual_time_s'])
                max_roundoff=max(max_roundoff,error);assert error<1e-8
        assert sum(costs.values())==row['virtual_us']
        assert {k:v/1e6 for k,v in costs.items()}==row['time_breakdown_s']
        assert len(cleared)==row['cleared'] and measurements==row['measurements'] and switches==row['switches']
        assert clear_attempts==row['clear_attempts'] and clear_attempts-len(cleared)==row['failed_clears']
        if row['success']:
            assert data['history'][0]['path']=='/enter' and data['history'][-1]['path']=='/exit'
            assert row['normal_exit'] and row['certificate'] and row['all_cleared'] and not row['error']
        actions+=len(data['history']);failed+=not row['success']
        if (i+1)%1000==0:print(f'Audited {i+1}/{expected}',flush=True)
    assert len(scenes)==meta['cases']
    assert all(sorted(v)==sorted(METHODS) for v in group.values())
    result=dict(status='verified',runs=expected,paired_cases=len(scenes),repeats=meta['repeats'],
                accepted_actions=actions,failed_runs=failed,paired_scenarios_identical=True,
                all_replays_matched=True,max_cost_disagreement_s=max_roundoff,
                results_sha256=hashlib.sha256((folder/'results.jsonl').read_bytes()).hexdigest())
    (folder/'validation.json').write_text(json.dumps(result,indent=2))
    return rows,result

def ci(values,rng):
    a=np.asarray(values,dtype=float)
    means=[]
    for _ in range(20):means.extend(a[rng.integers(0,len(a),size=(500,len(a)))].mean(axis=1))
    return np.quantile(means,[.025,.975]).tolist()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--main',type=Path,default=HERE/'results/paired2000')
    ap.add_argument('--timing',type=Path,default=HERE/'results/timing50')
    args=ap.parse_args();check_inputs()
    rows,validation=audit(args.main)
    timing,timing_validation=audit(args.timing)
    assert validation['paired_cases']==2000 and timing_validation['paired_cases']==50
    # An unsuccessful run stays in the audit and success denominator; do not
    # silently turn conditional-on-success means into the main ranking.
    if any(not r['success'] for r in rows+timing):
        (args.main/'failures.json').write_text(json.dumps([r for r in rows+timing if not r['success']],ensure_ascii=False,indent=2))
        raise SystemExit('Failures retained; inspect before reporting a complete ranking')
    rng=np.random.default_rng(20260912)
    by={m:sorted([r for r in rows if r['method']==m],key=lambda r:r['index']) for m in METHODS}
    stats=[];baseline=np.array([r['seconds_per_source'] for r in by['v3']])
    for method in METHODS:
        rr=by[method];a=np.array([r['seconds_per_source'] for r in rr]);diff=a-baseline
        tr=[r for r in timing if r['method']==method];wall=np.array([r['wall_s'] for r in tr])
        stats.append(dict(method=method,label=LABELS[method],n=len(rr),mean=float(a.mean()),ci95=ci(a,rng),
                          median=float(np.median(a)),p95=float(np.quantile(a,.95)),minimum=float(a.min()),maximum=float(a.max()),
                          all_clear=len(rr),sources=sum(r['cleared'] for r in rr),failed_clear_attempts=sum(r['failed_clears'] for r in rr),
                          below200=int((a<200).sum()),paired_delta_vs_v3=float(diff.mean()),paired_ci95_vs_v3=ci(diff,rng),
                          paired_faster_vs_v3=int((diff< -1e-9).sum()),paired_equal_vs_v3=int((np.abs(diff)<=1e-9).sum()),
                          paired_slower_vs_v3=int((diff>1e-9).sum()),wall_mean=float(wall.mean()),wall_median=float(np.median(wall)),
                          wall_p95=float(np.quantile(wall,.95)),nonrequest_mean=float(np.mean([r['nonrequest_wall_s'] for r in tr])),
                          mean_measurements=float(np.mean([r['measurements'] for r in rr])),mean_movement_m=float(np.mean([r['movement_m'] for r in rr])),
                          strata={n:dict(n=sum(r['source_total']==n for r in rr),mean=float(np.mean([r['seconds_per_source'] for r in rr if r['source_total']==n]))) for n in range(10,17)}))
    stats.sort(key=lambda s:s['mean'])
    result=dict(kind='local_static_reconstruction',paired=True,main_validation=validation,timing_validation=timing_validation,
                methods=stats,bootstrap_resamples=10000,source_count_distribution=dict(Counter(r['source_total'] for r in by['v3'])))
    report=HERE/'本地模拟器前五名结果.md'
    lines=['# 第三问前五名：本地复原模拟器结果','',
           '2000 个共同案例 × 5 种方法，共 10000 轮。所有方法使用完全相同的地图和空间测向误差，参数按之前版本固定。另用 50 个独立地图，每法两遍串行执行，比较实际运行速度。','',
           '模拟器来自工作区 `local-jammers-simulator`，按当前代码固定副本；基于 exe 静态复原，尚未完成与官方运行时逐输入输出对照。以下属于本地复原仿真结果，与先前 184 例官方演练分开。','',
           '## 成绩：秒/源，越低越好','',
           '每例先计算虚拟总耗时 ÷ 成功清除数，再对 2000 例等权平均。全清状态单独核对，失败的光学尝试保留并计费。95% 区间由 10000 次案例重采样得到。','',
           '| 方法 | 全清案例 | 平均成绩 | 均值 95% 区间 | 中位数 | P95 | 最好～最差 | 低于 200 秒/源 |','|---|---:|---:|---|---:|---:|---:|---:|']
    for s in stats:lines.append(f"| {s['label']} | {s['all_clear']}/{s['n']} | {s['mean']:.3f} | [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] | {s['median']:.3f} | {s['p95']:.3f} | {s['minimum']:.2f}～{s['maximum']:.2f} | {s['below200']}/{s['n']} |")
    lines+=['','## 同一案例相对 v3 的差值','',
            '差值 = 本方法成绩 − v3 成绩，负值代表更快。以每例差值重采样，所以区间保留了同图配对关系；这些区间未作多重比较校正。','',
            '| 方法 | 平均差值（秒/源） | 差值 95% 区间 | 更快 / 相同 / 更慢案例 |','|---|---:|---|---:|']
    for s in stats:
        lines.append(f"| {s['label']} | {s['paired_delta_vs_v3']:+.3f} | [{s['paired_ci95_vs_v3'][0]:+.3f}, {s['paired_ci95_vs_v3'][1]:+.3f}] | {s['paired_faster_vs_v3']} / {s['paired_equal_vs_v3']} / {s['paired_slower_vs_v3']} |")
    lines+=['','## 本机实际执行速度','',
            '50 个额外共同地图 × 两轮，单进程顺序执行，并旋转/反转方法顺序。单位秒/例，包括策略创建、规划、进入/退出及进程内模拟接口；不含场景生成、JIT 预热、导出和重放。非请求部分包括策略和少量接入开销，不等于纯 CPU 时间。','',
            '| 方法 | 平均执行秒 | 中位数 | P95 | 平均非请求秒 | 主批失败光学尝试 |','|---|---:|---:|---:|---:|---:|']
    for s in stats:lines.append(f"| {s['label']} | {s['wall_mean']:.4f} | {s['wall_median']:.4f} | {s['wall_p95']:.4f} | {s['nonrequest_mean']:.4f} | {s['failed_clear_attempts']} |")
    lines+=['','## 按干扰源数分层','',
            '| 源数 | 每法案例数 | '+ ' | '.join(LABELS[m] for m in METHODS)+' |',
            '|---:|---:|'+ '|'.join(['---:']*len(METHODS))+'|']
    by_stat={s['method']:s for s in stats}
    for n in range(10,17):lines.append(f"| {n} | {by_stat['v3']['strata'][n]['n']} | "+' | '.join(f"{by_stat[m]['strata'][n]['mean']:.3f}" for m in METHODS)+' |')
    lines+=['','## 验证与复现','',
            f"- 主批 {validation['runs']} 轮、计时批 {timing_validation['runs']} 轮全部保存场景与接受的请求响应，逐轮重放一致；主批共 {validation['accepted_actions']} 个动作。",
            '- 独立审计核对压缩记录 SHA-256、相同案例的五份场景一致、成功清除距离、动作数、频道切换、每段移动取整及总微秒时间；不读取真值指导策略。',
            '- 本地模拟器原有 23 项测试通过；接入测试对 2 张地图 × 5 种方法比较原官方协议适配器，全部逐动作一致。另有 25 轮小批试跑，与主统计分开。',
            '- 所有策略与模拟器源码固定在 vendor，源文件与摘要见 source_manifest.json；命令、运行环境及计时范围见各结果目录 metadata.json。',
            '- 有限样本全清不代表全部可能场景都全清；本地复原结果也不能替代正式测试成绩。','',
            '[2000 个案例的逐例成绩](逐例成绩.md) · [完整统计](statistics.json) · [主批审计](results/paired2000/validation.json) · [运行说明](README.md)','']
    report.write_text('\n'.join(lines))
    (HERE/'statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=['# 前五名：2000 个共同案例的逐例成绩','',
           '单位秒/源，越低越好；每个单元格对应同一地图和同一空间误差场。均已全清。','',
           '| 序号 | 案例编号 | 源数 | '+' | '.join(LABELS[m] for m in METHODS)+' |',
           '|---:|---|---:|'+ '|'.join(['---:']*len(METHODS))+'|']
    for i in range(2000):
        r=by['v3'][i];lines.append(f"| {i} | {r['case_id']} | {r['source_total']} | "+' | '.join(f"{by[m][i]['seconds_per_source']:.3f}" for m in METHODS)+' |')
    (HERE/'逐例成绩.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([dict(method=s['method'],mean=s['mean'],wall_s=s['wall_mean'],delta_vs_v3=s['paired_delta_vs_v3'],delta_ci=s['paired_ci95_vs_v3']) for s in stats],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
