"""Read the four completed development batches; never construct/run a case."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.q4_sector_release import build_selection

OUT=Path(__file__).resolve().parent
SPLITS=('development','development-stress')
CONFIGS=('sector_1','sector_3')


def read(path): return json.loads(path.read_bytes())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def trace(record):
    # Access only row and accepted action history; evaluation truth is unused.
    row,summary=record['row'],record['summary']
    actions=summary['action_history']; last_time=0.;channel=1
    phases={};detected=set();clears=[];coverage=[];known_scans=0;new_on_cover=0
    for i,a in enumerate(actions):
        phase=a['phase'];kind=a['action'];t=a['virtual_time_s'];delta=t-last_time
        measure=kind=='measure';success=kind=='clear' and a['result']=='success'
        switch=float(measure and a['channel']!=channel)
        parts=dict(movement_s=delta-5*measure-switch-3*(kind=='clear')-2*success,
                   detection_s=5.*measure,switching_s=switch,optical_s=3.*(kind=='clear'),removal_s=2.*success)
        if parts['movement_s'] < -1e-7: raise ValueError('Negative reconstructed movement')
        p=phases.setdefault(phase,dict(actions=0,measures=0,no_signal=0,successful_clears=0,
            movement_s=0.,detection_s=0.,switching_s=0.,optical_s=0.,removal_s=0.,total_s=0.))
        p['actions']+=1;p['measures']+=measure;p['no_signal']+=measure and a['result']=='no_signal'
        p['successful_clears']+=success;p['total_s']+=delta
        for k,v in parts.items():p[k]+=v
        if measure:
            if phase=='coverage':
                coverage.append(i);known_scans+=a['channel'] in detected
                new_on_cover+=a['channel'] not in detected and a['result'] in ('direction','near')
            if a['result'] in ('direction','near'):detected.add(a['channel'])
            channel=a['channel']
        if success:clears.append((i,t,a['channel'],phase))
        last_time=t
    if abs(last_time-row['virtual_time_s'])>1e-6:raise ValueError('Trace does not close at actual final time')
    for key in ('movement_s','detection_s','switching_s','optical_s','removal_s'):
        if abs(sum(p[key] for p in phases.values())-row[key])>1e-5:raise ValueError('Phase cost reconstruction differs: '+key)
    last_coverage=max(coverage,default=-1)
    return dict(seed=row['seed'],source_total=row['source_total'],virtual_time_s=row['virtual_time_s'],
        time_per_source_s=row['virtual_time_s']/row['source_total'],lower_bound_s=row['common_lower_bound_s'],
        time_over_lower_bound=row['virtual_time_s']/row['common_lower_bound_s'],
        phases=phases,coverage_points_visited=summary['coverage_points_visited'],
        coverage_measurements=len(coverage),known_channel_coverage_measures=known_scans,
        first_discoveries_on_coverage=new_on_cover,
        first_clear_s=clears[0][1] if clears else None,
        mean_source_clear_timestamp_s=statistics.mean(v[1] for v in clears) if clears else None,
        last_clear_s=clears[-1][1] if clears else None,
        completion_tail_s=row['virtual_time_s']-clears[-1][1] if clears else None,
        cleared_before_last_actual_coverage=sum(i<last_coverage for i,_,_,_ in clears),
        clear_order=[c for _,_,c,_ in clears],clear_timestamps_s=[t for _,t,_,_ in clears])


def main():
    selected=read(OUT/'selection.json')
    actual=build_selection(selected['development_directories'],ROOT)
    if selected!=actual:raise ValueError('Selection differs from full evidence reconstruction')
    report=dict(schema='q4-sector-development-analysis-v1',source_commit='866b0c268540c60ae75b7ca2d2cc46c26dae9efe',
        selection=selected,analysis_script_sha256=sha(Path(__file__)),selection_sha256=sha(OUT/'selection.json'),
        mechanism_scope='Only accepted action history and row costs. Phase movement is incoming travel to an action, not an isolated counterfactual route cost.',
        batches={},paired_sector3_minus_sector1={})
    all_rows={};all_traces={}
    for config in CONFIGS:
        for split in SPLITS:
            folder=ROOT/'results/q4_sector_service'/config/split
            summary=read(folder/'summary.json');audit=read(folder/'independent_audit.json')
            if not audit['all_passed']:raise ValueError('Mechanism report awaits all first audits passing')
            traces=[]
            for row in summary['rows']:
                with gzip.open(folder/'records'/f"{row['strategy']}-{row['seed']}.json.gz",'rt',encoding='utf-8') as stream:
                    record=json.load(stream)
                traces.append(trace(record))
            means={k:statistics.mean(t[k] for t in traces) for k in (
                'coverage_points_visited','coverage_measurements','known_channel_coverage_measures','first_discoveries_on_coverage',
                'first_clear_s','mean_source_clear_timestamp_s','last_clear_s','completion_tail_s','cleared_before_last_actual_coverage')}
            phase_names=sorted({name for t in traces for name in t['phases']})
            phases={name:{k:statistics.mean(t['phases'].get(name,{}).get(k,0.) for t in traces)
                for k in ('actions','measures','no_signal','successful_clears','movement_s','detection_s','switching_s','optical_s','removal_s','total_s')}
                for name in phase_names}
            key=f'{config}/{split}'
            report['batches'][key]=dict(summary={k:v for k,v in summary.items() if k!='rows'},trace_means=means,
                mean_phase_costs=phases,per_case=traces,evidence_sha256={name:sha(folder/name) for name in
                    ('plan.json','manifest.json','freeze.json','source.zip','summary.json','independent_audit.json')})
            all_rows[key]={r['seed']:r for r in summary['rows']};all_traces[key]={t['seed']:t for t in traces}
    for split in SPLITS:
        left,right=all_rows[f'sector_1/{split}'],all_rows[f'sector_3/{split}']
        pairs=[]
        for seed in sorted(left):
            a,b=left[seed],right[seed]
            if a['case_sha256']!=b['case_sha256'] or a['common_lower_bound_s']!=b['common_lower_bound_s']:raise ValueError('Pair identity differs')
            keys=('movement_s','detection_s','switching_s','optical_s','removal_s')
            ta,tb=all_traces[f'sector_1/{split}'][seed],all_traces[f'sector_3/{split}'][seed]
            pairs.append(dict(seed=seed,N=a['source_total'],sector_1_T_s=a['virtual_time_s'],sector_3_T_s=b['virtual_time_s'],
                LB_s=a['common_lower_bound_s'],sector_1_T_LB=a['time_over_lower_bound'],sector_3_T_LB=b['time_over_lower_bound'],
                delta_T_s=b['virtual_time_s']-a['virtual_time_s'],delta_T_per_N_s=(b['virtual_time_s']-a['virtual_time_s'])/a['source_total'],
                delta_components_s={k:b[k]-a[k] for k in keys},
                delta_mean_clear_timestamp_s=tb['mean_source_clear_timestamp_s']-ta['mean_source_clear_timestamp_s'],
                delta_coverage_measurements=tb['coverage_measurements']-ta['coverage_measurements']))
        report['paired_sector3_minus_sector1'][split]=dict(
            mean_delta_T_s=statistics.mean(p['delta_T_s'] for p in pairs),
            mean_delta_T_per_N_s=statistics.mean(p['delta_T_per_N_s'] for p in pairs),
            faster=sum(p['delta_T_s']<0 for p in pairs),slower=sum(p['delta_T_s']>0 for p in pairs),ties=sum(p['delta_T_s']==0 for p in pairs),
            mean_delta_components_s={k:statistics.mean(p['delta_components_s'][k] for p in pairs) for k in keys},
            pairs=pairs,best_three=sorted(pairs,key=lambda p:p['delta_T_s'])[:3],worst_three=sorted(pairs,key=lambda p:p['delta_T_s'])[-3:])
    (OUT/'development-metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# R26 开发结果：局部分组访问未通过继续门槛','',
        '两候选各119局（随机70+压力49），共238局全部执行。四批均全清、原始费用/覆盖/继承前缀首审全部通过，源码未变。没有候选同时满足两组 mean(T/N)≤500，因此 selection.passed=false；不打开独立6322001/6324001数据。未作参数修订。','',
        '以下均为本地假定生成分布的开发结果。主指标逐局等权 mean(T/N)；T/LB始终为 mean(T)/mean(LB)。500仅是继续投入门槛，460才是最终总体目标。','',
        '| 候选/批次 | 全清 | mean(T/N) | P95(T/N) | mean(T) | mean(LB) | T/LB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,v in report['batches'].items():
        s=v['summary'];lines.append(f"| {key} | {s['successful']}/{s['runs']} | {s['mean_time_per_source_s']:.3f} | {s['p95_time_per_source_s']:.3f} | {s['mean_time_s']:.3f} | {s['mean_lower_bound_s']:.3f} | {s['mean_time_over_mean_lower_bound']:.6f} |")
    lines+=['','完整每N/家族分层、失败列表、主指标区间、均值比与pooled口径见 development-metrics.json 的各批 summary。开发压力每N×家族仅1局，bootstrap的零宽区间没有层内变异估计，不能解释为确定性。','',
        '## 同场景两候选的实际差异','',
        '| 批次 | sector_3 − sector_1：mean(T/N) | mean(T)差 | 移动差 | 检测差 | 换频差 | 光学差 | s1 / s3 T/LB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split,p in report['paired_sector3_minus_sector1'].items():
        c=p['mean_delta_components_s'];a=report['batches'][f'sector_1/{split}']['summary'];b=report['batches'][f'sector_3/{split}']['summary']
        lines.append(f"| {split} | {p['mean_delta_T_per_N_s']:.3f} | {p['mean_delta_T_s']:.3f} | {c['movement_s']:.3f} | {c['detection_s']:.3f} | {c['switching_s']:.3f} | {c['optical_s']:.3f} | {a['mean_time_over_mean_lower_bound']:.6f} / {b['mean_time_over_mean_lower_bound']:.6f} |")
    lines+=['','移除费用差为零（同场景源数相同、全部清除）。只比较这两个新候选；未在632复跑原R12，不能把不同630数据集的均值差当作R26相对原方法的因果收益。','',
        '## 实际清除时机与扫描开销','',
        '| 候选/批次 | 首清时刻 | 平均源清除时刻 | 末次实际coverage前已清源数 | coverage测量次数 | 已知频道coverage次数 | mean(T) | mean(LB) | T/LB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,v in report['batches'].items():
        m=v['trace_means'];s=v['summary'];lines.append(f"| {key} | {m['first_clear_s']:.3f} | {m['mean_source_clear_timestamp_s']:.3f} | {m['cleared_before_last_actual_coverage']:.3f} | {m['coverage_measurements']:.3f} | {m['known_channel_coverage_measures']:.3f} | {s['mean_time_s']:.3f} | {s['mean_lower_bound_s']:.3f} | {s['mean_time_over_mean_lower_bound']:.6f} |")
    lines+=['','时间按每局统计后等权平均。末次实际coverage不必等于扫完整22站（N=16可合法提前停止发现）；本项只是轨迹时机描述，不是反事实可省的清除数。每个phase的移动按到该动作的实际入程计，不把它混称纯覆盖几何长度。','']
    for split in SPLITS:
        av,bv=report['batches'][f'sector_1/{split}'],report['batches'][f'sector_3/{split}']
        a,b=av['trace_means'],bv['trace_means'];p=report['paired_sector3_minus_sector1'][split]
        lines.append(f"{split}：sector_3首清平均提前{a['first_clear_s']-b['first_clear_s']:.3f}秒，平均源清除时刻提前{a['mean_source_clear_timestamp_s']-b['mean_source_clear_timestamp_s']:.3f}秒，但整局仅平均少{-p['mean_delta_T_s']:.3f}秒。实际coverage测量平均减少{a['coverage_measurements']-b['coverage_measurements']:.3f}次；已知频道coverage反而增加{b['known_channel_coverage_measures']-a['known_channel_coverage_measures']:.3f}次，不能把总减少全归因为少测已知频道。两者coverage阶段仍占实际总时间的{100*av['mean_phase_costs']['coverage']['total_s']/av['summary']['mean_time_s']:.2f}% / {100*bv['mean_phase_costs']['coverage']['total_s']/bv['summary']['mean_time_s']:.2f}%；对应LB及T/LB见上表。")
    lines+=['','## 两个实际退化例（sector_3相对sector_1）','',
        '| 批次/seed/N | s1 T / T/N | s3 T / T/N | LB | s1 / s3 T/LB | 多用时间 | 移动/检测/换频/光学变化 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for split,p in report['paired_sector3_minus_sector1'].items():
        w=p['worst_three'][-1];c=w['delta_components_s'];n=w['N']
        lines.append(f"| {split}/{w['seed']}/{n} | {w['sector_1_T_s']:.3f} / {w['sector_1_T_s']/n:.3f} | {w['sector_3_T_s']:.3f} / {w['sector_3_T_s']/n:.3f} | {w['LB_s']:.3f} | {w['sector_1_T_LB']:.6f} / {w['sector_3_T_LB']:.6f} | {w['delta_T_s']:.3f} | {c['movement_s']:.3f}/{c['detection_s']:.3f}/{c['switching_s']:.3f}/{c['optical_s']:.3f} |")
    lines+=['','各例均来自同一个已完成场景的真实动作计费差。访问顺序改变后，获取方位的先后、可清时机及调度入口一起变化；只能证明费用实际从哪些分量增加，不能把这些关联当作某一次决策的反事实因果证明。','',
        '本方法没有减少必保留的22个站，仅改变访问顺序。原纯覆盖路线17612.419米，两候选分别18451.731和18171.960米，先天增加约167.862和111.908秒纯行走；期望收益必须靠更早形成可清区域并减少后续折返抵消。实际扫描费用仍需支付，早清和总时间也不等价。上表与逐局分量给出了这一假设在本轮的实际表现，但没有隔离各机制的因果贡献。','',
        '首审证据、source.zip及四份plan全部保留。development-metrics.json包含逐局清除次序/时刻、phase费用、两方法最好/最差三局及所有输入SHA；分析只使用已结束记录的row和真实行动，不以隐藏位置生成动作。复核脚本：`python -B research/q4_sector_service/analyze_development.py`。','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(passed=selected['passed'],selected=selected['selected'],batches=len(report['batches']))))


if __name__=='__main__':main()
