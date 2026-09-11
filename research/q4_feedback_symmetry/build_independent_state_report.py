"""Close the existing R16 state evidence after the user cancelled new RL runs.

This never constructs a case, runs a policy, changes the old protocol or writes
qualification.json. The new per-source metric is explicitly retrospective.
"""
from pathlib import Path
import gzip, hashlib, json, statistics, sys, zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.evaluate_q4_feedback_symmetry import (RESEARCH,RESULTS,INDEPENDENT,AUDITS,
    development_identity,stage_specs,validate_matrix,validate_audit,candidate_passes)
from experiments.evaluate_q4_round2 import comparison
from experiments.run_q4_round2 import digest,report_rows
from experiments.run_q4_state_study import percentile

def read(p): return json.loads(p.read_bytes())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(rows):
    assert rows and all(type(r['source_total']) is int and r['source_total']>0 for r in rows)
    ratios=[r['penalized_time_s']/r['source_total'] for r in rows]
    total=sum(r['penalized_time_s'] for r in rows); sources=sum(r['source_total'] for r in rows)
    lower=statistics.mean(r['common_lower_bound_s'] for r in rows)
    return dict(runs=len(rows),successful=sum(r['successful'] for r in rows),all_clear=all(r['successful'] for r in rows),
        mean_time_s=total/len(rows),mean_time_per_source_s=statistics.mean(ratios),pooled_time_per_source_s=total/sources,
        p95_time_per_source_s=percentile(ratios,.95),total_time_s=total,total_sources=sources,
        mean_lower_bound_s=lower,mean_time_over_mean_lower_bound=(total/len(rows))/lower,
        mean_individual_time_over_lower_bound=statistics.mean(r['penalized_time_over_lower_bound'] for r in rows),
        user_target_seconds=460,below_user_target=statistics.mean(ratios)<460,
        failure_rows=[r for r in rows if not r['successful']])

def main():
    destination=RESEARCH/'independent-state-report.json';doc=RESEARCH/'INDEPENDENT_STATE_RESULTS.md'
    assert not destination.exists() and not doc.exists()
    frozen=read(RESEARCH/'development-freeze.json'); selection=read(RESEARCH/'selection.json')
    assert all(frozen[k]==v for k,v in development_identity().items())
    evidence={}; datasets={}; comparisons={}
    for stage,seeds in INDEPENDENT.items():
        d=RESULTS/stage;specs=stage_specs(stage);manifest=read(d/'manifest.json');freeze=read(d/'freeze.json')
        assert manifest['source_sha256']==frozen['source_sha256'] and manifest['specs']==specs and manifest['seeds']==seeds
        assert manifest['selection_sha256']==sha(RESEARCH/'selection.json') and freeze['manifest_sha256']==digest(manifest)
        with zipfile.ZipFile(d/'source.zip') as z:
            assert all(hashlib.sha256(z.read(p)).hexdigest()==v for p,v in frozen['source_sha256'].items())
        report=read(d/'summary.json');rows=report['rows'];validate_matrix(rows,seeds,specs,stage)
        assert report_rows(sorted(rows,key=lambda r:(r['seed'],r['strategy'])))==report
        assert {p.name for p in (d/'records').glob('*.json.gz')}=={f'{label}-{seed}.json.gz' for label in specs for seed in seeds}
        for row in rows:
            p=d/'records'/f"{row['strategy']}-{row['seed']}.json.gz";r=json.loads(gzip.decompress(p.read_bytes()))
            assert r['row']==row and r['spec']==specs[row['strategy']]
            # Count only: truth coordinates/type/orientation never inform this
            # analysis or a policy decision. N comes from the after-exit score.
            assert len(r['evaluation']['ground_truth']['sources'])==row['source_total']
            evidence[p.relative_to(ROOT).as_posix()]=sha(p)
        for filename,label in AUDITS.items():
            if label is not None and label not in specs: continue
            validate_audit(read(d/filename),label,seeds,rows,d);evidence[(d/filename).relative_to(ROOT).as_posix()]=sha(d/filename)
        for filename in ('source.zip','manifest.json','freeze.json','summary.json'):
            evidence[(d/filename).relative_to(ROOT).as_posix()]=sha(d/filename)
        groups={}
        for label in specs:
            rs=[r for r in rows if r['strategy']==label]
            groups[label]=metrics(rs)
            groups[label]['by_source_count']={str(n):metrics([r for r in rs if r['source_total']==n]) for n in sorted({r['source_total'] for r in rs})}
        datasets[stage]=groups
        comparisons[stage]=comparison(report,'compact_feedback_centers','compact_joint_continuation')
    output=dict(report_complete=False,state_only_complete=True,all_state_audits_passed=True,
        original_full_qualification='Not run: user cancelled the not-yet-started 212 independent RL cases and routine multi-arm follow-up',
        independent_rl_completed=0,independent_rl_planned_but_cancelled=212,
        metric_change='User changed main metric after these fixed independent state runs: equal-run mean(T_i/N_i)<460 s. Pooled sum(T)/sum(N) and T/LB are auxiliary. This report does not rewrite the old pre-registration.',
        failure_policy='Use existing penalized_time_s: failed run 360000 s; all-clear is a separate required gate. All 848 retained state records succeeded.',
        old_state_numeric_gate_passed=candidate_passes(comparisons['confirmation'],comparisons['stress'],'confirmation'),
        old_frozen_comparisons_vs_r12=comparisons,selected=selection['selected'],selection_sha256=sha(RESEARCH/'selection.json'),
        primary_metric='arithmetic mean over cases of T_i/N_i',datasets=datasets,evidence_sha256=evidence)
    with destination.open('x',encoding='utf-8',newline='\n') as f: json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    selected='compact_feedback_centers'; names={'confirmation':'随机128','stress':'困难84'}
    text=['# R16 独立状态结果：用户变更指标后的保留报告','',
        '848条已完成状态记录和两组各四层审计均通过，全部全清。用户在状态运行结束后，将主指标改为各局等权的 **mean(T_i/N_i)<460秒/源**，并取消尚未启动的212条独立RL和后续常规多臂横比。本报告只闭合现有状态证据；旧完整qualification缺少已取消RL，未运行完整选择器，`report_complete=false`、`state_only_complete=true`。没有回写625预登记、伪造RL或补抽场景。','',
        '新的mean(T/N)在这批数据上是事后按用户要求计算的指标，不冒充预登记主检验；后续新实验应提前固定源数分层。N只取终局评分中的真实源数量，未用于策略。全部病例成功，因此本表实际T与既有惩罚T相同。','',
        '|数据|主指标mean(T/N)|辅助总T/总N|P95(T/N)|平均T/s|辅助均T/均LB|全清|','|---|---:|---:|---:|---:|---:|---:|']
    for stage,title in names.items():
        m=datasets[stage][selected];text.append(f"|{title}|{m['mean_time_per_source_s']:.6f}|{m['pooled_time_per_source_s']:.6f}|{m['p95_time_per_source_s']:.6f}|{m['mean_time_s']:.6f}|{m['mean_time_over_mean_lower_bound']:.6f}|{m['successful']}/{m['runs']}|")
    text+=['','两组主指标均约504秒/源，均未达到460。总T/总N会给予源更多的局更大权重，不能替代各局等权主指标；困难案例也不与随机案例混池制造更低的数字。T/LB仍是理想化事后理论下界的辅助比值，不是在线可实现最优性证明。','','## 按真实源数量分组（选定centers）','','|N|随机局数|随机mean(T/N)|随机P95(T/N)|困难局数|困难mean(T/N)|困难P95(T/N)|','|---:|---:|---:|---:|---:|---:|---:|']
    for n in range(10,17):
        a=datasets['confirmation'][selected]['by_source_count'][str(n)];b=datasets['stress'][selected]['by_source_count'][str(n)]
        text.append(f"|{n}|{a['runs']}|{a['mean_time_per_source_s']:.6f}|{a['p95_time_per_source_s']:.6f}|{b['runs']}|{b['mean_time_per_source_s']:.6f}|{b['p95_time_per_source_s']:.6f}|")
    text+=['','## 原冻结统计事实','','保留R12同批记录的旧比较，但不据此新增任何横向运行或修改旧门槛。','','|数据|平均节省/s|配对95%区间/s|改善比例|P95时间比|胜/负/平|','|---|---:|---|---:|---:|---:|']
    for stage,title in names.items():
        c=comparisons[stage];lo,hi=c['saving_ci95_s'];text.append(f"|{title}|{c['mean_saved_s']:.6f}|[{lo:.6f},{hi:.6f}]|{100*c['mean_reduction_fraction']:.4f}%|{c['p95_ratio']:.6f}|{c['wins']}/{c['losses']}/{c['pairs']-c['wins']-c['losses']}|")
    text+=['','随机平均改善不足原0.5%门槛，且两组节省区间均跨零；旧状态数值门槛不通过。除此以外，旧完整资格流程的RL部分已按用户指示中止，不能写完整资格完成。开发中较大的收益没有在这批独立数据上形成同等幅度的稳定证据。','','全部四臂既有记录的新指标、逐N统计、原比较全部退化行及完整输入SHA保存在 [independent-state-report.json](independent-state-report.json)。原始848条及八份审计在 `results/q4_feedback_symmetry/{confirmation,stress}`，独立RL目录没有创建。原625协议和开发选择文件保持原字节，不追加qualification.json。']
    with doc.open('x',encoding='utf-8',newline='\n') as f: f.write('\n'.join(text)+'\n')
    print(json.dumps(dict(state_only_complete=True,original_report_complete=False,old_state_gate=output['old_state_numeric_gate_passed'],primary={s:datasets[s][selected]['mean_time_per_source_s'] for s in datasets})))

if __name__=='__main__': main()
