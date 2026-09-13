"""Summarize official random cases; never label them paired scenarios."""
from collections import Counter
from datetime import datetime
import json, math, random, statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
METHODS=['phased','joint','v2','v3','v3_origin20','optical','scenario','future_cover','scenario_future']
NAMES=['分阶段','初版联合','v2','v3','v3 原点扫描','v3 提前光学','多目标场景预测','位置联合设计','两方向组合']

def quantile(values,p):
    a=sorted(values);x=(len(a)-1)*p;i=int(x)
    return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)

def bootstrap(values,rng):
    if len(values)<2:return None
    means=[statistics.fmean(rng.choices(values,k=len(values))) for _ in range(10000)]
    return [quantile(means,.025),quantile(means,.975)]

def main():
    cfg=json.loads((HERE/'bulk_config.json').read_text())
    folder=HERE/'results/bulk'/cfg['batch_id']
    status=json.loads((folder/'status.json').read_text(encoding='utf-8-sig'))
    rows=[json.loads(s) for s in (folder/'completed.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    assert len({r['case'] for r in rows})==len(rows)
    assert len({r['index'] for r in rows})==len(rows)
    for r in rows:
        plan=cfg['schedule'][r['index']-1]
        assert plan['method']==r['method'] and plan['block']==r['block']
        assert r['total']==r['cleared'] and math.isclose(r['virtual_s']/r['total'],r['seconds_per_source'])
    ns=Counter(r['total'] for r in rows)
    weights={n:c/len(rows) for n,c in ns.items()}
    rng=random.Random(20260912)
    stats=[]
    for method,label in zip(METHODS,NAMES):
        rs=[r for r in rows if r['method']==method]
        if not rs:continue
        vals=[r['seconds_per_source'] for r in rs];wall=[r['wall_s'] for r in rs]
        strata={n:[r['seconds_per_source'] for r in rs if r['total']==n] for n in ns}
        standardized=sum(weights[n]*statistics.fmean(a) for n,a in strata.items()) if all(strata.values()) else None
        stats.append(dict(method=method,label=label,n=len(rs),mean=statistics.fmean(vals),median=statistics.median(vals),
                          sd=statistics.stdev(vals) if len(vals)>1 else None,p95=quantile(vals,.95),ci95=bootstrap(vals,rng),
                          source_standardized_mean=standardized,mean_virtual_total=statistics.fmean(r['virtual_s'] for r in rs),
                          wall_mean=statistics.fmean(wall),wall_median=statistics.median(wall),wall_p95=quantile(wall,.95),
                          nonrequest_mean=statistics.fmean(r['nonrequest_s'] for r in rs),
                          full_clear_cases=sum(r['total']==r['cleared'] for r in rs),sources=sum(r['total'] for r in rs),
                          failed_clear_attempts=sum(r['failed_clear_attempts'] for r in rs),
                          per_source_count={n:dict(n=len(a),mean=statistics.fmean(a) if a else None) for n,a in strata.items()}))
    complete=status['status']=='completed' and len(rows)==len(cfg['schedule'])
    out=dict(updated_at=datetime.now().astimezone().isoformat(),complete=complete,cases=len(rows),target=len(cfg['schedule']),
             total_cleared=sum(r['cleared'] for r in rows),source_count_weights=weights,paired_scenarios=False,methods=stats)
    (folder/'statistics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 第三问官方演练：大样本对比','',
           f"状态：{'已完成' if complete else '进行中，以下为阶段性结果'}。已核对 {len(rows)}/{len(cfg['schedule'])} 轮，清除 {out['total_cleared']} 个干扰源。",'',
           '九种方法每组各运行一次，组内顺序按固定随机种子打乱。每轮是新生成的独立官方演练案例，各方法使用不同布局；不称为相同场景配对试验。最初的九轮试跑不并入这批。','',
           '## 虚拟任务耗时','',
           '主指标为每例先计算虚拟总秒数/源数，再对案例等权平均。95% 区间是各方法独立重采样 10000 次的 bootstrap 均值区间。分层均值按这批合并样本的源数分布加权，以减轻各方法源数比例不同造成的影响；任何源数层尚缺样本时不显示。','',
           '| 方法 | 样本 | 全清 | 平均秒/源 | 95% 区间 | 中位数 | P95 | 源数分层均值 |','|---|---:|---:|---:|---|---:|---:|---:|']
    for s in stats:
        ci='样本不足' if s['ci95'] is None else f"[{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]"
        adj='样本不足' if s['source_standardized_mean'] is None else f"{s['source_standardized_mean']:.3f}"
        lines.append(f"| {s['label']} | {s['n']} | {s['full_clear_cases']}/{s['n']} | {s['mean']:.3f} | {ci} | {s['median']:.3f} | {s['p95']:.3f} | {adj} |")
    lines+=['','## 实际运行耗时','',
            '单位为秒。实际耗时包含创建客户端、进入接口、策略计算与通信、退出接口及关闭客户端；不含 Python 进程/模块启动、模拟器准备案例和界面自动化等待。非请求部分含规划及适配开销，不等同于纯 CPU 时间。','',
            '| 方法 | 平均执行 | 中位数 | P95 | 平均非请求部分 | 失败清除尝试总数 |','|---|---:|---:|---:|---:|---:|']
    for s in stats:lines.append(f"| {s['label']} | {s['wall_mean']:.3f} | {s['wall_median']:.3f} | {s['wall_p95']:.3f} | {s['nonrequest_mean']:.3f} | {s['failed_clear_attempts']} |")
    lines+=['','## 按源数分组','',
            '| 方法 | 源数 | 样本数 | 平均秒/源 |','|---|---:|---:|---:|']
    for s in stats:
        for n,a in sorted(s['per_source_count'].items()):
            if a['n']:lines.append(f"| {s['label']} | {n} | {a['n']} | {a['mean']:.3f} |")
    lines+=['','## 记录与解释边界','',
            '- 清除成功数来自已接受的公开接口响应，总源数来自演练结束界面；每轮另保存原始请求响应和官方加密 `.jlog`。案例编号以官方导出日志的文件名为准，界面 OCR 原始读数单独保留。',
            '- 提前光学策略的失败尝试保留并计入虚拟耗时。若某一轮发生程序、通信或全清核对异常，批量脚本会暂停并保留现场，不静默丢弃失败后继续报成功率。',
            '- 有一次自动循环调试在倒计时期间过早发送 `/enter`，未执行策略动作；已按同一请求编号恢复并退出，原始证据保留于 guest 批目录的 `setup_attempts/countdown_enter`，与算法样本分开统计。',
            '- 第 176 例启动时曾停在演练列表并等待超时，尚未创建策略运行记录。核验旧工作进程已结束、界面仍可开始后，从该索引恢复；启动失败现场保留于 `automation_events/0176_start_guard_timeout`，不计为已经执行的算法样本。',
            '- 第 183 例策略已正常完成，登记时遇到 Windows 文件共享锁。保留原案例并补登记，不重新执行；异常现场及文件占用恢复测试记录保留于 `automation_events/0183_ledger_sharing_lock`。',
            '- 当前 Windows 11 ARM 环境使用 Python 3.13.15、NumPy 2.5.3、无 Numba。运行顺序分组随机化，Mac 的其他工作仍可能影响实际运行时间。',
            '- 阶段性结果随样本增加变化；有限样本全清不代表未来所有场景必然全清。',
            '', '[逐例结果](completed.jsonl) · [统计 JSON](statistics.json) · [运行状态](status.json)', '']
    (folder/'大样本结果.md').write_text('\n'.join(lines),encoding='utf-8')
    labels=dict(zip(METHODS,NAMES))
    case_lines=['# 当前逐例成绩','',
                f'已登记 {len(rows)} 例。平均定位清除时间 = 虚拟总耗时 / 成功清除数，单位秒/源，越低越好。以下均为官方演练案例。','',
                '| 序号 | 方法 | 案例编号 | 清除/总数 | 成绩（秒/源） | 虚拟总秒 | 实际执行秒 |',
                '|---:|---|---|---:|---:|---:|---:|']
    for r in rows:
        case_lines.append(f"| {r['index']} | {labels[r['method']]} | {r['case']} | {r['cleared']}/{r['total']} | {r['seconds_per_source']:.2f} | {r['virtual_s']:.2f} | {r['wall_s']:.3f} |")
    (folder/'逐例成绩.md').write_text('\n'.join(case_lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=len(rows),complete=complete,per_method={s['method']:s['n'] for s in stats}),ensure_ascii=False))

if __name__=='__main__':main()
