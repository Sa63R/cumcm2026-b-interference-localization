"""Unfiltered paired analysis of the shipped local synthetic experiments."""
from __future__ import annotations
import argparse,collections,csv,json,math,random,statistics as st
from pathlib import Path

MAIN={'mixed25','mixed50','mixed75','boundary','plus','minus'}
OOD={'cluster','all_radius_1000','smooth_error'}

def quantile(x,q):
    x=sorted(x);z=(len(x)-1)*q;i=int(z)
    return x[i]+(x[min(i+1,len(x)-1)]-x[i])*(z-i)

def aggregate(rows,base):
    ts=[r['seconds_per_cleared'] for r in rows]
    bs=[base[r['seed']]['seconds_per_cleared'] for r in rows]
    dd=[a-b for a,b in zip(ts,bs)]
    se=st.stdev(dd)/math.sqrt(len(dd)) if len(dd)>1 else 0.
    ret=[(a/b-1)*100 for a,b in zip(ts,bs)]
    out=dict(cases=len(rows),total_sources=sum(r['targets'] for r in rows),
      full_clear_cases=sum(r['targets']==r['cleared'] for r in rows),mean_seconds_per_source=st.mean(ts),
      baseline_mean_seconds_per_source=st.mean(bs),reduction_percent=(1-st.mean(ts)/st.mean(bs))*100,
      paired_saved_mean=-st.mean(dd),paired_saved_normal95=[-st.mean(dd)-1.96*se,-st.mean(dd)+1.96*se],
      p95_seconds_per_source=quantile(ts,.95),faster=sum(x< -1e-7 for x in dd),
      slower=sum(x>1e-7 for x in dd),tied=sum(abs(x)<=1e-7 for x in dd),
      worst_relative_regression_percent=max(ret),best_relative_reduction_percent=-min(ret),
      mean_distance_m=st.mean(r['distance_m'] for r in rows),mean_detections=st.mean(r['detections'] for r in rows),
      mean_switches=st.mean(r['switches'] for r in rows),mean_optical_failures=st.mean(r['failed_clears'] for r in rows),
      mean_tail_after_last_clear=st.mean(r['tail_after_last_clear'] for r in rows),
      mean_wall_seconds=st.mean(r['wall_seconds'] for r in rows),max_wall_seconds=max(r['wall_seconds'] for r in rows),
      mean_planner_calls=st.mean(r['planner_calls'] for r in rows),mean_planner_accepts=st.mean(r['planner_accepts'] for r in rows),
      total_planner_fallbacks=sum(sum(r['failures'].values()) for r in rows),
      total_rb_fallbacks=sum(r['rb_share_failures'] for r in rows))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default='results');a=p.parse_args();root=Path(a.directory)
    rows=[]
    for f in ['holdout.jsonl','holdout_global.jsonl']:
        path=root/f
        if path.exists():rows.extend(json.loads(l) for l in path.read_text().splitlines() if l.strip())
    errors=[r for r in rows if 'error' in r]
    good=[r for r in rows if 'error' not in r];base={r['seed']:r for r in good if r['version']=='v4'}
    versions=['v4','rb_only','local_only','v5','global'];out={'warning':'LOCAL SYNTHETIC, NOT OFFICIAL; model misspecification possible','errors':errors,'groups':{}}
    for label,names in [('main',MAIN),('ood',OOD),('all',MAIN|OOD)]+[(g,{g}) for g in sorted(MAIN|OOD)]:
        out['groups'][label]={}
        for v in versions:
            rr=[r for r in good if r['version']==v and r['scenario'] in names and r['seed'] in base]
            if rr:out['groups'][label][v]=aggregate(rr,base)
    # All variants on EXACTLY the same subset used by the full-rollout study.
    keys={r['seed'] for r in good if r['version']=='global'}
    out['global_subset']={v:aggregate([r for r in good if r['version']==v and r['seed'] in keys],base) for v in versions if any(r['version']==v and r['seed'] in keys for r in good)}
    (root/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    fields=sorted({k for r in rows for k in r})
    with (root/'paired_records.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    text=['# 冻结配置留出对照（自建模拟，不是官方成绩）','',
      '每行在完全相同的案例上独立从初始状态运行。时间先算每例T/N再取均值；累计实际搜索确认无遗漏的时间包含在T中。',
      '主测试180例，分布外60例；全局续跑另在主测试预定的60例子集比较。概率模型失配和逐例退步均未排除。','',
      '|场景|例数|V4秒/源|V5秒/源|平均下降|','|---|---:|---:|---:|---:|']
    for g in ['mixed25','mixed50','mixed75','boundary','plus','minus','main','cluster','all_radius_1000','smooth_error','ood','all']:
        a=out['groups'][g].get('v5')
        if a:text.append(f"|{g}|{a['cases']}|{a['baseline_mean_seconds_per_source']:.4f}|{a['mean_seconds_per_source']:.4f}|{a['reduction_percent']:.4f}%|")
    text+=['','## 主测试消融','', '|版本|例数|秒/源|平均下降|更快/更慢/持平|','|---|---:|---:|---:|---|']
    for v in versions[:-1]:
        a=out['groups']['main'][v];text.append(f"|{v}|{a['cases']}|{a['mean_seconds_per_source']:.4f}|{a['reduction_percent']:.4f}%|{a['faster']}/{a['slower']}/{a['tied']}|")
    text+=['','## 完整全局续跑的同子集比较','', '|版本|例数|秒/源|平均下降|平均本地计算秒|','|---|---:|---:|---:|---:|']
    for v,a in out['global_subset'].items():text.append(f"|{v}|{a['cases']}|{a['mean_seconds_per_source']:.4f}|{a['reduction_percent']:.4f}%|{a['mean_wall_seconds']:.3f}|")
    text+=['','完整统计、P95、最大退步、误差区间、无信号支持不足时的保底次数见 summary.json。并发运行中的墙钟时间不是受控硬件性能基准，未计网络延迟。','',f'记录错误数：{len(errors)}。','']
    (root/'RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(out['groups']['main'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
