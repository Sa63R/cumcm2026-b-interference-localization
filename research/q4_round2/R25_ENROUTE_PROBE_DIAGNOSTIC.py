"""Fixed old-621 prefix geometry only; never call a policy or decode truth.

Take the first ten completed coverage blocks per case before eligibility
filtering. A proposed probe is not appended to the real prefix. Selected
channels are suppressed only to enforce the diagnostic's once/channel cap.
"""
from collections import Counter
import hashlib,json,math,statistics,sys,time
from pathlib import Path

CORE=Path(__file__).resolve().parents[2]
SOURCE=CORE.parent/'q4-r12-joint-continuation'
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE)]
from localization import CandidateRegion
from experiments.diagnose_q4_joint_visibility import observation_record
from experiments.audit_q4_joint_continuation import wire_prefix

STAGES={'development':range(621001,621025),'development-stress':range(621031,621045)}
SPEC={'entrypoint':'strategies.q4_joint_continuation:run_q4_joint_continuation',
      'kwargs':{'config':'after_active_miss_optical','max_expansions':200}}
MAX_PREFIXES=10

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def candidates(start,end,center):
    dx,dy=end[0]-start[0],end[1]-start[1]
    t=max(0.,min(1.,((center[0]-start[0])*dx+(center[1]-start[1])*dy)/(dx*dx+dy*dy)))
    values=[.25,.5,.75]+([t] if .1<=t<=.9 else [])
    return [(u,(start[0]+u*dx,start[1]+u*dy)) for u in sorted(set(values))]

def scan(record,seed,stage):
    assert record['spec']==SPEC
    h,before=wire_prefix(record)
    stations=[tuple(p) for p in record['summary']['coverage_points']]
    assert len(stations)==22 and len(set(stations))==22
    regions={};known=set();cleared=set();near=set();selected_channels=set()
    blocks=0;index=0;prefixes=[];counts=Counter()
    def update(a):
        c=a['channel'];p=tuple(a['position'])
        if a['action']=='clear':
            if a['result']=='success':cleared.add(c);known.add(c)
        elif a['result']=='direction':
            known.add(c);regions.setdefault(c,CandidateRegion()).observe(p,a['bearing_deg'])
        elif a['result']=='near':known.add(c);near.add(c)
    while index<len(h) and blocks<MAX_PREFIXES:
        a=h[index]
        if a['action']!='measure' or a['phase']!='coverage':
            update(a);index+=1;continue
        point=tuple(a['position']);station=blocks
        assert point==stations[station], 'Coverage order differs'
        required=set(range(1,21))-known;measured=set();begin=index
        while index<len(h):
            a=h[index]
            if a['action']!='measure' or a['phase']!='coverage' or tuple(a['position'])!=point:break
            assert a['channel'] not in measured
            measured.add(a['channel']);update(a);index+=1
        assert required<=measured,'Partial block is not a completed coverage prefix'
        blocks+=1;counts['sampled_earliest_complete_coverage_prefixes']+=1
        start,tuned,_=before[index]
        assert start==point
        row=dict(seed=seed,stage=stage,prefix=index,coverage_block_start=begin,
                 completed_stations=blocks,current_position=list(start),known_channels=sorted(known),
                 cleared_channels=sorted(cleared),near_channels=sorted(near),current_channel=tuned,
                 next_cover=list(stations[blocks]) if blocks<22 else None,
                 candidate_channels=[],opportunities=[],selected=None)
        prefixes.append(row)
        if blocks>=22 or len(known)>=16:
            row['skip']='discovery_done_or_known16';counts[row['skip']]+=1;continue
        end=stations[blocks];segment=math.dist(start,end);row['segment_m']=segment
        row['next_actual_action_is_next_cover']=bool(index<len(h) and h[index]['action']=='measure'
            and h[index]['phase']=='coverage' and tuple(h[index]['position'])==end)
        if segment<100.:
            row['skip']='segment_under_100m';counts[row['skip']]+=1;continue
        options=[]
        for channel in sorted(known-cleared-near-selected_channels):
            region=regions.get(channel)
            if not region or not region.vertices:continue
            disk=region.enclosing_disk()
            if not 19.9<disk.radius<=120.:continue
            original=tuple(region.vertices);tests=[]
            for t,q in candidates(start,end,disk.center):
                counts['candidate_geometry_evaluations']+=1
                distance=math.dist(q,disk.center)
                event=dict(channel=channel,t=t,point=list(q),old_radius_m=disk.radius,
                    old_center=list(disk.center),nominal_source_distance_m=distance,
                    extra_single_radio_fee_s=5.+int(tuned!=channel),
                    segment_split_distance_excess_m=math.dist(start,q)+math.dist(q,end)-segment,
                    nominal_new_radius_m=None,nominal_bearing_deg=None,eligible=False)
                assert abs(event['segment_split_distance_excess_m'])<1e-8
                if distance<1e-9:
                    event['skip']='zero_direction_vector';tests.append(event);continue
                angle=round(math.degrees(math.atan2(disk.center[1]-q[1],disk.center[0]-q[0]))%360.,2)%360.
                copied=region.copy().observe(q,angle)
                assert tuple(region.vertices)==original,'Diagnostic mutated canonical region'
                event['nominal_bearing_deg']=angle
                if not copied.vertices:
                    event['skip']='nominal_observation_empty';tests.append(event);continue
                event['nominal_new_radius_m']=copied.enclosing_disk().radius
                event['eligible']=event['nominal_new_radius_m']<=19.9
                if event['eligible']:options.append(event)
                tests.append(event)
            row['candidate_channels'].append(dict(channel=channel,radius_m=disk.radius,
                positive_observations=len(region.observations),vertices=[list(p) for p in original],tests=tests))
        row['opportunities']=options
        if options:
            chosen=min(options,key=lambda e:(e['nominal_new_radius_m'],e['t'],e['channel']))
            row['selected']=chosen;selected_channels.add(chosen['channel'])
            counts['selected_once_per_channel']+=1
            if row['next_actual_action_is_next_cover']:counts['selected_on_actual_immediate_cover_segment']+=1
    return dict(seed=seed,stage=stage,counts=dict(counts),prefixes=prefixes)

def aggregate(cases):
    prefixes=[p for c in cases for p in c['prefixes']]
    chosen=[p for p in prefixes if p['selected']]
    strict=[p for p in chosen if p['next_actual_action_is_next_cover']]
    opportunities={(p['seed'],e['channel']) for p in prefixes for e in p['opportunities']}
    total=Counter()
    for c in cases:total.update(c['counts'])
    return dict(cases=len(cases),counts=dict(total),
        cases_with_any_nominal_opportunity=len({p['seed'] for p in prefixes if p['opportunities']}),
        distinct_source_opportunities=len(opportunities),selected_cases=len({p['seed'] for p in chosen}),
        selected_sources=len(chosen),immediate_segment_selected_cases=len({p['seed'] for p in strict}),
        immediate_segment_selected_sources=len(strict),
        selected_old_radii_m=[p['selected']['old_radius_m'] for p in chosen],
        selected_nominal_new_radii_m=[p['selected']['nominal_new_radius_m'] for p in chosen],
        selected_single_radio_fees_s=[p['selected']['extra_single_radio_fee_s'] for p in chosen],
        selected_prefixes=[[p['seed'],p['prefix'],p['selected']['channel']] for p in chosen],
        additional_switch_or_future_route_cost_included=False)

def main():
    output=Path(__file__).with_suffix('.json');report=Path(__file__).with_suffix('.md')
    assert not output.exists() and not report.exists(),'Refuse diagnostic overwrite'
    began=time.perf_counter();cases=[];inputs={}
    for stage,seeds in STAGES.items():
        for seed in seeds:
            path=SOURCE/f'results/q4_joint_continuation/{stage}/records/compact_joint_continuation-{seed}.json.gz'
            inputs[path.relative_to(CORE.parent).as_posix()]=sha(path)
            cases.append(scan(observation_record(path),seed,stage))
    grouped={stage:aggregate([c for c in cases if c['stage']==stage]) for stage in STAGES}
    combined=aggregate(cases)
    result=dict(schema='R25-old-prefix-enroute-probe-diagnostic-v1',source_case_count=38,
        sampling='First up to ten completed coverage blocks per old case, before eligibility filtering; no replacement',
        candidate_rule='t=.25,.5,.75 plus clamped center projection if .1<=t<=.9; positive nominal center bearing round .01deg; radius in (19.9,120]',
        choice_rule='minimum nominal new radius, then t, then channel; once/channel, no hypothetical feedback persists',
        no_hidden_evaluation_decoded=True,no_simulation_or_actual_measurement=True,
        input_sha256=inputs,diagnostic_sha256=sha(__file__),
        dependency_sha256={p:sha(SOURCE/p) for p in ['src/localization/__init__.py','src/geometry/__init__.py','experiments/diagnose_q4_joint_visibility.py','experiments/audit_q4_joint_continuation.py']},
        grouped=grouped,combined=combined,cases=cases,runtime_s=time.perf_counter()-began,
        limitations=['Nominal positive is hypothetical; directional source may be silent.','Only geometric region copies change; no real or optical certificate is created.','A split fixed segment has equal Euclidean length; floating microsecond rounding and subsequent switch/route changes are not total-cost guarantees.','Actual next action may service a source first; those prefixes are separately marked, not called direct old-cover insertion.','Repeated diagnostic prefixes come from unchanged old history, not rollout of the proposed actions.'])
    with output.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    lines=['# R25：沿覆盖线路补测的旧前缀诊断','','仅分析旧R12的621开发24随机+14困难。每局最早最多10个完整coverage块结束前缀先取样、再筛资格，不以机会好坏向后补。仅重建真实已知未清、非near频道的canonical正反馈区域；从不解码evaluation、真值或生成新场景。','','## 与旧实验的区别','','已查阅R2顺路服务、R5联合路线重排、R15未知频道补扫、R11接收核和R17条件观测材料，未找到同样的“保留原覆盖边、在边内给已知未ready源只加一次radio”的已执行Q4实验。这个结论限于检索到的研究材料。R2可以离开线路完整定位；R15在停点扫未知频道；R5改站顺序；本诊断均不采用。','','|集合|取样前缀|有机会局数|不同源机会|选中局数/源数|下一实际动作直接cover的选中局数/源数|','|---|---:|---:|---:|---|---|']
    for stage,label in [('development','随机24'),('development-stress','困难14')]:
        g=grouped[stage];lines.append(f"|{label}|{g['counts']['sampled_earliest_complete_coverage_prefixes']}|{g['cases_with_any_nominal_opportunity']}|{g['distinct_source_opportunities']}|{g['selected_cases']}/{g['selected_sources']}|{g['immediate_segment_selected_cases']}/{g['immediate_segment_selected_sources']}|")
    lines+=['','候选固定t=.25/.5/.75及MEC中心在线段上的截断投影（只留.1至.9）。只在C副本添加朝MEC中心、量化到.01°的名义方位；名义新半径≤19.9才叫机会，按新半径、t、频道稳定选择，每频道最多选一次。原C完全不变，之后各前缀仍读旧真实轨迹，不能当作新策略的连贯仿真。','','选中机会明细：','','|seed|prefix|频道|原r/m|名义新r/m|t|直接radio费/s|下一实际动作直接cover|','|---:|---:|---:|---:|---:|---:|---:|---|']
    for c in cases:
        for p in c['prefixes']:
            e=p['selected']
            if e:lines.append(f"|{c['seed']}|{p['prefix']}|{e['channel']}|{e['old_radius_m']:.6f}|{e['nominal_new_radius_m']:.6f}|{e['t']:.6f}|{e['extra_single_radio_fee_s']:.0f}|{p['next_actual_action_is_next_cover']}|")
    lines+=['','## 费用与判断边界','','对于固定起终点和线上q，|Aq|+|qB|=|AB|，本脚本逐候选检查拆段长度误差小于1e−8米。额外单次radio直接费用为5秒加至多1秒换频；这不是整局额外费用上界：后续换频、立即清除绕路、cover/source调度改变和虚拟移动微秒舍入仍会改变总用时。','','名义positive不保证真实接收，Q4的未知方向可造成no_signal；即使有信号，真实含误差bearing也不等于名义角。≤19.9的名义半径不是现有可执行清除证书，不能据此宣布ready或承诺省掉后来回访。需要新独立策略与真实反馈实验才能评估净收益，本诊断不产生T/N或T/LB性能反事实。','',''+(f"在固定38局窗口中，仅{combined['immediate_segment_selected_cases']}局、{combined['immediate_segment_selected_sources']}源同时满足名义收缩和下一实际动作直接cover；机会量有限，不能由此预期大幅整体提速。" if combined['immediate_segment_selected_cases']<10 else f"固定窗口有{combined['immediate_segment_selected_cases']}局、{combined['immediate_segment_selected_sources']}源满足严格可直接插入的名义机会，值得审慎讨论，但没有净时间收益证据。"),'','全部输入SHA、有限候选及未选原因见同名JSON；脚本固定规模，无参数搜索，未改变策略或打开新场景。']
    with report.open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(combined,ensure_ascii=False))

if __name__=='__main__':main()
