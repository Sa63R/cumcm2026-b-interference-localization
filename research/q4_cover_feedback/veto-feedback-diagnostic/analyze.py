"""Actual feedback after the 13 first-audited completed R39 cover vetoes.

Only observation fields are JSON-decoded from raw records. No case generation,
policy execution, hidden evaluation, or alternative trajectory is consulted.
"""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.audit_q4_cover_feedback import Prefix


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(path):return sha(path.read_bytes())


def skip_value(text,at):
    if text[at] not in '[{"':
        while at<len(text) and text[at] not in ',}] \t\r\n':at+=1
        return at
    stack=[];quoted=text[at]=='"';escaped=False
    if not quoted:stack.append(text[at])
    at+=1
    while at<len(text):
        c=text[at];at+=1
        if quoted:
            if escaped:escaped=False
            elif c=='\\':escaped=True
            elif c=='"':
                quoted=False
                if not stack:return at
        elif c=='"':quoted=True
        elif c in '[{':stack.append(c)
        elif c in ']}':
            require(stack and (stack.pop(),c) in (('[',']'),('{','}')),'Invalid skipped JSON structure')
            if not stack:return at
    raise ValueError('Unterminated skipped JSON value')


def observed_record(data):
    text=gzip.decompress(data).decode('utf-8');decoder=json.JSONDecoder();at=len(text)-len(text.lstrip())
    require(text[at]=='{','Record must be object');at+=1;out={};keys=set()
    while True:
        while text[at].isspace() or text[at]==',':at+=1
        if text[at]=='}':break
        key,at=decoder.raw_decode(text,at)
        require(key not in keys,'Repeated top-level field');keys.add(key)
        while text[at].isspace():at+=1
        require(text[at]==':','Invalid record field');at+=1
        while text[at].isspace():at+=1
        if key in {'summary','history','spec'}:out[key],at=decoder.raw_decode(text,at)
        else:at=skip_value(text,at)
    require(set(out)=={'summary','history','spec'},'Incomplete observed record')
    return out


def source_state(prefix,n,c):
    known,cleared,near,regions=prefix.snapshot(n);r=regions.get(c)
    circle=r.enclosing_disk() if r is not None and r.vertices else None
    return dict(channel=c,known=c in known,cleared=c in cleared,near_point=near.get(c),
        ready=prefix.ready(n,c),positive_bearing_count=len(r.observations) if r else 0,
        radius_m=circle.radius if circle else None,target=prefix.target(n,c),
        vertices=[list(v) for v in r.vertices] if r else [])


def coverage_blocks(h):
    result=[];i=0
    while i<len(h):
        if h[i]['phase']!='coverage':i+=1;continue
        start=i;p=h[i]['position']
        while i<len(h) and h[i]['phase']=='coverage' and h[i]['position']==p:i+=1
        result.append(dict(start=start,end=i,point=p,measurements=i-start))
    return result


def one(record,event,split,seed):
    prefix=Prefix(record);h=prefix.h;start=event['after_actual_action_count'];end=event['end_actual_action_count']
    require(event['veto_selected'] and event['executed_cover'] and event['status']=='completed' and end>start,
            'Not a completed actual veto')
    q=event['next_cover'];channel=event['channel'];blocks=coverage_blocks(h)
    require(any(b['start']==start and b['end']==end and b['point']==q for b in blocks),
            'Veto does not exactly own its complete actual coverage block')
    require(all(a['action']=='measure' and a['position']==q and a['phase']=='coverage' for a in h[start:end]),
            'Veto block includes noncoverage actions')
    known,cleared,_,_=prefix.snapshot(start);after_known,after_cleared,_,_=prefix.snapshot(end)
    require(known==set(event['known_channels']) and cleared==set(event['cleared_channels']),'Event actual known prefix differs')
    require(prefix.ready(start,channel) and channel not in cleared,'Deferred source was not live ready')
    nonready=sorted(c for c in known-cleared if not prefix.ready(start,c));updates=[]
    for c in nonready:
        before=source_state(prefix,start,c);after=source_state(prefix,end,c)
        feedback=[dict(action_index=i,result=h[i]['result'],bearing_deg=h[i].get('bearing_deg'))
                  for i in range(start,end) if h[i]['channel']==c]
        updates.append(dict(channel=c,before=before,after=after,actual_feedback=feedback,
            geometry_changed=before['vertices']!=after['vertices'] or before['near_point']!=after['near_point'],
            became_ready=not before['ready'] and after['ready']))
    newly_known=sorted(after_known-known)
    new_sources=[dict(channel=c,after=source_state(prefix,end,c),actual_feedback=[dict(action_index=i,
        result=h[i]['result'],bearing_deg=h[i].get('bearing_deg')) for i in range(start,end) if h[i]['channel']==c])
        for c in newly_known]
    clear_index=next((i for i in range(end,len(h)) if h[i]['action']=='clear'
        and h[i]['channel']==channel and h[i]['result']=='success'),None)
    finish=(clear_index+1) if clear_index is not None else len(h)
    later_blocks=[b for b in blocks if b['start']>=end and b['end']<=finish]
    clear_time=h[clear_index]['virtual_time_s'] if clear_index is not None else None
    return dict(split=split,seed=seed,event_id=event['id'],channel=channel,next_cover=q,
        after_actual_action_count=start,end_actual_action_count=end,
        prefix_time_s=prefix.before[start][2],q_finished_time_s=prefix.before[end][2],
        actual_q_block_time_s=prefix.before[end][2]-prefix.before[start][2],
        known_count_before=len(known),known_count_after=len(after_known),
        known_nonready_before=nonready,known_nonready_updates=updates,
        known_nonready_geometry_changed=[x['channel'] for x in updates if x['geometry_changed']],
        known_nonready_became_ready=[x['channel'] for x in updates if x['became_ready']],
        newly_discovered_sources=new_sources,newly_discovered_count=len(new_sources),
        new_sources_already_ready_after_q=[x['channel'] for x in new_sources if x['after']['ready']],
        actual_q_feedback_counts=dict(Counter(a['result'] for a in h[start:end])),
        deferred_source_clear_action_index=clear_index,deferred_source_cleared=clear_index is not None,
        deferred_source_clear_time_s=clear_time,
        elapsed_prefix_to_clear_s=clear_time-prefix.before[start][2] if clear_time is not None else None,
        elapsed_q_end_to_clear_s=clear_time-prefix.before[end][2] if clear_time is not None else None,
        accepted_actions_after_q_through_clear=(clear_index+1-end) if clear_index is not None else None,
        further_completed_cover_count=len(later_blocks),further_completed_covers=later_blocks,
        completed_covers_from_veto_through_clear=1+len(later_blocks),
        observed_followup_time_s=prefix.before[finish][2]-prefix.before[end][2],
        source_uncleared_at_observed_end=channel not in prefix.snapshot(len(h))[1],
        scope='Elapsed waiting is actual chronology, not extra loss relative to a nonexistent counterfactual.')


def summarize(events):
    waits=[e['elapsed_q_end_to_clear_s'] for e in events if e['deferred_source_cleared']]
    return dict(cases=len({(e['split'],e['seed']) for e in events}),completed_veto_blocks=len(events),
        known_nonready_source_events=sum(len(e['known_nonready_before']) for e in events),
        known_nonready_geometry_changed=sum(len(e['known_nonready_geometry_changed']) for e in events),
        blocks_with_known_geometry_change=sum(bool(e['known_nonready_geometry_changed']) for e in events),
        known_nonready_became_ready=sum(len(e['known_nonready_became_ready']) for e in events),
        blocks_with_known_new_ready=sum(bool(e['known_nonready_became_ready']) for e in events),
        newly_discovered_sources=sum(e['newly_discovered_count'] for e in events),
        blocks_discovering_new_source=sum(e['newly_discovered_count']>0 for e in events),
        new_sources_ready_immediately=sum(len(e['new_sources_already_ready_after_q']) for e in events),
        deferred_sources_later_cleared=sum(e['deferred_source_cleared'] for e in events),
        further_cover_count_distribution=dict(sorted(Counter(e['further_completed_cover_count'] for e in events).items())),
        q_end_to_clear_s=dict(count=len(waits),min=min(waits) if waits else None,
            mean=statistics.mean(waits) if waits else None,max=max(waits) if waits else None),
        per_event_waits_not_additive_loss=True)


def main():
    require(not (HERE/'results.json').exists(),'Preserve diagnostic output')
    inputs={};events=[];raw_count=0
    for split,expected in [('development',7),('development-stress',6)]:
        directory=ROOT/'results/q4_cover_feedback'/split;audit_path=directory/'independent_audit.json'
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        require(audit['all_passed'] is True and audit['errors']==[],'Complete original first audit did not pass')
        inputs[audit_path.relative_to(ROOT).as_posix()]=file_sha(audit_path)
        selected=[a for a in audit['audits'] if a['cover_feedback']['prefix']['cover_feedback_vetoes']>0]
        require(len(selected)==expected,'Audited selected case count changed')
        for item in selected:
            require(item['passed'] and item['errors']==[],'Selected original case audit failed')
            path=directory/f"records/{item['strategy']}-{item['seed']}.json.gz";data=path.read_bytes()
            require(sha(data)==audit['input_sha256'][path.relative_to(directory).as_posix()],'Audited raw bytes changed')
            inputs[path.relative_to(ROOT).as_posix()]=sha(data);record=observed_record(data);raw_count+=1
            owned=[e for e in record['summary']['strategy_parameters']['cover_feedback_log']
                   if e['veto_selected'] and e['executed_cover'] and e['status']=='completed']
            require(len(owned)==item['cover_feedback']['prefix']['cover_feedback_vetoes'],'Actual/audited veto count differs')
            events.extend(one(record,e,split,item['seed']) for e in owned)
    require(raw_count==13 and len(events)==13,'Expected precisely thirteen audited actual cover blocks')
    sources=('experiments/audit_q4_cover_feedback.py','experiments/audit_q4_joint_continuation.py',
        'experiments/audit_q4_clear_before_probe.py','src/localization/__init__.py','src/geometry/__init__.py')
    result=dict(kind='r39_actual_veto_feedback_v1',script_sha256=file_sha(Path(__file__)),input_sha256=inputs,
        prefix_reconstruction_source_sha256={p:file_sha(ROOT/p) for p in sources},
        summary=summarize(events),by_split={split:summarize([e for e in events if e['split']==split])
            for split in ('development','development-stress')},events=events,
        observation_reader='Only summary/history/spec top-level fields decoded; row/evaluation subtrees lexically skipped.',
        limitations=['Selected intervention events are not independent performance samples or a calibration set.',
            'C/near means canonical actual geometry only, not auxiliary joint regions or hidden source positions.',
            'Later waiting time includes real coverage and other services; it is not an extra cost, saving or regret.',
            'New discoveries show a part of the real future omitted by the all-known proxy, not the sign of its causal impact.'])
    with (HERE/'results.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result['summary'],ensure_ascii=False))


if __name__=='__main__':main()
