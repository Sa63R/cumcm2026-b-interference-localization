"""Independently audit official practice journals and publish aggregate metrics.
Run on the copied batch folder, never calls the simulator.
"""
from pathlib import Path
from datetime import datetime
import argparse,collections,hashlib,json,math,statistics


def audit(run, row, batch):
    result = json.loads((run /  'result.json').read_text())
    raw = (run /  'requests.jsonl').read_bytes()
    assert raw.endswith(b'\n'), 'Truncated final journal line'
    rows = [json.loads(line) for line in raw.splitlines()]
    counts = collections.Counter(row['event'] for row in rows)
    assert rows[0]['event'] == 'session_created'
    assert rows[-1]['event'] == 'client_closed'
    assert rows[-1]['pending'] is None
    assert len({r['run_id'] for r in rows}) == 1
    assert not any(r['event'] in {'attempt_failed', 'outcome_unknown'} for r in rows)
    requests = [r for r in rows if r['event'] == 'request']
    responses = [r for r in rows if r['event'] == 'response']
    states = [r for r in rows if r['event'] == 'state']
    assert len(requests) == len(responses) == len(states) == result['client_state']['accepted_actions']
    assert len(rows) == 3 * len(requests) + 2
    assert len({r['payload']['request_id'] for r in requests}) == len(requests)
    position = (0.0, 0.0)
    channel = 1
    virtual = 0.0
    costs = dict(movement_s=0.0, switching_s=0.0, detection_s=0.0, optical_s=0.0, removal_s=0.0)
    cleared = set()
    paths = collections.Counter()
    measurement_results = collections.Counter()
    clear_results = collections.Counter()
    timeline = []
    for index, (req, resp, state) in enumerate(zip(requests, responses, states), 1):
        assert rows[1 + (index-1)*3:1 + index*3] == [req, resp, state]
        request_id = req['payload']['request_id']
        assert resp['request_id'] == state['request_id'] == request_id
        assert request_id.endswith('-' + str(index))
        assert req['attempt'] == resp['attempt'] == 1
        data = resp['response']
        assert resp['http_status'] == 200 and data['accepted'] is True
        path = req['path']
        paths[path] += 1
        if index == 1:
            assert path == '/enter'
        elif index == len(requests):
            assert path == '/exit' and data['exit_reason'] == 'user_exit'
        else:
            assert path in {'/measure', '/clear'}
        target = req['payload'].get('position')
        target_channel = req['payload'].get('channel')
        if target:
            new_position = (target['x'], target['y'])
            move = math.dist(position, new_position) / 5.0
            costs['movement_s'] += move
            virtual += move
            position = new_position
            if path == '/measure':
                switching = float(target_channel != channel)
                costs['switching_s'] += switching
                costs['detection_s'] += 5.0
                virtual += switching + 5.0
                channel = target_channel
                measurement_results[data['measure_result']] += 1
            else:
                success = data['clear_result'] == 'success'
                costs['optical_s'] += 3.0
                costs['removal_s'] += 2.0 * success
                virtual += 3.0 + 2.0 * success
                clear_results[data['clear_result']] += 1
                if success:
                    assert target_channel not in cleared
                    cleared.add(target_channel)
        assert math.isclose(virtual, data['virtual_time_s'], abs_tol=1e-5)
        assert math.isclose(state['state']['virtual_time_s'], data['virtual_time_s'], abs_tol=1e-8)
        assert state['state']['accepted_actions'] == index
        assert state['state']['cleared_count'] == len(cleared)
        assert state['state']['current_channel'] == channel
        assert (state['state']['position']['x'], state['state']['position']['y']) == position
        kind = data.get('measure_result', data.get('clear_result', data.get('exit_reason', 'entered')))
        if 'svd_deg' in data:
            kind += f" {data['svd_deg']} deg"
        timeline.append(dict(step=index, request_id=request_id, recorded_at=req['recorded_at'],
                             official_timestamp_ms=data['real_timestamp_ms'], path=path,
                             position=target, channel=target_channel, result=kind,
                             virtual_time_s=data['virtual_time_s']))
    assert states[-1]['state'] == rows[-1]['state'] == result['client_state']
    assert result['status'] == 'policy_completed_and_exited'
    assert result['pending_request'] is None
    assert result['client_state']['session'] == 'exited'

    q=row['question']
    versions=json.loads(Path(__file__).with_name('expected_versions.json').read_text())
    assert result['source_manifest']==versions[str(q)], 'Source manifest differs from formal-test version'
    absent=set(range(1,21))-cleared
    if q==3:
        assert result['method']=='v3_origin20'
        assert set(result['policy']['cleared'])==cleared
        assert set(result['policy']['absent'])==absent
        assert result['policy']['complete_channel_certificate'] is True
    else:
        assert result['method']=='v6_lite'
        assert set(result['successful_clear_channels'])==cleared
        stop=result['policy']['stop_certificate']
        if len(cleared)==16:
            assert stop==result['completion_audit']['kind']=='source_upper_bound'
        else:
            assert stop==result['completion_audit']['kind']=='coverage_complete'
            assert result['completion_audit']['certificate']['ok'] is True
            assert result['completion_audit']['all_undiscovered_channels_measured_at_21_sites'] is True
            assert sorted(result['policy']['visited_stations'])==list(range(21))
            n1,n2,r1,r2=result['policy']['configuration']['base']['ring_sites']
            sites=[(0.,0.)]+[(r*math.cos(2*math.pi*k/n),r*math.sin(2*math.pi*k/n)) for n,r in [(n1,r1),(n2,r2)] for k in range(n)]
            measurements={c:[] for c in range(1,21)}
            for req in requests:
                if req['path']=='/measure':
                    p=req['payload']['position'];measurements[req['payload']['channel']].append((p['x'],p['y']))
            for c in absent:
                for site in sites:
                    assert any(math.dist(site,p)<=1e-6 for p in measurements[c]),(c,site)
    for key,value in costs.items():
        assert math.isclose(value,result['client_state']['time_breakdown'][key],abs_tol=1e-6)
    assert math.isclose(virtual/len(cleared),result['seconds_per_accepted_clear'],abs_tol=1e-6)
    log=batch/'official_logs'/row['official_log']
    official=json.loads(log.with_suffix('.result.json').read_text('utf-8-sig'))
    end_ms=round(datetime.fromisoformat(official['ended_at_utc'].replace('Z','+00:00')).timestamp()*1000)
    delta=result['exit_response']['real_timestamp_ms']-end_ms
    assert abs(delta)<=10
    assert official['problem_no']==q and official['case_code']==row['case']
    assert official['jammer_count']==row['total']==row['cleared']==len(cleared)
    assert hashlib.sha256(log.read_bytes()).hexdigest()==official['package_sha256']==row['official_log_sha256']
    assert row['actions']==len(requests)
    assert math.isclose(row['virtual_s'],result['client_state']['virtual_time_s'],abs_tol=1e-8)
    assert math.isclose(row['wall_s'],result['wall_seconds'],abs_tol=1e-8)
    assert sum(s['failed_clear_count'] for s in result['client_state']['sources'].values())==row['failed_clear_attempts']
    report=dict(status='passed',index=row['index'],question=q,case=row['case'],actual_sources=len(cleared),actions=len(requests),event_counts=dict(counts),action_counts=dict(paths),measurements=dict(measurement_results),clear_results=dict(clear_results),costs=costs,official_end_delta_ms=delta,coverage_certificate_verified=True,jlog_sha256_verified=True,requests_sha256=hashlib.sha256(raw).hexdigest(),virtual_s=row['virtual_s'],wall_s=row['wall_s'])
    (run/'independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('batch',type=Path);p.add_argument('--expected',type=int);args=p.parse_args()
    batch=args.batch
    rows=json.loads((batch/'completed.json').read_text('utf-8-sig'))
    if isinstance(rows,dict):rows=[rows]
    if args.expected is not None:assert len(rows)==args.expected
    assert len({r['index'] for r in rows})==len(rows)
    assert len({r['case'] for r in rows})==len(rows)
    reports=[];failures=[]
    for row in rows:
        run=batch/'runs'/f"{row['index']:03d}_q{row['question']}_{row['ordinal']:03d}"
        try:reports.append(audit(run,row,batch))
        except Exception as ex:failures.append({'index':row['index'],'error':repr(ex)});raise
    out={'audited':len(reports),'passed':len(reports),'failures':failures,'questions':{}}
    for q in (3,4):
        a=[r for r in rows if r['question']==q];rp=[r for r in reports if r['question']==q]
        if not a:continue
        scores=[r['seconds_per_source'] for r in a]
        out['questions'][str(q)]={'method':a[0]['method'],'cases':len(a),'total_sources':sum(r['total'] for r in a),'cleared':sum(r['cleared'] for r in a),'total_virtual_s':sum(r['virtual_s'] for r in a),'pooled_s_per_source':sum(r['virtual_s'] for r in a)/sum(r['cleared'] for r in a),'case_mean_s_per_source':statistics.mean(scores),'case_median_s_per_source':statistics.median(scores),'case_stdev_s_per_source':statistics.stdev(scores) if len(scores)>1 else None,'case_min_s_per_source':min(scores),'case_max_s_per_source':max(scores),'mean_wall_s':statistics.mean(r['wall_s'] for r in a),'actions':sum(r['actions'] for r in a),'failed_clear_attempts':sum(r['failed_clear_attempts'] for r in a),'costs':{k:sum(r['costs'][k] for r in rp) for k in rp[0]['costs']}}
    (batch/'independent_audit_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
