from pathlib import Path
import collections
from datetime import datetime
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent
result = json.loads((ROOT / 'result.json').read_text())
raw = (ROOT / 'requests.jsonl').read_bytes()
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
assert set(result['successful_clear_channels']) == cleared
absent = set(range(1, 21)) - cleared
assert result['policy']['stop_certificate'] == 'coverage_complete'
assert result['completion_audit']['kind'] == 'coverage_complete'
assert result['completion_audit']['certificate']['ok'] is True
assert result['completion_audit']['all_undiscovered_channels_measured_at_21_sites'] is True
assert sorted(result['policy']['visited_stations']) == list(range(21))
n1, n2, r1, r2 = result['policy']['configuration']['base']['ring_sites']
sites = [(0., 0.)] + [(radius * math.cos(2*math.pi*k/n), radius * math.sin(2*math.pi*k/n))
                       for n, radius in [(n1, r1), (n2, r2)] for k in range(n)]
measurements = {c: [] for c in range(1, 21)}
for request in requests:
    if request['path'] == '/measure':
        p = request['payload']['position']
        measurements[request['payload']['channel']].append((p['x'], p['y']))
for c in absent:
    for site in sites:
        assert any(math.dist(site, p) <= 1e-6 for p in measurements[c]), (c, site)
assert all(state['status'] != 'detected' for state in result['client_state']['sources'].values())
for key, value in costs.items():
    assert math.isclose(value, result['client_state']['time_breakdown'][key], abs_tol=1e-6)
assert math.isclose(virtual / len(cleared), result['seconds_per_accepted_clear'], abs_tol=1e-6)
base = 'formal-p4-1-WUE8-Z5DG-6DXB-387N'
official = json.loads((ROOT / (base + '.result.json')).read_text())
end_ms = round(datetime.fromisoformat(official['ended_at_utc'].replace('Z', '+00:00')).timestamp() * 1000)
end_delta_ms = result['exit_response']['real_timestamp_ms'] - end_ms
assert abs(end_delta_ms) <= 1, 'Official end timestamps differ by more than 1 ms'
assert official['problem_no'] == 4 and official['formal_index'] == 1
assert official['team_no'] == rows[0]['robot_id']
assert hashlib.sha256((ROOT / (base + '.jlog')).read_bytes()).hexdigest() == official['package_sha256']
audit = dict(status='passed', source_directory=r'C:\Users\baiwc\Downloads\Q4V6Lite_20260913\results\run-q4-1',
             method=result['method'], run_label=result['case_label'], official_mode='formal', formal_index=1,
             official_case=official['case_code'], ended_at_utc=official['ended_at_utc'],
             successful_clears=len(cleared), absent_channels=sorted(absent), complete_channel_certificate=True, coverage_stations=21,
             undiscovered_channels_21_site_coverage_verified=True,
             virtual_time_s=result['client_state']['virtual_time_s'], seconds_per_cleared_source=result['seconds_per_accepted_clear'],
             wall_seconds=result['wall_seconds'], event_counts=dict(counts), action_counts=dict(paths),
             measurement_results=dict(measurement_results), clear_results=dict(clear_results),
             request_response_state_pairs_complete=True, unknown_requests=0, retries=0,
             step_cost_accounting_verified=True, official_end_timestamp_matches=(end_delta_ms == 0), official_end_timestamp_difference_ms=end_delta_ms,
             official_end_timestamp_within_1ms=True, official_jlog_hash_matches=True,
             official_source_truth_count_available=False, upload_status_verified=False,
             costs=costs,
             file_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir()
                          if p.is_file() and p.name in {'result.json','requests.jsonl',base+'.jlog',base+'.result.json'}})
(ROOT / 'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
(ROOT / '逐步过程.json').write_text(json.dumps(timeline,ensure_ascii=False,indent=2)+'\n')
lines = ['# run-q4-1 记录核对', '', f"正式案例：{official['case_code']}（问题 4，第 1 次正式测试）", '',
         f"程序正常退出；成功清除 {len(cleared)} 个源，{len(absent)} 个频道判定不存在，覆盖全部 20 个频道。官方正式模式不显示源真值总数。", '',
         f"虚拟耗时 {result['client_state']['virtual_time_s']:.6f} 秒，平均 {result['seconds_per_accepted_clear']:.6f} 秒/成功清除；程序实际耗时 {result['wall_seconds']:.6f} 秒。", '',
         f"核对了 {len(requests)} 组请求、响应与状态快照，均有完整对应；无重试或结果未知请求。逐步计时、最终状态和清除计数一致。官方结束时间与退出响应相差 {end_delta_ms} 毫秒，官方加密日志 SHA-256 匹配。", '',
         '请求日志记录每步坐标、频道、测向角或无信号/近距离反馈、清除反馈、现实时间、累计虚拟时间及已知源状态。内部候选点评分和选点理由没有单独记录。', '',
         '本次只读取和核对已有记录，没有启动测试。未核实官方服务器的上传接收状态。', '',
         '| 步骤 | 动作 | x | y | 频道 | 返回结果 | 累计虚拟秒 |', '|---:|---|---:|---:|---:|---|---:|']
for step in timeline:
    p=step['position']
    lines.append(f"| {step['step']} | {step['path']} | {p['x'] if p else ''} | {p['y'] if p else ''} | {step['channel'] or ''} | {step['result']} | {step['virtual_time_s']:.6f} |")
(ROOT / '运行过程核对.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k!='file_sha256'},ensure_ascii=False,indent=2))
