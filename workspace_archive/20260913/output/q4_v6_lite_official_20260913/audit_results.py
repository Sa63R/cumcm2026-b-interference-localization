"""Independently audit accepted official responses against saved case metadata."""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def audit(run, official, index, ui):
    result = read(run / 'result.json')
    meta = read(official)
    log = official.with_name(official.name.removesuffix('.result.json') + '.jlog')
    assert log.stat().st_size > 0
    assert hashlib.sha256(log.read_bytes()).hexdigest() == meta['package_sha256']
    events = [json.loads(line) for line in (run / 'requests.jsonl').read_text().splitlines()]
    requests = [x for x in events if x['event'] == 'request']
    responses = [x for x in events if x['event'] == 'response']
    assert len(requests) == len(responses) == result['client_state']['accepted_actions']
    assert len({x['payload']['request_id'] for x in requests}) == len(requests)
    assert requests[0]['path'] == '/enter' and requests[-1]['path'] == '/exit'
    assert collections.Counter(x['path'] for x in requests)['/enter'] == 1
    assert collections.Counter(x['path'] for x in requests)['/exit'] == 1
    counts = collections.Counter(); measured = collections.defaultdict(list); clear = set()
    position = (0., 0.); channel = 1; prior = 0.; last_clear = None
    costs = dict(movement_s=0., switching_s=0., detection_s=0., optical_s=0., removal_s=0.)
    max_step_error = 0.
    for req, resp in zip(requests, responses):
        assert req['attempt'] == resp['attempt'] == 1
        assert req['payload']['request_id'] == resp['request_id']
        assert resp['http_status'] == 200 and resp['response']['accepted'] is True
        body = resp['response']; path = req['path']; payload = req['payload']; increment = 0.
        counts[path] += 1
        if path in ('/measure','/clear'):
            dest = (payload['position']['x'], payload['position']['y'])
            movement = math.dist(position,dest)/5.; costs['movement_s'] += movement; increment += movement; position = dest
            if path == '/measure':
                switch = int(payload['channel'] != channel); costs['switching_s'] += switch
                channel = payload['channel']; costs['detection_s'] += 5.; increment += switch+5.
                measured[channel].append(dest)
                assert body['measure_result'] in ('direction','near','no_signal')
            else:
                success = body['clear_result'] == 'success'
                assert body['clear_result'] in ('success','no_target_in_range')
                costs['optical_s'] += 3.; costs['removal_s'] += 2.*success; increment += 3.+2.*success
                if success:
                    assert payload['channel'] not in clear
                    clear.add(payload['channel']); last_clear = body['virtual_time_s']
                else: counts['failed_clear'] += 1
        error = abs((body['virtual_time_s']-prior)-increment)
        max_step_error = max(max_step_error,error)
        assert error < 1e-5, (path,error)
        assert body['virtual_time_s'] >= prior
        prior = body['virtual_time_s']
    assert meta['problem_no'] == 4
    assert len(clear) == meta['jammer_count'] == result['client_state']['cleared_count'] == ui['total']
    assert meta['omnidirectional_jammer_count'] == ui['omni']
    assert meta['directional_jammer_count'] == ui['directional']
    assert sum([ui['omni'],ui['directional']]) == len(clear)
    assert sorted(clear) == result['successful_clear_channels']
    assert result['status'] == 'policy_completed_and_exited' and result['pending_request'] is None
    assert result['client_state']['session'] == 'exited'
    assert result['exit_response']['exit_reason'] == 'user_exit'
    sites = [(0.,0.)] + [(r*math.cos(2*math.pi*k/n),r*math.sin(2*math.pi*k/n)) for n,r in [(8,998),(12,1865)] for k in range(n)]
    if len(clear)<16:
        assert result['completion_audit']['kind'] == 'coverage_complete'
        for c in set(range(1,21))-clear:
            assert all(any(math.dist(s,p)<1e-7 for p in measured[c]) for s in sites)
    else:
        assert result['completion_audit']['kind'] == 'source_upper_bound'
        assert result['policy']['stop_certificate'] == 'source_upper_bound'
    assert prior == result['client_state']['virtual_time_s']
    for key,value in costs.items(): assert abs(value-result['client_state']['time_breakdown'][key])<1e-5
    return dict(index=index,case=meta['case_code'],case_label_as_recorded=result['case_label'],all_cleared=True,
        total=len(clear),omni=ui['omni'],directional=ui['directional'],virtual_s=prior,
        seconds_per_source=prior/len(clear),wall_s=result['wall_seconds'],cpu_s=result['cpu_seconds'],
        actions=len(requests),measurements=counts['/measure'],clear_attempts=counts['/clear'],
        failed_clear_attempts=counts['failed_clear'],last_clear_virtual_s=last_clear,
        coverage_tail_s=prior-last_clear,costs=costs,max_step_error_s=max_step_error,
        official_log=log.name,official_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        request_log_sha256=hashlib.sha256((run/'requests.jsonl').read_bytes()).hexdigest(),
        started_utc=result['started_utc'],source_commit=result['source_commit'])

def main():
    first = ROOT/'official_results'/'EHKR-PC7N-PNWY-K6A9'
    first_meta = next((ROOT/'official_logs').glob('*.result.json'))
    rows=[audit(first,first_meta,0,read(first/'ui_verification.json'))]
    batch = ROOT/'additional_10'
    if (batch/'completed.json').exists():
        entries=read(batch/'completed.json'); entries=[entries] if isinstance(entries,dict) else entries
        for entry in entries:
            run=batch/'runs'/f"{entry['index']:02d}"
            meta=batch/'official_logs'/entry['official_log'].replace('.jlog','.result.json')
            rows.append(audit(run,meta,entry['index'],read(run/'ui_verification.json')))
    assert len({x['case'] for x in rows})==len(rows)
    (ROOT/'audited_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print(json.dumps(dict(audited_cases=len(rows),cleared=sum(x['total'] for x in rows),
        case_mean_s_per_source=sum(x['seconds_per_source'] for x in rows)/len(rows),
        pooled_s_per_source=sum(x['virtual_s'] for x in rows)/sum(x['total'] for x in rows)),indent=2))
if __name__=='__main__': main()
