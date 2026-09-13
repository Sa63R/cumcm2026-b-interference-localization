"""Cross-check every archived official response against committed bulk rows."""
from collections import Counter
from pathlib import Path
import json,hashlib,math

here=Path(__file__).resolve().parent
cfg=json.loads((here/'bulk_config.json').read_text())
folder=here/'results/bulk'/cfg['batch_id']
collection=json.loads((folder/'collection.json').read_text())
evidence=Path(collection['extracted_to'])
rows=[json.loads(s) for s in (evidence/'completed.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
max_error=0.;accepted=0
for row in rows:
    prefix=f"{row['index']:04d}_{row['method']}"
    found=list((evidence/'runs'/prefix).glob('*/result.json'))
    assert len(found)==1
    result=json.loads(found[0].read_text(encoding='utf-8'))
    assert result['status']=='policy_completed_and_exited'
    assert result['method']==row['method']
    assert result['pending_request'] is None
    assert result['client_state']['session']=='exited'
    assert result['policy']['complete_channel_certificate']
    assert set(result['policy']['cleared']).isdisjoint(result['policy']['absent'])
    assert set(result['policy']['cleared'])|set(result['policy']['absent'])==set(range(1,21))
    assert len(result['policy']['cleared'])==row['total']==row['cleared']
    assert result['exit_response']['accepted'] and result['exit_response']['exit_reason']=='user_exit'
    assert result['exit_response']['virtual_time_s']==row['virtual_s']
    assert math.isclose(result['wall_seconds'],row['wall_s'],abs_tol=1e-10)
    events=[json.loads(s) for s in (found[0].parent/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    req={e['payload']['request_id']:e for e in events if e['event']=='request'}
    res={e['request_id']:e for e in events if e['event']=='response'}
    assert set(req)==set(res)
    assert all(e['http_status']==200 and e['response']['accepted'] for e in res.values())
    clear=Counter(e['response'].get('clear_result') for k,e in res.items() if req[k]['path']=='/clear')
    assert clear['success']==row['cleared']
    assert sum(clear.values())-clear['success']==row['failed_clear_attempts']
    for event in events:
        if event['event']=='state':
            state=event['state'];error=abs(state['estimated_total_virtual_time_s']-state['virtual_time_s'])
            max_error=max(max_error,error);assert error<=1e-5
    official=evidence/'official_logs'/row['official_log']
    assert row['case'] in official.name
    assert hashlib.sha256(official.read_bytes()).hexdigest()==row['official_log_sha256']
    accepted+=len(res)
result=dict(status='verified',archive_cases=len(rows),total_cleared=sum(r['cleared'] for r in rows),
            accepted_requests=accepted,max_cost_error_s=max_error,
            per_method=dict(Counter(r['method'] for r in rows)),source_archive=collection['local_archive'])
(folder/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
