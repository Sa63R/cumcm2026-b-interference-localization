import json,time
from pathlib import Path
from utm_guest import call,VM
root=Path(__file__).resolve().parent
prior=None
for _ in range(120):
    raw=call('file','pull',VM,r'C:\Users\baiwc\Downloads\Q4V6Lite_20260913\additional_10\status.json')
    try:s=json.loads(raw.decode('utf-8-sig'))
    except ValueError:time.sleep(5);continue
    (root/'bulk_status.json').write_bytes(raw)
    marker=(s['status'],s['completed'],s['phase'],s['index'])
    if marker!=prior:
        print(json.dumps({**{k:s[k] for k in ['status','index','completed','phase','message','utc']},'last_result':s['last_results'][-1] if s['last_results'] else None},ensure_ascii=False),flush=True);prior=marker
    if s['status'] in ('error','completed','stopped'):break
    time.sleep(8)
