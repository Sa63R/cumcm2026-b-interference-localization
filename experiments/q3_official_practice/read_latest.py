"""Read guest log and preserve each completed practice result on the host."""
import json
from pathlib import Path
from utm_guest import call, VM
raw=call('file','pull',VM,r'C:\Users\baiwc\Downloads\Q3Practice\latest_run.log')
text=raw.decode('utf-8',errors='replace')
rows=[]
for line in text.splitlines():
    if line.startswith('{"method":'):
        rows.append(json.loads(line))
if not rows:
    print(text[-4000:]); raise SystemExit(0)
r=rows[-1]
folder=Path(__file__).resolve().parent/'results'/'live_cases';folder.mkdir(parents=True,exist_ok=True)
name=r['case_label']+'_'+r['method']
(folder/(name+'.json')).write_text(json.dumps(r,indent=2,ensure_ascii=False))
(folder/(name+'.log')).write_bytes(raw)
s=r['client_state'];p=r.get('policy',{})
print(json.dumps(dict(method=r['method'],case=r['case_label'],status=r['status'],cleared=s['cleared_count'],virtual_s=s['virtual_time_s'],seconds_per_source=r['seconds_per_accepted_clear'],wall_s=r['wall_seconds'],nonrequest_s=p.get('nonrequest_wall_s'),failed_clears=sum(v['failed_clear_count'] for v in s['sources'].values()),exception=r.get('exception')),ensure_ascii=False))
