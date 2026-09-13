"""Snapshot completed cases via guest-agent file reads; never touch active runs."""
from pathlib import Path
import json,hashlib,re
from utm_guest import VM,call
ROOT=Path(__file__).resolve().parent
GUEST=r'C:\Users\baiwc\Downloads\Q4V6Lite_20260913\additional_10'
raw=call('file','pull',VM,GUEST+r'\completed.json')
rows=json.loads(raw.decode('utf-8-sig'));rows=[rows] if isinstance(rows,dict) else rows
for row in rows:
    index=int(row['index']);assert 1<=index<=10
    name=row['official_log'];assert re.fullmatch(r'practice-p4-\d+-[A-Z0-9-]+\.jlog',name)
    paths=[f'runs/{index:02d}/'+n for n in ['result.json','requests.jsonl','ui_verification.json']]
    paths += ['official_logs/'+name,'official_logs/'+name.replace('.jlog','.result.json')]
    for rel in paths:
        target=ROOT/'additional_10'/rel
        if target.exists() and target.stat().st_size:continue
        data=call('file','pull',VM,GUEST+'\\'+rel.replace('/','\\'))
        assert data,rel
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    assert hashlib.sha256((ROOT/'additional_10'/'official_logs'/name).read_bytes()).hexdigest()==row['official_log_sha256']
    print('Backed up official additional case',index,row['case'],flush=True)
(ROOT/'additional_10'/'completed.json').write_bytes(raw)
