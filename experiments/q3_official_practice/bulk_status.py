"""Download a fresh, validated snapshot of the Windows bulk experiment."""
import json
from pathlib import Path
from collections import Counter
from utm_guest import call, VM

HERE=Path(__file__).resolve().parent
cfg=json.loads((HERE/'bulk_config.json').read_text())
guest=r'C:\Users\baiwc\Downloads\Q3Practice\batches'+'\\'+cfg['batch_id']
local=HERE/'results'/'bulk'/cfg['batch_id']
local.mkdir(parents=True,exist_ok=True)
raw=call('file','pull',VM,guest+'\\status.json')
status=json.loads(raw.decode('utf-8-sig'))
(local/'status.json').write_bytes(raw)
rows=[]
if status['completed']:
    raw=call('file','pull',VM,guest+'\\completed.jsonl')
    rows=[json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    assert len(rows)>=status['completed']
    assert len({r['index'] for r in rows})==len(rows)
    assert len({r['case'] for r in rows})==len(rows)
    (local/'completed.jsonl').write_bytes(raw)
print(json.dumps(dict(status=status['status'],phase=status['phase'],completed=len(rows),
    target=status['target'],updated_at=status['updated_at'],current=status['current'],
    per_method=dict(Counter(r['method'] for r in rows)),
    cleared=sum(r['cleared'] for r in rows),failed_clear_attempts=sum(r['failed_clear_attempts'] for r in rows),
    latest=rows[-1] if rows else None,message=status['message']),ensure_ascii=False,indent=2))
