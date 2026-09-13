"""Windows-side snapshot archive of committed bulk cases, plus setup evidence."""
from pathlib import Path
import hashlib,json,zipfile

root=Path(r'C:\Users\baiwc\Downloads\Q3Practice')
cfg=json.loads((root/'bulk_config.json').read_text(encoding='utf-8-sig'))
batch=root/'batches'/cfg['batch_id']
ledger=(batch/'completed.jsonl').read_bytes()
rows=[json.loads(s) for s in ledger.decode('utf-8-sig').splitlines() if s.strip()]
assert len({r['index'] for r in rows})==len(rows)
files=set()
for row in rows:
    result=Path(row['result_path'])
    assert result.is_relative_to(batch/'runs')
    files.update(p for p in result.parent.parent.rglob('*') if p.is_file())
    log=batch/'official_logs'/row['official_log']
    assert hashlib.sha256(log.read_bytes()).hexdigest()==row['official_log_sha256']
    files.add(log)
    prefix=f"{row['index']:04d}_{row['method']}"
    for phase in ['list','ready','done','done_page','return_list']:
        for suffix in ['.png','.png.json']:
            p=batch/'screenshots'/(prefix+'_'+phase+suffix)
            if p.exists():files.add(p)
for evidence_dir in ['setup_attempts','automation_events']:
    if (batch/evidence_dir).exists():
        files.update(p for p in (batch/evidence_dir).rglob('*') if p.is_file())
archive=batch/f'evidence_{len(rows):04d}.zip'
index=[]
with zipfile.ZipFile(archive.with_suffix('.tmp.zip'),'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files):
        name=p.relative_to(batch).as_posix();data=p.read_bytes()
        z.writestr(name,data)
        index.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    z.writestr('completed.jsonl',ledger)
    z.writestr('bulk_config.json',json.dumps(cfg,indent=2))
    z.writestr('status_at_collection.json',(batch/'status.json').read_bytes())
    z.writestr('file_index.json',json.dumps(index,indent=2))
archive.with_suffix('.tmp.zip').replace(archive)
info=dict(batch_id=cfg['batch_id'],completed=len(rows),archive=str(archive),bytes=archive.stat().st_size,
          sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(index))
(batch/'collection.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
print(json.dumps(info),flush=True)
