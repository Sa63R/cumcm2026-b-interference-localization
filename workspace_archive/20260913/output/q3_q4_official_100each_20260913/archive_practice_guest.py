from pathlib import Path
import argparse,json,zipfile,hashlib
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=200);args=p.parse_args()
root=Path(r'C:\Users\baiwc\Downloads\Q4V6Lite_20260913');batch=root/'practice_100each_20260913'
rows=json.loads((batch/'completed.json').read_text('utf-8-sig'))
if isinstance(rows,dict):rows=[rows]
assert len(rows)>=args.limit,(len(rows),args.limit)
rows=rows[:args.limit]
archive=root/f'practice_checkpoint_{args.limit:03d}.zip'
files={}
for row in rows:
 prefix=f"{row['index']:03d}_q{row['question']}_{row['ordinal']:03d}"
 for f in (batch/'runs'/prefix).rglob('*'):
  if f.is_file():files[f.relative_to(batch).as_posix()]=f
 for ext in ('.jlog','.result.json','.psum'):
  f=(batch/'official_logs'/row['official_log']).with_suffix(ext)
  if f.is_file():files[f.relative_to(batch).as_posix()]=f
 for f in list((batch/'screenshots').glob(prefix+'*'))+list(batch.glob(prefix+'*')):
  if f.is_file():files[f.relative_to(batch).as_posix()]=f
for name in ('batch_config.json','status.json'):
 f=batch/name
 if f.is_file():files[name]=f
manifest={'exported_utc':datetime.now(timezone.utc).isoformat(),'completed':len(rows),'files':{}}
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
 for arc,f in files.items():
  raw=f.read_bytes();z.writestr(arc,raw);manifest['files'][arc]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 z.writestr('completed.json',json.dumps(rows,ensure_ascii=False,indent=2))
 z.writestr('export_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
(root/f'practice_checkpoint_{args.limit:03d}_status.json').write_text(json.dumps({'status':'complete','archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'runs':len(rows),'files':len(files)},indent=2))
