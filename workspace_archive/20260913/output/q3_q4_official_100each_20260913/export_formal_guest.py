from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,zipfile
root=Path(r'C:\Users\baiwc\Downloads\Q4V6Lite_20260913')
logs=Path(r'C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\JammersSimulatorData\behavior-logs')
items=[(3,1,'q3_run01','Z6ST-HDB2-SF5J-PGN7'),(3,2,'q3_run2','FN76-V6XC-EA3X-PTN8'),(3,3,'q3_run3','C2TW-9ZRR-SGVQ-CQDA'),(4,1,'run-q4-1','WUE8-Z5DG-6DXB-387N'),(4,2,'run-q4-2','53EM-93JQ-FT23-ZCBD'),(4,3,'run-q4-3','8BDU-U9XS-4FT9-652S')]
manifest={'exported_utc':datetime.now(timezone.utc).isoformat(),'files':{},'runs':[]}
archive=root/'formal_6_raw_20260913.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for q,n,name,case in items:
  files=list((root/'results'/name).rglob('*'))
  pairs=[(f,f'Q{q}/run{n:02d}/algorithm/'+f.relative_to(root/'results'/name).as_posix()) for f in files if f.is_file()]
  official=list(logs.glob(f'formal-*{case}*'))
  assert len([f for f in official if f.suffix=='.jlog'])==1
  pairs.extend((f,f'Q{q}/run{n:02d}/official/'+f.name) for f in official if f.is_file())
  for f,arc in pairs:
   raw=f.read_bytes();z.writestr(arc,raw);manifest['files'][arc]={'source':str(f),'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
  manifest['runs'].append({'question':q,'formal_index':n,'source_folder':name,'case':case})
 for f in (root/'results'/'q3_run02').rglob('*'):
  if f.is_file():
   arc='extra_failed_enter/q3_run02/'+f.name;raw=f.read_bytes();z.writestr(arc,raw);manifest['files'][arc]={'source':str(f),'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 z.writestr('export_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
(root/'formal_6_export_status.json').write_text(json.dumps({'status':'complete','archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(manifest['files']),'runs':len(items)},indent=2))
