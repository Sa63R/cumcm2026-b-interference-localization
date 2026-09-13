"""Run inside Windows to archive only Q3 practice evidence and our outputs."""
from pathlib import Path
import hashlib
import json
import zipfile
root=Path(r'C:\Users\baiwc\Downloads\Q3Practice')
original=Path(r'C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator')
files=list((root/'practice_results').rglob('*')) if (root/'practice_results').exists() else []
files=[p for p in files if p.is_file()]
files+=list(original.rglob('practice-p3-*.jlog'))
files+=list(Path(r'C:\Users\baiwc\Desktop').glob('practice-p3-*.jlog'))
archive=root/'q3_evidence.zip'
entries=[]
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(files)):
        name=str(p.relative_to(root)) if p.is_relative_to(root) else 'official_logs/'+p.name
        z.write(p,name)
        entries.append(dict(path=str(p),archive_path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
results=[]
for p in sorted((root/'practice_results').glob('*/result.json')):
    r=json.loads(p.read_text(encoding='utf-8'))
    results.append(dict(path=str(p.relative_to(root)),method=r['method'],case_label=r['case_label'],status=r['status'],cleared=r['client_state']['cleared_count'],virtual_s=r['client_state']['virtual_time_s'],wall_s=r['wall_seconds']))
index=dict(results=results,files=entries,archive_bytes=archive.stat().st_size)
(root/'q3_evidence_index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
print(json.dumps(index,indent=2),flush=True)
