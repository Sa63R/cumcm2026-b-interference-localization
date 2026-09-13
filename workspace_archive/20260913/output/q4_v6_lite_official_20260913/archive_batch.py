"""Archive only a finished batch; keep logs exactly as written."""
from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).resolve().parent
batch=root/'additional_10'
s=json.loads((batch/'status.json').read_text(encoding='utf-8-sig'))
assert s['status']=='completed' and s['completed']==10
archive=root/'additional_10_evidence.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(batch.rglob('*')):
        if p.is_file() and not p.name.startswith('RESUME_READY_') and not p.name.endswith('.tmp'):z.write(p,p.relative_to(root))
report=dict(status='archived',bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),path=str(archive))
(root/'archive_status.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
