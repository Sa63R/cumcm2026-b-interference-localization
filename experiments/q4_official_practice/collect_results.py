"""Collect completed practice outputs only; never collect account/config data."""
from pathlib import Path
import json
import zipfile

root = Path(__file__).resolve().parents[3]
files = sorted(f for f in (root / 'practice_results').rglob('*')
               if f.is_file() and f.suffix in ('.json', '.jsonl'))
files += sorted(f for f in (root / 'Jammers-simulator').rglob('practice-p4-*.jlog') if f.is_file())
archive = root / 'q4_completed_evidence.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, str(f.relative_to(root)))
results = []
for f in (root / 'practice_results').glob('*/result.json'):
    r = json.loads(f.read_text(encoding='utf-8'))
    state = r.get('client_state', {})
    results.append(dict(path=str(f.relative_to(root)), method=r.get('method'),
                        case_label=r.get('case_label'), status=r.get('status'),
                        wall_seconds=r.get('wall_seconds'), cleared=state.get('cleared_count'),
                        virtual_seconds=state.get('virtual_time_s')))
index = dict(results=results, archive_bytes=archive.stat().st_size,
             files=[str(f.relative_to(root)) for f in files])
(root / 'q4_evidence_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(index, ensure_ascii=False, indent=2))
