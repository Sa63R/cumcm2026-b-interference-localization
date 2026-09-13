from pathlib import Path
import hashlib,json,zipfile
root=Path(r'C:\Users\baiwc\Downloads\Q4Practice')
download=root.parent
archive=download/'q4_speedup_guest_evidence.zip'
files=[]
for parent in (root/'speedup_practice_results',root/'output/q4_speedup/schedule_dev/adapter_speedup'):
 if parent.exists(): files.extend(p for p in parent.rglob('*') if p.is_file())
console=download/'q4_speedup_adapter_console.txt'
if console.exists(): files.append(console)
for p in download.glob('q4_speedup_*_console.txt'):
 if p not in files:files.append(p)
code=list((root/'code/experiments/q4_speedup').glob('*.py'))+[root/'code/experiments/q4_official_practice/run_speedup.py']
audit={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,p.relative_to(download).as_posix())
 z.writestr('guest_source_audit.json',json.dumps(audit,indent=2))
