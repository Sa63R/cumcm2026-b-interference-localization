"""Copy this task's deliverables into an isolated repository checkout; no deletes."""
from pathlib import Path
import argparse,shutil,subprocess

p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);a=p.parse_args()
src=Path(__file__).resolve().parents[1];checkout=Path(a.checkout).resolve()
top=Path(subprocess.check_output(['git','-C',str(checkout),'rev-parse','--show-toplevel'],text=True).strip()).resolve()
if top!=checkout:raise SystemExit('Must specify the checkout root.')
dest=(checkout/'final').resolve()
if dest.parent!=checkout:raise SystemExit('Destination escaped checkout.')
count=0
for f in src.rglob('*'):
    if not f.is_file():continue
    rel=f.relative_to(src)
    if 'qa' in rel.parts or '__pycache__' in rel.parts:continue
    if rel.parts[0]=='references' and (f.suffix in ['.pdf','.html','.png']):continue
    if f.suffix in ['.aux','.log','.out','.xdv']:continue
    q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,q);count+=1
print(f'Copied {count} deliverable files into {dest}; no files deleted.')
