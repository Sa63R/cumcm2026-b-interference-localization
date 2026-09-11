"""Build a portable third-round handoff; include test imports transitively.

The package intentionally includes all Python tests (not only collected files)
so helpers such as tests.test_strategy remain available. Runtime credentials,
local exchange helpers, raw databases and SSH configuration are never inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def collect_files(root=ROOT,include_results=True):
    selected={root/'pyproject.toml'}
    for folder in ('src','tests'):
        selected.update(p for p in (root/folder).rglob('*.py') if '__pycache__' not in p.parts)
    selected.update(p for p in (root/'experiments').glob('*.py')
        if 'q3_fresh' in p.name or 'q3_round3' in p.name or p.name in
        ('__init__.py','q3_confirmation_bound.py','session_lower_bounds.py'))
    for folder in ('research/q3_fresh_round3',):
        selected.update(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix not in ('.zip','.tar','.pyc'))
    selected.add(root/'research/q3_fresh_round2/confirmation_certificate.json')
    selected.add(root/'research/q3_fresh_round2/FROZEN_CANDIDATE.json')
    selected.add(root/'research/q3_fresh_round2/VALIDATION_SELECTION.json')
    selected.add(root/'results/q3_fresh/validation/results.json')
    if include_results:
        selected.update(p for p in (root/'results/q3_fresh_round3').rglob('*') if p.is_file() and p.name!='reproduction-source.tar.gz')
        # The reused reference metadata points to these immutable original files.
        reference=root/'results/q3_fresh_round3/reference_development/results.json'
        if reference.is_file():
            m=json.loads(reference.read_text(encoding='utf-8'))['manifest']
            selected.update(root/e['source'] for e in m['source_evidence'])
    return sorted(p for p in selected if p.is_file())


def build(output,include_results=True):
    files=collect_files(include_results=include_results)
    if ROOT/'tests/test_strategy.py' not in files:raise RuntimeError('Missing transitive test helper')
    manifest=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  includes_raw_database=False,includes_connection_credentials=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in files:archive.write(p,p.relative_to(ROOT))
        archive.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(output) as archive:
        bad=archive.testzip()
        if bad:raise RuntimeError('Archive integrity failure: '+bad)
    return dict(files=len(files),bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--without-results',action='store_true');a=p.parse_args()
    print(json.dumps(build(a.out,not a.without_results)))


if __name__=='__main__':main()
