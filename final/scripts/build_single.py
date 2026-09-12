"""Compile/check/package the editable single-file paper; never regenerate main.tex."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pypdf import PdfReader

FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL / 'overleaf-single'
QA = FINAL / 'qa/single-latex'
QA.mkdir(parents=True, exist_ok=True)
for i in range(2):
    run = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                          '-file-line-error', f'-output-directory={QA}', 'main.tex'],
                         cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    (QA / f'run-{i+1}.txt').write_text(run.stdout + run.stderr, encoding='utf-8')
    if run.returncode:
        print(run.stdout[-9000:])
        raise SystemExit(run.returncode)
shutil.copy2(QA/'main.pdf', ROOT/'main.pdf')
source = (ROOT/'main.tex').read_text(encoding='utf-8')
labels = re.findall(r'\\label\{([^}]+)\}', source)
refs = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', source)
cites = re.findall(r'\\cite\{([^}]+)\}', source)
bibs = re.findall(r'\\bibitem\{([^}]+)\}', source)
log = (QA/'run-2.txt').read_text(encoding='utf-8')
serious = [s for s in log.splitlines() if any(x in s for x in
           ['Overfull', 'Missing character:', 'Undefined control sequence', 'undefined references', 'LaTeX Error'])]
report = {'pages':len(PdfReader(ROOT/'main.pdf').pages),
          'numbered_equations':source.count(r'\begin{equation}'),
          'algorithm_count':source.count(r'\refstepcounter{algorithm}'),
          'duplicate_labels':sorted({v for v in labels if labels.count(v)>1}),
          'unresolved_labels':sorted(set(refs)-set(labels)),
          'unresolved_citations':sorted(set(cites)-set(bibs)),
          'serious_compile_messages':serious,
          'pdf_sha256':hashlib.sha256((ROOT/'main.pdf').read_bytes()).hexdigest(),
          'main_sha256':hashlib.sha256((ROOT/'main.tex').read_bytes()).hexdigest(),
          'scope':'Single-file formatting of existing Q1/Q2/Q3; no new experiments or source claims.'}
(ROOT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with zipfile.ZipFile(FINAL/'overleaf-single-upload.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in [ROOT/'main.tex', *sorted((ROOT/'figures').glob('*'))]:
        z.write(p,p.relative_to(ROOT).as_posix())
print(json.dumps(report,ensure_ascii=False))
if any(report[k] for k in ['duplicate_labels','unresolved_labels','unresolved_citations','serious_compile_messages']):
    raise SystemExit(1)
