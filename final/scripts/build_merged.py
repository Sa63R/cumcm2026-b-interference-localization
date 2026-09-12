"""Compile the manually editable Q1/Q2/Q3 project; never overwrite section text."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pypdf import PdfReader

FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL / 'overleaf-merged'
QA = FINAL / 'qa/merged-latex'
QA.mkdir(parents=True, exist_ok=True)
for fig in (FINAL / 'overleaf-import/q23_figures').glob('*.pdf'):
    shutil.copy2(fig, ROOT / 'figures' / fig.name)

cmd = ['xelatex', '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
       f'-output-directory={QA}', 'main.tex']
for i in range(2):
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    (QA / f'run-{i+1}.txt').write_text(run.stdout + run.stderr, encoding='utf-8')
    if run.returncode:
        print(run.stdout[-9000:])
        raise SystemExit(run.returncode)
shutil.copy2(QA / 'main.pdf', ROOT / 'main.pdf')

texts = {p.relative_to(ROOT).as_posix(): p.read_text(encoding='utf-8')
         for p in [ROOT/'main.tex', ROOT/'preamble.tex', *sorted((ROOT/'sections').glob('*.tex'))]
         if p.name != 'symbols_q23.tex'}
joined = '\n'.join(texts.values())
labels = re.findall(r'\\label\{([^}]+)\}', joined)
refs = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', joined)
citations = re.findall(r'\\cite\{([^}]+)\}', joined)
bibitems = re.findall(r'\\bibitem\{([^}]+)\}', joined)
log = (QA / 'run-2.txt').read_text(encoding='utf-8')
serious = [line for line in log.splitlines() if any(x in line for x in
           ['Overfull', 'Missing character:', 'Undefined control sequence', 'undefined references', 'LaTeX Error'])]
pdf = PdfReader(ROOT/'main.pdf')
report = {
    'pages': len(pdf.pages),
    'numbered_equations': len(re.findall(r'\\begin\{equation\}', joined)),
    'algorithm_count': joined.count(r'\refstepcounter{algorithm}'),
    'duplicate_labels': sorted({v for v in labels if labels.count(v) > 1}),
    'unresolved_labels': sorted(set(refs) - set(labels)),
    'unresolved_citations': sorted(set(citations) - set(bibitems)),
    'serious_compile_messages': serious,
    'pdf_sha256': hashlib.sha256((ROOT/'main.pdf').read_bytes()).hexdigest(),
    'scope': 'Formatting and integration of existing Q1/Q2/Q3. No experiment or simulator execution.',
}
(ROOT/'merge_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
(QA/'page_texts.json').write_text(json.dumps([p.extract_text() for p in pdf.pages], ensure_ascii=False, indent=2), encoding='utf-8')

# Only the active project is packaged: reviews, old drafts and PDFs stay outside the upload ZIP.
active = [ROOT/'main.tex', ROOT/'preamble.tex', ROOT/'README.md']
active += [ROOT/'sections'/n for n in ['symbols.tex','q1.tex','q2.tex','q3.tex','references.tex','appendix.tex']]
active += sorted((ROOT/'figures').glob('*'))
with zipfile.ZipFile(FINAL/'overleaf-merged-upload.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in active:
        z.write(p, p.relative_to(ROOT).as_posix())
print(json.dumps(report, ensure_ascii=False))
if any(report[k] for k in ['duplicate_labels','unresolved_labels','unresolved_citations','serious_compile_messages']):
    raise SystemExit(1)
