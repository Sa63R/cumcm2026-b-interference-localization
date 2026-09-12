"""Check the assembled paper and figure artifacts; never runs experiments."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
from pypdf import PdfReader
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
checks = []

def check(name, passed, detail=None):
    checks.append({'name': name, 'passed': bool(passed), 'detail': detail})

pdf = ROOT / '第二问与第三问论文稿.pdf'
reader = PdfReader(pdf)
texts = [page.extract_text() or '' for page in reader.pages]
check('nonempty_pages', all(len(t.strip()) > 150 for t in texts), len(texts))
for i, page in enumerate(reader.pages, 1):
    w, h = map(float, (page.mediabox.width, page.mediabox.height))
    check(f'page_{i}_a4', abs(w - 595.28) < 1 and abs(h - 841.89) < 1)
log = (ROOT / 'qa/latex_run_2.txt').read_text(encoding='utf-8')
for marker in ('Overfull', 'Missing character:', 'Undefined control sequence', 'LaTeX Error:', 'undefined references'):
    check('compile_no_' + marker, marker not in log)
md = (ROOT / '第二问与第三问论文稿.md').read_text(encoding='utf-8')
tex = (ROOT / '第二问与第三问论文稿.tex').read_text(encoding='utf-8')
check('two_complete_algorithm_blocks', tex.count(r'\begin{minipage}{\linewidth}') == 2)
check('no_draft_markers', not re.search(r'FIG:|TODO|TBD|EVIDENCE|待填|待补', md))
check('eight_figure_references', len(re.findall(r'!\[.*?\]\(figures/', md)) == 8)
check('four_table_captions', all(f'Table: 表{i} ' in md for i in range(1, 5)))
check('q2_equation_tags', all(f'\\tag{{2-{i}}}' in md for i in range(1, 9)))
check('q3_equation_tags', all(f'\\tag{{3-{i}}}' in md for i in range(1, 18)))
figures = sorted((ROOT / 'figures').glob('fig*.pdf'))
check('eight_vector_figures', len(figures) == 8)
for path in figures:
    check(path.stem + '_one_page', len(PdfReader(path).pages) == 1)
    for suffix in ('.svg', '.png', '.export.json'):
        check(path.stem + suffix + '_exists', path.with_suffix(suffix).is_file())
    with Image.open(path.with_suffix('.png')) as img:
        dpi = img.info.get('dpi', (0, 0))
        check(path.stem + '_400dpi', all(abs(d - 400) < 1 for d in dpi), dpi)

doc = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Document compilation, structure, PDF pages and figure exports. Visual review is recorded separately.',
    'pdf_pages': len(texts),
    'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
    'total_checks': len(checks),
    'failed_checks': sum(not x['passed'] for x in checks),
    'checks': checks,
}
(ROOT / 'sources/validation_document.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
(ROOT / 'qa/pdf_text.json').write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: doc[k] for k in ('pdf_pages', 'total_checks', 'failed_checks', 'pdf_sha256')}, ensure_ascii=False))
for item in checks:
    if not item['passed']:
        print(item)
raise SystemExit(int(doc['failed_checks'] > 0))
