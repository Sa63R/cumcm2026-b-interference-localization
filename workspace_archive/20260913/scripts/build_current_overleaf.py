"""Build the current Markdown as a portable XeLaTeX/Overleaf project.

Keeps all earlier editions and the root Markdown unchanged.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import zipfile

import rebuild_model_pdf as markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf/当前版'
QA = ROOT / 'tmp/pdfs/current_document'
SOURCE = ROOT / '问题一_定位区域模型.md'
OLD = ROOT / 'output/pdf/问题一_定位区域模型.tex'
CASE = ROOT / 'output/pdf/当前算例配图/问题一_三测点反例.pdf'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = SOURCE.read_text()
    previous = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (ROOT / 'output/pdf').rglob('*')
        if p.suffix in ('.tex', '.pdf') and OUT not in p.parents
    }
    OUT.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    (OUT / 'source_original.md').write_text(source)

    # This previously identified typo disagrees with both the circle geometry
    # and the decimal value in the source. Keep its original in the snapshot.
    corrections = []
    old_radius = r'R^*=\frac{5\sqrt{13}}{6}'
    new_radius = r'R^*=\frac{65}{6}'
    corrected = source
    if old_radius in corrected:
        corrected = corrected.replace(old_radius, new_radius, 1)
        corrections.append({'from': old_radius, 'to': new_radius,
                            'reason': 'Circumradius of A=(-10,0), B=(10,0), U=(0,15) is 65/6 m.'})

    title = corrected.splitlines()[0].removeprefix('# ')
    body = markdown.convert(corrected)
    body = body.replace(r'\Needspace{21\baselineskip}', r'\Needspace{16\baselineskip}')
    for line in corrected.splitlines():
        if line.startswith('> '):
            body = body.replace(markdown.inline(line),
                r'\begin{quote}' + '\n' + markdown.inline(line[2:]) + '\n' + r'\end{quote}')
    body = body.replace(
        r'>{\centering\arraybackslash}p{0.25\linewidth}>{\centering\arraybackslash}p{0.24\linewidth}>{\centering\arraybackslash}X',
        r'>{\centering\arraybackslash}p{0.22\linewidth}>{\centering\arraybackslash}p{0.20\linewidth}>{\raggedright\arraybackslash}X')

    audit = re.sub(r'\\addcontentsline\{toc\}\{(?:sub)?section\}\{[^{}]*\}', '', body)
    han = lambda value: ''.join(re.findall('[\u4e00-\u9fff]', value))
    assert han(source.split('\n', 1)[1]) == han(audit), 'Chinese source text was changed or omitted.'

    anchor = markdown.inline('超出半径 $5$ m，式 (6) 不成立。')
    assert anchor in body
    case_figure = r'''
% Added case illustration; the six constraints use the exact stated angles.
\begin{center}
\begin{minipage}{\linewidth}
\centering
\QOneCaseFigure
\captionof{figure}{三次测向的角域交集与直径圆覆盖检验。左图按实际比例绘制三个误差角域；右图放大定位三角形，红色圆的半径为 $10$ m，上顶点超出 $5$ m。绿色虚线为第 5.4 节求得的最小包围圆。}
\label{fig:counterexample}
\end{minipage}
\end{center}
'''
    body = body.replace(anchor, anchor + '\n' + case_figure, 1)
    body = body.replace(r'\section*{六、问题一小结}',
                        r'\Needspace{14\baselineskip}' + '\n' + r'\section*{六、问题一小结}', 1)

    preamble = OLD.read_text().split(r'\begin{document}', 1)[0]
    preamble = re.sub(r'\\hypersetup\{pdftitle=\{[^{}]*\}\}',
                     lambda _: r'\hypersetup{pdftitle={' + title + '}}', preamble)
    preamble += '\n' + r'''\usepackage{graphicx}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,labelsep=quad,hypcap=false}
\renewcommand{\figurename}{图}
\input{figures/q1_case}
'''
    title_tex = markdown.inline(title).replace('的干扰源', r'的\par 干扰源', 1)
    tex = (preamble + '\n' + r'\begin{document}\thispagestyle{plain}' + '\n'
           + r'\begin{center}{\LARGE\bfseries ' + title_tex
           + r'\par}\end{center}' + '\n'
           + '% Source radius typo corrected: 5*sqrt(13)/6 -> 65/6.\n'
           + body + '\n' + r'\end{document}' + '\n')
    formulas = re.findall(r'\$\$\s*\n(.*?)\n\$\$', corrected, re.S)
    assert all(f in tex for f in formulas)
    assert re.findall(r'\\tag\{(\d+)\}', source) == re.findall(r'\\tag\{(\d+)\}', tex)
    (OUT / 'main.tex').write_text(tex)
    drawing = CASE.with_suffix('.tex').read_text()
    colors = drawing[drawing.index(r'\definecolor'):drawing.index(r'\begin{document}')]
    picture = drawing[drawing.index(r'\begin{tikzpicture}'):drawing.index(r'\end{tikzpicture}') + len(r'\end{tikzpicture}')]
    picture = picture.replace('(0,0) rectangle (23,10.6)', '(0,.64) rectangle (23,9.2)')
    picture = re.sub(r'\\node\[font=\\Large\\bfseries\].*?;\n', '', picture, count=1, flags=re.S)
    picture = re.sub(r'\\node\[black!65\].*?;\n', '', picture, count=1, flags=re.S)
    picture = re.sub(r'\\node\[font=\\small\\bfseries\] at \(11.5,.22\).*?;\n', '', picture, count=1, flags=re.S)
    figure_source = (r'\usepackage{tikz}' + '\n' + r'\usetikzlibrary{arrows.meta,calc,patterns}' + '\n'
        + colors + '\n' + r'\newcommand{\QOneCaseFigure}{\resizebox{\linewidth}{!}{%' + '\n'
        + picture + '\n}}\n')
    figure_source = re.sub(r'\b(ink|blue|orange|purple|region|fail|cover)\b', lambda m: 'qone' + m[0], figure_source)
    (OUT / 'figures/q1_case.tex').write_text(figure_source)
    (OUT / 'latexmkrc').write_text("$pdf_mode = 5;\n$xelatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';\n")
    (OUT / 'README.md').write_text('''# 问题一：当前 Markdown 的 LaTeX 版

主文件：`main.tex`。编译器：XeLaTeX。建议 TeX Live 2025 或 2026。

在现有 Overleaf 项目上传 `main.tex`、`latexmkrc` 和 `figures/`，将主文档设为 `main.tex`、编译器设为 XeLaTeX。
旧版文件在本地保留，上传前请保留 Overleaf 项目中的其他章节和原主文件。

本地编译：

```sh
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

`source_original.md` 是转换时的原稿快照。根目录 Markdown 未改动。
TeX 中已将原稿的最小包围圆半径笔误 `5*sqrt(13)/6` 修正为 `65/6`，约 10.833 m。
正文插入了当前三测点算例图；`figures/q1_case.tex` 为可编辑 TikZ 图源，无外部图片依赖，可通过文本文件 MCP 同步。
''')

    args = ['/Library/TeX/texbin/xelatex', '-interaction=nonstopmode', '-halt-on-error',
            '-file-line-error', '-output-directory=' + str(QA), 'main.tex']
    for n in (1, 2):
        result = subprocess.run(args, cwd=OUT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (QA / f'build-{n}.txt').write_text(result.stdout)
        if result.returncode:
            print(result.stdout[-5000:])
            raise RuntimeError('XeLaTeX failed')
    log = (QA / 'main.log').read_text()
    issues = re.findall(r'^.*(?:Overfull|Underfull|Missing character|Warning|undefined).*$', log, re.M)
    shutil.copy2(QA / 'main.pdf', OUT / 'main.pdf')
    subprocess.run(['/opt/homebrew/bin/pdftoppm', '-r', '95', '-png', str(OUT / 'main.pdf'), str(QA / 'page')], check=True)
    assert SOURCE.read_text() == source, 'Markdown changed while building; rebuild required.'
    for name, sha in previous.items():
        assert digest(ROOT / name) == sha, 'An earlier edition changed: ' + name

    report = {'source_sha256': digest(SOURCE), 'tex_sha256': digest(OUT / 'main.tex'),
              'pdf_sha256': digest(OUT / 'main.pdf'), 'display_formula_count': len(formulas),
              'figure_count': 1, 'corrections': corrections, 'warnings': issues,
              'source_prose_preserved': True, 'earlier_editions_preserved': True,
              'visual_review_passed': False}
    (OUT / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    archive = ROOT / 'output/overleaf/问题一_当前版.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for filename in ('main.tex', 'latexmkrc', 'README.md', 'source_original.md',
                         'figures/q1_case.tex'):
            z.write(OUT / filename, filename)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Overleaf archive:', archive)
    print('Visual QA pages:', QA)


if __name__ == '__main__':
    main()
