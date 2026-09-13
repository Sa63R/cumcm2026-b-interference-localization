"""Build a separate illustrated edition; never overwrite the previous TeX/PDF."""
from pathlib import Path
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile

import rebuild_model_pdf as markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf/新版_含示意图'
OLD_TEX = ROOT / 'output/pdf/问题一_定位区域模型.tex'
OLD_PDF = OLD_TEX.with_suffix('.pdf')
SOURCE = ROOT / '问题一_定位区域模型.md'
SNAPSHOT = OUT / '问题一_新版_原稿.md'
TEX = OUT / '问题一_新版_含示意图.tex'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def figure(command, caption, label):
    return ('\n% BEGIN ADDED FIGURE\n'
            + r'\begin{center}\begin{minipage}{\linewidth}' + '\n' + r'\centering' + '\n'
            + '\\' + command + '\n' + r'\captionof{figure}{' + caption + '}\n'
            + r'\label{' + label + '}\n' + r'\end{minipage}\end{center}' + '\n'
            + '% END ADDED FIGURE\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preserved = {str(p.relative_to(ROOT)): sha(p) for p in (OLD_TEX, OLD_PDF)}
    # Keep the input used for this edition stable across layout-only rebuilds.
    if not SNAPSHOT.exists():
        shutil.copy2(SOURCE, SNAPSHOT)
    source = SNAPSHOT.read_text(encoding='utf-8')
    body = markdown.convert(source)
    for line in source.splitlines():
        if line.startswith('> '):
            body = body.replace(markdown.inline(line),
                                r'\begin{quote}' + '\n' + markdown.inline(line[2:]) + '\n' + r'\end{quote}')
    # Audit original content before the addition of figure descriptions.
    clean = re.sub(r'\\addcontentsline\{toc\}\{(?:sub)?section\}\{[^{}]*\}', '', body)
    han = lambda s: ''.join(re.findall('[\u4e00-\u9fff]', s))
    assert han(source.split('\n', 1)[1]) == han(clean)
    base_body = body

    f1 = figure('AngleDiagram',
        '单次测向的误差角域。为便于辨认，图中误差角作放大示意；模型计算仍取半张角 $\\varepsilon=1^\\circ$。',
        'fig:angle-domain')
    anchor = markdown.inline('将式 (1) 展开，一次测向对应两个线性不等式：')
    assert anchor in body
    body = body.replace(anchor, f1 + '\n' + anchor, 1)

    f2 = figure('IntersectionDiagram',
        '两次测向的角域交会与定位区域。采用第 5.3 节算例数据，边界角按真实的 $\\pm1^\\circ$ 绘制；两幅子图各自保持等比例坐标，右图为交会区域放大。',
        'fig:intersection')
    anchor = r'\subsection*{5.2 定位区域及其直径的求解}'
    assert anchor in body
    body = body.replace(anchor, f2 + '\n' + anchor, 1)

    f3 = figure('CoverageDiagram',
        '直径圆覆盖失败与最小包围圆的修正。红色虚线圆在上顶点 $V_3$ 处留下约 $0.307$ m 的覆盖缺口；蓝色实线圆覆盖全部顶点。左图按等比例绘制，右图单独放大顶部差异。',
        'fig:coverage')
    anchor = r'\subsection*{5.5 模型说明}'
    assert anchor in body
    body = body.replace(anchor, f3 + '\n' + anchor, 1)

    # Separate the abstract from the body while retaining all supplied headings.
    body = body.replace(r'\section*{一、问题背景与重述}',
                        r'\clearpage' + '\n' + r'\section*{一、问题背景与重述}', 1)
    body = body.replace(r'\section*{摘要}', r'\section*{\centering 摘要}', 1)
    body = body.replace(r'\section*{三、模型假设}',
                        r'\Needspace{10\baselineskip}' + '\n' + r'\section*{三、模型假设}', 1)

    t = math.tan(math.radians(43.5))
    u = math.tan(math.radians(45.5))
    a = 500 * (u - t) / (u + t)
    middle = 1000 * t * u / (u + t)
    bottom, top = 500 * t, 500 * u
    b = top - middle
    shift = (b*b - a*a)/(2*b)
    radius = (a*a + b*b)/(2*b)
    cover_y = middle + shift
    vertices = [(0,bottom), (a,middle), (0,top), (-a,middle)]
    distances = [math.dist(x,y) for i,x in enumerate(vertices) for y in vertices[i+1:]]
    assert abs(max(distances) - 2*a) < 1e-10
    assert all(math.hypot(x,y-cover_y) <= radius+1e-10 for x,y in vertices)
    for x,y in vertices:
        assert t*(x+500) <= y+1e-10 and y <= u*(x+500)+1e-10
        assert t*(500-x) <= y+1e-10 and y <= u*(500-x)+1e-10
    assert b-a > 0
    values = dict(TanLower=t, TanUpper=u, TanCenter=math.tan(math.radians(44.5)),
                  RayLowerEnd=800*t, RayUpperEnd=800*u, HalfD=a,
                  MiddleY=middle, BottomY=bottom, TopY=top,
                  CoverY=cover_y, CoverR=radius, CircleTop=middle+a)
    data = '\n'.join('\\def\\' + key + '{' + f'{value:.12f}' + '}' for key,value in values.items())

    title = source.splitlines()[0][2:]
    preamble = OLD_TEX.read_text().split(r'\begin{document}', 1)[0]
    preamble = re.sub(r'\\hypersetup\{pdftitle=\{[^{}]*\}\}',
                      lambda _: r'\hypersetup{pdftitle={' + title + '}}', preamble)
    preamble = re.sub(r'\\fancyhead\[L\]\{\\small [^{}]*\}',
                      lambda _: r'\fancyhead[L]{\small 无源测向快速定位模型}', preamble)
    preamble += '\n% BEGIN EDITABLE VECTOR FIGURES\n' + data + '\n'
    preamble += (ROOT/'scripts/model_illustrations.tex').read_text()
    preamble += '\n% END EDITABLE VECTOR FIGURES\n'
    title_tex = markdown.inline(title).replace('的无源', r'的\par\vspace{0.25em}无源', 1)
    tex = (preamble + '\n' + r'\begin{document}' + '\n' + r'\thispagestyle{plain}' + '\n'
           + r'\begin{center}{\LARGE\bfseries ' + title_tex + r'\par}\end{center}' + '\n'
           + r'\vspace{0.4em}' + '\n% BEGIN DOCUMENT BODY\n'
           + body + '\n% END DOCUMENT BODY\n' + r'\end{document}' + '\n')
    formulas = re.findall(r'\$\$\s*\n(.*?)\n\$\$',source,re.S)
    assert all(f in tex for f in formulas)
    assert re.findall(r'\\tag\{(\d+)\}',source) == re.findall(r'\\tag\{(\d+)\}',tex)
    TEX.write_text(tex, encoding='utf-8')
    scratch = Path(tempfile.mkdtemp(prefix='model-illustrated-', dir='/private/tmp'))
    args = ['/Library/TeX/texbin/xelatex', '-interaction=nonstopmode', '-halt-on-error',
            '-file-line-error', '-output-directory=' + str(scratch), str(TEX)]
    print('QA directory:', scratch, flush=True)
    for n in (1,2):
        with (scratch/f'build-{n}.txt').open('w') as out:
            result = subprocess.run(args,cwd=OUT,stdout=out,stderr=subprocess.STDOUT)
        if result.returncode:
            print((scratch/f'build-{n}.txt').read_text()[-4500:])
            raise RuntimeError('XeLaTeX compilation failed')
    log = (scratch/(TEX.stem+'.log')).read_text()
    issues = re.findall(r'^.*(?:Overfull|Underfull|Missing character|Warning|undefined).*$' ,log,re.M)
    if issues:
        print('\n'.join(issues))
    pdf = scratch/TEX.with_suffix('.pdf').name
    subprocess.run(['/opt/homebrew/bin/pdftoppm','-r','100','-png',str(pdf),str(scratch/'page')],check=True)
    subprocess.run(['/opt/homebrew/bin/pdftotext',str(pdf),str(scratch/'extracted.txt')],check=True)
    # Retain the converted original body for content-preservation checks.
    (scratch/'source_body.tex').write_text(base_body)
    for name, digest in preserved.items():
        assert sha(ROOT/name) == digest, 'Previous edition changed: ' + name
    manifest = dict(source_sha256=sha(SNAPSHOT), original_files=preserved,
                    display_formulas=len(formulas), formula_tags=re.findall(r'\\tag\{(\d+)\}',source),
                    figure_count=3, geometry=values,
                    vertex_excess=b-a, center_shift=shift,
                    qa_directory=str(scratch), warnings=issues)
    (OUT/'build_verification.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    shutil.copy2(pdf,TEX.with_suffix('.pdf'))
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
