"""Synchronize the model Markdown to LaTeX and rebuild its PDF with XeLaTeX."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '问题一_定位区域模型.md'
TEX = ROOT / 'output/pdf/问题一_定位区域模型.tex'
PDF = TEX.with_suffix('.pdf')


def escaped(text):
    mapping = {'&': r'\&', '%': r'\%', '#': r'\#', '_': r'\_',
               '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
               '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'}
    return ''.join(mapping.get(c, c) for c in text).replace('——', '---').replace('—', '---')


def inline(text):
    text = re.sub(r'"([^"]*)"', r'“\1”', text)
    parts = []
    for token in re.split(r'(\$[^$]+\$|\*\*.*?\*\*)', text):
        if token.startswith('$') and token.endswith('$'):
            parts.append(token)
        elif token.startswith('**') and token.endswith('**'):
            parts.append(r'\textbf{' + inline(token[2:-2]) + '}')
        else:
            parts.append(escaped(token))
    return ''.join(parts)


def algorithm(line):
    for a, b in [('S_i', '$S_i$'), ('θ_i', r'$\theta_i$'),
                 ('i = 1,…,n', r'$i=1,\dots,n$'),
                 ('±x、±y', r'$\pm x$、$\pm y$'),
                 ('(A, B)', '$(A,B)$'), ('D = |AB|', '$D=|AB|$')]:
        line = line.replace(a, b)
    line = re.sub(r'(?<![A-Za-z_$])D(?![A-Za-z_$])', '$D$', line)
    return (r'\noindent\hspace*{1.5em}' if line.startswith('   ')
            else r'\noindent\hangindent=1.5em\hangafter=1 ') + inline(line.lstrip()) + r'\par'


def convert(source):
    lines = source.splitlines()
    body = []
    i = 1
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            body.append('')
            i += 1
        elif re.match(r'^#{2,3} ', line):
            level = 'subsection' if line.startswith('### ') else 'section'
            heading = line.split(' ', 1)[1]
            if heading == '四、符号说明':
                body.append(r'\Needspace{21\baselineskip}')
            body.append('\\' + level + '*{' + inline(heading) + '}')
            body.append(r'\addcontentsline{toc}{' + level + '}{' + inline(heading) + '}')
            i += 1
        elif line.strip() == '$$':
            i += 1
            math = []
            while i < len(lines) and lines[i].strip() != '$$':
                math.append(lines[i])
                i += 1
            body.extend([r'\[', '\n'.join(math), r'\]'])
            i += 1
        elif line.startswith('```'):
            i += 1
            body.extend([r'\begin{tcolorbox}[colback=black!2,colframe=black!22,boxrule=0.4pt,arc=0pt,left=9pt,right=9pt,top=7pt,bottom=7pt]',
                         r'\small\setlength{\parskip}{2pt}'])
            while i < len(lines) and not lines[i].startswith('```'):
                body.append(algorithm(lines[i]))
                i += 1
            body.append(r'\end{tcolorbox}')
            i += 1
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [x.strip() for x in re.split(r'(?<!\\)\|', lines[i].strip().strip('|'))]
                if not all(re.fullmatch(r':?-+:?', x) for x in cells):
                    rows.append(cells)
                i += 1
            if len(rows[0]) == 3:
                spec = r'>{\centering\arraybackslash}p{0.25\linewidth}>{\centering\arraybackslash}p{0.24\linewidth}>{\centering\arraybackslash}X'
            else:
                spec = r'>{\raggedright\arraybackslash}p{0.52\linewidth}X'
            body.extend([r'\begin{center}', r'\renewcommand{\arraystretch}{1.3}',
                         r'\begin{tabularx}{0.98\linewidth}{' + spec + '}', r'\toprule'])
            body.append(' & '.join(r'\textbf{' + inline(x) + '}' for x in rows[0]) + r' \\')
            body.append(r'\midrule')
            body.extend(' & '.join(inline(x) for x in row) + r' \\' for row in rows[1:])
            body.extend([r'\bottomrule', r'\end{tabularx}', r'\end{center}'])
        elif re.match(r'^\d+\. ', line) or line.startswith('- '):
            numbered = bool(re.match(r'^\d+\. ', line))
            env = 'enumerate' if numbered else 'itemize'
            pattern = r'^\d+\. ' if numbered else r'^- '
            body.append(r'\begin{' + env + '}')
            while i < len(lines) and re.match(pattern, lines[i]):
                body.append(r'\item ' + inline(re.sub(pattern, '', lines[i])))
                i += 1
            body.append(r'\end{' + env + '}')
        else:
            if line.startswith('**表 '):
                body.append(r'\Needspace{19\baselineskip}')
                body.extend([r'\begin{center}', inline(line), r'\end{center}'])
            else:
                body.append(inline(line))
            i += 1
    return '\n'.join(body)


def main():
    source = SOURCE.read_text(encoding='utf-8')
    title = source.splitlines()[0][2:]
    old_tex = TEX.read_text(encoding='utf-8')
    preamble = old_tex.split(r'\begin{document}', 1)[0]
    preamble = re.sub(r'\\hypersetup\{pdftitle=\{[^{}]*\}\}',
                      lambda _: r'\hypersetup{pdftitle={' + title + '}}', preamble)
    preamble = re.sub(r'\\fancyhead\[L\]\{\\small [^{}]*\}',
                      lambda _: r'\fancyhead[L]{\small 问题一：干扰源定位区域}', preamble)
    body = convert(source)
    tex = (preamble + r'\begin{document}' + '\n' + r'\thispagestyle{plain}' + '\n'
           + r'\begin{center}{\LARGE\bfseries ' + inline(title)
           + r'\par}\end{center}' + '\n' + r'\vspace{0.4em}' + '\n'
           + body + '\n' + r'\end{document}' + '\n')
    formulas = re.findall(r'\$\$\s*\n(.*?)\n\$\$', source, re.S)
    assert all(f in tex for f in formulas)
    assert re.findall(r'\\tag\{(\d+)\}', source) == re.findall(r'\\tag\{(\d+)\}', tex)
    # Bookmark strings intentionally repeat headings; ignore these in the text audit.
    clean_body = re.sub(r'\\addcontentsline\{toc\}\{(?:sub)?section\}\{[^{}]*\}', '', body)
    han = lambda t: ''.join(re.findall('[\u4e00-\u9fff]', t))
    assert han(source.split('\n', 1)[1]) == han(clean_body)
    TEX.write_text(tex, encoding='utf-8')
    scratch = Path(tempfile.mkdtemp(prefix='model-latex-', dir='/private/tmp'))
    args = ['/Library/TeX/texbin/xelatex', '-interaction=nonstopmode', '-halt-on-error',
            '-file-line-error', '-output-directory=' + str(scratch), str(TEX)]
    for n in (1, 2):
        with (scratch / f'build-{n}.txt').open('w') as output:
            subprocess.run(args, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)
    log = (scratch / (TEX.stem + '.log')).read_text()
    issues = re.findall(r'^.*(?:Overfull|Underfull|Missing character|Warning|undefined).*$' , log, re.M)
    if issues:
        print('\n'.join(issues))
        print('QA directory:', scratch)
        raise RuntimeError('Inspect the LaTeX warnings before delivering the PDF.')
    result = scratch / PDF.name
    subprocess.run(['/opt/homebrew/bin/pdftoppm', '-r', '105', '-png', str(result), str(scratch / 'page')], check=True)
    subprocess.run(['/opt/homebrew/bin/pdftotext', str(result), str(scratch / 'extracted.txt')], check=True)
    assert SOURCE.read_text(encoding='utf-8') == source, 'Markdown changed during compilation; rebuild again.'
    shutil.copy2(result, PDF)
    print(json.dumps({'pdf': str(PDF), 'tex': str(TEX), 'qa_directory': str(scratch),
                      'display_formulas': len(formulas), 'source_sha256': hashlib.sha256(source.encode()).hexdigest()},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
